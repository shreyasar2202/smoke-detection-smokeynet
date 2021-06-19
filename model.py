"""
Created by: Anshuman Dewangan
Date: 2021

Description: Main file with PyTorch Lightning LightningModule. Defines model, forward pass, and training.
"""

# Torch imports
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pytorch_lightning as pl
import torchvision
import torchmetrics

# Other package imports
import numpy as np
import datetime
import csv
from pathlib import Path

# File imports
import util_fns

    

class ResNet50Backbone(nn.Module):
    """
    Description: Simple model with ResNet backbone and a few linear layers
    """
    def __init__(self, series_length, freeze_backbone=True):
        super().__init__()

        resnet = torchvision.models.resnet50(pretrained=True)
        resnet.fc = nn.Identity()

        if freeze_backbone:
            for param in resnet.parameters():
                param.requires_grad = False

        self.conv = resnet

        self.fc1 = nn.Linear(in_features=series_length * 2048, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=64)
        self.fc3 = nn.Linear(in_features=64, out_features=1)

    def forward(self, x):
        x = x.float()
        batch_size, num_tiles, series_length, num_channels, height, width = x.size()

        x = x.view(batch_size * num_tiles * series_length, num_channels, height, width)
        x = self.conv(x) # [batch_size * num_tiles * series_length, 2048]

        x = x.view(batch_size, num_tiles, -1) # [batch_size, num_tiles, series_length * 2048]
        x = F.relu(self.fc1(x)) # [batch_size, num_tiles, 512]
        x = F.relu(self.fc2(x)) # [batch_size, num_tiles, 64]
        x = self.fc3(x) # [batch_size, num_tiles, 1]

        return x

    
#####################
## Training Model
#####################

class LightningModel(pl.LightningModule):

    ### Initialization ###

    def __init__(self,
                 model,
                 learning_rate=0.001,
                 lr_schedule=True,
                 parsed_args=None):
        """
        Args:
            - model (torch.nn.Module): model to use for training/evaluation
            - learning_rate (float): learning rate for optimizer
            - lr_schedule (bool): should ReduceLROnPlateau learning rate schedule be used?
            - parsed_args (dict): full dict of parsed args to log as hyperparameters

        Other Attributes:
            - example_input_array (tensor): example of input to log computational graph in tensorboard
            - criterion (obj): objective function used to calculate loss
            - self.metrics (dict): contains many properties related to logging metrics, including:
                - torchmetrics (torchmetrics module): keeps track of metrics per step and per epoch
                - split (list of str): name of splits e.g. ['train/', 'val/', 'test/']
                - category (list of str): metric subcategories
                    - 'tile_': metrics on per-tile basis
                    - 'image-gt': labels based on if image name has '+' in it)
                    - 'image_xml': labels based on if image has XML file associated
                    - 'image-pos-tile': labels based on if image has at least one positive tile 
                - name (list of str): name of metric e.g. ['accuracy', 'precision', ...]
                - function (list of torchmetrics functions): used to initiate torchmetric modules
        """
        super().__init__()

        # Initialize model
        self.model = model
        self.example_input_array = torch.randn((parsed_args.batch_size,108,parsed_args.series_length, 3, 224, 224))

        # Initialize model params
        self.criterion = nn.BCEWithLogitsLoss()
        self.learning_rate = learning_rate
        self.lr_schedule = lr_schedule

        # Save hyperparameters
        self.save_hyperparameters('learning_rate', 'lr_schedule')
        self.save_hyperparameters(parsed_args)

        # Initialize evaluation metrics
        self.metrics = {}
        self.metrics['torchmetric'] = {}
        self.metrics['split']       = ['train/', 'val/', 'test/']
        self.metrics['category']    = ['tile_', 'image-gt_', 'image-xml_', 'image-pos-tile_']
        self.metrics['name']        = ['accuracy', 'precision', 'recall', 'f1']
        self.metrics['function']    = [torchmetrics.Accuracy, torchmetrics.Precision, torchmetrics.Recall, torchmetrics.F1]

        for split in self.metrics['split']:
            for title in self.metrics['category']:
                for label, func in zip(self.metrics['name'], self.metrics['function']):
                    self.metrics['torchmetric'][split+title+label] = func(mdmc_average='global') \
                    if title == self.metrics['category'][0] else func()

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        
        if self.lr_schedule:
            # Includes learning rate scheduler
            scheduler = ReduceLROnPlateau(optimizer)
            return {"optimizer": optimizer,
                    "lr_scheduler": scheduler,
                    "monitor": "val_loss"}
        else:
            return optimizer

    def forward(self, x):
        return self.model(x).squeeze(dim=2) # [batch_size, num_tiles]

    ### Step Functions ###

    def step(self, batch, split):
        image_name, x, tile_labels, ground_truth_labels, has_xml_labels, has_positive_tiles = batch

        # Compute loss
        outputs = self(x)
        loss = self.criterion(outputs, tile_labels)

        # Compute predictions
        tile_preds = util_fns.predict_tile(outputs)
        image_preds = util_fns.predict_image_from_tile_preds(tile_preds)

        # Compute metrics
        for label in self.metrics['name']:
            self.metrics['torchmetric'][split+self.metrics['category'][0]+label].to(self.device)(tile_preds, tile_labels.int())
            self.metrics['torchmetric'][split+self.metrics['category'][1]+label].to(self.device)(image_preds, ground_truth_labels)
            self.metrics['torchmetric'][split+self.metrics['category'][2]+label].to(self.device)(image_preds, has_xml_labels)
            self.metrics['torchmetric'][split+self.metrics['category'][3]+label].to(self.device)(image_preds, has_positive_tiles)

        # Log loss (on_step only if split='train')
        self.log(split+'loss', loss, on_step=(split==self.metrics['split'][0]),on_epoch=True)

        # Log other metrics
        for title in self.metrics['category']:
            for label in self.metrics['name']:
                name = split+title+label
                self.log(name, self.metrics['torchmetric'][name], on_step=False, on_epoch=True)

        return image_name, outputs, loss, tile_preds, image_preds

    def training_step(self, batch, batch_idx):
        image_name, outputs, loss, tile_preds, image_preds = self.step(batch, self.metrics['split'][0])
        return loss

    def validation_step(self, batch, batch_idx):
        image_name, outputs, loss, tile_preds, image_preds = self.step(batch, self.metrics['split'][1])

    def test_step(self, batch, batch_idx):
        image_name, outputs, loss, tile_preds, image_preds = self.step(batch, self.metrics['split'][2])
        return image_name, tile_preds, image_preds

    ### Test Metric Logging ###
    
    def test_epoch_end(self, test_step_outputs):
        fire_preds = {}
        
        with open(self.logger.log_dir+'/image_preds.csv', 'w') as image_preds_csv:
            image_preds_csv_writer = csv.writer(image_preds_csv)

            # Loop through batch
            for image_names, tile_preds, image_preds in test_step_outputs:
                # Loop through entry in batch
                for image_name, tile_pred, image_pred in zip(image_names, tile_preds, image_preds):
                    fire_name = util_fns.get_fire_name(image_name)
                    image_pred = image_pred.item()
                    
                    # Save image predictions
                    image_preds_csv_writer.writerow([image_name, image_pred])

                    # Save tile predictions
                    tile_preds_path = self.logger.log_dir+'/tile_preds/'+fire_name
                    Path(tile_preds_path).mkdir(parents=True, exist_ok=True)
                    np.save(self.logger.log_dir+'/tile_preds/'+\
                            image_name+\
                            '.npy', tile_pred.cpu().numpy())
                    
                    if fire_name not in fire_preds:
                        fire_preds[fire_name] = [0] * 81
