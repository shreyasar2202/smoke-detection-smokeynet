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
    Args:
        - series_length: number of sequential video frames to process during training
        - freeze_backbone: disables freezing of layers on pre-trained backbone
    
    Other Attributes:
        - criterion (obj): objective function used to calculate loss
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
        
        self.criterion = nn.BCEWithLogitsLoss()

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
    
    def compute_loss(self, outputs, tile_labels):
        return self.criterion(outputs, tile_labels)

    
#####################
## Training Model
#####################

class LightningModel(pl.LightningModule):

    ### Initialization ###

    def __init__(self,
                 model=None,
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
        print("Initializing LightningModel... ")
        super().__init__()

        # Initialize model
        self.model = model
        # ASSUMPTION: num_tiles=54, num_channels=3, image_height=224, image_width=224 
        self.example_input_array = torch.randn((parsed_args.batch_size,54,parsed_args.series_length, 3, 224, 224))

        # Initialize model params
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
        
        # WARNING: torchmetrics has a very weird way of calculating metrics! 
        for split in self.metrics['split']:
            # Use mdmc_average='global' for tile_preds only
            self.metrics['torchmetric'][split+self.metrics['category'][0]+'accuracy'] = torchmetrics.Accuracy(mdmc_average='global')
            self.metrics['torchmetric'][split+self.metrics['category'][0]+'precision-recall'] = torchmetrics.Precision(num_classes=2, average='none', mdmc_average='global')
            self.metrics['torchmetric'][split+self.metrics['category'][0]+'f1'] = torchmetrics.F1(num_classes=2, average='none', mdmc_average='global')
            
            for category in self.metrics['category'][1:]:
                self.metrics['torchmetric'][split+category+'accuracy'] = torchmetrics.Accuracy()
                self.metrics['torchmetric'][split+category+'precision-recall'] = torchmetrics.Precision(num_classes=2, average='none')
                self.metrics['torchmetric'][split+category+'f1'] = torchmetrics.F1(num_classes=2, average='none')
            
        print("Initializing LightningModel Complete. ")

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

        
    #####################
    ## Model Functions
    #####################
    
    def forward(self, x):
        """
        Description: compute forward pass of all model_parts as well as adds additional layers
        Args:
            - x (tensor): input provided by dataloader
        Returns:
            - outputs (tensor): outputs after going through forward passes of all layers
        """
        
        outputs = self.model(x).squeeze(dim=2) # [batch_size, num_tiles]
        return outputs
    
    def compute_loss(self, outputs, tile_labels, ground_truth_labels):
        """
        Description: computes total loss by computing loss of each sub-module
        Args:
            - outputs (tensor): outputs after going through forward passes of all layers
            - tile_labels: see metadata.pkl documentation for more info
            - ground_truth_labels: see metadata.pkl documentation for more info
        Returns:
            - loss (float): total loss for model
        """
        
        loss = self.model.compute_loss(outputs, tile_labels)
        return loss
    
    def compute_predictions(self, outputs):
        """
        Description: computes tile-level and image-level predictions
        Args:
            - outputs (tensor): outputs after going through forward passes of all layers
        Returns:
            - tile_preds (tensor): 0/1 prediction for each tile in each image of the batch
            - image_preds (tensor): 0/1 prediction for each image in the batch
        """
        
        tile_preds = util_fns.predict_tile(outputs)
        image_preds = util_fns.predict_image_from_tile_preds(tile_preds)
        
        return tile_preds, image_preds

    #####################
    ## Step Functions
    #####################

    def step(self, batch, split):
        image_names, x, tile_labels, ground_truth_labels, has_xml_labels, has_positive_tiles = batch

        # Compute outputs, loss, and predictions
        outputs = self(x)
        loss = self.compute_loss(outputs, tile_labels, ground_truth_labels)
        tile_preds, image_preds = self.compute_predictions(outputs)
        
        # Log loss (on_step only if split='train')
        self.log(split+'loss', loss, on_step=(split==self.metrics['split'][0]),on_epoch=True)

        # Calculate & log evaluation metrics
        for category, args in zip(self.metrics['category'], 
                                  ((tile_preds, tile_labels.int()), 
                                   (image_preds, ground_truth_labels), 
                                   (image_preds, has_xml_labels), 
                                   (image_preds, has_positive_tiles))
                                 ):
            # Have to move the metric to self.device 
            accuracy = self.metrics['torchmetric'][split+category+'accuracy'].to(self.device)(args[0], args[1])
            # Returns a tuple: (precision, recall)
            precision_recall = self.metrics['torchmetric'][split+category+'precision-recall'].to(self.device)(args[0], args[1])
            # Return a tuple: (_, f1_score)
            f1 = self.metrics['torchmetric'][split+category+'f1'].to(self.device)(args[0], args[1])
            
            self.log(split+category+'accuracy', accuracy, on_step=False, on_epoch=True)
            self.log(split+category+'precision', precision_recall[0], on_step=False, on_epoch=True)
            self.log(split+category+'recall', precision_recall[1], on_step=False, on_epoch=True)
            self.log(split+category+'f1', f1[1], on_step=False, on_epoch=True)
        
        return image_names, outputs, loss, tile_preds, image_preds

    def training_step(self, batch, batch_idx):
        image_names, outputs, loss, tile_preds, image_preds = self.step(batch, self.metrics['split'][0])
        return loss

    def validation_step(self, batch, batch_idx):
        image_names, outputs, loss, tile_preds, image_preds = self.step(batch, self.metrics['split'][1])

    def test_step(self, batch, batch_idx):
        image_names, outputs, loss, tile_preds, image_preds = self.step(batch, self.metrics['split'][2])
        return image_names, tile_preds, image_preds

    
    #####################
    ## Test Metric Logging
    #####################

    def test_epoch_end(self, test_step_outputs):
        """
        Description: saves predictions to .txt files and computes additional evaluation metrics for test set (e.g. time-to-detection)
        Args:
            - test_step_outputs (list of {image_names, tile_preds, image_preds}): what's returned from test_step
        """
        
        print("Computing Test Evaluation Metrics... ")
        fire_preds_dict = {}
        
        ### Save predictions as .txt files ###
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
                    
                    # Add prediction to fire_preds_dict list
                    # ASSUMPTION: images are in order and test data has not been shuffled
                    if fire_name not in fire_preds_dict:
                        fire_preds_dict[fire_name] = []
                        
                    fire_preds_dict[fire_name].append(image_pred)
        
        ### Create data structures ###
        # Create data structure to store preds of all relevant fires
        fire_preds = []
        for fire in fire_preds_dict:
            # ASSUMPTION: only calculate statistics for fires with 81 images
            if len(fire_preds_dict[fire]) == 81:
                fire_preds.append(fire_preds_dict[fire])
                
        fire_preds = torch.as_tensor(fire_preds, dtype=int)

        negative_preds = fire_preds[:,:fire_preds.shape[1]//2] 
        positive_preds = fire_preds[:,fire_preds.shape[1]//2:]
        
        ### Compute & log metrics ###
        self.log(self.metrics['split'][2]+'negative_accuracy',
                 util_fns.calculate_negative_accuracy(negative_preds))
        self.log(self.metrics['split'][2]+'negative_accuracy_by_fire',
                 util_fns.calculate_negative_accuracy_by_fire(negative_preds))
        self.log(self.metrics['split'][2]+'positive_accuracy',
                 util_fns.calculate_positive_accuracy(positive_preds))
        self.log(self.metrics['split'][2]+'positive_accuracy_by_fire',
                 util_fns.calculate_positive_accuracy_by_fire(positive_preds))
        
        # Use 'global_step' to graph positive_accuracy_by_time and positive_cumulative_accuracy
        positive_accuracy_by_time = util_fns.calculate_positive_accuracy_by_time(positive_preds)
        positive_cumulative_accuracy = util_fns.calculate_positive_cumulative_accuracy(positive_preds)
        
        for i in range(len(positive_accuracy_by_time)):
            self.logger.experiment.add_scalar(self.metrics['split'][2]+'positive_accuracy_by_time',
                                             positive_accuracy_by_time[i], global_step=i)
            self.logger.experiment.add_scalar(self.metrics['split'][2]+'positive_cumulative_accuracy',
                                             positive_cumulative_accuracy[i], global_step=i)
        
        self.log(self.metrics['split'][2]+'average_time_to_detection',
                 util_fns.calculate_average_time_to_detection(positive_preds))
        
        print("Computing Test Evaluation Metrics Complete. ")
