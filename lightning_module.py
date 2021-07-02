"""
Created by: Anshuman Dewangan
Date: 2021

Description: PyTorch Lightning LightningModule that defines optimizers, training step and metrics.
"""

# Torch imports
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pytorch_lightning as pl
import torchmetrics

# Other package imports
import numpy as np
import datetime
import csv
from pathlib import Path

# File imports
import util_fns
from main_model import MainModel


class LightningModule(pl.LightningModule):

    #####################
    ## Initialization
    #####################

    def __init__(self,
                 model,
                 
                 optimizer_type='AdamW',
                 optimizer_weight_decay=0.001,
                 learning_rate=0.0001,
                 lr_schedule=True,
                 
                 series_length=1,
                 parsed_args=None):
        """
        Args:
            - model (torch.nn.Module): model to use for training/evaluation
            
            - optimizer_type (str): type of optimizer to use. Options: [AdamW] [SGD]
            - optimizer_weight_decay (float): weight decay to use with optimizer
            - learning_rate (float): learning rate for optimizer
            - lr_schedule (bool): should ReduceLROnPlateau learning rate schedule be used?
            
            - series_length (int): number of sequential video frames to process during training
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
        """
        print("Initializing LightningModule...")
        super().__init__()
        
        # Initialize model
        self.model = model
        
        # Initialize optimizer params
        self.optimizer_type = optimizer_type
        self.optimizer_weight_decay = optimizer_weight_decay
        self.learning_rate = learning_rate
        self.lr_schedule = lr_schedule
        
        # Save hyperparameters
        self.save_hyperparameters(parsed_args)
        self.save_hyperparameters('learning_rate')
        
        # ASSUMPTION: num_tiles=45, num_channels=3, image_height=224, image_width=224 
        self.example_input_array = torch.randn((1,45,series_length, 3, 224, 224))

        # Initialize evaluation metrics
        self.metrics = {}
        self.metrics['torchmetric'] = {}
        self.metrics['split']       = ['train/', 'val/', 'test/']
        self.metrics['category']    = ['tile_', 'image-gt_', 'image-xml_', 'image-pos-tile_']
        self.metrics['name']        = ['accuracy', 'precision', 'recall', 'f1']
        
        # WARNING: torchmetrics has a very weird way of calculating metrics! 
        for split in self.metrics['split']:
            # Use mdmc_average='global' for tile_preds only
            self.metrics['torchmetric'][split+self.metrics['category'][0]+self.metrics['name'][0]] = torchmetrics.Accuracy(mdmc_average='global')
            # Recall and Precision are flipped in torchmetrics compared to sklearn
            self.metrics['torchmetric'][split+self.metrics['category'][0]+self.metrics['name'][1]] = torchmetrics.Recall(multiclass=False, mdmc_average='global')
            self.metrics['torchmetric'][split+self.metrics['category'][0]+self.metrics['name'][2]] = torchmetrics.Precision(multiclass=False, mdmc_average='global')
            self.metrics['torchmetric'][split+self.metrics['category'][0]+self.metrics['name'][3]] = torchmetrics.F1(multiclass=False, mdmc_average='global')
            
            for category in self.metrics['category'][1:]:
                self.metrics['torchmetric'][split+category+self.metrics['name'][0]] = torchmetrics.Accuracy()
                # Recall and Precision are flipped in torchmetrics compared to sklearn
                self.metrics['torchmetric'][split+category+self.metrics['name'][1]] = torchmetrics.Recall(multiclass=False)
                self.metrics['torchmetric'][split+category+self.metrics['name'][2]] = torchmetrics.Precision(multiclass=False)
                self.metrics['torchmetric'][split+category+self.metrics['name'][3]] = torchmetrics.F1(multiclass=False)
            
        print("Initializing LightningModule Complete.")

    def configure_optimizers(self):
        if self.optimizer_type == 'SGD':
            optimizer = torch.optim.SGD(self.model.parameters(), 
                                        lr=self.learning_rate, 
                                        momentum=0.9, 
                                        weight_decay=self.optimizer_weight_decay)
            print('Optimizer: SGD')
        elif self.optimizer_type == 'AdamW':
            optimizer = torch.optim.AdamW(self.model.parameters(), 
                                          lr=self.learning_rate, 
                                          weight_decay=self.optimizer_weight_decay)
            print('Optimizer: AdamW')
        
        if self.lr_schedule:
            # Includes learning rate scheduler
            scheduler = ReduceLROnPlateau(optimizer, 
                                          min_lr=self.learning_rate*1e-5, 
                                          patience=1,
                                          threshold=0.1,
                                          verbose=True)
            return {"optimizer": optimizer,
                    "lr_scheduler": scheduler,
                    "monitor": "val/loss"}
        else:
            return optimizer

    def forward(self, x):
        return self.model(x)
        
    #####################
    ## Step Functions
    #####################

    def step(self, batch, split):
        """Description: Takes a batch, calculates forward pass, losses, and predictions, and logs metrics"""
        image_names, x, tile_labels, ground_truth_labels, has_xml_labels, has_positive_tiles = batch

        # Compute outputs, loss, and predictions
        outputs, losses, total_loss, tile_preds, image_preds = self.model.forward_pass(x, tile_labels, ground_truth_labels)
        
        # Log losses (on_step only if split='train')
        for i, loss in enumerate(losses):
            self.log(split+'loss_'+str(i), tile_loss, on_step=(split==self.metrics['split'][0]),on_epoch=True)
        
        self.log(split+'loss', total_loss, on_step=(split==self.metrics['split'][0]),on_epoch=True)
        self.log('general/learning_rate', self.learning_rate, on_step=False, on_epoch=True)

        # Calculate & log evaluation metrics
        for category, args in zip(self.metrics['category'], 
                                  ((tile_preds, tile_labels.int()), 
                                   (image_preds, ground_truth_labels), 
                                   (image_preds, has_xml_labels), 
                                   (image_preds, has_positive_tiles))
                                 ):
            for name in self.metrics['name']:
                if args[0] is not None:
                    # Have to move the metric to self.device 
                    metric = self.metrics['torchmetric'][split+category+name].to(self.device)(args[0], args[1])
                    self.log(split+category+name, metric, on_step=False, on_epoch=True)
        
        return image_names, total_loss, tile_preds, image_preds

    def training_step(self, batch, batch_idx):
        image_names, loss, tile_preds, image_preds = self.step(batch, self.metrics['split'][0])
        return loss

    def validation_step(self, batch, batch_idx):
        image_names, loss, tile_preds, image_preds = self.step(batch, self.metrics['split'][1])

    def test_step(self, batch, batch_idx):
        image_names, loss, tile_preds, image_preds = self.step(batch, self.metrics['split'][2])
        return image_names, tile_preds, image_preds

    
    ########################
    ## Test Metric Logging
    ########################

    def test_epoch_end(self, test_step_outputs):
        """
        Description: saves predictions to .txt files and computes additional evaluation metrics for test set (e.g. time-to-detection)
        Args:
            - test_step_outputs (list of {image_names, tile_preds, image_preds}): what's returned from test_step
        """
        
        print("Computing Test Evaluation Metrics...")
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
        
        if len(fire_preds) == 0:
            print("Could not compute Test Evaluation Metrics.")
            return
        
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
        
        print("Computing Test Evaluation Metrics Complete.")
