"""
Created by: Anshuman Dewangan
Date: 2021

Description: Main model to use with lightning_module.py
"""

# Torch imports
import torch
from torch import nn
from torch.nn import functional as F
import torchvision

# Other imports
import numpy as np
import collections

# File imports
from model_components import *
import util_fns

    
#####################
## Main Model
#####################
    
class MainModel(nn.Module):
    """
    Description: Simple model with ResNet backbone and a few linear layers
    Args (see arg descriptions in main.py):
        - kwargs: any other args used in the models
    """
    def __init__(self, 
                 model_type_list=['RawToTile_MobileNetV3Large'], 
                 pretrain_epochs=None,
                 intermediate_supervision=True,
                 use_image_preds=False,
                 error_as_eval_loss=False,
                 
                 tile_loss_type='bce',
                 bce_pos_weight=36,
                 focal_alpha=0.25, 
                 focal_gamma=2,
                 image_loss_only=False,
                 image_pos_weight=1,
                 
                 
                 **kwargs):
        
        print("Initializing MainModel...")
        super().__init__()
        
        ### Initialize Model ###
        self.model_list = torch.nn.ModuleList()
                
        # Initializes each model using the class name and kwargs and adds it to model_list
        for model_type in model_type_list:
            self.model_list.append(globals()[model_type](**kwargs))
        
        # Saves the number of epochs to pretrain each part of the model
        if pretrain_epochs is not None:
            self.pretrain_epochs = np.array(pretrain_epochs).astype(int)
        else:
            self.pretrain_epochs = np.zeros(len(model_type_list)).astype(int)
            
        self.intermediate_supervision = intermediate_supervision
        self.error_as_eval_loss = error_as_eval_loss
        self.use_image_preds = use_image_preds
        self.image_loss_only = image_loss_only
        self.image_pos_weight = image_pos_weight
        

        ### Initialize Loss ###
        self.tile_loss = TileLoss(tile_loss_type=tile_loss_type,
                                  bce_pos_weight=bce_pos_weight,
                                  focal_alpha=focal_alpha, 
                                  focal_gamma=focal_gamma)
                
        print("Initializing MainModel Complete.")
        
    def forward(self, x):
        """Description: Maps raw inputs to outputs"""
        outputs = None
        
        for model in self.model_list:
            outputs, x = model(x, outputs)
            
        return outputs
        
    def forward_pass(self, x, tile_labels, bbox_labels, ground_truth_labels, omit_masks, split, num_epoch, device, outputs=None):
        """
        Description: compute forward pass of all model_list models
        Args:
            - x (tensor): raw image input
            - tile_labels (tensor): labels for tiles for tile_loss
            - bbox_labels: (list): labels for bboxes
            - ground_truth_labels (tensor): labels for images for image_loss
            - omit_masks (tensor): determines if tile predictions should be masked
            - split (str): if this is the train/val/test split for determining correct loss calculation
            - num_epoch (int): current epoch number (for pretrain_epochs)
            - device: current device being used to create new torch objects without getting errors
        Outputs:
            - losses (list of tensor): all tile-level or image-level losses (for logging purposes)
            - image_loss (list of tensor): all image-level losses (for logging purposes)
            - total_loss (tensor): overall loss (sum of all losses)
            - tile_probs (tensor): probabilities predicted for each tile
            - tile_preds (tensor): final predictions for each tile
            - image_preds (tensor): final predictions for each image
        """
        
        tile_outputs = None
        image_outputs = None
        
        losses = []
        total_loss = 0
        image_loss = None
        
        tile_probs = None
        tile_preds = None
        image_preds = None
        
        # Compute forward pass and loss for each model in model_list
        for i, model in enumerate(self.model_list):
            # Skip iteration if pretraining model
            if i != 0 and self.pretrain_epochs[:i].sum() > num_epoch:
                break
            
            # Compute forward pass
            if outputs is None or i > 0:
                outputs, x = model(x, bbox_labels=bbox_labels, tile_outputs=outputs)
                        
            # FPN ONLY: If outputs is a dictionary of FPN layers...
            if type(outputs) is collections.OrderedDict:
                returns = {}
                for key in outputs:
                    # Run through forward pass for each layer of FPN
                    # Outputs: (losses, image_loss, total_loss, tile_probs, tile_preds, image_preds)
                    returns[key] = self.forward_pass(x[key], tile_labels, bbox_labels, ground_truth_labels, omit_masks, split, num_epoch, device, outputs[key])
                    
                    # Compute correct statistics
                    losses = returns[key][0] if len(losses)==0 else losses + returns[key][0]
                    image_loss = returns[key][1] if image_loss is None else image_loss + returns[key][1]
                    total_loss += returns[key][2] / len(outputs)
                    tile_probs = returns[key][3] if tile_probs is None else tile_probs + returns[key][3]
                    tile_preds = returns[key][4] if tile_preds is None else torch.logical_or(tile_preds, returns[key][4])
                    image_preds = returns[key][5] if image_preds is None else torch.logical_or(image_preds, returns[key][5])
                    
                if tile_probs is not None:
                    tile_probs = tile_probs / len(outputs)
                    
                break
            

            
            # IMAGE ONLY: Else if model predicts images only...
            elif x is None:
                # Calculate image loss
                image_outputs = outputs
                image_loss = F.binary_cross_entropy_with_logits(image_outputs[:,-1], ground_truth_labels.float(), reduction='none', pos_weight=torch.as_tensor(self.image_pos_weight))
                
                # Always add image loss, even if no intermediate_supervision
                loss = image_loss.mean()
                total_loss += loss
                losses.append(loss)
                                        
            # TILE ONLY: Else if model predicts tiles only...
            elif len(x.shape) > 2:
                # Calculate tile loss
                tile_outputs = outputs
                loss = self.tile_loss(tile_outputs[omit_masks,:,-1], tile_labels[omit_masks], num_epoch=num_epoch) 
                
                # Only add loss if intermediate_supervision
                if self.intermediate_supervision and not self.image_loss_only:
                    total_loss += loss
                    losses.append(loss)
                else:
                    total_loss = loss
                
            # TILE & IMAGES: Else if model predicts tiles and images...
            else:
                # Calculate tile loss
                tile_outputs = outputs
                if not self.image_loss_only:
                    loss = self.tile_loss(tile_outputs[omit_masks,:,-1], tile_labels[omit_masks], num_epoch=num_epoch) 
                    total_loss += loss
                    losses.append(loss)
                
                # Calculate image loss
                image_outputs = x
                image_loss = F.binary_cross_entropy_with_logits(image_outputs[:,-1], ground_truth_labels.float(), reduction='none', pos_weight=torch.as_tensor(self.image_pos_weight))
                
                loss = image_loss.mean()
                total_loss += loss
                losses.append(loss)
            
        # Compute predictions for tiles
        if tile_outputs is not None:
            tile_probs = torch.sigmoid(tile_outputs[:,:,-1])
            tile_preds = (tile_probs > 0.5).int()
        
        # If created image_outputs, predict directly
        if self.use_image_preds and image_outputs is not None:
            image_probs = torch.sigmoid(image_outputs[:,-1])
            image_preds = (image_probs > 0.5).int()
        # Else, use tile_preds to determine image_preds
        elif tile_outputs is not None:
            image_preds = (tile_preds.sum(dim=1) > 0).int()
        
        # If error_as_eval_loss, replace total_loss with number of errors
        if self.error_as_eval_loss and split != 'train/':
            total_loss = torch.abs(image_preds.float() - ground_truth_labels.float()).sum()

        return losses, image_loss, total_loss, tile_probs, tile_preds, image_preds, image_probs