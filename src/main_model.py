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

# File imports
from model_components import *
import util_fns

    
#####################
## Main Model
#####################
    
class MainModel(nn.Module):
    """
    Description: Simple model with ResNet backbone and a few linear layers
    Args:
        - model_type_list (list of str): a sequential list of model_components to use to make up full model
        - pretrain_epochs (list of int): a sequential list of epochs to pretrain each model part for
        - intermediate_supervision (bool): whether or not to provide intermediate supervision between model components
        - use_image_preds (bool): uses image predictions from linear layers instead of tile preds.
        - kwargs: any other args used in the models
    """
    def __init__(self, 
                 model_type_list=['RawToTile_MobileNetV3Large'], 
                 pretrain_epochs=None,
                 intermediate_supervision=True,
                 use_image_preds=False,
                 
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
        
    def forward_pass(self, x, tile_labels, bbox_labels, ground_truth_labels, omit_masks, num_epoch, device):
        """
        Description: compute forward pass of all model_list models
        Args:
            - x (tensor): raw image input
            - tile_labels (tensor): labels for tiles for tile_loss
            - bbox_labels: (list): labels for bboxes
            - ground_truth_labels (tensor): labels for images for image_loss
            - omit_masks (tensor): determines if tile predictions should be masked
            - num_epoch (int): current epoch number (for pretrain_epochs)
        Outputs:
            - x (tensor): final outputs of model
            - tile_losses (list of tensor): all tile-level losses (for logging purposes)
            - image_losses (list of tensor): all image-level losses (for logging purposes)
            - total_loss (tensor): overall loss (sum of all losses)
            - tile_preds (tensor): final predictions for each tile
            - image_preds (tensor): final predictions for each image
        """
        
        outputs = None
        embeddings = None
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
            outputs, x = model(x, bbox_labels=bbox_labels, tile_outputs=outputs)
                        
            # If x is a dictionary of object detection losses...
            if type(x) is dict:
                # If training...
                if len(x) > 0: 
                    # Use losses from model
                    for loss in x:
                        total_loss += x[loss]
                # Else for val/test loss...
                else:
                    # Determine if there were any scores above confidence = 0
                    image_preds = torch.as_tensor([(output['scores'] > 0).sum() > 0 for output in outputs]).to(device)
#                     if image_preds.sum() > 0:
#                         print(outputs)
#                         import pdb; pdb.set_trace()
                    
                    # Use number of errors as loss
                    total_loss = torch.abs(image_preds.float() - ground_truth_labels.float()).sum()
                    
                return losses, image_loss, total_loss, tile_probs, tile_preds, image_preds
            
            # Else if model predicts images only...
            elif x is None:
                image_outputs = outputs
                image_loss = F.binary_cross_entropy_with_logits(image_outputs[:,-1], ground_truth_labels.float(), reduction='none', pos_weight=torch.as_tensor(self.image_pos_weight))
                
                # Always add image loss, even if no intermediate_supervision
                loss = image_loss.mean()
                total_loss += loss
                losses.append(loss)
                                        
            # Else if model predicts tiles only...
            elif len(x.shape) > 2:
                tile_outputs = outputs
                embeddings = x
                loss = self.tile_loss(tile_outputs[omit_masks,:,-1], tile_labels[omit_masks]) 
                
                # Only add loss if intermediate_supervision
                if self.intermediate_supervision and not self.image_loss_only:
                    total_loss += loss
                    losses.append(loss)
                else:
                    total_loss = loss
                
            # Else if model predicts tiles and images...
            else:
                tile_outputs = outputs
                if not self.image_loss_only:
                    loss = self.tile_loss(tile_outputs[omit_masks,:,-1], tile_labels[omit_masks]) 
                    total_loss += loss
                    losses.append(loss)
                
                image_outputs = x
                image_loss = F.binary_cross_entropy_with_logits(image_outputs[:,-1], ground_truth_labels.float(), reduction='none', pos_weight=torch.as_tensor(self.image_pos_weight))
                
                loss = image_loss.mean()
                total_loss += loss
                losses.append(loss)
            
        # Compute predictions for tiles and images 
        if tile_outputs is not None:
            tile_probs = torch.sigmoid(tile_outputs[:,:,-1])
            tile_preds = (tile_probs > 0.5).int()
        
        # If created image_outputs, predict directly
        if self.use_image_preds and image_outputs is not None:
            image_preds = (torch.sigmoid(image_outputs[:,-1]) > 0.5).int()
        # Else, use tile_preds to determine image_preds
        elif tile_outputs is not None:
            image_preds = (tile_preds.sum(dim=1) > 0).int()

        return losses, image_loss, total_loss, tile_probs, tile_preds, image_preds