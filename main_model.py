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

# File imports
import model_components
import util_fns


    
#####################
## Main Model
#####################
    
class MainModel(nn.Module):
    """
    Description: Simple model with ResNet backbone and a few linear layers
    Args:
        - model_type_list: a sequential list of model_components to use to make up full model
    """
    def __init__(self, 
                 model_type_list=['RawToTile_MobileNetV3Large'],
                 
                 # Backbone Args
                 series_length=1, 
                 freeze_backbone=True, 
                 pretrain_backbone=True,
                 
                 # Tile Loss args
                 tile_loss_type='bce',
                 bce_pos_weight=25,
                 focal_alpha=0.25, 
                 focal_gamma=2, 
                
                 # TileToImage args
                 num_tiles=45):
        
        print("Initializing MainModel...")
        super().__init__()
        
        self.model_list = torch.nn.ModuleList()
                
        for model_type in model_type_list:
            if model_type == 'RawToTile_ResNet50':
                model = model_components.RawToTile_ResNet50(
                                series_length=series_length, 
                                pretrain_backbone=pretrain_backbone,
                                freeze_backbone=freeze_backbone,
                    
                                tile_loss_type=tile_loss_type, 
                                bce_pos_weight=bce_pos_weight,
                                focal_alpha=focal_alpha, 
                                focal_gamma=focal_gamma)
            
            elif model_type == 'RawToTile_MobileNetV3Large':
                model = model_components.RawToTile_MobileNetV3Large(
                                series_length=series_length, 
                                pretrain_backbone=pretrain_backbone,
                                freeze_backbone=freeze_backbone,
                                                                  
                                tile_loss_type=tile_loss_type, 
                                bce_pos_weight=bce_pos_weight,
                                focal_alpha=focal_alpha, 
                                focal_gamma=focal_gamma)
                
            elif model_type == 'TileToImage_Linear':
                model = model_components.TileToImage_Linear(num_tiles=num_tiles)
            
            self.model_list.append(model)
        
        print("Initializing MainModel Complete.")
        
    def forward(self, x):
        """Description: Maps raw inputs to outputs"""
        for model in self.model_list:
            x = model(x)
            
        return x
        
    def forward_pass(self, x, tile_labels, ground_truth_labels):
        """
        Description: compute forward pass of all model_list models
        Args:
            - x (tensor): raw image input
            - tile_labels (tensor): labels for tiles for tile_loss
            - ground_truth_labels (tensor): labels for images for image_loss
        Outputs:
            - x (tensor): final outputs of model
            - tile_losses (list of tensor): all tile-level losses (for logging purposes)
            - image_losses (list of tensor): all image-level losses (for logging purposes)
            - total_loss (tensor): overall loss (sum of all losses)
            - tile_preds (tensor): final predictions for each tile
            - image_preds (tensor): final predictions for each image
        """
        
        tile_outputs = None
        image_outputs = None
        
        losses = []
        total_loss = 0
        
        tile_preds = None
        image_preds = None
        
        # Compute forward pass and loss for each model in model_list
        for model in self.model_list:
            # Compute forward pass
            x = model(x)
            
            # If model predicts tiles...
            if len(x.shape) > 1:
                tile_outputs = x
                loss = model.compute_loss(x, tile_labels)
                losses.append(loss)
            
            # Else if model predicts images...
            else:
                image_outputs = x
                loss = model.compute_loss(x, ground_truth_labels)
                losses.append(loss)
            
            # Add loss to total loss
            total_loss += loss
        
        # Compute predictions for tiles and images 
        if tile_outputs is not None:
            tile_preds = (torch.sigmoid(tile_outputs) > 0.5).int()
        
        # If created image_outputs, predict directly
        if image_outputs is not None:
            image_preds = (torch.sigmoid(image_outputs) > 0.5).int()
        # Else, use tile_preds to determine image_preds
        else:
            image_preds = (tile_preds.sum(dim=1) > 0).int()
            
        
        return x, losses, total_loss, tile_preds, image_preds