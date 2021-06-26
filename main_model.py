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
from model_components import ResNet50
import util_fns

    
#####################
## Main Model
#####################
    
class MainModel(nn.Module):
    """
    Description: Simple model with ResNet backbone and a few linear layers
    Args:
        - model_type: which backbone to use. Choices:
            - 'ResNet50'
        - series_length: number of sequential video frames to process during training
        - pretrain_backbone: pretrains backbone
        - freeze_backbone: freezes layers on pre-trained backbone
        - bce_pos_weight: how much to weight the positive class in BCE Loss
        - focal_alpha: focal loss, lower alpha -> more importance of positive class vs. negative class
        - focal_gamma: focal loss, higher gamma -> more importance of hard examples vs. easy examples
    """
    def __init__(self, 
                 model_type='ResNet50',
                 series_length=1, 
                 freeze_backbone=True, 
                 pretrain_backbone=True,
                 loss_type='bce',
                 bce_pos_weight=10,
                 focal_alpha=0.25, 
                 focal_gamma=2):
        
        print("Initializing MainModel...")
        super().__init__()
        
        if model_type == 'ResNet50':
            self.backbone = ResNet50(series_length=series_length, 
                                pretrain_backbone=pretrain_backbone,
                                freeze_backbone=freeze_backbone)
        
        self.loss_type = loss_type
        self.bce_pos_weight = bce_pos_weight
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        
        if loss_type == 'focal':
            print('Loss: Focal Loss')
        else:
            print('Loss: BCE Loss')
        
        print("Initializing MainModel Complete.")
        
    def forward(self, x):
        """
        Description: compute forward pass of all model_parts as well as adds additional layers
        Args:
            - x (tensor): input provided by dataloader
        Returns:
            - outputs (tensor): outputs after going through forward passes of all layers
        """
        
        outputs = self.backbone(x).squeeze(dim=2) # [batch_size, num_tiles]
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
        
        if self.loss_type == 'focal':
            loss = torchvision.ops.focal_loss.sigmoid_focal_loss(
                     outputs, 
                     tile_labels, 
                     reduction='mean', 
                     alpha=self.focal_alpha, 
                     gamma=self.focal_gamma)
        else:
            loss = F.binary_cross_entropy_with_logits(
                      outputs, 
                      tile_labels,
                      pos_weight=torch.as_tensor(self.bce_pos_weight))
        
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
                