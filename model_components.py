"""
Created by: Anshuman Dewangan
Date: 2021

Description: Different torch models to use with main_model.py. Models can be one of five types:
    1. RawToTile: Raw inputs -> tile predictions
    2. RawToImage: Raw inputs -> image predictions
    3. TileToTile: Tile predictions -> tile predictions
    4. TileToImage: Tile predictins -> image predictions
    5. ImageToImage: Image predictions -> image predictions
"""

# Torch imports
import torch
from torch import nn
from torch.nn import functional as F
import torchvision

# Other imports 
import numpy as np


#####################
## Parent Classes 
#####################

class TileClassifier(nn.Module):
    """
    Description: Takes any input and outputs tile predictions
    Args:
        - tile_loss_type: type of loss to use. Options: [bce] [focal]
        - bce_pos_weight: how much to weight the positive class in BCE Loss
        - focal_alpha: focal loss, lower alpha -> more importance of positive class vs. negative class
        - focal_gamma: focal loss, higher gamma -> more importance of hard examples vs. easy examples
    """
    
    def __init__(self, 
                 tile_loss_type='bce',
                 bce_pos_weight=25,
                 focal_alpha=0.25, 
                 focal_gamma=2):
        super().__init__()
        
        self.tile_loss_type = tile_loss_type
        self.bce_pos_weight = bce_pos_weight
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        
        if self.tile_loss_type == 'focal':
            print('-- Loss: Focal Loss')
        else:
            print('-- Loss: BCE Loss')
        
    def init_weights(self, layers):
        # Initialize weights as in RetinaNet paper
        for i, layer in enumerate(layers):
            torch.nn.init.normal_(layer.weight, 0, 0.01)

            # Set last layer bias to special value from paper
            if i == len(layers)-1:
                torch.nn.init.constant_(layer.bias, -np.log((1-0.01)/0.01))
            else:
                torch.nn.init.zeros_(layer.bias)
                
    def compute_loss(self, tile_outputs, tile_labels):
        if self.tile_loss_type == 'focal':
            tile_loss = torchvision.ops.focal_loss.sigmoid_focal_loss(
                     tile_outputs, 
                     tile_labels, 
                     reduction='sum', 
                     alpha=self.focal_alpha, 
                     gamma=self.focal_gamma)

            # Normalize by count of positives in ground truth only as in RetinaNet paper
            # Prevent divide by 0 error
            tile_loss = tile_loss / torch.maximum(torch.as_tensor(1), tile_labels.sum())
        elif self.tile_loss_type == 'bce':
            tile_loss = F.binary_cross_entropy_with_logits(
                      tile_outputs, 
                      tile_labels,
                      pos_weight=torch.as_tensor(self.bce_pos_weight))
        
        return tile_loss

class ImageClassifier(nn.Module):
    """
    Description: Takes any input and outputs image predictions
    """
    def __init__(self):
        super().__init__()
    
    def init_weights(self, layers):
        for layer in layers:
            torch.nn.init.xavier_uniform_(layer.weight, gain=torch.nn.init.calculate_gain('relu'))
            torch.nn.init.xavier_uniform_(layer.bias.reshape((-1,1)))
        
    def compute_loss(self, image_outputs, ground_truth_labels):
        image_loss = F.binary_cross_entropy_with_logits(image_outputs, ground_truth_labels.float())
        
        return image_loss

    
#####################
## RawToTile Models
#####################

class RawToTile_ResNet50(TileClassifier):
    """
    Description: ResNet backbone with a few linear layers.
    Args:
        - series_length (int)
        - freeze_backbone (bool): Freezes layers of pretrained backbone
        - pretrain_backbone (bool): Pretrains backbone
    """
    def __init__(self, 
                 series_length=1, 
                 freeze_backbone=True, 
                 pretrain_backbone=True,
                 
                 tile_loss_type='bce',
                 bce_pos_weight=25,
                 focal_alpha=0.25, 
                 focal_gamma=2):
        
        print('- RawToTile: ResNet50')
        super().__init__(tile_loss_type=tile_loss_type, 
                         bce_pos_weight=bce_pos_weight,
                         focal_alpha=focal_alpha, 
                         focal_gamma=focal_gamma)
        
        model = torchvision.models.resnet50(pretrained=pretrain_backbone)
        model.fc = nn.Identity()

        if pretrain_backbone and freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False

        self.conv = model

        self.fc1 = nn.Linear(in_features=series_length * 2048, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=64)
        self.fc3 = nn.Linear(in_features=64, out_features=1)
        
        self.init_weights([self.fc1, self.fc2, self.fc3])
        
    def forward(self, x):
        x = x.float()
        batch_size, num_tiles, series_length, num_channels, height, width = x.size()

        tile_outputs = x.view(batch_size * num_tiles * series_length, num_channels, height, width)
        tile_outputs = self.conv(tile_outputs) # [batch_size * num_tiles * series_length, 2048]

        tile_outputs = tile_outputs.view(batch_size, num_tiles, -1) # [batch_size, num_tiles, series_length * 2048]
        tile_outputs = F.relu(self.fc1(tile_outputs)) # [batch_size, num_tiles, 512]
        tile_outputs = F.relu(self.fc2(tile_outputs)) # [batch_size, num_tiles, 64]
        tile_outputs = self.fc3(tile_outputs) # [batch_size, num_tiles, 1]
        tile_outputs = tile_outputs.squeeze(2) # [batch_size, num_tiles]

        return tile_outputs
            
class RawToTile_MobileNetV3Large(TileClassifier):
    """
    Description: MobileNetV3Large backbone with a few linear layers.
    Args:
        - series_length (int)
        - freeze_backbone (bool): Freezes layers of pretrained backbone
        - pretrain_backbone (bool): Pretrains backbone
    """
    def __init__(self, 
                 series_length=1, 
                 freeze_backbone=True, 
                 pretrain_backbone=True,
                 
                 tile_loss_type='bce',
                 bce_pos_weight=25,
                 focal_alpha=0.25, 
                 focal_gamma=2):
        
        print('- RawToTile: MobileNetV3Large')
        super().__init__(tile_loss_type=tile_loss_type, 
                         bce_pos_weight=bce_pos_weight,
                         focal_alpha=focal_alpha, 
                         focal_gamma=focal_gamma)

        model = torchvision.models.mobilenet_v3_large(pretrained=pretrain_backbone)
        model.classifier = nn.Identity()

        if pretrain_backbone and freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False

        self.conv = model

        self.fc1 = nn.Linear(in_features=series_length * 960, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=64)
        self.fc3 = nn.Linear(in_features=64, out_features=1)
        
        self.init_weights([self.fc1, self.fc2, self.fc3])
        
    def forward(self, x):
        x = x.float()
        batch_size, num_tiles, series_length, num_channels, height, width = x.size()

        tile_outputs = x.view(batch_size * num_tiles * series_length, num_channels, height, width)
        tile_outputs = self.conv(tile_outputs) # [batch_size * num_tiles * series_length, 2048]

        tile_outputs = tile_outputs.view(batch_size, num_tiles, -1) # [batch_size, num_tiles, series_length * 2048]
        tile_outputs = F.relu(self.fc1(tile_outputs)) # [batch_size, num_tiles, 512]
        tile_outputs = F.relu(self.fc2(tile_outputs)) # [batch_size, num_tiles, 64]
        tile_outputs = self.fc3(tile_outputs) # [batch_size, num_tiles, 1]
        tile_outputs = tile_outputs.squeeze(2) # [batch_size, num_tiles]

        return tile_outputs

    
###########################
## TileToImage Models
########################### 

class TileToImage_Linear(ImageClassifier):
    """
    Description: Single linear layer to go from tile outputs to image predictions
    Args:
        - num_tiles (int): number of tiles in image
    """
    def __init__(self, num_tiles=45):
        print('- TileToImage: Linear')
        super().__init__()
        
        self.fc = nn.Linear(in_features=num_tiles, out_features=1)
        
        self.init_weights([self.fc])
        
    def forward(self, tile_outputs):
        image_outputs = F.relu(tile_outputs) # [batch_size, num_tiles]
        image_outputs = self.fc(image_outputs) # [batch_size, 1]
        image_outputs = image_outputs.squeeze(1) # [batch_size]

        return image_outputs