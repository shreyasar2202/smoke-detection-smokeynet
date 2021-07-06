"""
Created by: Anshuman Dewangan
Date: 2021

Description: Different torch models to use with main_model.py. Models can be one of five types:
    1. RawToTile: Raw inputs  -> tile predictions 
    2. RawToImage: Raw inputs -> image predictions
    3. TileToTile: Tile predictions -> tile predictions
    4. TileToImage: Tile predictins -> image predictions
    5. ImageToImage: Image predictions -> image predictions
Sizes:
    1. Raw inputs: [batch_size, num_tiles, series_length, num_channels, tile_height, tile_width]
    2. Tile predictions: [batch_size, num_tiles, series_length or 1]
    3. Image predictions: [batch_size, series_length or 1]
"""

# Torch imports
import torch
from torch import nn
from torch.nn import functional as F
import torchvision
import transformers

# Other imports 
import numpy as np

# File imports
import util_fns


#####################
## Loss Classes
#####################

class TileLoss():
    """
    Description: Class to calculate loss for tiles
    Args:
        - tile_loss_type: type of loss to use. Options: [bce] [focal]
        - bce_pos_weight: how much to weight the positive class in BCE Loss
        - focal_alpha: focal loss, lower alpha -> more importance of positive class vs. negative class
        - focal_gamma: focal loss, higher gamma -> more importance of hard examples vs. easy examples
    """
    def __init__(self, 
                 tile_loss_type='focal',
                 bce_pos_weight=25,
                 focal_alpha=0.25, 
                 focal_gamma=2):
        
        self.tile_loss_type = tile_loss_type
        self.bce_pos_weight = bce_pos_weight
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        
        if self.tile_loss_type == 'focal':
            print('-- Loss: Focal Loss')
        else:
            print('-- Loss: BCE Loss')
            
    def __call__(self, tile_outputs, tile_labels):
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
        
        
#####################
## RawToTile Models
#####################

# class RawToTile_ResNet50(nn.Module):
#     """
#     Description: ResNet backbone with a few linear layers.
#     Args:
#         - series_length (int)
#         - freeze_backbone (bool): Freezes layers of pretrained backbone
#         - pretrain_backbone (bool): Pretrains backbone
#     """
#     def __init__(self, freeze_backbone=True, pretrain_backbone=True, **kwargs): 
        
#         print('- RawToTile: ResNet50')
#         super().__init__()
        
#         model = torchvision.models.resnet50(pretrained=pretrain_backbone)
#         model.fc = nn.Identity()

#         if pretrain_backbone and freeze_backbone:
#             for param in model.parameters():
#                 param.requires_grad = False

#         self.conv = model

#         self.fc1 = nn.Linear(in_features=2048, out_features=512)
#         self.fc2 = nn.Linear(in_features=512, out_features=64)
#         self.fc3 = nn.Linear(in_features=64, out_features=1)
        
#         self.fc1, self.fc2, self.fc3 = util_fns.init_weights_RetinaNet(self.fc1, self.fc2, self.fc3)
        
#     def forward(self, x):
#         x = x.float()
#         batch_size, num_tiles, series_length, num_channels, height, width = x.size()

#         tile_outputs = x.view(batch_size * num_tiles * series_length, num_channels, height, width)
#         tile_outputs = self.conv(tile_outputs) # [batch_size * num_tiles * series_length, 2048]

#         tile_outputs = tile_outputs.view(batch_size, num_tiles * series_length, -1) # [batch_size, num_tiles * series_length, 2048]
#         tile_outputs = F.relu(self.fc1(tile_outputs)) # [batch_size, num_tiles * series_length, 512]
#         tile_outputs = F.relu(self.fc2(tile_outputs)) # [batch_size, num_tiles * series_length, 64]
#         tile_outputs = self.fc3(tile_outputs) # [batch_size, num_tiles * series_length, 1]
#         tile_outputs = tile_outputs.view(batch_size, num_tiles, series_length)

#         return tile_outputs # [batch_size, num_tiles, series_length]
            
class RawToTile_MobileNetV3Large(nn.Module):
    """
    Description: MobileNetV3Large backbone with a few linear layers.
    Args:
        - series_length (int)
        - freeze_backbone (bool): Freezes layers of pretrained backbone
        - pretrain_backbone (bool): Pretrains backbone
    """
    def __init__(self, freeze_backbone=True, pretrain_backbone=True, **kwargs):
        
        print('- RawToTile: MobileNetV3Large')
        super().__init__()

        model = torchvision.models.mobilenet_v3_large(pretrained=pretrain_backbone)
        model.classifier = nn.Identity()

        if pretrain_backbone and freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False

        self.conv = model

        self.fc1 = nn.Linear(in_features=960, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=64)
        self.fc3 = nn.Linear(in_features=64, out_features=1)
        
        self.fc1, self.fc2, self.fc3 = util_fns.init_weights_RetinaNet(self.fc1, self.fc2, self.fc3)
        
    def forward(self, x):
        x = x.float()
        batch_size, num_tiles, series_length, num_channels, height, width = x.size()

        tile_outputs = x.view(batch_size * num_tiles * series_length, num_channels, height, width)
        tile_outputs = self.conv(tile_outputs) # [batch_size * num_tiles * series_length, 960]

        tile_outputs = tile_outputs.view(batch_size, num_tiles * series_length, -1) # [batch_size, num_tiles * series_length, 960]
        tile_outputs = F.relu(self.fc1(tile_outputs)) # [batch_size, num_tiles * series_length, 512]
        tile_outputs = F.relu(self.fc2(tile_outputs)) # [batch_size, num_tiles * series_length, 64]
        tile_outputs = self.fc3(tile_outputs) # [batch_size, num_tiles * series_length, 1]
        tile_outputs = tile_outputs.view(batch_size, num_tiles, series_length)

        return tile_outputs # [batch_size, num_tiles, series_length]

    
class RawToTile_MobileNetV3LargeLSTM(nn.Module):
    """
    Description: MobileNetV3Large to LSTM
    """
    def __init__(self, freeze_backbone=True, pretrain_backbone=True, **kwargs):
        print('- RawToTile: MobileNetV3LargeLSTM')
        super().__init__()

        model = torchvision.models.mobilenet_v3_large(pretrained=pretrain_backbone)
        model.classifier = nn.Identity()

        if pretrain_backbone and freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False

        self.conv = model
        self.fc1 = nn.Linear(in_features=960, out_features=512)
        
        self.lstm = torch.nn.LSTM(input_size=512, hidden_size=512, num_layers=1, batch_first=True)
        
        self.fc2 = nn.Linear(in_features=512, out_features=64)
        self.fc3 = nn.Linear(in_features=64, out_features=1)
        
        self.fc1, = util_fns.init_weights_Xavier(self.fc1)
        self.fc2, self.fc3 = util_fns.init_weights_RetinaNet(self.fc2, self.fc3)
                
    def forward(self, x):
        batch_size, num_tiles, series_length, num_channels, height, width = x.size()

        tile_outputs = x.view(batch_size * num_tiles * series_length, num_channels, height, width).float()
        tile_outputs = self.conv(tile_outputs) # [batch_size * num_tiles * series_length, 960]

        tile_outputs = tile_outputs.view(batch_size, num_tiles * series_length, -1).float() # [batch_size, num_tiles * series_length, 960]
        tile_outputs = F.relu(self.fc1(tile_outputs)) # [batch_size, num_tiles * series_length, 512]
        
        tile_outputs = tile_outputs.view(batch_size * num_tiles, series_length, -1).float() # [batch_size * num_tiles, series_length, 512]
        tile_outputs, (hidden, cell) = self.lstm(tile_outputs) # [batch_size * num_tiles, series_length, 512]
        
        # Only save the last time step's outputs
        tile_outputs = tile_outputs[:,-1] # [batch_size * num_tiles, 512]
        tile_outputs = tile_outputs.view(batch_size, num_tiles, -1).float() # [batch_size, num_tiles, 512]
        
        tile_outputs = F.relu(self.fc2(tile_outputs)) # [batch_size, num_tiles, 64]
        tile_outputs = F.relu(self.fc3(tile_outputs)) # [batch_size, num_tiles, 1]
        
        return tile_outputs # [batch_size, num_tiles, 1]
    
    
###########################
## TileToTile Models
########################### 
    
class TileToTile_LSTM(nn.Module):
    """
    Description: Single linear layer to go from tile outputs to image predictions
    Args:
        - num_tiles (int): number of tiles in image
    """
    def __init__(self, **kwargs):
        print('- TileToTile: LSTM')
        super().__init__()
        
        self.lstm = torch.nn.LSTM(input_size=1, hidden_size=512, num_layers=1, batch_first=True)
        self.fc1 = nn.Linear(in_features=512, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=1)
        
        self.fc1, self.fc2 = util_fns.init_weights_RetinaNet(self.fc1, self.fc2)
                
    def forward(self, tile_outputs):
        batch_size, num_tiles, series_length = tile_outputs.size()
        tile_outputs = tile_outputs.view(-1, series_length, 1).float()
        
        tile_outputs, (hidden, cell) = self.lstm(tile_outputs) # [batch_size * num_tiles, series_length, lstm_num_hidden]
        
        # Only save the last time step's outputs
        tile_outputs = tile_outputs[:,-1] # [batch_size * num_tiles, 512]
        tile_outputs = tile_outputs.view(batch_size, num_tiles, -1).float()
        
        tile_outputs = F.relu(self.fc1(tile_outputs)) # [batch_size, num_tiles, 64]
        tile_outputs = F.relu(self.fc2(tile_outputs)) # [batch_size, num_tiles, 1]
        
        return tile_outputs # [batch_size, num_tiles, 1]

    
###########################
## TileToImage Models
########################### 

class TileToImage_Linear(nn.Module):
    """
    Description: Single linear layer to go from tile outputs to image predictions
    Args:
        - num_tiles (int): number of tiles in image
    """
    def __init__(self, num_tiles=45, **kwargs):
        print('- TileToImage: Linear')
        super().__init__()
        
        self.fc = nn.Linear(in_features=num_tiles, out_features=1)
        
        self.fc = util_fns.init_weights_Xavier(self.fc)
        
    def forward(self, tile_outputs):
        batch_size, num_tiles, series_length = tile_outputs.size()
        tile_outputs = tile_outputs.view(batch_size * series_length, num_tiles)
        
        image_outputs = F.relu(tile_outputs) # [batch_size * series_length, num_tiles]
        image_outputs = self.fc(image_outputs) # [batch_size * series_length]
        image_outputs = image_outputs.view(batch_size, series_length)

        return image_outputs # [batch_size, series_length]
    
class TileToImage_ViT(nn.Module):
    """
    Description: Vision Transformer operating on tiles to produce image prediction
    Args:
        - num_tiles_height (int): number of tiles that make up the height of the image
        - num_tiles_width (int): number of tiles that make up the width of the image
    """
    def __init__(self, num_tiles_height=5, num_tiles_width=9, **kwargs):
        print('- TileToImage: ViT')
        super().__init__()
        
        self.num_tiles_height = num_tiles_height
        self.num_tiles_width = num_tiles_width
        
        vit_config = transformers.ViTConfig(image_size=(num_tiles_height,num_tiles_width), patch_size=1, num_channels=1, num_labels=1)
        self.vit_model = transformers.ViTForImageClassification(vit_config)
                
    def forward(self, tile_outputs):
        batch_size, num_tiles, series_length = tile_outputs.size()
        tile_outputs = tile_outputs.view(batch_size * series_length, 1, self.num_tiles_height, self.num_tiles_width)
        
        image_outputs = self.vit_model(tile_outputs).logits # [batch_size * series_length]
        image_outputs = image_outputs.view(batch_size, series_length) 

        return image_outputs # [batch_size, series_length]
    
