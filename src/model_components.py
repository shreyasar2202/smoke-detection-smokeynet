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
    1. Raw inputs: [batch_size, num_tiles, series_length, num_channels, tile_height, tile_width]. Example: [8, 45, 4, 3, 224, 224]
    2. ToTile: tile_outputs=[batch_size, num_tiles, series_length or 1], embeddings=[batch_size, num_tiles, series_length or 1, 960]
    3. ToImage: image_outputs=[batch_size, series_length or 1], embeddings=None
"""

# Torch imports
import torch
from torch import nn
from torch.nn import functional as F
import torchvision
import transformers
from efficientnet_pytorch import EfficientNet

# Other imports 
import numpy as np

# File imports
import util_fns


#####################
## Helper Classes
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
                 tile_loss_type='bce',
                 bce_pos_weight=36,
                 focal_alpha=0.25, 
                 focal_gamma=2):
        
        self.tile_loss_type = tile_loss_type
        self.bce_pos_weight = bce_pos_weight
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        
        if self.tile_loss_type == 'focal':
            print('-- Tile Loss: Focal Loss')
        elif self.tile_loss_type == 'bce':
            print('-- Tile Loss: BCE Loss')
        else:
            raise ValueError('Tile Loss Type not recognized.')
            
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
                      reduction='sum', 
                      pos_weight=torch.as_tensor(self.bce_pos_weight))
            
            # Normalize by count of positives in ground truth only as in RetinaNet paper
            # Prevent divide by 0 error
            # Also divide by pos_weight+1 to normalize weights
            tile_loss = tile_loss / (torch.maximum(torch.as_tensor(1), tile_labels.sum()) * (self.bce_pos_weight+1) )
        
        return tile_loss

class TileEmbeddingsToOutput(nn.Module):
    """
    Description: Takes embeddings of dim=960 and converts them to outputs of dim=1 using linear layers
    """
    def __init__(self):
        super().__init__()
        
        self.fc1 = nn.Linear(in_features=960, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=64)
        self.fc3 = nn.Linear(in_features=64, out_features=1)
        
        self.fc1, self.fc2, self.fc3 = util_fns.init_weights_RetinaNet(self.fc1, self.fc2, self.fc3)
        
    def forward(self, tile_embeddings, batch_size, num_tiles):
        tile_outputs = F.relu(self.fc1(tile_embeddings)) # [batch_size, num_tiles, series_length, 512]
        tile_outputs = F.relu(self.fc2(tile_outputs)) # [batch_size, num_tiles, series_length, 64]
        tile_outputs = self.fc3(tile_outputs) # [batch_size, num_tiles, series_length, 1]
        
        tile_outputs = tile_outputs.view(batch_size, num_tiles, -1) 
        
        return tile_outputs # [batch_size, num_tiles, series_length]
        
#####################
## RawToTile Models
#####################
            
class RawToTile_MobileNetV3Large(nn.Module):
    """
    Description: MobileNetV3Large backbone with a few linear layers.
    Args:
        - series_length (int)
        - freeze_backbone (bool): Freezes layers of pretrained backbone
        - pretrain_backbone (bool): Pretrains backbone
    """
    def __init__(self, 
                 freeze_backbone=True, 
                 pretrain_backbone=True, 
                 backbone_checkpoint_path=None, 
                 **kwargs):
        print('- RawToTile_MobileNetV3Large')
        super().__init__()

        self.conv = torchvision.models.mobilenet_v3_large(pretrained=pretrain_backbone)
        self.conv.classifier = nn.Identity()

        # Initialize additional linear layers
        self.embeddings_to_output = TileEmbeddingsToOutput()
        
        if backbone_checkpoint_path is not None:
            self.load_state_dict(util_fns.get_state_dict(backbone_checkpoint_path))
        
        if freeze_backbone:
            for param in self.conv.parameters():
                param.requires_grad = False
        
    def forward(self, x, *args):
        x = x.float()
        batch_size, num_tiles, series_length, num_channels, height, width = x.size()

        # Run through conv model
        tile_outputs = x.view(batch_size * num_tiles * series_length, num_channels, height, width)
        tile_outputs = self.conv(tile_outputs) # [batch_size * num_tiles * series_length, 960]

        # Save embeddings of dim=960
        tile_outputs = tile_outputs.view(batch_size, num_tiles, series_length, 960)
        embeddings = tile_outputs
        
        # Use linear layers to get dim=1
        tile_outputs = self.embeddings_to_output(tile_outputs, batch_size, num_tiles) # [batch_size, num_tiles, series_length]

        return tile_outputs, embeddings # [batch_size, num_tiles, series_length], [batch_size, num_tiles, series_length, 960]

class RawToTile_EfficientNetB0(nn.Module):
    """
    Description: EfficientNetB0 backbone with a few linear layers.
    Args:
        - series_length (int)
        - freeze_backbone (bool): Freezes layers of pretrained backbone
        - pretrain_backbone (bool): Pretrains backbone
    """
    def __init__(self, freeze_backbone=True, pretrain_backbone=True, **kwargs):
        print('- RawToTile_EfficientNetB0')
        super().__init__()
        
        if pretrain_backbone:
            self.conv = EfficientNet.from_pretrained("efficientnet-b0")
        else:
            self.conv = EfficientNet.from_name("efficientnet-b0")
            
        self.avg_pooling = nn.AdaptiveAvgPool2d(1)

        if pretrain_backbone and freeze_backbone:
            for param in self.conv.parameters():
                param.requires_grad = False

        # Initialize additional linear layers
        self.fc = nn.Linear(in_features=1280, out_features=960)
        self.embeddings_to_output = TileEmbeddingsToOutput()
        
    def forward(self, x, *args):
        x = x.float()
        batch_size, num_tiles, series_length, num_channels, height, width = x.size()

        # Run through conv model
        tile_outputs = x.view(batch_size * num_tiles * series_length, num_channels, height, width)
        tile_outputs = self.conv.extract_features(tile_outputs) # [batch_size * num_tiles * series_length, 1280, 7, 7]
        tile_outputs = self.avg_pooling(tile_outputs) # [batch_size * num_tiles * series_length, 1280, 1, 1]
        
        # Save embeddings of dim=960
        tile_outputs = tile_outputs.view(batch_size, num_tiles, series_length, 1280)
        tile_outputs = self.fc(tile_outputs) # [batch_size, num_tiles, series_length, 960]
        embeddings = tile_outputs
        
        # Use linear layers to get dim=1
        tile_outputs = self.embeddings_to_output(tile_outputs, batch_size, num_tiles) # [batch_size, num_tiles, series_length]

        return tile_outputs, embeddings # [batch_size, num_tiles, series_length], [batch_size, num_tiles, series_length, 960]
    
class RawToTile_EfficientNetB3(nn.Module):
    """
    Description: EfficientNetB3 backbone with a few linear layers.
    Args:
        - series_length (int)
        - freeze_backbone (bool): Freezes layers of pretrained backbone
        - pretrain_backbone (bool): Pretrains backbone
    """
    def __init__(self, freeze_backbone=True, pretrain_backbone=True, **kwargs):
        print('- RawToTile_EfficientNetB3')
        super().__init__()
        
        if pretrain_backbone:
            self.conv = EfficientNet.from_pretrained("efficientnet-b3")
        else:
            self.conv = EfficientNet.from_name("efficientnet-b3")
            
        self.avg_pooling = nn.AdaptiveAvgPool2d(1)

        if pretrain_backbone and freeze_backbone:
            for param in self.conv.parameters():
                param.requires_grad = False

        # Initialize additional linear layers
        self.fc = nn.Linear(in_features=1536, out_features=960)
        self.embeddings_to_output = TileEmbeddingsToOutput()
        
    def forward(self, x, *args):
        x = x.float()
        batch_size, num_tiles, series_length, num_channels, height, width = x.size()

        # Run through conv model
        tile_outputs = x.view(batch_size * num_tiles * series_length, num_channels, height, width)
        tile_outputs = self.conv.extract_features(tile_outputs) # [batch_size * num_tiles * series_length, 1536, 7, 7]
        tile_outputs = self.avg_pooling(tile_outputs) # [batch_size * num_tiles * series_length, 1536, 1, 1]
        
        # Save embeddings of dim=960
        tile_outputs = tile_outputs.view(batch_size, num_tiles, series_length, 1536)
        tile_outputs = self.fc(tile_outputs) # [batch_size, num_tiles, series_length, 960]
        embeddings = tile_outputs
        
        # Use linear layers to get dim=1
        tile_outputs = self.embeddings_to_output(tile_outputs, batch_size, num_tiles) # [batch_size, num_tiles, series_length]

        return tile_outputs, embeddings # [batch_size, num_tiles, series_length], [batch_size, num_tiles, series_length, 960]
    
class RawToTile_EfficientNetB6(nn.Module):
    """
    Description: EfficientNetB6 backbone with a few linear layers.
    Args:
        - series_length (int)
        - freeze_backbone (bool): Freezes layers of pretrained backbone
        - pretrain_backbone (bool): Pretrains backbone
    """
    def __init__(self, freeze_backbone=True, pretrain_backbone=True, **kwargs):
        print('- RawToTile_EfficientNetB6')
        super().__init__()
        
        if pretrain_backbone:
            self.conv = EfficientNet.from_pretrained("efficientnet-b6")
        else:
            self.conv = EfficientNet.from_name("efficientnet-b6")
            
        self.avg_pooling = nn.AdaptiveAvgPool2d(1)

        if pretrain_backbone and freeze_backbone:
            for param in self.conv.parameters():
                param.requires_grad = False

        # Initialize additional linear layers
        self.fc = nn.Linear(in_features=2304, out_features=960)
        self.embeddings_to_output = TileEmbeddingsToOutput()
        
    def forward(self, x, *args):
        x = x.float()
        batch_size, num_tiles, series_length, num_channels, height, width = x.size()

        # Run through conv model
        tile_outputs = x.view(batch_size * num_tiles * series_length, num_channels, height, width)
        tile_outputs = self.conv.extract_features(tile_outputs) # [batch_size * num_tiles * series_length, 2304, 7, 7]
        tile_outputs = self.avg_pooling(tile_outputs) # [batch_size * num_tiles * series_length, 2304, 1, 1]
        
        # Save embeddings of dim=960
        tile_outputs = tile_outputs.view(batch_size, num_tiles, series_length, 2304)
        tile_outputs = self.fc(tile_outputs) # [batch_size, num_tiles, series_length, 960]
        embeddings = tile_outputs
        
        # Use linear layers to get dim=1
        tile_outputs = self.embeddings_to_output(tile_outputs, batch_size, num_tiles) # [batch_size, num_tiles, series_length]

        return tile_outputs, embeddings # [batch_size, num_tiles, series_length], [batch_size, num_tiles, series_length, 960]
    
class RawToTile_DeiT(nn.Module):
    """
    Description: Vision Transformer operating on raw inputs to produce tile predictions
    Args:
        - image_size (int, int): size of raw image
        - tile_size (int): size of square tile
    """
    def __init__(self, image_size=(1120,2016), tile_size=224, series_length=1, collapse_series=True, **kwargs):
        print('- RawToTile_DeiT')
        super().__init__()

        self.collapse_series = collapse_series
        self.image_size = (image_size[0] * series_length, image_size[1]) if collapse_series else image_size

        deit_config = transformers.DeiTConfig(image_size=self.image_size, 
                                            patch_size=tile_size, 
                                            num_channels=3, 
                                            num_labels=1,
                                            hidden_size=960,
                                            num_hidden_layers=4,
                                            num_attention_heads=3,
                                            intermediate_size=1024)

        self.deit_model = transformers.DeiTModel(deit_config)

        # Initialize additional linear layers
        self.embeddings_to_output = TileEmbeddingsToOutput()

    def forward(self, x, *args):
        x = x.float()
        batch_size, num_tiles, series_length, num_channels, height, width = x.size()

        # Run through DeiT
        if self.collapse_series:
            x = x.view(batch_size, num_channels, self.image_size[0], self.image_size[1])
            series_length = 1
        else:
            x = x.view(batch_size * series_length, num_channels, self.image_size[0], self.image_size[1])
        
        # Save only last_hidden_state and remove initial class token
        tile_outputs = self.deit_model(x).last_hidden_state[:,-num_tiles-1:-1] # [batch_size * series_length, num_tiles, embedding_size]
                
        # Save embeddings of dim=960
        tile_outputs = tile_outputs.view(batch_size, num_tiles, series_length, 960)
        embeddings = tile_outputs

        # Use linear layers to get dim=1
        tile_outputs = self.embeddings_to_output(tile_outputs, batch_size, num_tiles) # [batch_size, num_tiles, series_length]

        return tile_outputs, embeddings # [batch_size, num_tiles, series_length], [batch_size, num_tiles, series_length, 960]
    
###########################
## TileToTile Models
########################### 
    
class TileToTile_LSTM(nn.Module):
    """
    Description: LSTM that takes tile embeddings and outputs tile predictions
    """
    def __init__(self, **kwargs):
        print('- TileToTile_LSTM')
        super().__init__()
        
        self.lstm = torch.nn.LSTM(input_size=960, hidden_size=960, num_layers=2, batch_first=True)
        
        self.embeddings_to_output = TileEmbeddingsToOutput()
                
    def forward(self, tile_embeddings, *args):
        tile_embeddings = tile_embeddings.float()
        batch_size, num_tiles, series_length, embedding_size = tile_embeddings.size()
                
        # Run through LSTM
        tile_embeddings = tile_embeddings.view(batch_size * num_tiles, series_length, embedding_size).float()
        tile_outputs, (hidden, cell) = self.lstm(tile_embeddings) # [batch_size * num_tiles, series_length, lstm_num_hidden]
        
        # Only save the last time step's outputs
        tile_outputs = tile_outputs[:,-1] # [batch_size * num_tiles, embedding_size]
        
        # Save embeddings of dim=960
        tile_outputs = tile_outputs.view(batch_size, num_tiles, 1, embedding_size)
        embeddings = tile_outputs
        
        # Use linear layers to get dim=1
        tile_outputs = self.embeddings_to_output(tile_outputs, batch_size, num_tiles) # [batch_size, num_tiles, 1]
        
        return tile_outputs, embeddings # [batch_size, num_tiles, 1], [batch_size, num_tiles, 1, 960]

    
class TileToTile_Transformer(nn.Module):
    """
    Description: Transformer that takes tile embeddings and outputs tile predictions
    Args:
        - num_tiles (int): number of tiles in image
    """
    def __init__(self, series_length=1, **kwargs):
        print('- TileToTile_Transformer')
        super().__init__()
        
        # Initialize initial linear layers
        self.fc1 = nn.Linear(in_features=960, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=int(960/series_length))
        self.fc1, self.fc2 = util_fns.init_weights_Xavier(self.fc1, self.fc2)

        # Initialize transformer
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=960, nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer, num_layers=6)
        
        # Initialize additional linear layers
        self.embeddings_to_output = TileEmbeddingsToOutput()
                
    def forward(self, tile_embeddings, *args):
        tile_embeddings = tile_embeddings.float()
        batch_size, num_tiles, series_length, embedding_size = tile_embeddings.size()
        
        # Run through initial linear layers
        tile_embeddings = F.relu(self.fc1(tile_embeddings)) # [batch_size, num_tiles, series_length, 512]
        tile_embeddings = F.relu(self.fc2(tile_embeddings)) # [batch_size, num_tiles, series_length, 960 / series_length]
        
        # Run through transformer
        tile_embeddings = tile_embeddings.view(batch_size, num_tiles, embedding_size).float()
        tile_outputs = self.transformer_encoder(tile_embeddings) # [batch_size, num_tiles, 960]
        
        # Save embeddings of dim=960
        tile_outputs = tile_outputs.view(batch_size, num_tiles, 1, embedding_size)
        embeddings = tile_outputs
        
        # Use linear layers to get dim=1
        tile_outputs = self.embeddings_to_output(tile_outputs, batch_size, num_tiles) # [batch_size, num_tiles, 1]
        
        return tile_outputs, embeddings # [batch_size, num_tiles, 1], [batch_size, num_tiles, 1, 960]
    
    
class TileToTile_DeiT(nn.Module):
    """
    Description: Vision Transformer operating on tiles to produce tile predictions
    Args:
        - num_tiles_height (int): number of tiles that make up the height of the image
        - num_tiles_width (int): number of tiles that make up the width of the image
    """
    def __init__(self, num_tiles_height=5, num_tiles_width=9, **kwargs):
        print('- TileToTile_DeiT')
        super().__init__()
                
        # Initialize initial linear layers
        self.fc1 = nn.Linear(in_features=960, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=256)
        self.fc1, self.fc2 = util_fns.init_weights_Xavier(self.fc1, self.fc2)

        # Initialize DeiT
        self.embeddings_height = num_tiles_height * 16
        self.embeddings_width = num_tiles_width * 16
        
        deit_config = transformers.DeiTConfig(image_size=(self.embeddings_height,self.embeddings_width), 
                                            patch_size=16, 
                                            num_channels=1, 
                                            num_labels=1,
                                            hidden_size=960,
                                            num_hidden_layers=6, 
                                            num_attention_heads=8, 
                                            intermediate_size=2048,
                                            hidden_dropout_prob=0.1,)
        
        self.deit_model = transformers.DeiTModel(deit_config)
        
        # Initialize additional linear layers
        self.embeddings_to_output = TileEmbeddingsToOutput()
                
    def forward(self, tile_embeddings, *args):
        tile_embeddings = tile_embeddings.float()
        batch_size, num_tiles, series_length, embedding_size = tile_embeddings.size()
        
        # Run through initial linear layers
        tile_embeddings = F.relu(self.fc1(tile_embeddings)) # [batch_size, num_tiles, series_length, 512]
        tile_embeddings = F.relu(self.fc2(tile_embeddings)) # [batch_size, num_tiles, series_length, 256]
        
        # Run through DeiT
        tile_embeddings = tile_embeddings.view(batch_size * series_length, 1, self.embeddings_height, self.embeddings_width)
        # Save only last_hidden_state and remove initial class token
        tile_outputs = self.deit_model(tile_embeddings).last_hidden_state[:,1:-1] # [batch_size * series_length, num_tiles, embedding_size]
        
        # Save embeddings of dim=960
        tile_outputs = tile_outputs.contiguous().view(batch_size, num_tiles, series_length, embedding_size)
        embeddings = tile_outputs
        
        # Use linear layers to get dim=1
        tile_outputs = self.embeddings_to_output(tile_outputs, batch_size, num_tiles) # [batch_size, num_tiles, 1]
        
        return tile_outputs, embeddings # [batch_size, num_tiles, series_length], [batch_size, num_tiles, series_length, 960]

    
###########################
## TileToImage Models
########################### 

class TileToImage_LinearOutputs(nn.Module):
    """
    Description: Single linear layer to go from tile outputs to image predictions. Requires that series_length=1
    Args:
        - num_tiles (int): number of tiles in image
    """
    def __init__(self, num_tiles=45, **kwargs):
        print('- TileToImage_LinearOutputs')
        super().__init__()
        
        self.fc1 = nn.Linear(in_features=num_tiles, out_features=1)
        self.fc1, = util_fns.init_weights_Xavier(self.fc1)
        
    def forward(self, tile_embeddings, tile_outputs):
        batch_size, num_tiles, series_length = tile_outputs.size()
        tile_outputs = tile_outputs.view(batch_size, num_tiles)
        
        image_outputs = self.fc1(tile_outputs) # [batch_size, 1]

        return image_outputs, None # [batch_size, 1]
    
class TileToImage_LinearEmbeddings(nn.Module):
    """
    Description: Single linear layer to go from tile embeddings to image predictions. Requires that series_length=1
    Args:
        - num_tiles (int): number of tiles in image
    """
    def __init__(self, num_tiles=45, **kwargs):
        print('- TileToImage_LinearEmbeddings')
        super().__init__()
        
        self.embeddings_to_output = TileEmbeddingsToOutput()
        self.fc1 = nn.Linear(in_features=num_tiles, out_features=1)
        self.fc1, = util_fns.init_weights_Xavier(self.fc1)
        
    def forward(self, tile_embeddings, tile_outputs):
        batch_size, num_tiles, series_length, embedding_size = tile_embeddings.size()
        
        tile_outputs = self.embeddings_to_output(tile_embeddings, batch_size, num_tiles) # [batch_size, num_tiles, 1]
        tile_outputs = tile_outputs.view(batch_size, num_tiles)
        
        image_outputs = self.fc1(tile_outputs) # [batch_size, 1]

        return image_outputs, None # [batch_size, 1]
    
    
