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
    2. ToTile: tile_outputs=[batch_size, num_tiles, series_length or 1], embeddings=[batch_size, num_tiles, series_length or 1, tile_embedding_size]
    3. ToImage: image_outputs=[batch_size, series_length or 1], embeddings=None
"""

# Torch imports
import torch
from torch import nn
from torch.nn import functional as F
import torchvision
import transformers
from efficientnet_pytorch import EfficientNet
from torchvision.models.detection.backbone_utils import mobilenet_backbone, resnet_fpn_backbone
import torchvision.models.detection
import rcnn.faster_rcnn_noresize
import rcnn.retinanet_noresize

# Other imports 
import numpy as np

# File imports
import resnet
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
    Description: Takes embeddings of dim=tile_embedding_size and converts them to outputs of dim=1 using linear layers
    Args:
        - tile_embedding_size (int): size of input embeddings
    """
    def __init__(self, tile_embedding_size=960):
        super().__init__()
        
        self.tile_embedding_size = tile_embedding_size
        
        self.fc1 = nn.Linear(in_features=tile_embedding_size, out_features=np.minimum(512,tile_embedding_size))
        self.fc2 = nn.Linear(in_features=np.minimum(512,tile_embedding_size), out_features=64)
        self.fc3 = nn.Linear(in_features=64, out_features=1)
        
        self.fc1, self.fc2, self.fc3 = util_fns.init_weights_RetinaNet(self.fc1, self.fc2, self.fc3)
        
    def forward(self, tile_embeddings, batch_size, num_tiles, series_length):
        # Save embeddings of dim=tile_embedding_size
        tile_embeddings = tile_embeddings.view(batch_size, num_tiles, series_length, self.tile_embedding_size)
        embeddings = tile_embeddings # [batch_size, num_tiles, series_length, tile_embedding_size]
        
        # Use linear layers to get dim=1
        tile_outputs = F.relu(self.fc1(tile_embeddings)) # [batch_size, num_tiles, series_length, 512]
        tile_outputs = F.relu(self.fc2(tile_outputs)) # [batch_size, num_tiles, series_length, 64]
        tile_outputs = self.fc3(tile_outputs) # [batch_size, num_tiles, series_length, 1]
        
        tile_outputs = tile_outputs.view(batch_size, num_tiles, -1) # [batch_size, num_tiles, series_length]
        
        return tile_outputs, embeddings 
    
class FPNToEmbeddings(nn.Module):
    def __init__(self, out_channels=16):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels=256, out_channels=64, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=out_channels, kernel_size=1)
        
    def forward(self, fpn_outputs):
        fpn_outputs = F.relu(self.conv1(fpn_outputs)) # [batch_size * num_tiles * series_length, 64, 7, 7]
        fpn_outputs = F.relu(self.conv2(fpn_outputs)) # [batch_size * num_tiles * series_length, out_channels, 7, 7]
        fpn_outputs = fpn_outputs.flatten(1) # [batch_size * num_tiles * series_length, 784]
        
        return fpn_outputs
        
        
#####################
## RawToTile - General Models
#####################
            
class RawToTile_MobileNet(nn.Module):
    """
    Description: MobileNetV3Large backbone with a few linear layers.
    Args:
        - freeze_backbone (bool): Freezes layers of pretrained backbone
        - pretrain_backbone (bool): Pretrains backbone on ImageNet
        - backbone_checkpoint_path (str): path to pretrained checkpoint of the model
    """
    def __init__(self, 
                 freeze_backbone=True, 
                 pretrain_backbone=True, 
                 backbone_checkpoint_path=None,
                 **kwargs):
        print('- RawToTile_MobileNet')
        super().__init__()
        
        self.conv = torchvision.models.mobilenet_v3_large(pretrained=pretrain_backbone)
        self.conv.classifier = nn.Identity()

        self.embeddings_to_output = TileEmbeddingsToOutput(960)
        
        if backbone_checkpoint_path is not None:
            self.load_state_dict(util_fns.get_state_dict(backbone_checkpoint_path))
        
        if freeze_backbone:
            for param in self.conv.parameters():
                param.requires_grad = False
        
    def forward(self, x, **kwargs):
        x = x.float()
        batch_size, num_tiles, series_length, num_channels, height, width = x.size()

        # Run through conv model
        tile_outputs = x.view(batch_size * num_tiles * series_length, num_channels, height, width)
        tile_outputs = self.conv(tile_outputs) # [batch_size * num_tiles * series_length, tile_embedding_size]

        tile_outputs, embeddings = self.embeddings_to_output(tile_outputs, batch_size, num_tiles, series_length)
        
        return tile_outputs, embeddings
    
class RawToTile_MobileNet_NoPreTile(nn.Module):
    """
    Description: MobileNetV3Large backbone with a few linear layers. Inputs should not be pre-tiled
    Args:
        - freeze_backbone (bool): Freezes layers of pretrained backbone
        - pretrain_backbone (bool): Pretrains backbone on ImageNet
        - backbone_checkpoint_path (str): path to pretrained checkpoint of the model
    """
    def __init__(self, 
                 freeze_backbone=True, 
                 pretrain_backbone=True, 
                 backbone_checkpoint_path=None,
                 num_tiles_height=5,
                 num_tiles_width=9,
                 **kwargs):
        print('- RawToTile_MobileNetNoTile')
        super().__init__()
        
        self.tile_embedding_size = 960
        self.num_tiles_height = num_tiles_height
        self.num_tiles_width = num_tiles_width
        self.num_tiles = num_tiles_height * num_tiles_width

        self.conv = torchvision.models.mobilenet_v3_large(pretrained=pretrain_backbone)
        self.conv.avgpool = nn.Identity()
        self.conv.classifier = nn.Identity()

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.embeddings_to_output = TileEmbeddingsToOutput(self.tile_embedding_size)
        
        if backbone_checkpoint_path is not None:
            self.load_state_dict(util_fns.get_state_dict(backbone_checkpoint_path))
        
        if freeze_backbone:
            for param in self.conv.parameters():
                param.requires_grad = False
        
    def forward(self, x, **kwargs):
        x = x.float()
        batch_size, series_length, num_channels, height, width = x.size()

        # Run through conv model
        tile_outputs = x.view(batch_size * series_length, num_channels, height, width)
        tile_outputs = self.conv(tile_outputs) # [batch_size * series_length, num_tiles * 960 * 49]
        
        # Tile outputs
        tile_outputs = tile_outputs.view(batch_size, 
                                         series_length,
                                         self.tile_embedding_size,
                                         self.num_tiles_height, 
                                         7, 
                                         self.num_tiles_width,
                                         7)
        tile_outputs = tile_outputs.permute(0,1,3,5,2,4,6).contiguous() # [batch_size, series_length, num_tiles_height, num_tiles_width, tile_embedding_size, 7, 7]
        tile_outputs = tile_outputs.view(batch_size, series_length, self.num_tiles, self.tile_embedding_size, 7, 7)
        
        tile_outputs = self.avgpool(tile_outputs) # [batch_size, series_length, num_tiles, tile_embedding_size, 1, 1]
        tile_outputs = tile_outputs.squeeze().swapaxes(1,2).contiguous() # [batch_size, num_tiles, series_length, tile_embedding_size]
        tile_outputs, embeddings = self.embeddings_to_output(tile_outputs, batch_size, self.num_tiles, series_length)
        
        return tile_outputs, embeddings
    
class RawToTile_ResNet(nn.Module):
    """
    Description: ResNet backbone with a few linear layers.
    Args:
        - freeze_backbone (bool): Freezes layers of pretrained backbone
        - pretrain_backbone (bool): Pretrains backbone on ImageNet
        - backbone_checkpoint_path (str): path to pretrained checkpoint of the model
        - backbone_size (str): how big a model to train. Options: ['small'] ['medium'] ['large']
    """
    def __init__(self, 
                 freeze_backbone=True, 
                 pretrain_backbone=True, 
                 backbone_checkpoint_path=None,  
                 backbone_size='small', 
                 **kwargs):
        print('- RawToTile_ResNet_'+backbone_size)
        super().__init__()
        
        self.backbone_size = backbone_size
        self.size_to_embeddings = {'small': 1000, 'medium': 1000, 'large': 1000}

        if backbone_size == 'small':
            self.conv = torchvision.models.resnet34(pretrained=pretrain_backbone)
        elif backbone_size == 'medium':
            self.conv = torchvision.models.resnet50(pretrained=pretrain_backbone)
        elif backbone_size == 'large':
            self.conv = torchvision.models.resnet152(pretrained=pretrain_backbone)
        else:
            print('RawToTile_ResNet: backbone_size not recognized')
        
        self.conv.classifier = nn.Identity()

        self.embeddings_to_output = TileEmbeddingsToOutput(self.size_to_embeddings[self.backbone_size])
        
        if backbone_checkpoint_path is not None:
            self.load_state_dict(util_fns.get_state_dict(backbone_checkpoint_path))
        
        if freeze_backbone:
            for param in self.conv.parameters():
                param.requires_grad = False
        
    def forward(self, x, **kwargs):
        x = x.float()
        batch_size, num_tiles, series_length, num_channels, height, width = x.size()

        # Run through conv model
        tile_outputs = x.view(batch_size * num_tiles * series_length, num_channels, height, width)
        tile_outputs = self.conv(tile_outputs) # [batch_size * num_tiles * series_length, tile_embedding_size]

        tile_outputs, embeddings = self.embeddings_to_output(tile_outputs, batch_size, num_tiles, series_length)
        
        return tile_outputs, embeddings
    
class RawToTile_EfficientNet(nn.Module):
    """
    Description: EfficientNet backbone with a few linear layers.
    Args:
        - freeze_backbone (bool): Freezes layers of pretrained backbone
        - pretrain_backbone (bool): Loads pretrained backbone on ImageNet
        - backbone_checkpoint_path (str): path to self-pretrained checkpoint of the model
        - backbone_size (str): how big a model to train. Options: ['small'] ['medium'] ['large']
    """
    def __init__(self, 
                 freeze_backbone=True, 
                 pretrain_backbone=True, 
                 backbone_checkpoint_path=None, 
                 backbone_size='small', 
                 **kwargs):
        print('- RawToTile_EfficientNet_'+backbone_size)
        super().__init__()
        
        self.backbone_size = backbone_size
        self.size_to_embeddings = {'small': 1280, 'medium': 1408, 'large': 1792}
        size_to_name = {'small': 'b0', 'medium': 'b2', 'large': 'b4'}
        
        if pretrain_backbone:
            self.conv = EfficientNet.from_pretrained("efficientnet-"+size_to_name[backbone_size])
        else:
            self.conv = EfficientNet.from_name("efficientnet-"+size_to_name[backbone_size])
            
        self.avg_pooling = nn.AdaptiveAvgPool2d(1)

        if pretrain_backbone and freeze_backbone:
            for param in self.conv.parameters():
                param.requires_grad = False

        # Initialize additional linear layers
        self.fc = nn.Linear(in_features=self.size_to_embeddings[self.backbone_size], out_features=512)
        self.fc, = util_fns.init_weights_Xavier(self.fc)
        self.embeddings_to_output = TileEmbeddingsToOutput(self.size_to_embeddings[self.backbone_size])
        
    def forward(self, x, **kwargs):
        x = x.float()
        batch_size, num_tiles, series_length, num_channels, height, width = x.size()

        # Run through conv model
        tile_outputs = x.view(batch_size * num_tiles * series_length, num_channels, height, width)
        tile_outputs = self.conv.extract_features(tile_outputs) # [batch_size * num_tiles * series_length, embedding_size, 7, 7]
        tile_outputs = self.avg_pooling(tile_outputs) # [batch_size * num_tiles * series_length, embedding_size, 1, 1]
        
#         tile_outputs = tile_outputs.squeeze() # [batch_size * num_tiles * series_length, embedding_size]
#         tile_outputs = F.relu(self.fc(tile_outputs)) # [batch_size * num_tiles * series_length, 512]
        tile_outputs, embeddings = self.embeddings_to_output(tile_outputs, batch_size, num_tiles, series_length)
        
        return tile_outputs, embeddings
    
class RawToTile_DeiT(nn.Module):
    """
    Description: Vision Transformer operating on raw inputs to produce tile predictions
    Args:
        - freeze_backbone (bool): Freezes layers of pretrained backbone
        - pretrain_backbone (bool): Loads pretrained backbone on ImageNet
        - backbone_checkpoint_path (str): path to self-pretrained checkpoint of the model
        - backbone_size (str): how big a model to train. Options: ['small'] ['medium'] ['large']
    """
    def __init__(self, 
                 freeze_backbone=True, 
                 pretrain_backbone=True, 
                 backbone_checkpoint_path=None, 
                 backbone_size='small', 
                 **kwargs):
        print('- RawToTile_DeiT_'+backbone_size)
        super().__init__()
        
        self.backbone_size = backbone_size
        self.size_to_embeddings = {'small': 192, 'medium': 384, 'large': 768}
        size_to_name = {'small': 'tiny', 'medium': 'small', 'large': 'base'}

        if pretrain_backbone:
            self.deit_model = transformers.DeiTModel.from_pretrained('facebook/deit-'+size_to_name[backbone_size]+'-distilled-patch16-224')
        else:
            deit_config = transformers.DeiTConfig.from_pretrained('facebook/deit-'+size_to_name[backbone_size]+'-distilled-patch16-224')
            self.deit_model = transformers.DeiTModel(deit_config)
        
        self.embeddings_to_output = TileEmbeddingsToOutput(self.size_to_embeddings[self.backbone_size])

    def forward(self, x, **kwargs):
        x = x.float()
        batch_size, num_tiles, series_length, num_channels, height, width = x.size()

        # Run through DeiT
        x = x.view(batch_size * series_length * num_tiles, num_channels, height, width)
        tile_outputs = self.deit_model(x).pooler_output # [batch_size * series_length * num_tiles, embedding_size]
                
        tile_outputs, embeddings = self.embeddings_to_output(tile_outputs, batch_size, num_tiles, series_length)
        
        return tile_outputs, embeddings

###########################
## RawToTile - Feature Pyramid Network
########################### 
    
class RawToTile_MobileNetFPN(nn.Module):
    """
    Description: MobileNetV3Large backbone with a Feature Pyramid Network a few linear layers.
    Args:
        - pretrain_backbone (bool): Pretrains backbone on ImageNet
    """
    def __init__(self, 
                 pretrain_backbone=True, 
                 **kwargs):
        print('- RawToTile_MobileNetFPN')
        super().__init__()
        
        self.keys = ['0', '1', 'pool']
        
        self.conv = mobilenet_backbone('mobilenet_v3_large',pretrained=pretrain_backbone,fpn=True, trainable_layers=6)
        self.conv_list = nn.ModuleList([FPNToEmbeddings(16), FPNToEmbeddings(16), FPNToEmbeddings(49)])
        self.fc = nn.Linear(in_features=49*16*3, out_features=960)

        self.embeddings_to_output = TileEmbeddingsToOutput(960)
        
    def forward(self, x, **kwargs):
        x = x.float()
        batch_size, num_tiles, series_length, num_channels, height, width = x.size()

        # Run through conv model
        tile_outputs = x.view(batch_size * num_tiles * series_length, num_channels, height, width)
        
        # tile_outputs['0'] = [batch_size * num_tiles * series_length, 256, 7, 7]
        # tile_outputs['1'] = [batch_size * num_tiles * series_length, 256, 7, 7]
        # tile_outputs['pool'] = [batch_size * num_tiles * series_length, 256, 4, 4]
        tile_outputs = self.conv(tile_outputs) 
        
        outputs = []
        for i in range(len(self.keys)):
            outputs.append(self.conv_list[i](tile_outputs[self.keys[i]]))
        
        tile_outputs = torch.cat(outputs, dim=1) # [batch_size * num_tiles * series_length, 49*16*3]
        tile_outputs = F.relu(self.fc(tile_outputs)) # [batch_size * num_tiles * series_length, 960]

        tile_outputs, embeddings = self.embeddings_to_output(tile_outputs, batch_size, num_tiles, series_length)
        
        return tile_outputs, embeddings

class RawToTile_ResNetFPN(nn.Module):
    """
    Description: ResNet backbone with a Feature Pyramid Network a few linear layers.
    Args:
        - pretrain_backbone (bool): Pretrains backbone on ImageNet
        - backbone_size (str): how big a model to train. Options: ['small'] ['medium'] ['large']
    """
    def __init__(self, 
                 pretrain_backbone=True, 
                 backbone_size='small',
                 **kwargs):
        print('- RawToTile_ResNetFPN')
        super().__init__()
        
        self.keys = ['0', '1', '2', '3', 'pool']
        size_to_name = {'small': 'resnet18', 'medium': 'resnet34', 'large': 'resnext50_32x4d'}
        
        self.conv = resnet_fpn_backbone(size_to_name[backbone_size],pretrained=pretrain_backbone, trainable_layers=5)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=28)
        self.conv_list = nn.ModuleList([FPNToEmbeddings(1), FPNToEmbeddings(1), FPNToEmbeddings(4), FPNToEmbeddings(16), FPNToEmbeddings(49)])
        self.fc = nn.Linear(in_features=28*28*5, out_features=960)

        self.embeddings_to_output = TileEmbeddingsToOutput(960)
        
    def forward(self, x, **kwargs):
        x = x.float()
        batch_size, num_tiles, series_length, num_channels, height, width = x.size()

        # Run through conv model
        tile_outputs = x.view(batch_size * num_tiles * series_length, num_channels, height, width)
        
        # tile_outputs['0'] = [batch_size * num_tiles * series_length, 256, 56, 56]
        # tile_outputs['1'] = [batch_size * num_tiles * series_length, 256, 28, 28]
        # tile_outputs['2'] = [batch_size * num_tiles * series_length, 256, 14, 14]
        # tile_outputs['3'] = [batch_size * num_tiles * series_length, 256, 7, 7]
        # tile_outputs['pool'] = [batch_size * num_tiles * series_length, 256, 4, 4]
        tile_outputs = self.conv(tile_outputs) 
        tile_outputs['0'] = self.avgpool(tile_outputs['0']) # [batch_size * num_tiles * series_length, 256, 28, 28]
        
        outputs = []
        for i in range(len(self.keys)):
            outputs.append(self.conv_list[i](tile_outputs[self.keys[i]]))
        
        tile_outputs = torch.cat(outputs, dim=1) # [batch_size * num_tiles * series_length, 49*16*3]
        tile_outputs = F.relu(self.fc(tile_outputs)) # [batch_size * num_tiles * series_length, 960]

        tile_outputs, embeddings = self.embeddings_to_output(tile_outputs, batch_size, num_tiles, series_length)
        
        return tile_outputs, embeddings
    
###########################
## TileToTile Models
########################### 

class TileToTile_ViT(nn.Module):
    """Description: Vision Transformer operating on tiles to produce tile predictions"""
    
    def __init__(self, num_tiles_height=5, num_tiles_width=9, tile_embedding_size=960, **kwargs):
        print('- TileToTile_ViT')
        super().__init__()
                        
        # Initialize initial linear layers
        patch_size = int(np.floor(np.sqrt(tile_embedding_size)))
        self.fc1 = nn.Linear(in_features=tile_embedding_size, out_features=patch_size*patch_size)
        self.fc1, = util_fns.init_weights_Xavier(self.fc1)

        # Initialize ViT
        self.embeddings_height = num_tiles_height * patch_size
        self.embeddings_width = num_tiles_width * patch_size
        
        ViT_config = transformers.ViTConfig(image_size=(self.embeddings_height,self.embeddings_width), 
                                            patch_size=patch_size, 
                                            num_channels=1, 
                                            num_labels=1,
                                            hidden_size=516,
                                            num_hidden_layers=6, 
                                            num_attention_heads=6, 
                                            intermediate_size=1536,
                                            hidden_dropout_prob=0.1)
        
        self.ViT_model = transformers.ViTModel(ViT_config)
        
        # Initialize additional linear layers
        self.embeddings_to_output = TileEmbeddingsToOutput(516)
                
    def forward(self, tile_embeddings, **kwargs):
        tile_embeddings = tile_embeddings.float()
        batch_size, num_tiles, series_length, tile_embedding_size = tile_embeddings.size()
                
        # Run through initial linear layers
        tile_embeddings = F.relu(self.fc1(tile_embeddings)) # [batch_size, num_tiles, series_length, 512]
        
        # Run through ViT
        tile_embeddings = tile_embeddings.view(batch_size * series_length, 1, self.embeddings_height, self.embeddings_width)
        # Save only last_hidden_state and remove initial class token
        tile_outputs = self.ViT_model(tile_embeddings).last_hidden_state[:,1:] # [batch_size * series_length, num_tiles, embedding_size]
        # Avoid contiguous error
        tile_outputs = tile_outputs.contiguous()
        
        tile_outputs, embeddings = self.embeddings_to_output(tile_outputs, batch_size, num_tiles, series_length)
        
        return tile_outputs, embeddings
    
class TileToTile_LSTM(nn.Module):
    """Description: LSTM that takes tile embeddings and outputs tile predictions"""
    
    def __init__(self, tile_embedding_size=960, **kwargs):
        print('- TileToTile_LSTM')
        super().__init__()
        
        self.lstm = torch.nn.LSTM(input_size=tile_embedding_size, 
                                  hidden_size=tile_embedding_size, 
                                  num_layers=2, 
                                  bidirectional=False, 
                                  batch_first=True)
        
        self.embeddings_to_output = TileEmbeddingsToOutput(tile_embedding_size)
                
    def forward(self, tile_embeddings, **kwargs):
        tile_embeddings = tile_embeddings.float()
        batch_size, num_tiles, series_length, tile_embedding_size = tile_embeddings.size()
                
        # Run through LSTM
        tile_embeddings = tile_embeddings.view(batch_size * num_tiles, series_length, tile_embedding_size).float()
        tile_outputs, (hidden, cell) = self.lstm(tile_embeddings) # [batch_size * num_tiles, series_length, lstm_num_hidden]
        
        # Only save the last time step's outputs
        tile_outputs = tile_outputs[:,-1] # [batch_size * num_tiles, embedding_size]
        # Avoid contiguous error
        tile_outputs = tile_outputs.contiguous()
        
        tile_outputs, embeddings = self.embeddings_to_output(tile_outputs, batch_size, num_tiles, 1)
        
        return tile_outputs, embeddings
    
class TileToTile_ResNet3D(nn.Module):
    """Description: 3D ResNet operating on tiles to produce tile predictions"""
    
    def __init__(self, num_tiles_height=5, num_tiles_width=9, tile_embedding_size=960, **kwargs):
        print('- TileToTile_ResNet3D')
        super().__init__()
        
        self.tile_embedding_size = 512
        self.square_embedding_size = 31
        self.num_tiles_height = num_tiles_height
        self.num_tiles_width = num_tiles_width
                        
        # Initialize initial linear layers
        self.fc = nn.Linear(in_features=tile_embedding_size, out_features=self.square_embedding_size**2)
        self.fc, = util_fns.init_weights_Xavier(self.fc)
        
        # Initialize ResNet3D
        self.conv = resnet.resnet10()
        
        # Initialize additional linear layers
        self.embeddings_to_output = TileEmbeddingsToOutput(self.tile_embedding_size)
                
    def forward(self, tile_embeddings, **kwargs):
        tile_embeddings = tile_embeddings.float()
        batch_size, num_tiles, series_length, tile_embedding_size = tile_embeddings.size()
        
        # Run through initial linear layers
        tile_embeddings = F.relu(self.fc(tile_embeddings)) # [batch_size, num_tiles, series_length, squared_embedding_size**2]
        
        # Run through ResNet3D
        tile_embeddings = tile_embeddings.swapaxes(1,2).contiguous() # [batch_size, series_length, num_tiles, self.square_embedding_size**2]
        tile_embeddings = tile_embeddings.view(batch_size, 1, series_length, self.num_tiles_height*self.square_embedding_size, self.num_tiles_width*self.square_embedding_size)
        tile_outputs = self.conv(tile_embeddings) # [batch_size, tile_embedding_size, 1, num_tiles_height, num_tiles_width]
                
        tile_outputs = tile_outputs.view(batch_size, self.tile_embedding_size, num_tiles)
        tile_outputs = tile_outputs.swapaxes(1,2).contiguous() # [batch_size, num_tiles, tile_embedding_size]
        tile_outputs, embeddings = self.embeddings_to_output(tile_outputs, batch_size, num_tiles, 1)
        
        return tile_outputs, embeddings 

###########################
## TileToTileImage Models
########################### 

class TileToTileImage_SpatialViT(nn.Module):
    """Description: Vision Transformer operating on tiles to produce tile and image predictions"""
    
    def __init__(self, num_tiles_height=5, num_tiles_width=9, tile_embedding_size=960, **kwargs):
        print('- TileToTileImage_SpatialViT')
        super().__init__()
                        
        # Initialize initial linear layers
        patch_size = int(np.floor(np.sqrt(tile_embedding_size)))
        self.fc1 = nn.Linear(in_features=tile_embedding_size, out_features=patch_size*patch_size)
        self.fc1, = util_fns.init_weights_Xavier(self.fc1)

        # Initialize ViT
        self.embeddings_height = num_tiles_height * patch_size
        self.embeddings_width = num_tiles_width * patch_size
        
        ViT_config = transformers.ViTConfig(image_size=(self.embeddings_height,self.embeddings_width), 
                                            patch_size=patch_size, 
                                            num_channels=1, 
                                            num_labels=1,
                                            hidden_size=516,
                                            num_hidden_layers=6, 
                                            num_attention_heads=6, 
                                            intermediate_size=1536,
                                            hidden_dropout_prob=0.1)
        
        self.ViT_model = transformers.ViTModel(ViT_config)
        
        # Initialize additional linear layers
        self.embeddings_to_output = TileEmbeddingsToOutput(516)
                
    def forward(self, tile_embeddings, **kwargs):
        tile_embeddings = tile_embeddings.float()
        batch_size, num_tiles, series_length, tile_embedding_size = tile_embeddings.size()
                
        # Run through initial linear layers
        tile_embeddings = F.relu(self.fc1(tile_embeddings)) # [batch_size, num_tiles, series_length, 512]
        
        # Run through ViT
        tile_embeddings = tile_embeddings.view(batch_size * series_length, 1, self.embeddings_height, self.embeddings_width)
        # Save only last_hidden_state 
        tile_outputs = self.ViT_model(tile_embeddings).last_hidden_state # [batch_size * series_length, num_tiles+1, embedding_size]
        # Avoid contiguous error
        tile_outputs = tile_outputs.contiguous()
        
        tile_outputs, embeddings = self.embeddings_to_output(tile_outputs, batch_size, num_tiles+1, series_length)
        
        # Split CLS token
        image_outputs = tile_outputs[:,0]
        tile_outputs = tile_outputs[:,1:]
        
        return tile_outputs, image_outputs
    
class TileToTileImage_ViViT(nn.Module):
    """Description: Video Vision Transformer operating on tiles to produce tile predictions"""
    
    def __init__(self, num_tiles_height=5, num_tiles_width=9, tile_embedding_size=960, series_length=1, **kwargs):
        print('- TileToTileImage_ViViT')
        super().__init__()
                        
        # Initialize initial linear layers
        patch_size = int(np.floor(np.sqrt(tile_embedding_size)))
        self.fc1 = nn.Linear(in_features=tile_embedding_size, out_features=patch_size*patch_size)
        self.fc1, = util_fns.init_weights_Xavier(self.fc1)

        # Initialize ViT
        self.embeddings_height = num_tiles_height * patch_size
        self.embeddings_width = num_tiles_width * patch_size
        
        ViT_config = transformers.ViTConfig(image_size=(self.embeddings_height*series_length, self.embeddings_width), 
                                            patch_size=patch_size, 
                                            num_channels=1, 
                                            num_labels=1,
                                            hidden_size=516,
                                            num_hidden_layers=6, 
                                            num_attention_heads=6, 
                                            intermediate_size=1536,
                                            hidden_dropout_prob=0.1)
        
        self.ViT_model = transformers.ViTModel(ViT_config)
        
        # Initialize additional linear layers
        self.embeddings_to_output = TileEmbeddingsToOutput(516*series_length)
                
    def forward(self, tile_embeddings, **kwargs):
        tile_embeddings = tile_embeddings.float()
        batch_size, num_tiles, series_length, tile_embedding_size = tile_embeddings.size()
        
        # Run through initial linear layers
        tile_embeddings = F.relu(self.fc1(tile_embeddings)) # [batch_size, num_tiles, series_length, 1764]
        
        # Run through ViT
        tile_embeddings = tile_embeddings.view(batch_size, 1, self.embeddings_height * series_length, self.embeddings_width)
        # Save only last_hidden_state and remove initial class token
        tile_outputs = self.ViT_model(tile_embeddings).last_hidden_state # [batch_size, (num_tiles+1)*series_length, embedding_size]
        # Avoid contiguous error
        tile_outputs = tile_outputs.contiguous()
        
        tile_outputs = tile_outputs.view(batch_size, num_tiles+1, -1)
        tile_outputs, embeddings = self.embeddings_to_output(tile_outputs, batch_size, num_tiles+1, 1)
        
        # Split CLS token
        image_outputs = tile_outputs[:,0]
        tile_outputs = tile_outputs[:,1:]
        
        return tile_outputs, image_outputs 
    
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
        
    def forward(self, tile_embeddings, tile_outputs, **kwargs):
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
    def __init__(self, num_tiles=45, tile_embedding_size=960, **kwargs):
        print('- TileToImage_LinearEmbeddings')
        super().__init__()
        
        self.embeddings_to_output = TileEmbeddingsToOutput(516)
        
        self.fc1 = nn.Linear(in_features=num_tiles, out_features=1)
        self.fc1, = util_fns.init_weights_Xavier(self.fc1)
        
    def forward(self, tile_embeddings, tile_outputs, **kwargs):
        batch_size, num_tiles, series_length, tile_embedding_size = tile_embeddings.size()
        
        tile_outputs, _ = self.embeddings_to_output(tile_embeddings, batch_size, num_tiles, 1) # [batch_size, num_tiles, 1]
        tile_outputs = tile_outputs.view(batch_size, num_tiles)
        
        image_outputs = self.fc1(tile_outputs) # [batch_size, 1]

        return image_outputs, None # [batch_size, 1]

###########################
## Object Detection Models
########################### 
    
class RawToTile_ObjectDetection(nn.Module):
    def __init__(self, backbone_size='maskrcnn', **kwargs):
        print('- RawToTile_ObjectDetection')
        super().__init__()
        
        if backbone_size == 'retinanet_original':
            self.model = torchvision.models.detection.retinanet_resnet50_fpn(pretrained=False, num_classes=2, pretrained_backbone=True, trainable_backbone_layers=5)
        elif backbone_size == 'fasterrcnn_original':
            self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, num_classes=2, pretrained_backbone=True, trainable_backbone_layers=5)
        elif backbone_size == 'fasterrcnnmobile_original':
            self.model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained=False, num_classes=2, pretrained_backbone=True, trainable_backbone_layers=6)
        elif backbone_size == 'retinanet':
            self.model = rcnn.retinanet_noresize.retinanet_resnet50_fpn(pretrained=False, num_classes=2, pretrained_backbone=True, trainable_backbone_layers=5)
        elif backbone_size == 'fasterrcnn':
            self.model = rcnn.faster_rcnn_noresize.fasterrcnn_resnet50_fpn(pretrained=False, num_classes=2, pretrained_backbone=True, trainable_backbone_layers=5)
        elif backbone_size == 'fasterrcnnmobile':
            self.model = rcnn.faster_rcnn_noresize.fasterrcnn_mobilenet_v3_large_fpn(pretrained=False, num_classes=2, pretrained_backbone=True, trainable_backbone_layers=6)
        else:
            print('RawToTile_ObjectDetection: backbone_size not recognized.')
        
    def forward(self, x, bbox_labels, **kwargs):
        x = x.float()
        batch_size, series_length, num_channels, height, width = x.size()
                
        x = [item for sublist in x for item in sublist]
        bbox_labels = [item for sublist in bbox_labels for item in sublist]
                                                
        # losses: dict only returned when training, not during inference. Keys: ['loss_classifier', 'loss_box_reg', 'loss_mask', 'loss_objectness', 'loss_rpn_box_reg']
        # outputs: list of dict of len batch_size. Keys: ['boxes', 'labels', 'scores', 'masks']
        outputs = self.model(x, bbox_labels)
                
        # If training
        if type(outputs) is dict:
            return None, outputs
        else:
            return outputs, {}
    
class RawToTile_CustomMaskRCNN(nn.Module):
    def __init__(self, **kwargs):
        print('- RawToTile_CustomMaskRCNN')
        super().__init__()
        
        pretrain_object_detection = False
        
        self.model = custom_maskrcnn_resnet50_fpn(pretrained=pretrain_object_detection, 
                                                  num_classes=2, 
                                                  pretrained_backbone=True, 
                                                  trainable_backbone_layers=5)
        
        if pretrain_object_detection:
        # Source: https://haochen23.github.io/2020/06/fine-tune-mask-rcnn-pytorch.html#.YRN4JdNKhUI
            # Replace FastRCNN head
            self.model.roi_heads.box_predictor = FastRCNNPredictor(self.model.roi_heads.box_predictor.cls_score.in_features, num_classes)
            
            # Replace MaskRCNN head with hidden_dim=256 and num_classes=2
            self.model.roi_heads.mask_predictor = MaskRCNNPredictor(self.model.roi_heads.mask_predictor.conv5_mask.in_channels,256,2)
        
    def forward(self, x, bbox_labels, **kwargs):
        x = x.float()
        batch_size, series_length, num_channels, height, width = x.size()
        
        x = [item for sublist in x for item in sublist]
        bbox_labels = [item for sublist in bbox_labels for item in sublist]
        
        import pdb; pdb.set_trace()
                        
        # losses: dict only returned when training, not during inference. Keys: ['loss_classifier', 'loss_box_reg', 'loss_mask', 'loss_objectness', 'loss_rpn_box_reg']
        # outputs: list of dict of len batch_size. Keys: ['boxes', 'labels', 'scores', 'masks']
        losses, outputs = self.model(x, bbox_labels)
                        
        return outputs, losses