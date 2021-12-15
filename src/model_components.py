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
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.anchor_utils import AnchorGenerator
from gtrxl_torch.gtrxl_torch import GTrXL

# Other imports 
import numpy as np
import math

# File imports
import resnet
import util_fns
from rcnn import faster_rcnn_noresize, mask_rcnn_noresize, ssd_noresize, retinanet_noresize, ssd



#####################
## Helper Classes
#####################

class TileLoss():
    """
    Description: Class to calculate loss for tiles
    Args:
        - tile_loss_type: type of loss to use. Options: [bce] [focal] [weighted-sensitivity]
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
        elif self.tile_loss_type == 'weighted-sensitivity':
            print('-- Tile Loss: Weighted Sensitivity Loss')
        else:
            raise ValueError('Tile Loss Type not recognized.')
            
    def __call__(self, tile_outputs, tile_labels, num_epoch=0):
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
            
        elif self.tile_loss_type == 'weighted-sensitivity':
            # Source: Yash Pande
            outputs = torch.sigmoid(tile_outputs)
            eps = .001
            sensitivity_weight = 1
            specificity_weight = 1
            
            if num_epoch > 8:
                sensitivity_weight = 2
            if num_epoch > 18:
                sensitivity_weight = 3
            
            outputs = outputs.view(-1)
            labels = tile_labels.view(-1)
            true_positives = (outputs * labels).sum()
            
            sensitivity_loss = 1 - ((true_positives) / (labels.sum() + eps))
            precision_loss = 1 - ((true_positives) / (outputs.sum() + eps))
            
            tile_loss = sensitivity_weight * sensitivity_loss + specificity_weight * precision_loss
        
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
## Backbone Models
#####################
            
class RawToTile_MobileNet(nn.Module):
    """Description: MobileNetV3Large backbone with a few linear layers."""
    
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
    """Description: MobileNetV3Large backbone with a few linear layers. Inputs should not be pre-tiled. Outputs are tiled."""
    
    def __init__(self, 
                 freeze_backbone=True, 
                 pretrain_backbone=True, 
                 backbone_checkpoint_path=None,
                 num_tiles_height=5,
                 num_tiles_width=9,
                 **kwargs):
        print('- RawToTile_MobileNet_NoPreTile')
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
    """Description: ResNet backbone with a few linear layers."""
    
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
    """Description: EfficientNet backbone with a few linear layers."""
    
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
        self.embeddings_to_output = TileEmbeddingsToOutput(self.size_to_embeddings[self.backbone_size])
        
    def forward(self, x, **kwargs):
        x = x.float()
        batch_size, num_tiles, series_length, num_channels, height, width = x.size()

        # Run through conv model
        tile_outputs = x.view(batch_size * num_tiles * series_length, num_channels, height, width)
        tile_outputs = self.conv.extract_features(tile_outputs) # [batch_size * num_tiles * series_length, embedding_size, 7, 7]
        tile_outputs = self.avg_pooling(tile_outputs) # [batch_size * num_tiles * series_length, embedding_size, 1, 1]
        
        tile_outputs, embeddings = self.embeddings_to_output(tile_outputs, batch_size, num_tiles, series_length)
        
        return tile_outputs, embeddings
    
class RawToTile_DeiT(nn.Module):
    """Description: Vision Transformer (Data Efficient Image Transformer) backbone with a few linear layers."""
    
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
## Feature Pyramid Network Models
########################### 
    
class RawToTile_MobileNetFPN(nn.Module):
    """Description: MobileNetV3Large backbone with a Feature Pyramid Network in which the layers of the FPN are concatenated and passed through linear layers."""
    
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
        
        # Results in dictionary with 3 keys:
        # tile_outputs['0'] = [batch_size * num_tiles * series_length, 256, 7, 7]
        # tile_outputs['1'] = [batch_size * num_tiles * series_length, 256, 7, 7]
        # tile_outputs['pool'] = [batch_size * num_tiles * series_length, 256, 4, 4]
        tile_outputs = self.conv(tile_outputs) 
        
        # Run each layer through conv independently
        outputs = []
        for i in range(len(self.keys)):
            outputs.append(self.conv_list[i](tile_outputs[self.keys[i]]))
        
        # Concatenate outputs and run through linear layers
        tile_outputs = torch.cat(outputs, dim=1) # [batch_size * num_tiles * series_length, 49*16*3]
        tile_outputs = F.relu(self.fc(tile_outputs)) # [batch_size * num_tiles * series_length, 960]

        tile_outputs, embeddings = self.embeddings_to_output(tile_outputs, batch_size, num_tiles, series_length)
        
        return tile_outputs, embeddings
    
class RawToTile_MobileNetFPN_Real(nn.Module):
    """Description: MobileNetV3Large backbone with a Feature Pyramid Network in which predictions are generated on each layer of the FPN."""
    
    def __init__(self, 
                 pretrain_backbone=True, 
                 **kwargs):
        print('- RawToTile_MobileNetFPN_Real')
        super().__init__()
                
        self.conv = mobilenet_backbone('mobilenet_v3_large',pretrained=pretrain_backbone,fpn=True, trainable_layers=6)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.embeddings_to_output = TileEmbeddingsToOutput(256)
        
    def forward(self, x, **kwargs):
        x = x.float()
        batch_size, num_tiles, series_length, num_channels, height, width = x.size()

        # Run through conv model
        tile_outputs = x.view(batch_size * num_tiles * series_length, num_channels, height, width)
        
        # Results in dictionary with 3 keys:
        # tile_outputs['0'] = [batch_size * num_tiles * series_length, 256, 7, 7]
        # tile_outputs['1'] = [batch_size * num_tiles * series_length, 256, 7, 7]
        # tile_outputs['pool'] = [batch_size * num_tiles * series_length, 256, 4, 4]
        tile_outputs = self.conv(tile_outputs) 
        
        # Run each FPN layer through linear layers independently
        embeddings = {}
        for key in tile_outputs:
            tile_outputs[key] = self.avgpool(tile_outputs[key]) # [batch_size * num_tiles * series_length, 256, 1, 1]
            tile_outputs[key] = tile_outputs[key].view(batch_size * num_tiles * series_length, -1) # [batch_size * num_tiles * series_length, 256]
            tile_outputs[key], embeddings[key] = self.embeddings_to_output(tile_outputs[key], batch_size, num_tiles, series_length)
        
        return tile_outputs, embeddings

class RawToTile_ResNetFPN(nn.Module):
    """Description: ResNet backbone with a Feature Pyramid Network in which the layers of the FPN are concatenated and passed through linear layers."""
    
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
        
        # Run each layer through conv independently
        outputs = []
        for i in range(len(self.keys)):
            outputs.append(self.conv_list[i](tile_outputs[self.keys[i]]))
        
        # Concatenate outputs and run through linear layers
        tile_outputs = torch.cat(outputs, dim=1) # [batch_size * num_tiles * series_length, 49*16*3]
        tile_outputs = F.relu(self.fc(tile_outputs)) # [batch_size * num_tiles * series_length, 960]

        tile_outputs, embeddings = self.embeddings_to_output(tile_outputs, batch_size, num_tiles, series_length)
        
        return tile_outputs, embeddings

###########################
## Optical Flow Models
########################### 
    
class RawToTile_MobileNet_FlowSimple(nn.Module):
    """Description: Two MobileNetV3Large backbones, one for the raw image and one for optical flow, in which outputs are concatenated and passed through linear layers."""
    
    def __init__(self, 
                 freeze_backbone=True, 
                 pretrain_backbone=True, 
                 backbone_checkpoint_path=None,
                 is_background_removal=False,
                 **kwargs):
        print('- RawToTile_MobileNet_FlowSimple')
        super().__init__()
        
        self.conv = torchvision.models.mobilenet_v3_large(pretrained=pretrain_backbone)
        self.conv.classifier = nn.Identity()
        
        self.conv_flow = torchvision.models.mobilenet_v3_large(pretrained=False)
        self.conv_flow.classifier = nn.Identity()
        if is_background_removal:
            self.conv_flow.features[0][0] = nn.Conv2d(1, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)

        self.fc = nn.Linear(in_features=960*2, out_features=960)
        self.embeddings_to_output = TileEmbeddingsToOutput(960)
        
        if backbone_checkpoint_path is not None:
            self.load_state_dict(util_fns.get_state_dict(backbone_checkpoint_path))
        
        if freeze_backbone:
            for param in self.conv.parameters():
                param.requires_grad = False
        
    def forward(self, x, **kwargs):
        x = x.float()
        batch_size, num_tiles, series_length, num_channels, height, width = x.size()

        # Run raw image through conv model
        tile_outputs = x[:,:,:,:3]
        tile_outputs = tile_outputs.view(batch_size * num_tiles * series_length, 3, height, width)
        tile_outputs = self.conv(tile_outputs) # [batch_size * num_tiles * series_length, tile_embedding_size]
        
        # Run flow through conv model
        tile_outputs_flow = x[:,:,:,3:]
        tile_outputs_flow = tile_outputs_flow.view(batch_size * num_tiles * series_length, num_channels-3, height, width)
        tile_outputs_flow = self.conv_flow(tile_outputs_flow) # [batch_size * num_tiles * series_length, tile_embedding_size]
        
        # Concatenate outputs and run through linear layers
        tile_outputs = torch.cat([tile_outputs, tile_outputs_flow], dim=1) # [batch_size * num_tiles * series_length, tile_embedding_size * 2]
        tile_outputs = F.relu(self.fc(tile_outputs)) # [batch_size * num_tiles * series_length, tile_embedding_size]
        tile_outputs, embeddings = self.embeddings_to_output(tile_outputs, batch_size, num_tiles, series_length)
        
        return tile_outputs, embeddings
    
class RawToTile_MobileNet_Flow(nn.Module):
    """Description: Two MobileNetV3Large backbones, one for the raw image and one for optical flow, in which outputs are kept separate."""
    
    def __init__(self, 
                 freeze_backbone=True, 
                 pretrain_backbone=True, 
                 backbone_checkpoint_path=None,
                 is_background_removal=False,
                 **kwargs):
        print('- RawToTile_MobileNet_Flow')
        super().__init__()
        
        self.conv = torchvision.models.mobilenet_v3_large(pretrained=pretrain_backbone)
        self.conv.classifier = nn.Identity()
        
        self.conv_flow = torchvision.models.mobilenet_v3_large(pretrained=False)
        self.conv_flow.classifier = nn.Identity()
        if is_background_removal:
            self.conv_flow.features[0][0] = nn.Conv2d(1, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)

        self.embeddings_to_output = TileEmbeddingsToOutput(960)
        self.embeddings_to_output_flow = TileEmbeddingsToOutput(960)
                
        if backbone_checkpoint_path is not None:
            self.load_state_dict(util_fns.get_state_dict(backbone_checkpoint_path))
        
        if freeze_backbone:
            for param in self.conv.parameters():
                param.requires_grad = False
        
    def forward(self, x, **kwargs):
        x = x.float()
        batch_size, num_tiles, series_length, num_channels, height, width = x.size()

        # Run raw image through conv model
        tile_outputs = x[:,:,:,:3]
        tile_outputs = tile_outputs.view(batch_size * num_tiles * series_length, 3, height, width)
        tile_outputs = self.conv(tile_outputs) # [batch_size * num_tiles * series_length, tile_embedding_size]
        tile_outputs, embeddings = self.embeddings_to_output(tile_outputs, batch_size, num_tiles, series_length)
        
        # Run flow through conv model
        tile_outputs_flow = x[:,:,:,3:]
        tile_outputs_flow = tile_outputs_flow.view(batch_size * num_tiles * series_length, num_channels-3, height, width)
        tile_outputs_flow = self.conv_flow(tile_outputs_flow) # [batch_size * num_tiles * series_length, tile_embedding_size]
        tile_outputs_flow, embeddings_flow = self.embeddings_to_output_flow(tile_outputs_flow, batch_size, num_tiles, series_length)
        
        # Return raw image outputs and flow outputs separately
        return (tile_outputs, tile_outputs_flow), (embeddings, embeddings_flow)
    
class TileToTile_LSTM_Flow(nn.Module):
    """Description: LSTM that takes tile embeddings for the raw image and flow separately and outputs tile predictions. For use with RawToTile_MobileNet_Flow."""
    
    def __init__(self, tile_embedding_size=960, **kwargs):
        print('- TileToTile_LSTM_Flow')
        super().__init__()
        
        self.lstm = torch.nn.LSTM(input_size=tile_embedding_size, 
                                  hidden_size=tile_embedding_size, 
                                  num_layers=2, 
                                  bidirectional=False, 
                                  batch_first=True)
        
        self.lstm_flow = torch.nn.LSTM(input_size=tile_embedding_size, 
                                  hidden_size=tile_embedding_size, 
                                  num_layers=2, 
                                  bidirectional=False, 
                                  batch_first=True)
        
        self.fc = nn.Linear(in_features=tile_embedding_size*2, out_features=tile_embedding_size)
        self.embeddings_to_output = TileEmbeddingsToOutput(tile_embedding_size)
        self.embeddings_to_output_flow = TileEmbeddingsToOutput(tile_embedding_size)
                
    def forward(self, tile_embeddings, **kwargs):
        tile_embeddings, tile_embeddings_flow = tile_embeddings[0].float(), tile_embeddings[1].float()
        batch_size, num_tiles, series_length, tile_embedding_size = tile_embeddings.size()
        
        # Run raw image outputs through LSTM
        tile_outputs = tile_embeddings.view(batch_size * num_tiles, series_length, tile_embedding_size).float()
        tile_outputs, (hidden, cell) = self.lstm(tile_outputs) # [batch_size * num_tiles, series_length, lstm_num_hidden]
        tile_outputs = tile_outputs[:,-1] # [batch_size * num_tiles, embedding_size]
        tile_outputs = tile_outputs.contiguous()
        tile_outputs, embeddings = self.embeddings_to_output(tile_outputs, batch_size, num_tiles, 1)
                
        # Run flow outputs through LSTM
        tile_outputs_flow = tile_embeddings_flow.view(batch_size * num_tiles, series_length, tile_embedding_size).float()
        tile_outputs_flow, (hidden, cell) = self.lstm(tile_outputs_flow) # [batch_size * num_tiles, series_length, lstm_num_hidden]
        tile_outputs_flow = tile_outputs_flow[:,-1] # [batch_size * num_tiles, embedding_size]
        tile_outputs_flow = tile_outputs_flow.contiguous()
        tile_outputs_flow, embeddings_flow = self.embeddings_to_output_flow(tile_outputs_flow, batch_size, num_tiles, 1)
        
        # Concatenate outputs and run through linear layers
        embeddings = torch.cat([embeddings, embeddings_flow], dim=3) # [batch_size, num_tiles, series_length, tile_embedding_size * 2]
        embeddings = F.relu(self.fc(embeddings)) # [batch_size, num_tiles, series_length, tile_embedding_size]
        
        return (tile_outputs, tile_outputs_flow), embeddings
    
###########################
## TileToTile Models
########################### 
    
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
    
class TileToTile_Transformer(nn.Module):
    """Description: Base transformer module that takes tile embeddings and outputs tile predictions"""
    
    def __init__(self, tile_embedding_size=960, **kwargs):
        print('- TileToTile_Transformer')
        super().__init__()
        
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=tile_embedding_size, nhead=8, dim_feedforward=tile_embedding_size, batch_first=True, norm_first=True)
        self.transformer = torch.nn.TransformerEncoder(self.encoder_layer, num_layers=2, norm=None)
        
        self.embeddings_to_output = TileEmbeddingsToOutput(tile_embedding_size)
                
    def forward(self, tile_embeddings, **kwargs):
        tile_embeddings = tile_embeddings.float()
        batch_size, num_tiles, series_length, tile_embedding_size = tile_embeddings.size()
                
        # Run through LSTM
        tile_embeddings = tile_embeddings.view(batch_size * num_tiles, series_length, tile_embedding_size).float()
        tile_outputs = self.transformer(tile_embeddings) # [batch_size * num_tiles, series_length, dim_feedforward]
        # Only save the last time step's outputs
        tile_outputs = tile_outputs[:,-1,:] # [batch_size * num_tiles, 2, embedding_size]
        tile_outputs = tile_outputs.squeeze(1) # [batch_size * num_tiles, embedding_size]
        # Avoid contiguous error
        tile_outputs = tile_outputs.contiguous()
        
        tile_outputs, embeddings = self.embeddings_to_output(tile_outputs, batch_size, num_tiles, 1)
        
        return tile_outputs, embeddings
    
class TileToTile_GTrXL(nn.Module):
    """Description: Gated Transformer module that takes tile embeddings and outputs tile predictions with attention between T and T-1 of each tile (attention dim = series len)"""
    
    def __init__(self, tile_embedding_size=960, **kwargs):
        print('- TileToTile_GTrXL')
        super().__init__()
        self.gtrxl = GTrXL(d_model=tile_embedding_size, nheads=8, transformer_layers=2, hidden_dims=tile_embedding_size, n_layers=1, batch_first=True)
        self.embeddings_to_output = TileEmbeddingsToOutput(tile_embedding_size)
                
    def forward(self, tile_embeddings, **kwargs):
        tile_embeddings = tile_embeddings.float()
        batch_size, num_tiles, series_length, tile_embedding_size = tile_embeddings.size()
                
        # Run through GTrXL
        tile_embeddings = tile_embeddings.view(batch_size * num_tiles, series_length, tile_embedding_size).float()
        tile_outputs = self.gtrxl(tile_embeddings) # [batch_size * num_tiles, series_length, hidden_dims]            
        # Only save the last time step's outputs
        tile_outputs = tile_outputs[:,-1,:] # [batch_size * num_tiles, 2, embedding_size]
        tile_outputs = tile_outputs.squeeze(1) # [batch_size * num_tiles, embedding_size]
        # Avoid contiguous error
        tile_outputs = tile_outputs.contiguous()
        
        tile_outputs, embeddings = self.embeddings_to_output(tile_outputs, batch_size, num_tiles, 1)
        
        return tile_outputs, embeddings

    
class TileToTile_GTrXL_DispersedAttention(nn.Module):
    """Description: Gated Transformer module that takes tile embeddings and outputs tile predictions with open attention between all tiles T and T-1 (attention dim = series len * num_tiles)"""
    
    def __init__(self, tile_embedding_size=960, **kwargs):
        print('- TileToTile_GTrXL_DispersedAttention')
        super().__init__()
        self.gtrxl = GTrXL(d_model=tile_embedding_size, nheads=4, transformer_layers=2, hidden_dims=tile_embedding_size, n_layers=1, batch_first=True)
        self.time2vec = Time2Vec(hidden_dim=2)
        self.embeddings_to_output = TileEmbeddingsToOutput(tile_embedding_size)
                
    def forward(self, tile_embeddings, **kwargs):
        tile_embeddings = tile_embeddings.float()
        batch_size, num_tiles, series_length, tile_embedding_size = tile_embeddings.size()
                
        # Run through GTrXL
        tile_embeddings = tile_embeddings.view(batch_size, series_length * num_tiles, tile_embedding_size).float()
        # time2vec_embeddings = self.time2vec(tile_embeddings)
        tile_outputs = self.gtrxl(tile_embeddings) # [batch_size, series_length * num_tiles, hidden_dims]            
        # Only save the last time step's outputs
        tile_outputs = tile_outputs[:,-1,:] # [batch_size * num_tiles, 2, embedding_size]
        tile_outputs = tile_outputs.squeeze(1) # [batch_size * num_tiles, embedding_size]
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
                        
        self.fc = nn.Linear(in_features=tile_embedding_size, out_features=self.square_embedding_size**2)
        self.fc, = util_fns.init_weights_Xavier(self.fc)
        
        self.conv = resnet.resnet10()
        
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
                
        # Run through linear layers
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
    """Description: Single linear layer to go from tile outputs to image predictions. Requires that series_length=1"""
    
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
    """Description: Single linear layer to go from tile embeddings to image predictions. Requires that series_length=1"""
    
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
    """Description: Class for any object detection model: [retinanet] [fasterrcnn] [fasterrcnnmobile] [ssd] [maskrcnn]"""
    
    def __init__(self, backbone_size='maskrcnn', pretrain_backbone=True, **kwargs):
        print('- RawToTile_ObjectDetection')
        super().__init__()
        
        num_classes = 91 if pretrain_backbone else 2
                
        # RetinaNet
        if backbone_size == 'retinanet':
            self.model = torchvision.models.detection.retinanet_resnet50_fpn(pretrained=pretrain_backbone, pretrained_backbone=True, num_classes=num_classes, trainable_backbone_layers=5)
            
            if pretrain_backbone:
                # Source: https://datascience.stackexchange.com/questions/92724/fine-tune-the-retinanet-model-in-pytorch
                in_features = self.model.head.classification_head.conv[0].in_channels
                num_anchors = self.model.head.classification_head.num_anchors
                self.model.head.classification_head.num_classes = 2

                cls_logits = torch.nn.Conv2d(in_features, num_anchors * 2, kernel_size = 3, stride=1, padding=1)
                torch.nn.init.normal_(cls_logits.weight, std=0.01)
                torch.nn.init.constant_(cls_logits.bias, -math.log((1 - 0.01) / 0.01)) 
                self.model.head.classification_head.cls_logits = cls_logits
       
        # FasterRCNN
        elif backbone_size == 'fasterrcnn':
            self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=pretrain_backbone, pretrained_backbone=True, num_classes=num_classes, trainable_backbone_layers=5)
            
            if pretrain_backbone:
                # Source: https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
                self.model.roi_heads.box_predictor = FastRCNNPredictor(self.model.roi_heads.box_predictor.cls_score.in_features, 2)
        
        # Faster RCNN Mobile
        elif backbone_size == 'fasterrcnnmobile':
            self.model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained=pretrain_backbone, pretrained_backbone=True, num_classes=num_classes, trainable_backbone_layers=6)
            
            if pretrain_backbone:
                # Source: https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
                self.model.roi_heads.box_predictor = FastRCNNPredictor(self.model.roi_heads.box_predictor.cls_score.in_features, 2)
        
        # SSD
        elif backbone_size == 'ssd':
            self.model = ssd.ssd300_vgg16(pretrained=pretrain_backbone, pretrained_backbone=True, num_classes=num_classes, trainable_backbone_layers=5)
            
            if pretrain_backbone:
                in_features = self.model.head.classification_head.in_channels
                num_anchors = self.model.head.classification_head.num_anchors
                self.model.head.classification_head = ssd.SSDClassificationHead(in_features, num_anchors, 2)
        
        # MaskRCNN
        elif backbone_size == 'maskrcnn':
            self.model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=pretrain_backbone, pretrained_backbone=True, num_classes=num_classes, trainable_backbone_layers=5)
            
            if pretrain_backbone:
                # Source: https://haochen23.github.io/2020/06/fine-tune-mask-rcnn-pytorch.html#.YS_sUdNKhUI
                self.model.roi_heads.box_predictor = FastRCNNPredictor(self.model.roi_heads.box_predictor.cls_score.in_features, 2)
                self.model.roi_heads.mask_predictor = MaskRCNNPredictor(self.model.roi_heads.mask_predictor.conv5_mask.in_channels,256,2)
        
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

class CrazyBackbone(nn.Module):
    """Description: Implements a custom backbone to use with object detection models"""
    
    def __init__(self, num_tiles_height=5, num_tiles_width=8, **kwargs):
        super().__init__()
        
        self.num_tiles_height = num_tiles_height
        self.num_tiles_width = num_tiles_width
        self.out_channels = 516
        
        self.cnn = RawToTile_MobileNet_NoPreTile(num_tiles_height=num_tiles_height, num_tiles_width=num_tiles_width, **kwargs)
        self.vit = TileToTile_ViT(num_tiles_height=num_tiles_height, num_tiles_width=num_tiles_width,**kwargs)
        
    def forward(self, x, **kwargs):
        x = x.unsqueeze(1)
        outputs1, x = self.cnn(x)
        outputs2, x = self.vit(x) # [batch_size, num_tiles, series_length, tile_embedding_size]
        
        batch_size, num_tiles, series_length, tile_embedding_size = x.size()
        x = x.squeeze(2) # [batch_size, num_tiles, tile_embedding_size]
        x = torch.swapaxes(x, 1, 2) # [batch_size, tile_embedding_size, num_tiles]
        x = x.reshape(batch_size, tile_embedding_size, self.num_tiles_height, self.num_tiles_width)
        
        return x, outputs1, outputs2
        
class CrazyFasterRCNN(nn.Module):
    """Description: Uses CrazyBackbone custom backbone with FasterRCNN object detection model"""
    
    def __init__(self, **kwargs):
        print('- CrazyFasterRCNN')
        super().__init__()
        
        backbone = CrazyBackbone(**kwargs)
        
        anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                           aspect_ratios=((0.5, 1.0, 2.0),))
        roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
                                                        output_size=7,
                                                        sampling_ratio=2)

        self.model = faster_rcnn_noresize.FasterRCNN(backbone,
                   num_classes=2,
                   rpn_anchor_generator=anchor_generator,
                   box_roi_pool=roi_pooler)
        
    def forward(self, x, bbox_labels, **kwargs):
        x = x.float()
                
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