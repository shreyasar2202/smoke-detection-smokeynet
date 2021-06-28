"""
Created by: Anshuman Dewangan
Date: 2021

Description: Different torch models to use with main_model.py
"""

# Torch imports
import torch
from torch import nn
from torch.nn import functional as F
import torchvision

    
class ResNet50(nn.Module):
    """
    Description: Simple model with ResNet backbone and a few linear layers
    Args:
        - series_length: number of sequential video frames to process during training
        - pretrain_backbone: pretrains backbone
        - freeze_backbone: freezes layers on pre-trained backbone
    """
    def __init__(self, series_length=1, freeze_backbone=True, pretrain_backbone=True):
        print('Model: ResNet50')
        super().__init__()

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

        x = x.view(batch_size * num_tiles * series_length, num_channels, height, width)
        x = self.conv(x) # [batch_size * num_tiles * series_length, 2048]

        x = x.view(batch_size, num_tiles, -1) # [batch_size, num_tiles, series_length * 2048]
        x = F.relu(self.fc1(x)) # [batch_size, num_tiles, 512]
        x = F.relu(self.fc2(x)) # [batch_size, num_tiles, 64]
        x = self.fc3(x) # [batch_size, num_tiles, 1]

        return x
    
    def init_weights(self, layers):
        for layer in layers:
            torch.nn.init.xavier_uniform_(layer.weight)
            torch.nn.init.xavier_uniform_(layer.bias.reshape((-1,1)))
            

class MobileNetV3Large(nn.Module):
    """
    Description: Simple model with MobileNetV3 Large backbone and a few linear layers
    Args:
        - series_length: number of sequential video frames to process during training
        - pretrain_backbone: pretrains backbone
        - freeze_backbone: freezes layers on pre-trained backbone
    """
    def __init__(self, series_length=1, freeze_backbone=True, pretrain_backbone=True):
        print('Model: MobileNetV3Large')
        super().__init__()

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

        x = x.view(batch_size * num_tiles * series_length, num_channels, height, width)
        x = self.conv(x) # [batch_size * num_tiles * series_length, 2048]

        x = x.view(batch_size, num_tiles, -1) # [batch_size, num_tiles, series_length * 2048]
        x = F.relu(self.fc1(x)) # [batch_size, num_tiles, 512]
        x = F.relu(self.fc2(x)) # [batch_size, num_tiles, 64]
        x = self.fc3(x) # [batch_size, num_tiles, 1]

        return x
    
    def init_weights(self, layers):
        for layer in layers:
            torch.nn.init.xavier_uniform_(layer.weight)
            torch.nn.init.xavier_uniform_(layer.bias.reshape((-1,1)))