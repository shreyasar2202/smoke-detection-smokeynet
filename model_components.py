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

# File imports
import util_fns

    
class ResNet50(nn.Module):
    """
    Description: Simple model with ResNet backbone and a few linear layers
    Args:
        - series_length: number of sequential video frames to process during training
        - pretrain_backbone: pretrains backbone
        - freeze_backbone: freezes layers on pre-trained backbone
    """
    def __init__(self, series_length=1, freeze_backbone=True, pretrain_backbone=True):
        super().__init__()
        
        print('Model: ResNet50')

        resnet = torchvision.models.resnet50(pretrained=pretrain_backbone)
        resnet.fc = nn.Identity()

        if pretrain_backbone and freeze_backbone:
            for param in resnet.parameters():
                param.requires_grad = False

        self.conv = resnet

        self.fc1 = nn.Linear(in_features=series_length * 2048, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=64)
        self.fc3 = nn.Linear(in_features=64, out_features=1)
        
        util_fns.init_weights([self.fc1, self.fc2, self.fc3])
        
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