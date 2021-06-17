# Torch imports
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pytorch_lightning as pl
import torchvision
import torchmetrics

# File imports
import metric_utils


class TileResNet(nn.Module):
    def __init__(self, freeze_backbone=True):
        super().__init__()
        
        resnet = torchvision.models.resnet50(pretrained=True)
        resnet.fc = nn.Identity()
        
        if freeze_backbone:
            for param in resnet.parameters():
                param.requires_grad = False
        
        self.conv = resnet
        
        self.fc1 = nn.Linear(in_features=2048, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=64)
        self.fc3 = nn.Linear(in_features=64, out_features=1)

    def forward(self, x):
        x = x.float()
        batch_size, num_tiles, series_length, num_channels, height, width = x.size()
        
        x = x.view(batch_size * num_tiles, num_channels * series_length, height, width)
        x = self.conv(x) # [batch_size * num_tiles * series_length, 2048]
        
        x = x.view(batch_size, num_tiles, -1) # [batch_size, num_tiles, 2048]
        x = F.relu(self.fc1(x)) # [batch_size, num_tiles, 512]
        x = F.relu(self.fc2(x)) # [batch_size, num_tiles, 64]
        x = self.fc3(x) # [batch_size, num_tiles, 1]
        
        return x

    
class LightningModel(pl.LightningModule):

    def __init__(self, learning_rate=0.001, lr_schedule=True, freeze_backbone=True):
        super().__init__()
        
        # Initialize model
        self.model = TileResNet(freeze_backbone=freeze_backbone)
        
        # Initialize model params
        self.criterion = nn.BCEWithLogitsLoss()
        self.learning_rate = learning_rate
        self.lr_schedule = lr_schedule        
        
        # Initialize evaluation metrics
        self.metrics = {}
        self.metric_splits = ['train/', 'val/', 'test/']
        self.metric_titles = ['tile_', 'image-gt_', 'image-xml_', 'image-pos-tile_']
        self.metric_labels = ['accuracy', 'precision', 'recall', 'f1']
        self.metric_functions = [torchmetrics.Accuracy, torchmetrics.Precision, torchmetrics.Recall, torchmetrics.F1]
        
        for split in self.metric_splits:
            for title in self.metric_titles:
                for label, func in zip(self.metric_labels, self.metric_functions):
                    self.metrics[split+title+label] = func(mdmc_average='global') if title == self.metric_titles[0] else func()
        
    
    def step(self, batch, split):
        x, tile_labels, ground_truth_labels, has_xml_labels, has_positive_tiles = batch
        
        # Compute loss
        output = self.model(x).squeeze(dim=2) # [batch_size, num_tiles]
        loss = self.criterion(output, tile_labels)
        
        # Compute predictions
        tile_preds = metric_utils.predict_tile(output)
        image_preds = metric_utils.predict_image_from_tile_preds(tile_preds)
        
        # Compute metrics
        for label in self.metric_labels:
            self.metrics[split+self.metric_titles[0]+label].to(self.device)(tile_preds, tile_labels.int())
            self.metrics[split+self.metric_titles[1]+label].to(self.device)(image_preds, ground_truth_labels)
            self.metrics[split+self.metric_titles[2]+label].to(self.device)(image_preds, has_xml_labels)
            self.metrics[split+self.metric_titles[3]+label].to(self.device)(image_preds, has_positive_tiles)
            
        # Log metrics
        self.log(split+'loss', loss, on_step=(split==self.metric_splits[0]),on_epoch=True)
        
        for title in self.metric_titles:
            for label in self.metric_labels:
                name = split+title+label
                self.log(name, self.metrics[name], on_step=False, on_epoch=True)
        
        return loss
        
    def training_step(self, batch, batch_idx):
        return self.step(batch, self.metric_splits[0])

    def validation_step(self, batch, batch_idx):
        self.step(batch, self.metric_splits[1])

    def test_step(self, batch, batch_idx):
        self.step(batch, self.metric_splits[2])

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        
        if self.lr_schedule:
            scheduler = ReduceLROnPlateau(optimizer)
            return {"optimizer": optimizer, 
                    "lr_scheduler": scheduler, 
                    "monitor": "val_loss"}
        else:
            return optimizer