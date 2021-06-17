# Torch imports
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pytorch_lightning as pl
import torchvision
import torchmetrics

# Other package imports
import numpy as np
import datetime
import csv
from pathlib import Path

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

    def __init__(self, 
                 model, 
                 learning_rate=0.001, 
                 lr_schedule=True,
                 parsed_args=None, 
                 start_time=datetime.datetime.now()):
        super().__init__()
        
        # Initialize model
        self.model = model
        
        # Initialize model params
        self.criterion = nn.BCEWithLogitsLoss()
        self.learning_rate = learning_rate
        self.lr_schedule = lr_schedule
        self.start_time = start_time
        
        # Save hyperparameters
        self.save_hyperparameters('learning_rate', 'lr_schedule')
        self.save_hyperparameters(parsed_args)
        
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
        
    def forward(self, x):
        return self.model(x).squeeze(dim=2) # [batch_size, num_tiles]
    
    def step(self, batch, split):
        image_name, x, tile_labels, ground_truth_labels, has_xml_labels, has_positive_tiles = batch
        
        # Compute loss
        outputs = self(x)
        loss = self.criterion(outputs, tile_labels)
        
        # Compute predictions
        tile_preds = metric_utils.predict_tile(outputs)
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

        return image_name, outputs, loss, tile_preds, image_preds
        
    def training_step(self, batch, batch_idx):
        image_name, outputs, loss, tile_preds, image_preds = self.step(batch, self.metric_splits[0])
        return loss

    def validation_step(self, batch, batch_idx):
        image_name, outputs, loss, tile_preds, image_preds = self.step(batch, self.metric_splits[1])

    def test_step(self, batch, batch_idx):
        image_name, outputs, loss, tile_preds, image_preds = self.step(batch, self.metric_splits[2])
        return image_name, tile_preds, image_preds
        
    def test_epoch_end(self, test_step_outputs):
        with open(self.logger.log_dir+'/image_preds.csv', 'w') as image_preds_csv:
            image_preds_csv_writer = csv.writer(image_preds_csv)
                
            for image_names, tile_preds, image_preds in test_step_outputs:
                for image_name, tile_pred, image_pred in zip(image_names, tile_preds, image_preds):
                    import pdb; pdb.set_trace()
                    image_preds_csv_writer.writerow([image_name, image_preds.item()])
                    
                    tile_preds_path = self.logger.log_dir+'/tile_preds/'+image_name.split('/')[0]
                    Path(tile_preds_path).mkdir(parents=True, exist_ok=True)
                    np.save(self.logger.log_dir+'/tile_preds/'+image_name+'.npy', tile_preds.cpu().numpy())
            
        total_runtime_mins = (datetime.datetime.now() - self.start_time).total_seconds()/60
        self.log('total_runtime_mins', total_runtime_mins)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        
        if self.lr_schedule:
            scheduler = ReduceLROnPlateau(optimizer)
            return {"optimizer": optimizer, 
                    "lr_scheduler": scheduler, 
                    "monitor": "val_loss"}
        else:
            return optimizer