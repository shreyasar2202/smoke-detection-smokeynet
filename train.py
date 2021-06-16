import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchvision

import pickle
import numpy as np
from argparse import ArgumentParser

from batched_tiled_dataloader import BatchedTiledDataModule, BatchedTiledDataloader


#####################
## Argument Parser
#####################

parser = ArgumentParser(description='Takes raw wildfire images and saves tiled images')

# Path args
parser.add_argument('--data-path', type=str, default='/userdata/kerasData/data/new_data/batched_tiled_data',
                    help='Path to batched & tiled data')
parser.add_argument('--metadata-path', type=str, default='/userdata/kerasData/data/new_data/batched_tiled_data/metadata.pkl',
                    help='Path to batched & tiled data')
parser.add_argument('--train-split-path', type=str, default='/userdata/kerasData/data/new_data/mask_rcnn_preprocessed/20210420/train_list.txt',
                    help='Path to batched & tiled data')
parser.add_argument('--val-split-path', type=str, default='/userdata/kerasData/data/new_data/mask_rcnn_preprocessed/20210420/val_list.txt',
                    help='Path to batched & tiled data')
parser.add_argument('--test-split-path', type=str, default='/userdata/kerasData/data/new_data/mask_rcnn_preprocessed/20210420/test_list.txt',
                    help='Path to batched & tiled data')

# Dataloader args
parser.add_argument('--batch-size', type=int, default=1,
                    help='Desired height to resize inputted images')
parser.add_argument('--series-length', type=int, default=1,
                    help='Desired height to resize inputted images')
parser.add_argument('--time-range-min', type=int, default=-2400,
                    help='Desired height to resize inputted images')
parser.add_argument('--time-range-max', type=int, default=2400,
                    help='Desired height to resize inputted images')

# Model args
parser.add_argument('--learning-rate', type=float, default=0.001,
                    help='Desired height to resize inputted images')
parser.add_argument('--no-lr-schedule', action='store_true',
                    help='Desired height to resize inputted images')

# Training args
parser.add_argument('--min-epochs', type=int, default=10,
                    help='Desired height to resize inputted images')
parser.add_argument('--max-epochs', type=int, default=50,
                    help='Desired height to resize inputted images')
parser.add_argument('--no-auto-lr-find', action='store_true',
                    help='Desired height to resize inputted images')
parser.add_argument('--no-early-stopping', action='store_true',
                    help='Desired height to resize inputted images')
parser.add_argument('--no-sixteen-bit', action='store_true',
                    help='Desired height to resize inputted images')
parser.add_argument('--no-stochastic-weight-avg', action='store_true',
                    help='Desired height to resize inputted images')
parser.add_argument('--gradient-clip-val', type=float, default=0.5,
                    help='Desired height to resize inputted images')
parser.add_argument('--accumulate-grad-batches', type=int, default=16,
                    help='Desired height to resize inputted images')


#####################
## Model
#####################

class LightningModel(pl.LightningModule):

    def __init__(self, learning_rate=0.001, lr_schedule=True):
        super().__init__()
        resnet = torchvision.models.resnet50(pretrained=True)
        resnet.fc = nn.Identity()
        for param in resnet.parameters():
            param.requires_grad = False
        
        self.conv = resnet
        
        self.fc1 = nn.Linear(in_features=2048, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=64)
        self.fc3 = nn.Linear(in_features=64, out_features=1)
        
        self.criterion = nn.BCEWithLogitsLoss()
        self.learning_rate = learning_rate
        self.lr_schedule = lr_schedule
    
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
        
    def step(self, batch, batch_idx):
        x, y, ground_truth_label, has_xml_label, has_positive_tile = batch
        
        output = self.forward(x).squeeze(dim=2) # [batch_size, num_tiles]
        loss = self.criterion(output, y)
        
        return loss
        
    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx)

    def test_step(self, batch, batch_idx):
        # --------------------------
        # REPLACE WITH YOUR OWN
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        self.log('test_loss', loss)
        # --------------------------

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate)
        
        if self.lr_schedule:
            scheduler = ReduceLROnPlateau(optimizer)
            return {"optimizer": optimizer, 
                    "lr_scheduler": scheduler, 
                    "monitor": "val_loss"}
        else:
            return optimizer

    
#####################
## Main
#####################
    
def main(
         # Path args
         data_path, 
         metadata_path, 
         train_split_path=None, 
         val_split_path=None, 
         test_split_path=None,
    
         # Dataloader args
         batch_size=1, 
         series_length=1, 
         time_range=(-2400,2400), 
    
         # Model args
         learning_rate=0.001,
         lr_schedule=True,
    
         # Trainer args 
         min_epochs=10,
         max_epochs=50,
         auto_lr_find=True,
         early_stopping=True,
         sixteen_bit=True,
         stochastic_weight_avg=True,
         gradient_clip_val=0,
         accumulate_grad_batches=1):
    
    # Initialize data_module
    data_module = BatchedTiledDataModule(
        # Path args
        data_path=data_path,
        metadata_path=metadata_path,
        train_split_path=train_split_path,
        val_split_path=val_split_path,
        test_split_path=test_split_path,
        
        # Dataloader args
        batch_size=batch_size,
        series_length=series_length,
        time_range=time_range)
    
    # Initialize model
    model = LightningModel(learning_rate=learning_rate,
                           lr_schedule=lr_schedule)

    # Implement EarlyStopping
    early_stop_callback = EarlyStopping(
       monitor='val_loss',
       min_delta=0.00,
       patience=3,
       verbose=False,
       mode='max')

    # Initialize a trainer
    trainer = pl.Trainer(
        # Trainer args
        min_epochs=min_epochs,
        max_epochs=max_epochs,
        auto_lr_find=auto_lr_find,
        callbacks=[early_stop_callback] if early_stopping else None,
        precision=16 if sixteen_bit else 32,
        stochastic_weight_avg=stochastic_weight_avg,
        gradient_clip_val=gradient_clip_val,
        accumulate_grad_batches=accumulate_grad_batches,
        
        # Dev args
        fast_dev_run=True, 
        progress_bar_refresh_rate=20,
#         profiler="simple", # "advanced" "pytorch"
        gpus=1)
    
    # Auto find learning rate
    if auto_lr_find:
        trainer.tune(model)
        
    # Train the model ⚡
    trainer.fit(model, data_module)

    
if __name__ == '__main__':
    args = parser.parse_args()
        
    main(
         # Path args
         data_path=args.data_path, 
         metadata_path=args.metadata_path, 
         train_split_path=args.train_split_path, 
         val_split_path=args.val_split_path, 
         test_split_path=args.test_split_path,
         
         # Dataloader args
         batch_size=args.batch_size, 
         series_length=args.series_length, 
         time_range=(args.time_range_min,args.time_range_max), 
        
         # Model args
         learning_rate=args.learning_rate,
         lr_schedule=not args.no_lr_schedule,
         
         # Trainer args
         min_epochs=args.min_epochs,
         max_epochs=args.max_epochs,
         auto_lr_find=not args.no_auto_lr_find,
         early_stopping=not args.no_early_stopping,
         sixteen_bit=not args.no_sixteen_bit,
         stochastic_weight_avg=not args.no_stochastic_weight_avg,
         gradient_clip_val=args.gradient_clip_val,
         accumulate_grad_batches=args.accumulate_grad_batches)