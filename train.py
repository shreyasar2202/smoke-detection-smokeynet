import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
import torchvision

import pickle
import numpy as np

from batched_tiled_dataloader import BatchedTiledDataModule, BatchedTiledDataloader



class LightningModel(pl.LightningModule):

    def __init__(self):
        super().__init__()
        resnet = torchvision.models.resnet50(pretrained=True)
        resnet.fc = nn.Identity()
        self.conv = resnet
        self.fc1 = nn.Linear(in_features=2048, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=64)
        self.fc3 = nn.Linear(in_features=64, out_features=1)

    def forward(self, x, embeddings_out=False):
        x = x.float()
        batch_size, grid_size, C, H, W = x.size()
        x = x.view(batch_size * grid_size, C, H, W)
        x = self.conv(x)
        x = x.view(batch_size, grid_size, -1)
        if embeddings_out:
            return x
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
        
    def custom_loss(self, outputs, labels):
        binary_loss = nn.BCEWithLogitsLoss()
        pos_idx = labels > 0
        pos_labels = labels[pos_idx]
        neg_labels = labels[~pos_idx]
        pos_outputs = outputs[pos_idx]
        neg_outputs = outputs[~pos_idx]
        loss = 2 * (binary_loss(pos_outputs, pos_labels) if len(pos_labels)>0 else 0)
        return loss + (binary_loss(neg_outputs, neg_labels)if len(neg_labels)>0 else 0)
        
    def training_step(self, batch, batch_idx):
        x, y, ground_truth_label, has_xml_label, has_positive_tile = batch
        
        pos_idx = y[0, :] > 0
        neg_idx = torch.where(~pos_idx)
        neg_idx = neg_idx[0][torch.tensor([.1]*32-pos_idx.sum()).multinomial(num_samples=32-pos_idx.sum())]
#         neg_idx = torch.random.choice(neg_idx[0], size=32-pos_idx.sum(), replace=False)
        pos_idx[neg_idx] = True
        inputs = torch.as_tensor(x[:, pos_idx, -1, :])
        labels = y[:, pos_idx].float().unsqueeze(2)
        outputs = self.forward(inputs)
        loss = self.custom_loss(outputs, labels)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, ground_truth_label, has_xml_label, has_positive_tile = batch
        
        pos_idx = y[0, :] > 0
        neg_idx = np.where(~pos_idx)
        neg_idx = np.random.choice(neg_idx[0], size=32-pos_idx.sum(), replace=False)
        pos_idx[neg_idx] = True
        inputs = torch.as_tensor(x[:, pos_idx, -1, :])
        labels = y[:, pos_idx].float().unsqueeze(2)
        outputs = self.forward(inputs)
        loss = self.custom_loss(outputs, labels)
        return loss

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
        optimizer = torch.optim.SGD(self.parameters(), lr=.001)
        
        return optimizer
    
    
def main():
    # Initialize data_module
    data_module = BatchedTiledDataModule(
        data_path='/userdata/kerasData/data/new_data/batched_tiled_data',
        metadata_path='/userdata/kerasData/data/new_data/batched_tiled_data/metadata.pkl',
        batch_size=1,
        series_length=1,
        time_range=(-2400,2400))
    
    # Initialize model
    model = LightningModel()

    # Initialize a trainer
    trainer = pl.Trainer(
        gpus=1, 
        progress_bar_refresh_rate=20,
        fast_dev_run=True)

    # Train the model ⚡
    trainer.fit(model, data_module)

if __name__ == '__main__':
    main()