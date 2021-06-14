import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import torchvision

import pickle
import numpy as np

from tiled_dataloader import TiledDataloader

train_split, val_split, test_split = pickle.load(open('/userdata/kerasData/data/new_data/vision_transformer/gridded_batched/small_split.pkl', 'rb'))

train_dataset = TiledDataloader('/userdata/kerasData/data/new_data/vision_transformer/gridded_batched', img_list=train_split, series_length=5)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

val_dataset = TiledDataloader('/userdata/kerasData/data/new_data/vision_transformer/gridded_batched', img_list=val_split, series_length=5)
val_loader = DataLoader(val_dataset, batch_size=1)

test_dataset = TiledDataloader('/userdata/kerasData/data/new_data/vision_transformer/gridded_batched', img_list=test_split, series_length=5)
test_loader = DataLoader(test_dataset, batch_size=1)


class Model(pl.LightningModule):

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
        labels = batch['class'].cpu().numpy()
        pos_idx = labels[0, :] > 0
        neg_idx = np.where(~pos_idx)
        neg_idx = np.random.choice(neg_idx[0], size=32-pos_idx.sum(), replace=False)
        pos_idx[neg_idx] = True
        inputs = torch.as_tensor(batch['image'][:, pos_idx, -1, :])
        labels = batch['class'][:, pos_idx].float().unsqueeze(2)
        outputs = self.forward(inputs)
        loss = self.custom_loss(outputs, labels)
        return loss

    def validation_step(self, batch, batch_idx):
        labels = batch['class'].cpu().numpy()
        pos_idx = labels[0, :] > 0
        neg_idx = np.where(~pos_idx)
        neg_idx = np.random.choice(neg_idx[0], size=32-pos_idx.sum(), replace=False)
        pos_idx[neg_idx] = True
        inputs = torch.as_tensor(batch['image'][:, pos_idx, -1, :])
        labels = batch['class'][:, pos_idx].float().unsqueeze(2)
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
    
# init model
ae = Model()

# Initialize a trainer
trainer = pl.Trainer(gpus=1, max_epochs=1, progress_bar_refresh_rate=20)

# Train the model ⚡
trainer.fit(ae, train_loader, val_loader)