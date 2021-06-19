"""
Created by: Anshuman Dewangan
Date: 2021

Description: Loads tiled & batched data that was generated from generate_batched_tiled_data.py
"""
# Torch imports
import pytorch_lightning as pl
import pickle
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split

# File imports
import util_fns
from generate_batched_tiled_data import save_batched_tiled_images


#####################
## Data Module
#####################

class BatchedTiledDataModule(pl.LightningDataModule):
    def __init__(self, 
                 data_path='./data/', 
                 metadata_path='./data/metadata.pkl',  
                 train_split_path=None,
                 val_split_path=None,
                 test_split_path=None,
                 batch_size=1, 
                 num_workers=0, 
                 series_length=5, 
                 time_range=(-2400, 2400),
                 generate_data=False,
                 generate_data_params=None):
        """
        Args:
            - data_path (str): path to batched_tiled_data
            - metadata_path (str): path to metadata.pkl file generated by generate_batched_tiled_data.py. Usually data_path/metadata.pkl
            
            - train_split_path (str): path to existing train split .txt file
            - val_split_path (str): path to existing val split .txt file
            - test_split_path (str): path to existing test split .txt file
            
            - batch_size (int): batch_size for training
            - num_workers (int): number of workers for dataloader
            - series_length (int): how many sequential images should be used for training
            - time_range (int, int): The time range of images to consider for training by time stamp
            
            - generate_data (bool): if generate_batched_tiled_data.py should be run to generate the batched & tiled dataset. 
                WARNING: will take ~1 hour and 250 GB of storage for full dataset
            - generate_data_params (dict): params to generate the data. Must include:
                - raw_images_path (str): path to raw images
                - labels_path (str): path to XML labels
                - output_path (str): desired path of outputted Numpy files
                - image_dimensions (int, int): dimensions original image should be resized to
                - tile_dimensions (int, int): desired dimensions of tiles
                - overlap_amount (int): how much the tiles should overlap in pixels
                - smoke_threshold (int): how many pixels of smoke to label tile as a positive sample? 
        
        Other Attributes:
            - self.train_split (list): list of image names to be used for train dataloader
            - self.val_split (list): list of image names to be used for val dataloader
            - self.test_split (list): list of image names to be used for test dataloader
            - self.has_setup (bool): if setup has already occurred to prevent from doing twice
        """
        super().__init__()
        
        self.data_path = data_path
        self.metadata = pickle.load(open(metadata_path, 'rb'))
        
        self.train_split_path = train_split_path
        self.val_split_path = val_split_path
        self.test_split_path = test_split_path
        
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.series_length = series_length
        self.time_range = time_range
        
        self.generate_data = generate_data
        self.generate_data_params = generate_data_params
        
        self.has_setup = False

        
    def prepare_data(self):
        # Generates batched & tiled data
        # WARNING: will take ~1 hour and 250 GB of storage for full dataset
        if self.generate_data:
            print("Preparing Data... ")
            
            save_batched_tiled_images(
                raw_images_path=self.generate_data_params['raw_images_path'], 
                labels_path=self.generate_data_params['labels_path'], 
                output_path=self.generate_data_params['output_path'], 
                image_dimensions=self.generate_data_params['image_dimensions'], 
                tile_dimensions=self.generate_data_params['tile_dimensions'], 
                overlap_amount=self.generate_data_params['overlap_amount'], 
                smoke_threshold=self.generate_data_params['smoke_threshold'])
            
            print("Preparing Data Complete. ")
        
        
    def setup(self, stage=None):
        if self.has_setup: return
        print("Setting Up Data... ")
        
        # If any split not provided, randomly create own train/val/test splits
        if self.train_split_path is None or self.val_split_path is None or self.test_split_path is None:
            train_fires, val_fires = train_test_split(list(self.metadata['fire_to_images'].keys()), test_size=0.4)
            val_fires, test_fires = train_test_split(val_fires, test_size=0.5)
        else:
            # Load existing splits, saving fires only
            train_list = np.loadtxt(self.train_split_path, dtype=str)
            val_list = np.loadtxt(self.val_split_path, dtype=str)
            test_list = np.loadtxt(self.test_split_path, dtype=str)
            
            train_fires = {util_fns.get_fire_name(item) for item in train_list}
            val_fires = {util_fns.get_fire_name(item) for item in val_list}
            test_fires = {util_fns.get_fire_name(item) for item in test_list}
        
        # Shorten fire_to_images to relevant time frame
        self.metadata['fire_to_images'] = util_fns.shorten_time_range(self.metadata, self.time_range, train_fires)
        
        # Save arrays representing series of images
        self.metadata['image_series'] = util_fns.generate_series(self.metadata, self.series_length) 
        
        # Create train/val/test split of Images
        if self.train_split_path is None or self.val_split_path is None or self.test_split_path is None:
            self.train_split = util_fns.unpack_fire_images(self.metadata, train_fires)
            self.val_split = util_fns.unpack_fire_images(self.metadata, val_fires)
            self.test_split = util_fns.unpack_fire_images(self.metadata, test_fires, is_test=True)
        else:
            self.train_split = [util_fns.get_image_name(item) for item in train_list]
            self.val_split   = [util_fns.get_image_name(item) for item in val_list]
            self.test_split  = [util_fns.get_image_name(item) for item in test_list]
        
        self.has_setup = True
        print("Setting Up Data Complete. ")
            

    def train_dataloader(self):
        train_dataset = BatchedTiledDataloader(self.data_path, self.train_split, self.metadata)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)
        return train_loader

    def val_dataloader(self):
        val_dataset = BatchedTiledDataloader(self.data_path, self.val_split, self.metadata)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
        return val_loader

    def test_dataloader(self):
        test_dataset = BatchedTiledDataloader(self.data_path, self.test_split, self.metadata)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
        return test_loader

    
#####################
## Dataloader
#####################
    
class BatchedTiledDataloader(Dataset):
    def __init__(self, data_path, data_split, metadata):
        """
        Args / Attributes:
            - data_path (str): path to batched_tiled_data
            - data_split (list): list of images of the current split
            - metadata (dict): metadata dictionary from DataModule
        """
        self.data_path = Path(data_path)
        self.data_split = data_split
        self.metadata = metadata

    def __len__(self):
        return len(self.data_split)

    def __getitem__(self, idx):
        image_name = self.data_split[idx]
        
        # Load all pixel inputs of each image in the series
        x = []
        for file_name in self.metadata['image_series'][image_name]:
            x.append(np.load(self.data_path/f'{file_name}_img.npy'))

        # x.shape = [num_tiles, series_length, num_channels, height, width]
        # e.g. [108, 5, 3, 224, 224]
        x = np.transpose(np.stack(x), (1, 0, 2, 3, 4))/255 
        
        # Load tile-level labels for current image
        # y.shape = [num_tiles] e.g. [108,]
        tile_labels = np.load(self.data_path/f'{image_name}_lbl.npy').astype(float)

        # Load image-level labels for current image
        ground_truth_label = self.metadata['ground_truth_label'][image_name]
        has_xml_label = self.metadata['has_xml_label'][image_name]
        has_positive_tile = self.metadata['has_positive_tile'][image_name]
                
        return image_name, x, tile_labels, ground_truth_label, has_xml_label, has_positive_tile