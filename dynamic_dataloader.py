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

import os

# File imports
import util_fns


#####################
## Data Module
#####################

class DynamicDataModule(pl.LightningDataModule):
    def __init__(self, 
                 raw_data_path='./data/raw_data', 
                 labels_path='./data/labels', 
                 train_split_path=None,
                 val_split_path=None,
                 test_split_path=None,
                 batch_size=1, 
                 num_workers=0, 
                 series_length=5, 
                 time_range=(-2400, 2400), 
                 image_dimensions = (1536, 2016),
                 crop_height = 1344,
                 tile_dimensions = (224, 224),
                 smoke_threshold = 10):
        """
        Args:
            - raw_data_path (str): path to raw data
            - labels_path (str): path to XML labels
            
            - train_split_path (str): path to existing train split .txt file
            - val_split_path (str): path to existing val split .txt file
            - test_split_path (str): path to existing test split .txt file
            
            - batch_size (int): batch_size for training
            - num_workers (int): number of workers for dataloader
            - series_length (int): how many sequential images should be used for training
            - time_range (int, int): The time range of images to consider for training by time stamp
            
            - image_dimensions (int, int): desired dimensions of image before cropping
            - crop_height (int): height to crop image to
            - tile_dimensions (int, int): desired size of tiles
            - smoke_threshold (int): # of pixels of smoke to consider tile positive
        
        Other Attributes:
            - self.train_split (list): list of image names to be used for train dataloader
            - self.val_split (list): list of image names to be used for val dataloader
            - self.test_split (list): list of image names to be used for test dataloader
            - self.has_setup (bool): if setup has already occurred to prevent from doing twice
        """
        super().__init__()
        
        self.raw_data_path = raw_data_path
        self.labels_path = labels_path
        
        self.train_split_path = train_split_path
        self.val_split_path = val_split_path
        self.test_split_path = test_split_path
        
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.series_length = series_length
        self.time_range = time_range
        
        self.image_dimensions = image_dimensions
        self.crop_height = crop_height
        self.tile_dimensions = tile_dimensions
        self.smoke_threshold = smoke_threshold
        
        self.has_setup = False
        
        
    def setup(self, stage=None, log_dir=None):
        """
        Args:
            - log_dir (str): logging directory to save train/val/test splits 
        """
        if self.has_setup: return
        print("Setting Up Data... ")

        # Create dictionary of fires and images
        self.fire_to_images = util_fns.generate_fire_to_images(self.raw_data_path, self.labels_path)
        self.omit_images_list = util_fns.generate_omit_images_list(self.fire_to_images, self.labels_path)
        print(self.fire_to_images)

        # If any split not provided, randomly create own train/val/test splits
        if self.train_split_path is None or self.val_split_path is None or self.test_split_path is None:
            train_fires, val_fires = train_test_split(list(self.fire_to_images.keys()), test_size=0.4)
            val_fires, test_fires = train_test_split(val_fires, test_size=0.5)
            
            # Shorten fire_to_images to relevant time frame
            self.fire_to_images = util_fns.shorten_time_range(self.fire_to_images, self.time_range, train_fires)

            # Save arrays representing series of images
            self.image_series = util_fns.generate_series(self.fire_to_images, self.series_length) 

            # Create train/val/test split of Images
            self.train_split = util_fns.unpack_fire_images(self.fire_to_images, train_fires, self.omit_images_list)
            self.val_split = util_fns.unpack_fire_images(self.fire_to_images, val_fires, self.omit_images_list)
            self.test_split = util_fns.unpack_fire_images(self.fire_to_images, test_fires, self.omit_images_list, is_test=True)

            # If logdir is provided, then save train/val/test splits
            if log_dir:
                os.mkdir(log_dir)
                np.savetxt(log_dir+'/train_fires.txt', self.train_split, fmt='%s')
                np.savetxt(log_dir+'/val_fires.txt', self.val_split, fmt='%s')
                np.savetxt(log_dir+'/test_fires.txt', self.test_split, fmt='%s')
        else:
            # Load existing splits
            train_list = np.loadtxt(self.train_split_path, dtype=str)
            val_list = np.loadtxt(self.val_split_path, dtype=str)
            test_list = np.loadtxt(self.test_split_path, dtype=str)
            
            self.train_split = [util_fns.get_image_name(item) for item in train_list]
            self.val_split   = [util_fns.get_image_name(item) for item in val_list]
            self.test_split  = [util_fns.get_image_name(item) for item in test_list]
        
        self.has_setup = True
        print("Setting Up Data Complete. ")
            

    def train_dataloader(self):
        train_dataset = DynamicDataloader(self.raw_data_path, 
                                          self.labels_path, 
                                          self.train_split,
                                          self.image_dimensions,
                                          self.crop_height,
                                          self.tile_dimensions,
                                          self.smoke_threshold)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)
        return train_loader

    def val_dataloader(self):
        val_dataset = DynamicDataloader(self.raw_data_path, 
                                        self.labels_path,  
                                        self.val_split,
                                        self.image_dimensions,
                                        self.crop_height,
                                        self.tile_dimensions,
                                        self.smoke_threshold)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
        return val_loader

    def test_dataloader(self):
        test_dataset = DynamicDataloader(self.raw_data_path, 
                                         self.labels_path,  
                                         self.test_split,
                                         self.image_dimensions,
                                         self.crop_height,
                                         self.tile_dimensions,
                                         self.smoke_threshold)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
        return test_loader
    
    
#####################
## Dataloader
#####################
    
class DynamicDataloader(Dataset):
    def __init__(self, 
                 raw_data_path, 
                 labels_path, 
                 data_split, 
                 image_dimensions = (1536, 2016),
                 crop_height = 1344,
                 tile_dimensions=(224,224), 
                 smoke_threshold=10):
        """
        Args / Attributes:
            - raw_data_path (Path): path to raw data
            - labels_path (Path): path to XML labels
            - data_split (list): list of images of the current split
            - image_dimensions (int, int): desired dimensions of image before cropping
            - crop_height (int): height to crop image to
            - tile_dimensions (int, int): desired size of tiles
            - smoke_threshold (int): # of pixels of smoke to consider tile positive
        """
        self.raw_data_path = raw_data_path
        self.labels_path = labels_path
        self.data_split = data_split
        
        self.image_dimensions = image_dimensions
        self.crop_height = crop_height
        self.tile_dimensions = tile_dimensions
        self.smoke_threshold = smoke_threshold

    def __len__(self):
        return len(self.data_split)

    def __getitem__(self, idx):
        image_name = self.data_split[idx]
        
        # Load all images in the series
        x = []
        for file_name in self.image_series[image_name]:
            img = cv2.imread(raw_data_path+'/'+file_name+'.jpg') # img.shape = [height, width, num_channels]
            # Resize and crop
            img = cv2.resize(img, (self.image_dimensions[1], self.image_dimensions[0]))[-self.crop_height:]
            x.append(img)
        
        # x.shape = [series_length, num_channels, height, width]
        # e.g. [5, 3, 1344, 2016]
        x = np.transpose(np.stack(x), (0, 4, 2, 3))/255 
           
        # Load XML labels
        labels = np.zeros(x[0].shape[:2], dtype=np.uint8) 
        has_xml_label = 0

        label_path = self.labels_path+'/'+util_fns.get_fire_name(image_name)+'/xml/'+util_fns.get_only_image_name(image_name)+'.xml'
        if Path(label_path).exists():
            cv2.fillPoly(mask_img, util_fns.xml_to_record(labels), 1)
            has_xml_label = 1
        
        # labels.shape = [height, width]
        labels = cv2.resize(labels, (self.image_dimensions[1], self.image_dimensions[0]))[-self.crop_height:]
        
        if self.tile_dimensions:
            # WARNING: Tile size must divide perfectly into image height and width
            # x.shape = [54, 5, 3, 224, 224]
            # labels.shape = [54, 224, 224]
            x = x.view(-1, len(self.image_series), 3, self.tile_dimensions[0], self.tile_dimensions[1])
            labels = labels.view(-1, self.tile_dimensions[0], self.tile_dimensions[1])

            # tile_labels.shape = [54,]
            labels = (labels.sum(dim=(1,2)) > self.smoke_threshold).float()

        # Load image-level labels for current image
        ground_truth_label = util_fns.get_ground_truth_label(image_name)
        has_positive_tile = util_fns.get_has_positive_tile(labels)
                
        return image_name, x, labels, ground_truth_label, has_xml_label, has_positive_tile