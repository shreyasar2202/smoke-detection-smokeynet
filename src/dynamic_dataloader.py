"""
Created by: Anshuman Dewangan
Date: 2021

Description: Loads data from raw image and XML files
"""
# Torch imports
import pytorch_lightning as pl
import pickle
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader

# Other package imports
import os
import cv2
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

# File imports
import util_fns


#####################
## Data Module
#####################

class DynamicDataModule(pl.LightningDataModule):
    def __init__(self, 
                 raw_data_path='./data/raw_data', 
                 labels_path='./data/labels', 
                 raw_labels_path='./data/raw_labels',
                 metadata_path='./data/metadata.pkl',
                 train_split_path=None,
                 val_split_path=None,
                 test_split_path=None,
                 train_split_size=0.7,
                 test_split_size=0.15,
                 batch_size=1, 
                 num_workers=0, 
                 series_length=1, 
                 time_range=(-2400, 2400), 
                 image_dimensions = (1536, 2016),
                 crop_height = 1120,
                 tile_dimensions = (224, 224),
                 smoke_threshold = 10,
                 num_tile_samples = 0,
                 flip_augment = False,
                 blur_augment = False,
                 create_data = False):
        """
        Args:
            - raw_data_path (str): path to raw data
            - labels_path (str): path to Numpy labels
            - raw_labels_path (str): path to XML labels
            - metadata_path (str): path to metadata.pkl
                - fire_to_images (dict): dictionary with fires as keys and list of corresponding images as values
                - num_fires (int): total number of fires in dataset
                - num_images (int): total number of images in dataset
                - ground_truth_label (dict): dictionary with fires as keys and 1 if fire has "+" in its file name
                - has_xml_label (dict): dictionary with fires as keys and 1 if fire has a .xml file associated with it
                - omit_no_xml (list of str): list of images that erroneously do not have XML files for labels
                - omit_no_bbox (list of str): list of images that erroneously do not have loaded bboxes for labels
                - omit_images_list (list of str): union of omit_no_xml and omit_no_bbox
            
            - train_split_path (str): path to existing train split .txt file
            - val_split_path (str): path to existing val split .txt file
            - test_split_path (str): path to existing test split .txt file
            
            - train_split_size (float): % of data to split for train
            - test_split_size (float): % of data to split for test
            - batch_size (int): batch_size for training
            - num_workers (int): number of workers for dataloader
            - series_length (int): how many sequential images should be used for training
            - time_range (int, int): The time range of images to consider for training by time stamp
            
            - image_dimensions (int, int): desired dimensions of image before cropping
            - crop_height (int): height to crop image to
            - tile_dimensions (int, int): desired size of tiles
            - smoke_threshold (int): # of pixels of smoke to consider tile positive
            
            - flip_augment (bool): enables data augmentation with horizontal flip
            - blur_augment (bool): enables data augmentation with Gaussian blur
            
            - num_tile_samples (int): number of random tile samples per batch. If < 1, then turned off
            
            - create_data (bool): should prepare_data be run?
        
        Other Attributes:
            - self.train_split (list): list of image names to be used for train dataloader
            - self.val_split (list): list of image names to be used for val dataloader
            - self.test_split (list): list of image names to be used for test dataloader
            - self.has_setup (bool): if setup has already occurred to prevent from doing twice
        """
        super().__init__()
                
        self.raw_data_path = raw_data_path
        self.labels_path = labels_path
        self.raw_labels_path = raw_labels_path
        self.metadata = pickle.load(open(metadata_path, 'rb'))
        
        self.train_split_path = train_split_path
        self.val_split_path = val_split_path
        self.test_split_path = test_split_path
        
        self.train_split_size = train_split_size
        self.test_split_size = test_split_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.series_length = series_length
        self.time_range = time_range
        
        self.image_dimensions = image_dimensions
        self.crop_height = crop_height
        self.tile_dimensions = tile_dimensions
        self.smoke_threshold = smoke_threshold
        self.num_tile_samples = num_tile_samples
        
        self.flip_augment = flip_augment
        self.blur_augment = blur_augment
        
        self.create_data = create_data
        self.has_setup = False
        
        
    def prepare_data(self):
        """
        Description: Creates metadata.pkl and saved labels for easier loading. Only needs to be run once.
        """
        if self.create_data:
            print("Preparing Data...")
            
            ### Create metadata.pkl ###
            self.metadata = {}

            self.metadata['fire_to_images'] = util_fns.generate_fire_to_images(self.raw_data_path, self.labels_path)
            self.metadata['ground_truth_label'] = {}
            self.metadata['has_xml_label'] = {}
            self.metadata['num_fires'] = 0
            self.metadata['num_images'] = 0
            self.metadata['omit_no_xml'] = []
            self.metadata['omit_no_bbox'] = []
            self.metadata['omit_images_list'] = []
            
            output_path = '/userdata/kerasData/data/new_data/drive_clone_new'

            for fire in self.metadata['fire_to_images']:
                self.metadata['num_fires'] += 1
                
                print('Preparing Folder ', self.metadata['num_fires'])
                
                for image in self.metadata['fire_to_images'][fire]:
                    self.metadata['num_images'] += 1
                    
                    self.metadata['ground_truth_label'][image] = util_fns.get_ground_truth_label(image)
                    self.metadata['has_xml_label'][image] = util_fns.get_has_xml_label(image, self.labels_path)
                    
                    if self.metadata['ground_truth_label'][image] != self.metadata['has_xml_label'][image]:
                        self.metadata['omit_no_xml'].append(image)
                    
                    if self.metadata['has_xml_label'][image]:
                        labels = util_fns.get_filled_labels(self.raw_data_path, self.raw_labels_path, image)
                        
                        if labels.sum() == 0:
                            self.metadata['omit_no_bbox'].append(image)

                        save_path = output_path + '/' + image + '.npy'

                        os.makedirs(output_path + '/' + fire, exist_ok=True)
                        np.save(save_path, labels.astype(np.uint8))

            self.metadata['omit_images_list'] = list(set().union(self.metadata['omit_no_xml'], self.metadata['omit_no_bbox']))
        
            with open(f'./metadata.pkl', 'wb') as pkl_file:
                pickle.dump(self.metadata, pkl_file)
                
            print("Preparing Data Complete.")
        
        
    def setup(self, stage=None, log_dir=None):
        """
        Args:
            - log_dir (str): logging directory to save train/val/test splits 
        """
        if self.has_setup: return
        print("Setting Up Data...")

        # If any split not provided, randomly create own train/val/test splits
        if self.train_split_path is None or self.val_split_path is None or self.test_split_path is None:
            train_fires, val_fires = train_test_split(list(self.metadata['fire_to_images'].keys()), test_size=(1-self.train_split_size))
            val_fires, test_fires = train_test_split(val_fires, test_size=self.test_split_size/(1-self.train_split_size))
            
            # Shorten fire_to_images to relevant time frame
            self.metadata['fire_to_images'] = util_fns.shorten_time_range(self.metadata['fire_to_images'], self.time_range, train_fires)

            # Save arrays representing series of images
            self.metadata['image_series'] = util_fns.generate_series(self.metadata['fire_to_images'], self.series_length) 

            # Create train/val/test split of Images
            self.train_split = util_fns.unpack_fire_images(self.metadata['fire_to_images'], train_fires, self.metadata['omit_images_list'])
            self.val_split = util_fns.unpack_fire_images(self.metadata['fire_to_images'], val_fires, self.metadata['omit_images_list'])
            self.test_split = util_fns.unpack_fire_images(self.metadata['fire_to_images'], test_fires, self.metadata['omit_images_list'], is_test=True)

            # If logdir is provided, then save train/val/test splits
            if log_dir:
                os.makedirs(log_dir)
                np.savetxt(log_dir+'/train_images.txt', self.train_split, fmt='%s')
                np.savetxt(log_dir+'/val_images.txt', self.val_split, fmt='%s')
                np.savetxt(log_dir+'/test_images.txt', self.test_split, fmt='%s')
        else:
            # Load existing splits
            train_list = np.loadtxt(self.train_split_path, dtype=str)
            val_list = np.loadtxt(self.val_split_path, dtype=str)
            test_list = np.loadtxt(self.test_split_path, dtype=str)
            
            self.train_split = [util_fns.get_image_name(item) for item in train_list]
            self.val_split   = [util_fns.get_image_name(item) for item in val_list]
            self.test_split  = [util_fns.get_image_name(item) for item in test_list]
            
            # Recreate fire_to_images and image_series
            self.metadata['fire_to_images'] = util_fns.generate_fire_to_images_from_splits([self.train_split, self.val_split, self.test_split])
            self.metadata['image_series'] = util_fns.generate_series(self.metadata['fire_to_images'], self.series_length) 
        
        self.has_setup = True
        print("Setting Up Data Complete.")
            

    def train_dataloader(self):
        train_dataset = DynamicDataloader(raw_data_path=self.raw_data_path, 
                                          labels_path=self.labels_path, 
                                          metadata=self.metadata,
                                          data_split=self.train_split,
                                          image_dimensions=self.image_dimensions,
                                          crop_height=self.crop_height,
                                          tile_dimensions=self.tile_dimensions,
                                          smoke_threshold=self.smoke_threshold,
                                          num_tile_samples=self.num_tile_samples,
                                          flip_augment=self.flip_augment,
                                          blur_augment=self.blur_augment)
        train_loader = DataLoader(train_dataset, 
                                  batch_size=self.batch_size, 
                                  num_workers=self.num_workers,
                                  pin_memory=True, 
                                  shuffle=True)
        return train_loader

    def val_dataloader(self):
        val_dataset = DynamicDataloader(raw_data_path=self.raw_data_path, 
                                          labels_path=self.labels_path, 
                                          metadata=self.metadata,
                                          data_split=self.val_split,
                                          image_dimensions=self.image_dimensions,
                                          crop_height=self.crop_height,
                                          tile_dimensions=self.tile_dimensions,
                                          smoke_threshold=self.smoke_threshold,
                                          num_tile_samples=0,
                                          flip_augment=self.flip_augment,
                                          blur_augment=self.blur_augment)
        val_loader = DataLoader(val_dataset, 
                                batch_size=self.batch_size, 
                                num_workers=self.num_workers,
                                pin_memory=True)
        return val_loader

    def test_dataloader(self):
        test_dataset = DynamicDataloader(raw_data_path=self.raw_data_path, 
                                          labels_path=self.labels_path, 
                                          metadata=self.metadata,
                                          data_split=self.test_split,
                                          image_dimensions=self.image_dimensions,
                                          crop_height=self.crop_height,
                                          tile_dimensions=self.tile_dimensions,
                                          smoke_threshold=self.smoke_threshold,
                                          num_tile_samples=0,
                                          flip_augment=self.flip_augment,
                                          blur_augment=self.blur_augment)
        test_loader = DataLoader(test_dataset, 
                                 batch_size=self.batch_size, 
                                 num_workers=self.num_workers,
                                 pin_memory=True)
        return test_loader
    
    
#####################
## Dataloader
#####################
    
class DynamicDataloader(Dataset):
    def __init__(self, 
                 raw_data_path, 
                 labels_path, 
                 metadata,
                 data_split, 
                 image_dimensions = (1536, 2016),
                 crop_height = 1120,
                 tile_dimensions = (224,224), 
                 smoke_threshold = 10,
                 num_tile_samples = 0,
                 flip_augment = False,
                 blur_augment = False):
        
        self.raw_data_path = raw_data_path
        self.labels_path = labels_path
        self.metadata = metadata
        self.data_split = data_split
        
        self.image_dimensions = image_dimensions
        self.crop_height = crop_height
        self.tile_dimensions = tile_dimensions
        self.smoke_threshold = smoke_threshold
        self.num_tile_samples = num_tile_samples
        
        self.flip_augment = flip_augment
        self.blur_augment = blur_augment

    def __len__(self):
        return len(self.data_split)

    def __getitem__(self, idx):
        image_name = self.data_split[idx]
        
        ### Load Images ###
        x = []
        series_length = len(self.metadata['image_series'][image_name])
        
        if self.flip_augment:
            should_flip = np.random.rand() > 0.5
        if self.blur_augment:
            should_blur = np.random.rand() > 0.5
            blur_size = np.maximum(int(np.random.randn()*3+10), 1)
        
        # Load all images in the series
        for file_name in self.metadata['image_series'][image_name]:
            # img.shape = [height, width, num_channels]
            img = cv2.imread(self.raw_data_path+'/'+file_name+'.jpg')
            # Resize and crop
            img = cv2.resize(img, (self.image_dimensions[1], self.image_dimensions[0]))[-self.crop_height:]
            
            # Add data augmentations
            if self.flip_augment and should_flip:
                img = cv2.flip(img, 1)
            if self.blur_augment and should_blur:
                img = cv2.blur(img, (blur_size,blur_size))
            
            x.append(img)
        
        # x.shape = [series_length, num_channels, height, width]
        # e.g. [5, 3, 1344, 2016]
        x = np.transpose(np.stack(x), (0, 3, 1, 2)) / 255 # Normalize by /255 (good enough normalization)
           
        ### Load XML labels ###
        label_path = self.labels_path+'/'+image_name+'.npy'
        if Path(label_path).exists():
            labels = np.load(label_path)
        else:
            labels = np.zeros(x[0].shape[:2], dtype=np.uint8) 
        
        # labels.shape = [height, width]
        labels = cv2.resize(labels, (self.image_dimensions[1], self.image_dimensions[0]))[-self.crop_height:]
        if self.flip_augment and should_flip:
            labels = cv2.flip(labels, 1)
        
        ### Tile Image ###
        if self.tile_dimensions:
            # WARNING: Tile size must divide perfectly into image height and width
            # x.shape = [45, 5, 3, 224, 224]
            # labels.shape = [45, 224, 224]
            x = np.reshape(x, (-1, series_length, 3, self.tile_dimensions[0], self.tile_dimensions[1]))
            labels = np.reshape(labels,(-1, self.tile_dimensions[0], self.tile_dimensions[1]))

            # tile_labels.shape = [45,]
            labels = (labels.sum(axis=(1,2)) > self.smoke_threshold).astype(float)
            
            if self.num_tile_samples > 0:
                # WARNING: Assumes that there are no labels with all 0s
                # Tip: Use --time-range-min 0
                x, labels = util_fns.randomly_sample_tiles(x, labels, self.num_tile_samples)
        else:
            # Pretend as if tile size = image size
            x = np.expand_dims(x, 0)
            labels = np.expand_dims(labels, 0)

        # Load Image-level Labels ###
        ground_truth_label = self.metadata['ground_truth_label'][image_name]
        has_xml_label = self.metadata['has_xml_label'][image_name]
        has_positive_tile = util_fns.get_has_positive_tile(labels)
                        
        return image_name, x, labels, ground_truth_label, has_xml_label, has_positive_tile