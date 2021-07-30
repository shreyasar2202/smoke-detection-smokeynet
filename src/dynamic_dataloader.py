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
                 omit_list=None,
                 omit_images_from_test=False,
                 mask_omit_images=False,
                 
                 raw_data_path=None, 
                 labels_path=None, 
                 raw_labels_path=None,
                 metadata_path='./data/metadata.pkl',
                 
                 train_split_path=None,
                 val_split_path=None,
                 test_split_path=None,
                 load_images_from_split=False,
                 
                 train_split_size=0.7,
                 test_split_size=0.15,
                 batch_size=1, 
                 num_workers=0, 
                 series_length=1, 
                 add_base_flow=False,
                 time_range=(-2400, 2400), 
                 
                 original_dimensions = (1536, 2016),
                 resize_dimensions = (1536, 2016),
                 crop_height = 1120,
                 tile_dimensions = (224, 224),
                 tile_overlap = 0,
                 smoke_threshold = 250,
                 num_tile_samples = 0,
                 
                 flip_augment = True,
                 resize_crop_augment = True,
                 blur_augment = True,
                 color_augment = True,
                 brightness_contrast_augment = True,
                 
                 create_data = False):
        """
        Args:
            - omit_list (list of str): list of metadata keys to omit from train/val sets
            - omit_images_from_test (bool): omits omit_list_images from the test set
            - mask_omit_images (bool): masks tile predictions for images in omit_list_images
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
                - omit_no_contour (list of str): list of images that erroneously do not have loaded contours for labels
                - omit_no_bbox (list of str): list of images that erroneously do not have bboxes
                - omit_mislabeled (list of str): list of images that erroneously have no XML files and are manually selected as mislabeled
                - train_fires_only (list of str): list of fires that should only be used for train (not 'mobo-c')
            
            - train_split_path (str): path to existing train split .txt file
            - val_split_path (str): path to existing val split .txt file
            - test_split_path (str): path to existing test split .txt file
            - load_images_from_split (bool): if images should be loaded exactly from split (as opposed to fires)
            
            - train_split_size (float): % of data to split for train
            - test_split_size (float): % of data to split for test
            - batch_size (int): batch_size for training
            - num_workers (int): number of workers for dataloader
            - series_length (int): how many sequential images should be used for training
            - add_base_flow (bool): if True, adds image from t=0 for fire
            - time_range (int, int): The time range of images to consider for training by time stamp
            
            - original_dimensions (int, int): original dimensions of image
            - resize_dimensions (int, int): desired dimensions of image before cropping
            - crop_height (int): height to crop image to
            - tile_dimensions (int, int): desired size of tiles
            - tile_overlap (int): amount to overlap each tile
            - smoke_threshold (int): # of pixels of smoke to consider tile positive
            - num_tile_samples (int): number of random tile samples per batch. If < 1, then turned off

            - flip_augment (bool): enables data augmentation with horizontal flip
            - resize_crop_augment (bool): enables data augmentation with random resize cropping
            - blur_augment (bool): enables data augmentation with Gaussian blur
            - color_augment (bool): enables data augmentation with color jitter
            - brightness_contrast_augment (bool): enables data augmentation with brightness contrast adjustment
            
            - create_data (bool): should prepare_data be run?
        
        Other Attributes:
            - self.train_split (list): list of image names to be used for train dataloader
            - self.val_split (list): list of image names to be used for val dataloader
            - self.test_split (list): list of image names to be used for test dataloader
            - self.has_setup (bool): if setup has already occurred to prevent from doing twice
        """
        super().__init__()
        
        self.omit_list = omit_list
        self.omit_images_from_test = omit_images_from_test
        self.mask_omit_images = mask_omit_images
           
        self.raw_data_path = raw_data_path
        self.labels_path = labels_path
        self.raw_labels_path = raw_labels_path
        self.metadata = pickle.load(open(metadata_path, 'rb'))
        
        self.train_split_path = train_split_path
        self.val_split_path = val_split_path
        self.test_split_path = test_split_path
        self.load_images_from_split = load_images_from_split
        
        self.train_split_size = train_split_size
        self.test_split_size = test_split_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.series_length = series_length
        self.add_base_flow = add_base_flow
        self.time_range = time_range
        
        self.original_dimensions = original_dimensions
        self.resize_dimensions = resize_dimensions
        self.crop_height = crop_height
        self.tile_dimensions = tile_dimensions
        self.tile_overlap = tile_overlap
        self.smoke_threshold = smoke_threshold
        self.num_tile_samples = num_tile_samples
        
        self.flip_augment = flip_augment
        self.resize_crop_augment = resize_crop_augment
        self.blur_augment = blur_augment
        self.color_augment = color_augment
        self.brightness_contrast_augment = brightness_contrast_augment
        
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
            self.metadata['omit_no_contour'] = []
            self.metadata['omit_no_contour_or_bbox'] = []
            self.metadata['omit_images_list'] = []
            self.metadata['train_only_fires'] = []
            self.metadata['eligible_fires'] = []
            
            images_output_path = '/userdata/kerasData/data/new_data/raw_images_numpy'
            labels_output_path = '/userdata/kerasData/data/new_data/drive_clone_numpy'

            # Loop through all fires
            for fire in self.metadata['fire_to_images']:
                self.metadata['num_fires'] += 1
                
                print('Preparing Folder ', self.metadata['num_fires'])
                
                if 'mobo-c' not in fire:
                    self.metadata['train_only_fires'].append(fire)
                else:
                    self.metadata['eligible_fires'].append(fire)
                    
                # Loop through each image in the fire
                for image in self.metadata['fire_to_images'][fire]:
                    # Save processed image as npy file
                    x = cv2.imread(self.raw_data_path + '/' + image + '.jpg')
                    x = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                    os.makedirs(images_output_path + '/' + fire, exist_ok=True)
                    np.save(images_output_path + '/' + image + '.npy', x.astype(np.uint8))
                    
                    # Create metadata
                    self.metadata['num_images'] += 1
                    
                    self.metadata['ground_truth_label'][image] = util_fns.get_ground_truth_label(image)
                    self.metadata['has_xml_label'][image] = util_fns.get_has_xml_label(image, self.labels_path)
                    
                    # If a positive image does not have an XML file associated with it, add it to omit_images_list
                    if self.metadata['ground_truth_label'][image] != self.metadata['has_xml_label'][image]:
                        self.metadata['omit_no_xml'].append(image)
                    
                    if self.metadata['has_xml_label'][image]:
                        label_path = raw_labels_path+'/'+\
                            get_fire_name(image_name)+'/xml/'+\
                            get_only_image_name(image_name)+'.xml'

                        poly = xml_to_contour(label_path)
                        
                        if poly is not None:
                            labels = np.zeros(x.shape[:2], dtype=np.uint8) 
                            cv2.fillPoly(labels, poly, 1)
                            os.makedirs(labels_output_path + '/' + fire, exist_ok=True)
                            np.save(labels_output_path + '/' + image + '.npy', labels.astype(np.uint8))
                        else:
                            self.metadata['omit_no_contour'].append(image)
                            
                            poly = xml_to_bbox(label_path)
                            
                            if poly is not None:
                                labels = np.zeros(x.shape[:2], dtype=np.uint8) 
                                cv2.rectangle(labels, *poly, 1, -1)
                                os.makedirs(labels_output_path + '/' + fire, exist_ok=True)
                                np.save(labels_output_path + '/' + image + '.npy', labels.astype(np.uint8))
                            else:
                                self.metadata['omit_no_contour_or_bbox'].append(image)                        

            self.metadata['omit_mislabeled'] = np.loadtxt('./data/omit_mislabeled.txt', dtype=str)
        
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

        # Determine which images to omit
        self.omit_images_list = []
        if self.omit_list is not None:
            for key in self.omit_list:
                self.omit_images_list.extend(self.metadata[key])
                        
        is_split_given = self.train_split_path is not None and self.val_split_path is not None and self.test_split_path is not None
        
        # If split is given, load existing splits
        if is_split_given:
            train_list = np.loadtxt(self.train_split_path, dtype=str)
            val_list = np.loadtxt(self.val_split_path, dtype=str)
            test_list = np.loadtxt(self.test_split_path, dtype=str)
                
        # If load_images_from_split, load images exactly from provided splits
        if is_split_given and self.load_images_from_split:
            self.train_split = [util_fns.get_image_name(item) for item in train_list]
            self.val_split   = [util_fns.get_image_name(item) for item in val_list]
            self.test_split  = [util_fns.get_image_name(item) for item in test_list]
            
            # Recreate fire_to_images and image_series
            self.metadata['fire_to_images'] = util_fns.generate_fire_to_images_from_splits([self.train_split, self.val_split, self.test_split])
            self.metadata['image_series'] = util_fns.generate_series(self.metadata['fire_to_images'], self.series_length, self.add_base_flow) 
        
        else:
            # If split is provided, extract fires from split
            if is_split_given:
                train_fires = {util_fns.get_fire_name(item) for item in train_list}
                val_fires   = {util_fns.get_fire_name(item) for item in val_list}
                test_fires  = {util_fns.get_fire_name(item) for item in test_list}
            # Else if split is not provided, randomly create our own splits
            else:
                train_fires, val_fires = train_test_split(self.metadata['eligible_fires'], test_size=(1-self.train_split_size))
                val_fires, test_fires = train_test_split(val_fires, test_size=self.test_split_size/(1-self.train_split_size))
                train_fires += self.metadata['train_only_fires']
                        
            # Save arrays representing series of images
            self.metadata['image_series'] = util_fns.generate_series(self.metadata['fire_to_images'], self.series_length, self.add_base_flow)
            
            # Shorten fire_to_images to relevant time frame
            self.metadata['fire_to_images'] = util_fns.shorten_time_range(list(train_fires)+list(val_fires), self.metadata['fire_to_images'], self.time_range, self.series_length,  self.add_base_flow)
            # Shorten test only by series_length
            self.metadata['fire_to_images'] = util_fns.shorten_time_range(test_fires, self.metadata['fire_to_images'], (-2400,2400), self.series_length, self.add_base_flow)
            
            # Create train/val/test split of Images
            # Only remove images from omit_images_list if not masking
            omit_images_list = None if self.mask_omit_images else self.omit_images_list
            self.train_split = util_fns.unpack_fire_images(self.metadata['fire_to_images'], train_fires, omit_images_list)
            self.val_split = util_fns.unpack_fire_images(self.metadata['fire_to_images'], val_fires, omit_images_list)
            # Only remove images from test is omit_images_from_test=True
            self.test_split = util_fns.unpack_fire_images(self.metadata['fire_to_images'], test_fires, omit_images_list if self.omit_images_from_test else None)

            # If logdir is provided, then save train/val/test splits
            if log_dir:
                os.makedirs(log_dir)
                np.savetxt(log_dir+'/train_images.txt', self.train_split, fmt='%s')
                np.savetxt(log_dir+'/val_images.txt', self.val_split, fmt='%s')
                np.savetxt(log_dir+'/test_images.txt', self.test_split, fmt='%s')
        
        self.has_setup = True
        print("Setting Up Data Complete.")
            

    def train_dataloader(self):
        train_dataset = DynamicDataloader(raw_data_path=self.raw_data_path,
                                          labels_path=self.labels_path, 
                                          
                                          metadata=self.metadata, 
                                          data_split=self.train_split,
                                          omit_images_list=self.omit_images_list if self.mask_omit_images else None,
                                          
                                          original_dimensions=self.original_dimensions,
                                          resize_dimensions=self.resize_dimensions,
                                          crop_height=self.crop_height,
                                          tile_dimensions=self.tile_dimensions,
                                          tile_overlap=self.tile_overlap,
                                          smoke_threshold=self.smoke_threshold,
                                          num_tile_samples=self.num_tile_samples,
                                          
                                          flip_augment=self.flip_augment,
                                          resize_crop_augment=self.resize_crop_augment,
                                          blur_augment=self.blur_augment,
                                          color_augment=self.color_augment,
                                          brightness_contrast_augment=self.brightness_contrast_augment)
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
                                          omit_images_list=self.omit_images_list if self.mask_omit_images else None,
                                        
                                          original_dimensions=self.original_dimensions,
                                          resize_dimensions=self.resize_dimensions,
                                          crop_height=self.crop_height,
                                          tile_dimensions=self.tile_dimensions,
                                          tile_overlap=self.tile_overlap,
                                          smoke_threshold=self.smoke_threshold,
                                          num_tile_samples=self.num_tile_samples,
                                        
                                          flip_augment=False,
                                          resize_crop_augment=False,
                                          blur_augment=False,
                                          color_augment=False,
                                          brightness_contrast_augment=False)
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
                                          omit_images_list=None,
                                         
                                          original_dimensions=self.original_dimensions,
                                          resize_dimensions=self.resize_dimensions,
                                          crop_height=self.crop_height,
                                          tile_dimensions=self.tile_dimensions,
                                          tile_overlap=self.tile_overlap,
                                          smoke_threshold=self.smoke_threshold,
                                          num_tile_samples=0,
                                         
                                          flip_augment=False,
                                          resize_crop_augment=False,
                                          blur_augment=False,
                                          color_augment=False,
                                          brightness_contrast_augment=False)
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
                 raw_data_path=None,
                 labels_path=None, 
                 
                 metadata=None,
                 data_split=None, 
                 omit_images_list=None,
                 
                 original_dimensions = (1536, 2016),
                 resize_dimensions = (1536, 2016),
                 crop_height = 1120,
                 tile_dimensions = (224,224), 
                 tile_overlap = 0,
                 smoke_threshold = 250,
                 num_tile_samples = 0,
                 
                 flip_augment = True,
                 resize_crop_augment = True,
                 blur_augment = True,
                 color_augment = True,
                 brightness_contrast_augment = True):
        
        self.raw_data_path = raw_data_path
        self.labels_path = labels_path
        
        self.metadata = metadata
        self.data_split = data_split
        self.omit_images_list = omit_images_list
        
        self.original_dimensions = original_dimensions
        self.resize_dimensions = resize_dimensions
        self.crop_height = crop_height
        self.tile_dimensions = tile_dimensions
        self.tile_overlap = tile_overlap
        self.smoke_threshold = smoke_threshold
        self.num_tile_samples = num_tile_samples
        
        self.flip_augment = flip_augment
        self.resize_crop_augment = resize_crop_augment
        self.blur_augment = blur_augment
        self.color_augment = color_augment
        self.brightness_contrast_augment = brightness_contrast_augment
        
        self.num_tiles_height, self.num_tiles_width = util_fns.calculate_num_tiles(resize_dimensions, crop_height, tile_dimensions, tile_overlap)

    def __len__(self):
        return len(self.data_split)

    def __getitem__(self, idx):
        image_name = self.data_split[idx]
        series_length = len(self.metadata['image_series'][image_name])
        
        data_augmentations = util_fns.DataAugmentations(self.original_dimensions,
                                                        self.resize_dimensions,
                                                        self.crop_height,
                                                        
                                                        self.flip_augment, 
                                                        self.resize_crop_augment, 
                                                        self.blur_augment, 
                                                        self.color_augment, 
                                                        self.brightness_contrast_augment)
        
        ### Load Images ###
        x = []
        
        for file_name in self.metadata['image_series'][image_name]:
            # Load image
            # img.shape = [height, width, num_channels]
            img = cv2.imread(self.raw_data_path+'/'+file_name+'.jpg')
            
            # Resize image to original dimensions if it isn't already
            if img.shape[:2] != self.original_dimensions:
                img = cv2.resize(img, (self.original_dimensions[1], self.original_dimensions[0]))
            
            # Apply data augmentations
            # img.shape = [crop_height, resize_dimensions[1], num_channels]
            img = data_augmentations(img, is_labels=False)
                
            # Tile image
            # img.shape = [num_tiles, tile_height, tile_width, num_channels]
            img = util_fns.tile_image(img, self.num_tiles_height, self.num_tiles_width, self.resize_dimensions, self.tile_dimensions, self.tile_overlap)
            
            # Rescale and normalize
            img = util_fns.normalize_image(img)

            x.append(img)
            
        # x.shape = [num_tiles, series_length, num_channels, tile_height, tile_width]
        x = np.transpose(np.stack(x), (1, 0, 4, 2, 3))
           
        ### Load Labels ###
        label_path = self.labels_path+'/'+image_name+'.npy'
        if Path(label_path).exists():
            # Repeat similar steps to images
            labels = np.load(label_path)
            
            if labels.shape[:2] != self.original_dimensions:
                labels = cv2.resize(labels, (self.original_dimensions[1], self.original_dimensions[0]))
                
            labels = data_augmentations(labels, is_labels=True)
        else:
            labels = np.zeros(x[0].shape[:2], dtype=np.uint8) 

        # Tile labels
        labels = util_fns.tile_labels(labels, self.num_tiles_height, self.num_tiles_width, self.resize_dimensions, self.tile_dimensions, self.tile_overlap)

        # labels.shape = [num_tiles,]
        labels = (labels.sum(axis=(1,2)) > self.smoke_threshold).astype(float)

        if self.num_tile_samples > 0:
            # WARNING: Assumes that there are no labels with all 0s. Use --time-range-min 0
            x, labels = util_fns.randomly_sample_tiles(x, labels, self.num_tile_samples)

        # Load Image-level Labels ###
        ground_truth_label = self.metadata['ground_truth_label'][image_name]
        has_positive_tile = util_fns.get_has_positive_tile(labels)
        
        # Determine if tile predictions should be masked
        omit_mask = False if (self.omit_images_list is not None and image_name in self.omit_images_list) else True
            
        return image_name, x, labels, ground_truth_label, has_positive_tile, omit_mask