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
                 raw_data_path=None, 
                 embeddings_path=None,
                 labels_path=None, 
                 raw_labels_path=None,
                 metadata_path='./data/metadata.pkl',
                 save_embeddings_path=None,
                 
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
                 
                 resize_dimensions = (1536, 2016),
                 crop_height = 1120,
                 tile_dimensions = (224, 224),
                 tile_overlap = 0,
                 smoke_threshold = 250,
                 num_tile_samples = 0,
                 
                 flip_augment = True,
                 blur_augment = True,
                 jitter_augment = True,
                 
                 create_data = False):
        """
        Args:
            - raw_data_path (str): path to raw data
            - embeddings_path (str): path to embeddings generated from pretrained model
            - labels_path (str): path to Numpy labels
            - raw_labels_path (str): path to XML labels
            - metadata_path (str): path to metadata.pkl
                - fire_to_images (dict): dictionary with fires as keys and list of corresponding images as values
                - num_fires (int): total number of fires in dataset
                - num_images (int): total number of images in dataset
                - ground_truth_label (dict): dictionary with fires as keys and 1 if fire has "+" in its file name
                - has_xml_label (dict): dictionary with fires as keys and 1 if fire has a .xml file associated with it
                - omit_no_xml (list of str): list of images that erroneously do not have XML files for labels
                - omit_no_contour (list of str): list of images that erroneously do not have loaded bboxes for labels
                - omit_images_list (list of str): union of omit_no_xml and omit_no_contour
            - save_embeddings_path (str): if not None, saves embeddings to this path
            
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
            
            - resize_dimensions (int, int): desired dimensions of image before cropping
            - crop_height (int): height to crop image to
            - tile_dimensions (int, int): desired size of tiles
            - tile_overlap (int): amount to overlap each tile
            - smoke_threshold (int): # of pixels of smoke to consider tile positive
            - num_tile_samples (int): number of random tile samples per batch. If < 1, then turned off

            - flip_augment (bool): enables data augmentation with horizontal flip
            - blur_augment (bool): enables data augmentation with Gaussian blur
            - jitter_augment (bool): enables data augmentation with slightly displaced cropping
            
            - create_data (bool): should prepare_data be run?
        
        Other Attributes:
            - self.train_split (list): list of image names to be used for train dataloader
            - self.val_split (list): list of image names to be used for val dataloader
            - self.test_split (list): list of image names to be used for test dataloader
            - self.has_setup (bool): if setup has already occurred to prevent from doing twice
        """
        super().__init__()
                
        self.raw_data_path = raw_data_path
        self.embeddings_path = embeddings_path
        self.labels_path = labels_path
        self.raw_labels_path = raw_labels_path
        self.metadata = pickle.load(open(metadata_path, 'rb'))
        self.save_embeddings_path = save_embeddings_path
        
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
        
        self.resize_dimensions = resize_dimensions
        self.crop_height = crop_height
        self.tile_dimensions = tile_dimensions
        self.tile_overlap = tile_overlap
        self.smoke_threshold = smoke_threshold
        self.num_tile_samples = num_tile_samples
        
        self.flip_augment = flip_augment
        self.blur_augment = blur_augment
        self.jitter_augment = jitter_augment
        
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
            self.metadata['omit_images_list'] = []
            
            output_path = '/userdata/kerasData/data/new_data/drive_clone_new'

            # Loop through all fires
            for fire in self.metadata['fire_to_images']:
                self.metadata['num_fires'] += 1
                
                print('Preparing Folder ', self.metadata['num_fires'])
                
                # Loop through each image in the fire
                for image in self.metadata['fire_to_images'][fire]:
                    self.metadata['num_images'] += 1
                    
                    self.metadata['ground_truth_label'][image] = util_fns.get_ground_truth_label(image)
                    self.metadata['has_xml_label'][image] = util_fns.get_has_xml_label(image, self.labels_path)
                    
                    # If a positive image does not have an XML file associated with it, add it to omit_images_list
                    if self.metadata['ground_truth_label'][image] != self.metadata['has_xml_label'][image]:
                        self.metadata['omit_no_xml'].append(image)
                    
                    if self.metadata['has_xml_label'][image]:
                        labels = util_fns.get_filled_labels(self.raw_data_path, self.raw_labels_path, image)
                        
                        # If a positive image has an XML file with no segmentation mask in it, add it to omit_images_list
                        if labels.sum() == 0:
                            self.metadata['omit_no_contour'].append(image)

                        save_path = output_path + '/' + image + '.npy'

                        os.makedirs(output_path + '/' + fire, exist_ok=True)
                        np.save(save_path, labels.astype(np.uint8))

            self.metadata['omit_images_list'] = list(set().union(self.metadata['omit_no_xml'], self.metadata['omit_no_contour']))
        
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
                train_fires, val_fires = train_test_split(list(self.metadata['fire_to_images'].keys()), test_size=(1-self.train_split_size))
                val_fires, test_fires = train_test_split(val_fires, test_size=self.test_split_size/(1-self.train_split_size))
                        
            # Save arrays representing series of images
            self.metadata['image_series'] = util_fns.generate_series(self.metadata['fire_to_images'], self.series_length, self.add_base_flow)
            
            # Shorten fire_to_images to relevant time frame
            self.metadata['fire_to_images'] = util_fns.shorten_time_range(self.metadata['fire_to_images'], self.time_range, self.series_length, list(train_fires)+list(val_fires))
            # Shorten test only by series_length
            self.metadata['fire_to_images'] = util_fns.shorten_time_range(self.metadata['fire_to_images'], (-2400,2400), self.series_length, test_fires)
            
            # Create train/val/test split of Images, removing images from omit_images_list
            self.train_split = util_fns.unpack_fire_images(self.metadata['fire_to_images'], train_fires, self.metadata['omit_images_list'])
            self.val_split = util_fns.unpack_fire_images(self.metadata['fire_to_images'], val_fires, self.metadata['omit_images_list'])
            self.test_split = util_fns.unpack_fire_images(self.metadata['fire_to_images'], test_fires, self.metadata['omit_images_list'])

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
                                          embeddings_path=self.embeddings_path,
                                          labels_path=self.labels_path, 
                                          
                                          metadata=self.metadata, 
                                          save_embeddings_path=self.save_embeddings_path,
                                          data_split=self.train_split,
                                          
                                          resize_dimensions=self.resize_dimensions,
                                          crop_height=self.crop_height,
                                          tile_dimensions=self.tile_dimensions,
                                          tile_overlap=self.tile_overlap,
                                          smoke_threshold=self.smoke_threshold,
                                          num_tile_samples=self.num_tile_samples,
                                          
                                          flip_augment=self.flip_augment,
                                          blur_augment=self.blur_augment,
                                          jitter_augment=self.jitter_augment)
        train_loader = DataLoader(train_dataset, 
                                  batch_size=self.batch_size, 
                                  num_workers=self.num_workers,
                                  pin_memory=True, 
                                  shuffle=True)
        return train_loader

    def val_dataloader(self):
        val_dataset = DynamicDataloader(raw_data_path=self.raw_data_path, 
                                          embeddings_path=self.embeddings_path,
                                          labels_path=self.labels_path, 
                                        
                                          metadata=self.metadata,
                                          save_embeddings_path=self.save_embeddings_path,
                                          data_split=self.val_split,
                                        
                                          resize_dimensions=self.resize_dimensions,
                                          crop_height=self.crop_height,
                                          tile_dimensions=self.tile_dimensions,
                                          tile_overlap=self.tile_overlap,
                                          smoke_threshold=self.smoke_threshold,
                                          num_tile_samples=self.num_tile_samples,
                                        
                                          flip_augment=False,
                                          blur_augment=False,
                                          jitter_augment=False)
        val_loader = DataLoader(val_dataset, 
                                batch_size=self.batch_size, 
                                num_workers=self.num_workers,
                                pin_memory=True)
        return val_loader

    def test_dataloader(self):
        test_dataset = DynamicDataloader(raw_data_path=self.raw_data_path, 
                                          embeddings_path=self.embeddings_path,
                                          labels_path=self.labels_path, 
                                         
                                          metadata=self.metadata,
                                          save_embeddings_path=self.save_embeddings_path,
                                          data_split=self.test_split,
                                         
                                          resize_dimensions=self.resize_dimensions,
                                          crop_height=self.crop_height,
                                          tile_dimensions=self.tile_dimensions,
                                          tile_overlap=self.tile_overlap,
                                          smoke_threshold=self.smoke_threshold,
                                          num_tile_samples=0,
                                         
                                          flip_augment=False,
                                          blur_augment=False,
                                          jitter_augment=False)
        test_loader = DataLoader(test_dataset, 
                                 batch_size=self.batch_size if self.save_embeddings_path is None else 1, 
                                 num_workers=self.num_workers,
                                 pin_memory=True)
        return test_loader
    
    
#####################
## Dataloader
#####################
    
class DynamicDataloader(Dataset):
    def __init__(self, 
                 raw_data_path=None, 
                 embeddings_path=None,
                 labels_path=None, 
                 
                 metadata=None,
                 save_embeddings_path=None,
                 data_split=None, 
                 
                 resize_dimensions = (1536, 2016),
                 crop_height = 1120,
                 tile_dimensions = (224,224), 
                 tile_overlap = 0,
                 smoke_threshold = 250,
                 num_tile_samples = 0,
                 
                 flip_augment = True,
                 blur_augment = True,
                 jitter_augment = True):
        
        self.raw_data_path = raw_data_path
        self.embeddings_path = embeddings_path
        self.labels_path = labels_path
        
        self.metadata = metadata
        self.save_embeddings_path = save_embeddings_path
        self.data_split = data_split
        
        self.resize_dimensions = resize_dimensions
        self.crop_height = crop_height
        self.tile_dimensions = tile_dimensions
        self.tile_overlap = tile_overlap
        self.smoke_threshold = smoke_threshold
        self.num_tile_samples = num_tile_samples
        
        self.flip_augment = flip_augment
        self.blur_augment = blur_augment
        self.jitter_augment = jitter_augment
        
        self.num_tiles_height, self.num_tiles_width = util_fns.calculate_num_tiles(resize_dimensions, crop_height, tile_dimensions, tile_overlap)

    def __len__(self):
        return len(self.data_split)
    
    def get_images(self, image_name, should_flip, should_blur, blur_size, jitter_amount):
        """Description: Loads series_length of raw images. Crops, resizes, and adds data augmentations"""
        x = []
        
        for file_name in self.metadata['image_series'][image_name]:
            # img.shape = [height, width, num_channels]
            img = cv2.imread(self.raw_data_path+'/'+file_name+'.jpg')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Resize and crop
            img = util_fns.crop_image(img, self.resize_dimensions, self.crop_height, self.tile_dimensions, jitter_amount)
            
            # Add data augmentations
            if should_flip:
                img = cv2.flip(img, 1)
            if should_blur:
                img = cv2.blur(img, (blur_size,blur_size))
                
            # Tile image
            # Source: https://towardsdatascience.com/efficiently-splitting-an-image-into-tiles-in-python-using-numpy-d1bf0dd7b6f7
            # WARNING: Tile size must divide perfectly into image height and width
            bytelength = img.nbytes // img.size
            # img.shape = [num_tiles_height, num_tiles_width, tile_height, tile_width, 3]
            img = np.lib.stride_tricks.as_strided(img, 
                shape=(self.num_tiles_height, 
                       self.num_tiles_width, 
                       self.tile_dimensions[0], 
                       self.tile_dimensions[1],
                       3), 
                strides=(self.resize_dimensions[1]*(self.tile_dimensions[0]-self.tile_overlap)*bytelength*3,
                         (self.tile_dimensions[1]-self.tile_overlap)*bytelength*3, 
                         self.resize_dimensions[1]*bytelength*3, 
                         bytelength*3, 
                         bytelength), writeable=False)
            
            # img.shape = [num_tiles, tile_height, tile_width, 3]
            img = img.reshape((-1, self.tile_dimensions[0], self.tile_dimensions[1], 3))
            
            # Rescale to [0,1]
            img = img / 255
            # Normalize to 0.5 mean & std
            img = (img - 0.5) / 0.5

            x.append(img)
            
        # x.shape = [num_tiles, series_length, num_channels, height, width]
        # e.g. [45, 5, 3, 224, 224]
        x = np.transpose(np.stack(x), (1, 0, 4, 2, 3))
        
        return x
        
    def get_embeddings(self, image_name, should_flip, should_blur, blur_size):
        """Description: Loads pre-saved embeddings"""
        x = []
        
        for file_name in self.metadata['image_series'][image_name]:
            if should_flip:
                img = np.load(self.embeddings_path+'/flip/'+file_name+'.npy').squeeze()
            elif should_blur:
                img = np.load(self.embeddings_path+'/blur/'+file_name+'.npy').squeeze()
            else:
                img = np.load(self.embeddings_path+'/raw/'+file_name+'.npy').squeeze()
                
            x.append(img)
                
        # x.shape = [num_tiles, series_length, embedding_size]
        # e.g. [45, 4, 960]
        x = np.transpose(np.stack(x), (1, 0, 2))
        
        return x
    
    def prep_save_embeddings(self, image_name, blur_size):
        """Description: Loads data augmented raw images to be saved as embeddings"""
        x = []
        
        img = cv2.imread(self.raw_data_path+'/'+image_name+'.jpg')
        img = util_fns.crop_image(img, self.resize_dimensions, self.crop_height, self.tile_dimensions)
        
        x.append(img)
        x.append(cv2.flip(img, 1))
        x.append(cv2.blur(img, (blur_size,blur_size)))
        
        # x.shape = [series_length, num_channels, height, width]
        # e.g. [5, 3, 1344, 2016]
        x = np.transpose(np.stack(x), (0, 3, 1, 2)) / 255
        
        return x

    def __getitem__(self, idx):
        image_name = self.data_split[idx]
        series_length = len(self.metadata['image_series'][image_name]) if self.save_embeddings_path is None else 3
        
        ### Initialize Data Augmentation ###
        should_flip = np.random.rand() > 0.5 if self.flip_augment else False
        
        should_blur = np.random.rand() > 0.5 if self.blur_augment else False
        blur_size = np.maximum(int(np.random.randn()*3+10), 1)
        
        # Always jitter if jitter_augment=True
        jitter_amount = np.random.randint(self.tile_dimensions[0]) if self.jitter_augment else 0
        
        ### Load Images or Embeddings ###
        if self.embeddings_path is not None:
            x = self.get_embeddings(image_name, should_flip, should_blur, blur_size)
        elif self.save_embeddings_path is not None:
            x = self.prep_save_embeddings(image_name, blur_size)
        else:
            x = self.get_images(image_name, should_flip, should_blur, blur_size, jitter_amount)
           
        ### Load Labels ###
        label_path = self.labels_path+'/'+image_name+'.npy'
        if Path(label_path).exists():
            labels = np.load(label_path)
        else:
            labels = np.zeros(x[0].shape[:2], dtype=np.uint8) 
        
        # labels.shape = [height, width]
        labels = util_fns.crop_image(labels, self.resize_dimensions, self.crop_height, self.tile_dimensions, jitter_amount)
        if should_flip:
            labels = cv2.flip(labels, 1)
        if should_blur:
            labels = cv2.blur(labels, (blur_size, blur_size))

        bytelength = labels.nbytes // labels.size
        labels = np.lib.stride_tricks.as_strided(labels, 
            shape=(self.num_tiles_height, 
                   self.num_tiles_width, 
                   self.tile_dimensions[0], 
                   self.tile_dimensions[1]), 
            strides=(self.resize_dimensions[1]*(self.tile_dimensions[0]-self.tile_overlap)*bytelength,
                     (self.tile_dimensions[1]-self.tile_overlap)*bytelength, 
                     self.resize_dimensions[1]*bytelength, 
                     bytelength), writeable=False)
        labels = labels.reshape(-1, self.tile_dimensions[0], self.tile_dimensions[1])

        # labels.shape = [45,]
        labels = (labels.sum(axis=(1,2)) > self.smoke_threshold).astype(float)

        if self.save_embeddings_path is None and self.num_tile_samples > 0:
            # WARNING: Assumes that there are no labels with all 0s. Use --time-range-min 0
            x, labels = util_fns.randomly_sample_tiles(x, labels, self.num_tile_samples)

        # Load Image-level Labels ###
        ground_truth_label = self.metadata['ground_truth_label'][image_name]
        has_positive_tile = util_fns.get_has_positive_tile(labels)
                        
        return image_name, x, labels, ground_truth_label, has_positive_tile