import pytorch_lightning as pl
import pickle
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split


class BatchedTiledDataModule(pl.LightningDataModule):
    """
    - raw_images_path (str): path to raw images
    - labels_path (str): path to XML labels
    - output_path (str): desired path of outputted Numpy files
    - image_dimensions (int, int): dimensions original image should be resized to
    - tile_dimensions (int, int): desired dimensions of tiles
    - overlap_amount (int): how much the tiles should overlap in pixels
    - smoke_threshold (int): how many pixels of smoke to label tile as a positive sample? 
    - time_created (datetime): date & time of when the dataset was created
    - fire_to_images (dict): dictionary with fires as keys and list of corresponding images as values
    - num_fires (int): total number of fires in dataset
    - num_images (int): total number of images in dataset
    - ground_truth_label (dict): dictionary with fires as keys and 1 if fire has "+" in its file name
    - has_xml_label (dict): dictionary with fires as keys and 1 if fire has a .xml file associated with it
    - has_positive_tile (dict): dictionary with fires as keys and 1 if at least one tiled image has a label of 1
    """
    def __init__(self, data_path, metadata_path, batch_size, series_length=5, time_range=(-2400, 2400)):
        super().__init__()
        self.data_path = data_path
        self.metadata = pickle.load(open(metadata_path, 'rb'))
        
        self.batch_size = batch_size
        self.series_length = series_length
        
        # Confirm time_range is between -2400 and 2400 and divisible by 60
        assert time_range[0] % 60 == 0 and time_range[0] >= -2400 and time_range[0] <= 2400-60
        assert time_range[1] % 60 == 0 and time_range[1] >= -2400+60 and time_range[1] <= 2400
        self.time_range = time_range

    def setup(self, stage):
        # Create train, val, test split of *fires*
        train_split, val_split = train_test_split(list(self.metadata['fire_to_images'].keys()), test_size=0.4)
        val_split, test_split = train_test_split(val_split, test_size=0.5)
        
        # Calculate indices related to desired time frame
        time_range_lower_idx = int((self.time_range[0] + 2400) / 60)
        time_range_upper_idx = int((self.time_range[1] + 2400) / 60) + 1 

        # Create dictionary to store series_length of images
        self.metadata['image_series'] = {}

        for fire in self.metadata['fire_to_images']:
            # Shorten fire_to_images to relevant time frame
            self.metadata['fire_to_images'][fire] = self.metadata['fire_to_images'][fire][time_range_lower_idx:time_range_upper_idx]
    
            # Add series_length of images to image_series for each image
            for i, img in enumerate(self.metadata['fire_to_images'][fire]):
                self.metadata['image_series'][img] = []
                idx = i
                
                while (len(self.metadata['image_series'][img]) < self.series_length):
                    self.metadata['image_series'][img].append(self.metadata['fire_to_images'][fire][idx])
                    if idx != 0: idx -= 1
                        
        self.train_split = [image for image in self.metadata['fire_to_images'][fire] for fire in train_split]
        self.val_split   = [image for image in self.metadata['fire_to_images'][fire] for fire in val_split]
        self.test_split  = [image for image in self.metadata['fire_to_images'][fire] for fire in test_split]

    def train_dataloader(self):
        train_dataset = BatchedTiledDataloader(self.data_path, self.train_split, self.metadata)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        return train_loader

    def val_dataloader(self):
        val_dataset = BatchedTiledDataloader(self.data_path, self.val_split, self.metadata)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size)
        return val_loader

    def test_dataloader(self):
        test_dataset = BatchedTiledDataloader(self.data_path, self.test_split, self.metadata)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size)
        return test_loader

    
class BatchedTiledDataloader(Dataset):
    def __init__(self, data_path, data_split, metadata):
        self.data_path = Path(data_path)
        self.data_split = data_split
        self.metadata = metadata

    def __len__(self):
        return len(self.data_split)

    def __getitem__(self, idx):
        cur_image = self.data_split[idx]
        
        x = []
        for file_name in self.metadata['image_series'][cur_image]:
            x.append(np.load(self.data_path/f'{file_name}_img.npy'))

        # x.shape = [num_tiles, series_length, num_channels, height, width]
        # e.g. [108, 5, 3, 224, 224]
        x = np.transpose(np.stack(x), (1, 0, 2, 3, 4))/255 
        
        # y.shape = [num_tiles] e.g. [108,]
        y = np.load(self.data_path/f'{cur_image}_lbl.npy') 

        # integers
        ground_truth_label = self.metadata['ground_truth_label'][cur_image]
        has_xml_label = self.metadata['has_xml_label'][cur_image]
        has_positive_tile = self.metadata['has_positive_tile'][cur_image]
        
        return x, y, ground_truth_label, has_xml_label, has_positive_tile