"""
Created by: Anshuman Dewangan
Date: 2021

Description: Utility and helper functions used in all files.
"""

# Torch imports
import torch
import torchmetrics

# Other package imports 
import os
import cv2
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from torch._six import string_classes
import collections
import re
import copy


#############
## General
#############

def get_fire_name(path):
    """
    Description: Gets fire name from longer path string
    Args:
        - path (str): Full path to filename or path of {fire}/{image}
    Returns:
        - fire_name (str): name of fire e.g. "20160718_FIRE_mg-s-iqeye"
    """
    # ASSUMPTION: Last two directories of path are {fire} and {image} respectively
    fire_name = path.split('/')[-2]
    
    return fire_name

def get_only_image_name(path):
    """
    Description: Gets only image name from longer path string (no fire name)
    Args:
        - path (str): Full path to filename or path of {fire}/{image}
    Returns:
        - only_image_name (str): name of image e.g. 1563305245_-01080
    """
    # ASSUMPTION: Last two directories of path are {fire} and {image} respectively
    only_image_name = path.split('/')[-1]
    
    only_image_name = str(Path(only_image_name).stem)
    
    # Remove '_lbl' or '_img' if it exists
    # ASSUMPTION: Last 4 chars of image_name may have '_lbl' or '_img'
    if only_image_name[-4:] == '_lbl' or only_image_name[-4:] == '_img':
        only_image_name = only_image_name[:-4]
    
    return only_image_name

def get_image_name(path):
    """
    Description: Gets image name from longer path string
    Args:
        - path (str): Full path to filename or path of {fire}/{image}
    Returns:
        - image_name (str): name of image e.g. 20190716_Meadowfire_hp-n-mobo-c/1563305245_-01080
    """
    # Get last two names of path: fire and image
    image_name = get_fire_name(path) + '/' + get_only_image_name(path)
    
    return image_name

def image_name_to_time_int(image_name):
    """
    Description: Given an image name (e.g. 20190716_Meadowfire_hp-n-mobo-c/1563305245_-01080), returns time index as integer (e.g. -1080)
    Args:
        - image_name (str): name of image
    Returns:
        - time_int (int): time index as integer
    """
    # ASSUMPTION: Last six characters of image name is the time stamp
    time_int = int(image_name[-6:])
    
    return time_int

def calculate_num_tiles(resize_dimensions, crop_height, tile_dimensions, tile_overlap):
    """Description: Give image size, calculates the number of tiles along the height and width"""
    num_tiles_height = 1 + (crop_height - tile_dimensions[0]) // (tile_dimensions[0] - tile_overlap)
    num_tiles_width = 1 + (resize_dimensions[1] - tile_dimensions[1]) // (tile_dimensions[1] - tile_overlap)
    
    return num_tiles_height, num_tiles_width

def get_ground_truth_label(image_name):
    """Description: Returns 1 if image_name has a + in it (ie. is a positive) or 0 otherwise"""
    ground_truth_label = 1 if "+" in image_name else 0
    return ground_truth_label

###############
## DataModule
###############

def generate_fire_to_images(raw_data_path):
    """
    Description: Given path to raw images, create dictionary mapping fire to list of images
    Args:
        - raw_data_path (str): path to raw data
    Returns:
        - fire_to_images (dict): maps fire to a list of images for that fire
    """
    raw_data_path = Path(raw_data_path)
        
    fire_to_images = {}
    all_fires = [folder.stem for folder in filter(Path.is_dir, raw_data_path.iterdir())]

    for fire in all_fires:
        # Skip if first character is '.'
        if fire[0] == '.':
            continue
        
        images = [get_image_name(str(item)) for item in (raw_data_path/fire).glob('*.jpg')]
        if len(images) > 0:
            fire_to_images[fire] = images
            fire_to_images[fire].sort()
        
    return fire_to_images

def generate_fire_to_images_from_splits(splits):
    """
    Description: Given train/val/test splits, create dictionary mapping fire to list of images
    Args:
        - splits (list of list): train/val/test splits of fires loaded from .txt file
    Returns:
        - fire_to_images (dict): maps fire to a list of images for that fire
    """
    fire_to_images = {}
    
    for split in splits:
        for item in split:
            fire = get_fire_name(item)
            image_name = get_image_name(item)
            
            if fire not in fire_to_images:
                fire_to_images[fire] = []
                
            fire_to_images[fire].append(image_name)
            
    for fire in fire_to_images:
        fire_to_images[fire].sort()
        
    return fire_to_images

def unpack_fire_images(fire_to_images, fires_list, omit_images_list=None):
    """
    Description: Returns images from a list of fires. If train or val, do not include images that are in 'omit_images_list'.
    Args:
        - fire_to_images (dict): maps fire to a list of images for that fire
        - fires_list (list of str): list of fire names to unpack
        - omit_images_list (list of str): list of images to not include (because of erroneous labeling)
    Returns:
        - unpacked_images (list of str): list of images from the fires
    """
    unpacked_images = []
    
    for fire in fires_list:
        for image in fire_to_images[fire]:
            if omit_images_list is None or image not in omit_images_list:
                unpacked_images.append(image)
                
    return unpacked_images

def shorten_time_range(fires_list, fire_to_images, time_range=(-2400,2400), series_length=1):
    """
    Description: From lists of images per fire, returns list of images within specified time range
    Args:
        - fires_list (list of str): list of fires in desired split
        - fire_to_images (dict): maps fire to a list of images for that fire
        - time_range (int, int): the time range of images to consider for training by time stamp
        - series_length (int): length of series to cut off starting of fires
    Returns:
        - fire_to_images (dict): list of images for each fire
    """
    # Calculate effective time start
    effective_series_length = series_length
    
    for fire in fires_list:
        for _ in range(effective_series_length-1):
            fire_to_images[fire].pop(0)
    
    effective_start = np.maximum(time_range[0], -2400)
        
    if effective_start > -2400 or time_range[1] < 2400:
        for fire in fires_list:
            images_to_keep = []

            for image in fire_to_images[fire]:
                image_time_index = image_name_to_time_int(image)
                if image_time_index >= effective_start and image_time_index <= time_range[1]:
                    images_to_keep.append(image)

            fire_to_images[fire] = images_to_keep
            
    return fire_to_images

def generate_series(fire_to_images, series_length):
    """
    Description: Creates a dict with image names as keys and lists of past <series_length> images as values
    Args:
        - fire_to_images (dict): maps fire to a list of images for that fire
        - series_length (int): how many sequential images should be used for training
        
    Returns:
        - image_series (dict): maps image names to lists of past <series_length> images in chronological order
    """
    image_series = {}
        
    for fire in fire_to_images:
        for i, img in enumerate(fire_to_images[fire]):
            image_series[img] = []
            idx = i
            

            while (len(image_series[img]) < series_length):
                image_series[img].insert(0, fire_to_images[fire][idx])
                if idx != 0: idx -= 1
    
    return image_series

def xml_to_contour(xml_file):
    """
    Description: Takes an XML label file and converts it to Numpy array
    Args:
        - xml_file (str): Path to XML file
    Returns:
        - all_polys (Numpy array): Numpy array with labels
    """
    tree = ET.parse(xml_file)
    
    for cur_object in tree.findall('object'):
        if cur_object.find('deleted').text=="1":
            continue
        
        if cur_object.find('name').text=="bp":
            all_polys = []
            
            for cur_poly in cur_object.findall('polygon'):
                cur_poly_pts = []
                for cur_pt in cur_poly.findall('pt'):
                    cur_poly_pts.append([int(cur_pt.find('x').text), int(cur_pt.find('y').text)])
                all_polys.append(cur_poly_pts)
            
            all_polys = np.array(all_polys, dtype=np.int32)
            return all_polys
    
    return None

def xml_to_bbox(xml_file):
    """
    Description: Takes an XML label file and converts it to Numpy array
    Args:
        - xml_file (str): Path to XML file
    Returns:
        - all_polys (Numpy array): Numpy array with labels
    """
    tree = ET.parse(xml_file)
    
    for cur_object in tree.findall('object'):
        if cur_object.find('deleted').text=="1":
            continue
        
        if cur_object.find('name').text=="sbb":
            x_s = []
            y_s = []
            for cur_pt in cur_object.find('polygon').findall('pt'):
                x_s.append(int(round(float(cur_pt.find('x').text))))
                y_s.append(int(round(float(cur_pt.find('y').text))))
            
            all_polys = [[min(x_s), min(y_s)], [max(x_s), max(y_s)]]
            all_polys = np.array(all_polys, dtype=np.int32)
            return all_polys
    
    return None


###############
## Dataloader
###############

class DataAugmentations():
    """Description: Data Augmentation class to ensure same augmentations are applied to image and labels"""
    def __init__(self, 
                 original_dimensions = (1536, 2048),
                 resize_dimensions = (1536, 2048),
                 tile_dimensions = (224, 224),
                 crop_height = 1244,
                 
                 flip_augment = True,
                 resize_crop_augment = True,
                 blur_augment = True,
                 color_augment = True,
                 brightness_contrast_augment = True):
        
        self.resize_dimensions = resize_dimensions
        self.crop_height = crop_height
        self.tile_dimensions = tile_dimensions
        
        self.resize_crop_augment = resize_crop_augment
        self.blur_augment = blur_augment
        self.color_augment = color_augment
        self.brightness_contrast_augment = brightness_contrast_augment
        
        # Determine amount to jitter height
        self.jitter_amount = np.random.randint(tile_dimensions[0]) if self.resize_crop_augment else 0
        
        # Determine if we should flip this time
        self.should_flip = np.random.rand() > 0.5 if flip_augment else False
        self.should_blur = np.random.rand() > 0.5 if blur_augment else False
        self.should_color = np.random.rand() > 0.5 if color_augment else False
        self.should_brightness = np.random.rand() > 0.5 if brightness_contrast_augment else False
        
        # Determines blur amount
        if self.should_blur:
            self.blur_size = np.maximum(int(np.random.randn()*3+10), 1)
        
    def __call__(self, img, is_labels=False):
        # Save resize factor for bbox labels
        self.resize_factor = np.array(self.resize_dimensions) / np.array(img.shape[:2])
        
        # Resize
        img = cv2.resize(img, (self.resize_dimensions[1],self.resize_dimensions[0]))
        
        # Save image_center for flipping bboxes
        self.img_center = np.array(img.shape[:2])[::-1]/2
        self.img_center = np.hstack((self.img_center, self.img_center))
        
        # Crop
        if self.jitter_amount == 0:
            img = img[-self.crop_height:]
        else:
            img = img[-(self.crop_height+self.jitter_amount):-self.jitter_amount]

        # Flip
        if self.should_flip:
            img = cv2.flip(img, 1)
            
        if not is_labels:
            # Color
            if self.should_color:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                img[:,:,0] = np.add(img[:,:,0], np.random.randn()*2, casting="unsafe")
                img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Brightness contrast
        if self.should_brightness and not is_labels:
            img = cv2.convertScaleAbs(img, alpha=np.random.uniform(0.95,1.05), beta=np.random.randint(-10,10))

        # Blur
        if self.should_blur:
            img = cv2.blur(img, (self.blur_size,self.blur_size))
        
        return img
    
    def process_bboxes(self, gt_bboxes):
        bboxes = []
        
        # Resize bboxes to appropriate image resize factor
        for i in range(len(gt_bboxes)):
            bboxes.append([0,0,0,0])
            
            # Scale bbox by resize_factor
            bboxes[i][0] = gt_bboxes[i][0]*self.resize_factor[1]
            # Move y-axis by crop_height & jitter_amount
            bboxes[i][1] = gt_bboxes[i][1]*self.resize_factor[0] - (self.resize_dimensions[0]-self.crop_height-self.jitter_amount)
            bboxes[i][2] = gt_bboxes[i][2]*self.resize_factor[1]
            bboxes[i][3] = gt_bboxes[i][3]*self.resize_factor[0] - (self.resize_dimensions[0]-self.crop_height-self.jitter_amount)

            # Make sure bbox isn't off-image
            bboxes[i][1] = np.maximum(bboxes[i][1], 0)
            bboxes[i][3] = np.maximum(bboxes[i][3], 1)
            
            if self.should_flip:
                # Source: https://blog.paperspace.com/data-augmentation-for-bounding-boxes/
                bboxes[i][0] += 2*(self.img_center[0] - bboxes[i][0])
                bboxes[i][2] += 2*(self.img_center[2] - bboxes[i][2])
                box_w = abs(bboxes[i][0] - bboxes[i][2])
                bboxes[i][0] -= box_w
                bboxes[i][2] += box_w

        return bboxes

def tile_image(img, num_tiles_height, num_tiles_width, resize_dimensions, tile_dimensions, tile_overlap):
    """
    Description: Tiles image with overlap
    Source: https://towardsdatascience.com/efficiently-splitting-an-image-into-tiles-in-python-using-numpy-d1bf0dd7b6f7
    WARNING: Tile size must divide perfectly into image height and width
    """
    bytelength = img.nbytes // img.size
    
    # img.shape = [num_tiles_height, num_tiles_width, tile_height, tile_width, 3]
    img = np.lib.stride_tricks.as_strided(img, 
        shape=(num_tiles_height, 
               num_tiles_width, 
               tile_dimensions[0], 
               tile_dimensions[1],
               3), 
        strides=(resize_dimensions[1]*(tile_dimensions[0]-tile_overlap)*bytelength*3,
                 (tile_dimensions[1]-tile_overlap)*bytelength*3, 
                 resize_dimensions[1]*bytelength*3, 
                 bytelength*3, 
                 bytelength), writeable=False)

    # img.shape = [num_tiles, tile_height, tile_width, num_channels]
    img = img.reshape((-1, tile_dimensions[0], tile_dimensions[1], 3))
    
    return img

def normalize_image(img):
    """Description: Rescales and normalizes an image"""
    # Rescale to [0,1]
    img = img / 255
    # Normalize to 0.5 mean & std
    img = (img - 0.5) / 0.5
    
    return img

def tile_labels(labels, num_tiles_height, num_tiles_width, resize_dimensions, tile_dimensions, tile_overlap):
    """
    Description: Tiles labels with overlap
    Source: https://towardsdatascience.com/efficiently-splitting-an-image-into-tiles-in-python-using-numpy-d1bf0dd7b6f7
    WARNING: Tile size must divide perfectly into image height and width
    """
    
    bytelength = labels.nbytes // labels.size
    labels = np.lib.stride_tricks.as_strided(labels, 
        shape=(num_tiles_height, 
               num_tiles_width, 
               tile_dimensions[0], 
               tile_dimensions[1]), 
        strides=(resize_dimensions[1]*(tile_dimensions[0]-tile_overlap)*bytelength,
                 (tile_dimensions[1]-tile_overlap)*bytelength, 
                 resize_dimensions[1]*bytelength, 
                 bytelength), writeable=False)
    labels = labels.reshape(-1, tile_dimensions[0], tile_dimensions[1])
    
    return labels

def randomly_sample_tiles(x, labels, num_samples=30):
    """
    Description: Randomly samples tiles to evenly balance positives and negatives
    Args:
        - x: pre-processed input (raw data -> resized, cropped, and tiled)
        - labels: pre-processed labels (resized, cropped, and tiled)
        - num_samples: total number of samples to keep
    """
    # Separate indices for positive and negative values
    pos_indices = np.where(labels == 1)[0]
    neg_indices = np.where(labels == 0)[0]
    
    # Randomly sample num_tiles of each
    size = num_samples//2
    sampled_pos_indices = np.random.choice(pos_indices, size=size, replace=len(pos_indices)<size)
    sampled_neg_indices = np.random.choice(neg_indices, size=size, replace=len(neg_indices)<size)

    # Update x and labels
    x = x[np.concatenate((sampled_pos_indices, sampled_neg_indices))]
    labels = labels[np.concatenate((sampled_pos_indices, sampled_neg_indices))]
    
    # Shuffle x and labels
    shuffle_idx = np.random.permutation(len(x))
    x = x[shuffle_idx]
    labels = labels[shuffle_idx]
    
    return x, labels

np_str_obj_array_pattern = re.compile(r'[SaUO]')
#####################
## LightningModule
#####################

def calculate_negative_accuracy(negative_preds_dict):
    """
    Description: Calculates accuracy on negative images
    Args:
        - negative_preds_dict (dict): predictions on negatives by fire
    """
    all_preds = []
    for fire in negative_preds_dict:
        all_preds.extend(negative_preds_dict[fire])
            
    return 1 - sum(all_preds) / len(all_preds)

def calculate_negative_accuracy_by_fire(negative_preds_dict):
    """
    Description: Calculates % of fires that DID NOT have a false positive
    Args:
        - negative_preds_dict (dict): predictions on negatives by fire
    """
    fire_preds = []
    
    for fire in negative_preds_dict:
        if sum(negative_preds_dict[fire]) > 0:
            fire_preds.append(0)
        else:
            fire_preds.append(1)
    
    return sum(fire_preds) / len(fire_preds)

def calculate_positive_accuracy(positive_preds_dict):
    """
    Description: Calculates accuracy on positive images
    Args:
        - positive_preds_dict (dict): predictions on positives by fire
    """
    all_preds = []
    for fire in positive_preds_dict:
        all_preds.extend(positive_preds_dict[fire])
            
    return sum(all_preds) / len(all_preds)

def calculate_positive_accuracy_by_fire(positive_preds_dict):
    """
    Description: Calculates % of fires that DID NOT have a false negative
    Args:
        - positive_preds_dict (dict): predictions on positives by fire
    """
    fire_preds = []
    
    for fire in positive_preds_dict:
        if sum(positive_preds_dict[fire]) < len(positive_preds_dict[fire]):
            fire_preds.append(0)
        else:
            fire_preds.append(1)
    
    return sum(fire_preds) / len(fire_preds)
    
def calculate_positive_accuracy_by_time(positive_preds_dict):
    """
    Description: Calculates accuracy per time step
    Args:
        - positive_preds_dict (dict): predictions on positives
    """
    positive_preds_dict = copy.deepcopy(positive_preds_dict)
    time_dict = {}
    time_dict_cumulative = {}
    
    for fire in positive_preds_dict:
        for i in range(len(positive_preds_dict[fire])):
            if str(i) not in time_dict:
                time_dict[str(i)] = []
                time_dict_cumulative[str(i)] = []
            
            time_dict[str(i)].append(positive_preds_dict[fire][i])
            
            if i != 0 and positive_preds_dict[fire][i-1] == 1:
                positive_preds_dict[fire][i] = 1
                
            time_dict_cumulative[str(i)].append(positive_preds_dict[fire][i])
    
    return_list = []
    return_list_cumulative = []
    for i in range(len(time_dict)):
        return_list.append(sum(time_dict[str(i)]) / len(time_dict[str(i)]))
        return_list_cumulative.append(sum(time_dict_cumulative[str(i)]) / len(time_dict_cumulative[str(i)]))
    
    return return_list, return_list_cumulative

def calculate_time_to_detection_stats(positive_preds_dict, positive_times_dict):
    """
    Description: Calculates average time to detection across all fires
    Args:
        - positive_preds_dict (dict): predictions on positives
        - positive_times_dict (dict): timestamps of positive predictions
    Returns:
        - average_time_to_detection (float)
        - median_time_to_detection (float)
        - std_time_to_detection (float)
    """
    times_to_detection = []
    
    for fire in positive_preds_dict:
        for i in range(len(positive_preds_dict[fire])):
            if positive_preds_dict[fire][i] == 1:
                times_to_detection.append(positive_times_dict[fire][i])
                break
    
    times_to_detection = np.array(times_to_detection) / 60
    
    average_time_to_detection = times_to_detection.mean()
    median_time_to_detection = np.median(times_to_detection)
    std_time_to_detection = times_to_detection.std()
    
    return average_time_to_detection, median_time_to_detection, std_time_to_detection


######################
## Model Components
######################

def init_weights_RetinaNet(*layers):
    """
    Description: Initialize weights as in RetinaNet paper
    Args:
        - layers (torch nn.Modules): layers to initialize
    Returns:
        - layers (torch nn.Modules): layers with weights initialized
    """
    for i, layer in enumerate(layers):
        torch.nn.init.normal_(layer.weight, 0, 0.01)

        # Set last layer bias to special value from paper
        if i == len(layers)-1:
            torch.nn.init.constant_(layer.bias, -np.log((1-0.01)/0.01))
        else:
            torch.nn.init.zeros_(layer.bias)
    
    return layers

def init_weights_Xavier(*layers):
    """
    Description: Initialize weights using xavier_uniform
    Args:
        - layers (torch nn.Modules): layers to initialize
    Returns:
        - layers (torch nn.Modules): layers with weights initialized
    """
    for layer in layers:
        torch.nn.init.xavier_uniform_(layer.weight, gain=torch.nn.init.calculate_gain('relu'))
        torch.nn.init.xavier_uniform_(layer.bias.reshape((-1,1)))
    
    return layers

def get_state_dict(backbone_checkpoint_path):
    """Description: Returns state dict of backbone_checkpoint_path after cleaning keys"""
    
    state_dict = torch.load(backbone_checkpoint_path)['state_dict']
    new_state_dict = {}
    
    for key in state_dict:
        # ASSUMPTION: First 19 characters of key need to be removed
        new_state_dict[str(key)[19:]] = state_dict[key]

    return new_state_dict