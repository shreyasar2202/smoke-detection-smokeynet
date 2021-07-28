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


###############
## DataModule
###############

def generate_fire_to_images(raw_data_path, labels_path):
    """
    Description: Given path to raw images and labels, create dictionary mapping fire to list of images
    Args:
        - raw_data_path (str): path to raw data
        - labels_path (str): path to XML labels
    Returns:
        - fire_to_images (dict): maps fire to a list of images for that fire
    """
    raw_data_path = Path(raw_data_path)
    labels_path = Path(labels_path)
        
    fire_to_images = {}
    all_fires = [folder.stem for folder in filter(Path.is_dir, labels_path.iterdir())]

    for fire in all_fires:
        # Skip if first character is '.' or if fire is monochrome
        if fire[0] == '.' or 'mobo-m' in fire:
            continue
        
        fire_to_images[fire] = [get_image_name(str(item)) for item in (raw_data_path/fire).glob('*.jpg')]
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

def shorten_time_range(fires_list, fire_to_images, time_range=(-2400,2400), series_length=1, add_base_flow=False):
    """
    Description: From lists of images per fire, returns list of images within specified time range
    Args:
        - fires_list (list of str): list of fires in desired split
        - fire_to_images (dict): maps fire to a list of images for that fire
        - time_range (int, int): the time range of images to consider for training by time stamp
        - series_length (int): length of series to cut off starting of fires
        - add_base_flow (bool): if True, adds image from t-5 for fire
    Returns:
        - fire_to_images (dict): list of images for each fire
    """
    # Calculate effective time start
    effective_series_length = np.maximum(int(add_base_flow)*5, series_length)
    effective_start = np.maximum(time_range[0], -2400 + (effective_series_length - 1) * 60)
        
    if effective_start > -2400 or time_range[1] < 2400:
        for fire in fires_list:
            images_to_keep = []

            for image in fire_to_images[fire]:
                image_time_index = image_name_to_time_int(image)
                if image_time_index >= effective_start and image_time_index <= time_range[1]:
                    images_to_keep.append(image)

            fire_to_images[fire] = images_to_keep
            
    return fire_to_images

def generate_series(fire_to_images, series_length, add_base_flow=False):
    """
    Description: Creates a dict with image names as keys and lists of past <series_length> images as values
    Args:
        - fire_to_images (dict): maps fire to a list of images for that fire
        - series_length (int): how many sequential images should be used for training
        - add_base_flow (bool): if True, adds image from t-5 for fire
    Returns:
        - image_series (dict): maps image names to lists of past <series_length> images in chronological order
    """
    image_series = {}
        
    for fire in fire_to_images:
        for i, img in enumerate(fire_to_images[fire]):
            image_series[img] = []
            idx = i
            
            # Add the t-5 image if add_base_flow
            if series_length != 1 and add_base_flow:
                image_series[img].append(fire_to_images[fire][np.maximum(0, idx-5)])

            while (len(image_series[img]) < series_length):
                image_series[img].insert(int(add_base_flow), fire_to_images[fire][idx])
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

def crop_image(img, resize_dimensions=(1536,2016), crop_height=1120, tile_dimensions=(224,224), jitter_amount=0):
    """
    Description: Resizes, crops, and randomly shifts image
    Args:
        - img (np array): raw image loaded from npy file
        - resize_dimensions (int, int): dimensions to resize raw image
        - crop_height (int): height after cropping image
        - tile_dimensions (int, int): dimensions of tile
        - jitter_amount (int): amount to jitter height
    Returns:
        - img (np array): after crop, resize, and jittering
    """
    # Resize image. Crop a little extra to account for jitter
    img = cv2.resize(img, (resize_dimensions[1], resize_dimensions[0]))[-(crop_height+tile_dimensions[0]):]

    # Crop image further with jitter
    jitter_amount = tile_dimensions[0] - jitter_amount # Normalize jitter such that 0 is the bottom of the image
    img = img[jitter_amount:crop_height+jitter_amount]
    
    return img

def tile_image(img, num_tiles_height, num_tiles_width, resize_dimensions, tile_dimensions, tile_overlap):
    """Description: Tiles image with overlap"""
    # Source: https://towardsdatascience.com/efficiently-splitting-an-image-into-tiles-in-python-using-numpy-d1bf0dd7b6f7
    # WARNING: Tile size must divide perfectly into image height and width
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

    # img.shape = [num_tiles, tile_height, tile_width, 3]
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
    """Description: Tiles labels with overlap"""
    
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



#########################
## Labels & Predictions
#########################

def get_ground_truth_label(image_name):
    """
    Description: Returns 1 if image_name has a + in it (ie. is a positive) or 0 otherwise
    """
    ground_truth_label = 1 if "+" in image_name else 0
    return ground_truth_label

def get_has_xml_label(image_name, labels_path):
    """
    Description: Returns 1 if image_name has an XML file associated with it or 0 otherwise
    """
    has_xml_label = os.path.isfile(labels_path+'/'+image_name+'.npy')
    return has_xml_label

def get_has_positive_tile(tile_labels):
    """
    Description: Returns 1 if tile_labels has a positive tile
    """
    has_positive_tile = 1 if tile_labels.sum() > 0 else 0
    return has_positive_tile


#####################
## LightningModule
#####################

def calculate_negative_accuracy(negative_preds):
    """
    Description: Calculates accuracy on negative images
    Args:
        - negative_preds (tensor): predictions on negatives. Shape = [num_fires, 40]
    Returns:
        - negative_accuracy (tensor): Shape = [1]
    """
    negative_accuracy = torchmetrics.functional.accuracy(negative_preds, torch.zeros(negative_preds.shape).int())
    
    return negative_accuracy

def calculate_negative_accuracy_by_fire(negative_preds):
    """
    Description: Calculates % of fires that DID NOT have a false positive
    Args:
        - negative_preds (tensor): predictions on negatives. Shape = [num_fires, 40]
    Returns:
        - negative_accuracy_by_fire (tensor): Shape = [1]
    """
    negative_accuracy_by_fire = torchmetrics.functional.accuracy(negative_preds, torch.zeros(negative_preds.shape).int(), subset_accuracy=True)
    
    return negative_accuracy_by_fire

def calculate_positive_accuracy(positive_preds):
    """
    Description: Calculates accuracy on positive images
    Args:
        - positive_preds (tensor): predictions on positives. Shape = [num_fires, 41]
    Returns:
        - positive_accuracy (tensor): Shape = [1]
    """
    positive_accuracy = torchmetrics.functional.accuracy(positive_preds, torch.ones(positive_preds.shape).int())
    
    return positive_accuracy

def calculate_positive_accuracy_by_fire(positive_preds):
    """
    Description: Calculates % of fires that DID NOT have a false negative
    Args:
        - positive_preds (tensor): predictions on positives. Shape = [num_fires, 41]
    Returns:
        - positive_accuracy_by_fire (tensor): Shape = [1]
    """
    positive_accuracy_by_fire = torchmetrics.functional.accuracy(positive_preds, torch.ones(positive_preds.shape).int(), subset_accuracy=True)
    
    return positive_accuracy_by_fire
    
def calculate_positive_accuracy_by_time(positive_preds):
    """
    Description: Calculates accuracy per time step
    Args:
        - positive_preds (tensor): predictions on positives. Shape = [num_fires, 41]
    Returns:
        - positive_accuracy_by_time (tensor): Shape = [41]
    """
    positive_accuracy_by_time = positive_preds.sum(dim=0)/positive_preds.shape[0]
    
    return positive_accuracy_by_time

def calculate_positive_cumulative_accuracy(positive_preds):
    """
    Description: Calculates % of fires predicted positive per time step
    Args:
        - positive_preds (tensor): predictions on positives. Shape = [num_fires, 41]
    Returns:
        - positive_cumulative_accuracy (tensor): Shape = [41]
    """
    cumulative_preds = (positive_preds.cumsum(dim=1) > 0).int()
    positive_cumulative_accuracy = cumulative_preds.sum(dim=0)/cumulative_preds.shape[0]
    
    return positive_cumulative_accuracy

def calculate_time_to_detection_stats(positive_preds):
    """
    Description: Calculates average time to detection across all fires
    Args:
        - positive_preds (tensor): predictions on positives. Shape = [num_fires, 41]
    Returns:
        - average_time_to_detection (tensor): Shape = [1]
    """
    cumulative_preds = (positive_preds.cumsum(dim=1) > 0).int()
    indices = cumulative_preds * torch.arange(cumulative_preds.shape[1], 0, -1)
    indices = torch.argmax(indices, 1, keepdim=True).float()
    
    average_time_to_detection = indices.mean()
    median_time_to_detection = indices.median()
    std_time_to_detection = indices.std()
    
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