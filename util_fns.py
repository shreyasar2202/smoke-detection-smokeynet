"""
Created by: Anshuman Dewangan
Date: 2021

Description: Kicks off training and evaluation. Contains many command line arguments for hyperparameters. 
"""
import torch
import torchmetrics

import os
from pathlib import Path


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

def get_image_name(path):
    """
    Description: Gets image name from longer path string
    Args:
        - path (str): Full path to filename or path of {fire}/{image}
    Returns:
        - image_name (str): name of image e.g. 20190716_Meadowfire_hp-n-mobo-c/1563305245_-01080
    """
    # Get last two names of path: fire and image
    image_name = path.split('/')[-2:]
    
    # Join last to names after removing any extensions from image
    image_name = '/'.join([image_name[0], str(Path(image_name[1]).stem)])
    
    # Remove '_lbl' or '_img' if it exists
    # ASSUMPTION: Last 4 chars of image_name may have '_lbl' or '_img'
    if image_name[-4:] == '_lbl' or image_name[-4:] == '_img':
        image_name = image_name[:-4]
    
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

def send_fb_message(message="Done"):
    """
    Source: https://www.nimrod-messenger.io/
    Description: Programmatically sends a Facebook message! Can be used when a job is finished or error has occurred. Make sure to set up $FB_API_KEY as an evnironment variable for it to work.
    Args:
        - message (str): Message to send
    """
    os.system(
        "curl -X POST -H \"Content-Type: application/json\" -d '{\"api_key\": \""+os.environ["FB_API_KEY"]+"\",\"message\":\""+message+"\"}' \"https://www.nimrod-messenger.io/api/v1/message\""
    )


###########################
## BatchedTiledDataModule
###########################

def unpack_fire_images(metadata, fires_list, is_test=False):
    """
    Description: Returns images from a list of fires. If train or val, do not include images that are in 'omit_images_list'.
    Args:
        - metadata (dict): metadata.pkl file contents
        - fires_list (list of str): list of fire names to unpack
    Returns:
        - unpacked_images (list of str): list of images from the fires
    """
    unpacked_images = []
    
    for fire in fires_list:
        for image in metadata['fire_to_images'][fire]:
            # ASSUMPTION: We want to keep all images for test set
            if is_test or image not in metadata['omit_images_list']:
                unpacked_images.append(image)
                
    return unpacked_images

def shorten_time_range(metadata, time_range, train_fires):
    """
    Description: From lists of images per fire, returns list of images within specified time range
    Args:
        - metadata (dict): metadata.pkl file contents
        - time_range (int, int): the time range of images to consider for training by time stamp
        - train_fires (list of str): list of fires in train split
    Returns:
        - metadata['fire_to_images'] (dict): list of images for each fire
    """
    if time_range[0] > -2400 or time_range[1] < 2400:
        for fire in train_fires:
            images_to_keep = []

            for image in metadata['fire_to_images'][fire]:
                image_time_index = image_name_to_time_int(image)
                if image_time_index >= time_range[0] and image_time_index <= time_range[1]:
                    images_to_keep.append(image)

            metadata['fire_to_images'][fire] = images_to_keep
            
    return metadata['fire_to_images']

def generate_series(metadata, series_length):
    """
    Description: Creates a dict with image names as keys and lists of past <series_length> images as values
    Args:
        - metadata (dict): metadata.pkl file contents
        - series_length (int): how many sequential images should be used for training
    Returns:
        - image_series (dict): maps image names to lists of past <series_length> images
    """
    image_series = {}
        
    for fire in metadata['fire_to_images']:
        for i, img in enumerate(metadata['fire_to_images'][fire]):
            image_series[img] = []
            idx = i

            while (len(image_series[img]) < series_length):
                image_series[img].append(metadata['fire_to_images'][fire][idx])
                if idx != 0: idx -= 1
    
    return image_series


#################################
## LightningModel - Predictions
#################################

def predict_tile(outputs):
    """
    Description: Takes raw outputs per tile and returns 0/1 prediction per tile
    Args:
        - outputs (tensor): raw outputs from model. Shape = [batch_size, num_tiles]
    Returns:
        - tile_preds (tensor): 0/1 predictions per tile. Shape = [batch_size, num_tiles]
    """
    tile_preds = torch.sigmoid(outputs)
    tile_preds = (tile_preds > 0.5).int()
    
    return tile_preds

def predict_image_from_tile_preds(tile_preds):
    """
    Description: Takes predictions per tile and returns if any are true
    Args:
        - tile_preds (tensor): 0/1 predictions per tile. Use predict_tile function.
    Returns:
        - image_preds (tensor): 0/1 prediction per image. shape = [batch_size]
    """
    image_preds = (tile_preds.sum(dim=1) > 0).int()
    
    return image_preds


#################################
## LightningModel - Test Metrics
#################################

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

def calculate_average_time_to_detection(positive_preds):
    """
    Description: Calculates average time to detection across all fires
    Args:
        - positive_preds (tensor): predictions on positives. Shape = [num_fires, 41]
    Returns:
        - average_time_to_detection (tensor): Shape = [1]
    """
    cumulative_preds = (positive_preds.cumsum(dim=1) > 0).int()
    indices = cumulative_preds * torch.arange(cumulative_preds.shape[1], 0, -1)
    indices = torch.argmax(indices, 1, keepdim=True)
    average_time_to_detection = indices.float().mean()
    
    return average_time_to_detection