"""
Created by: Anshuman Dewangan
Date: 2021

Description: Kicks off training and evaluation. Contains many command line arguments for hyperparameters. 
"""
import torch

from pathlib import Path


#####################
## Generating Predictions
#####################

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


#####################
## Manipulating Paths
#####################

def get_fire_name(path):
    """
    Description: Gets fire name from longer path string
    Args:
        - path (str): Full path to filename or path of {fire}/{image}
    Returns:
        - fire_name (str): name of fire e.g. "20160718_FIRE_mg-s-iqeye"
    """
    
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
    if image_name[-4:] == '_lbl' or image_name[-4:] == '_img':
        image_name = image_name[:-4]
    
    return image_name

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
            if is_test or image not in metadata['omit_images_list']:
                unpacked_images.append(image)
                
    return unpacked_images

#####################
## Manipulating Fire Image Indices
#####################

def image_name_to_time_index(image_name):
    """
    Description: Given an image name (e.g. 20190716_Meadowfire_hp-n-mobo-c/1563305245_-01080), returns time index as integer (e.g. -1080)
    Args:
        - image_name (str): name of image
    Returns:
        - time_index (int): time index as integer
    """
    
    return int(image_name[-6:])
    
