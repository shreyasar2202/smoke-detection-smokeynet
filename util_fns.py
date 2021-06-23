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
## DataModule & Dataloader
###########################

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
        fire_to_images[fire] = [get_image_name(str(item)) for item in (raw_data_path/fire).glob('*.jpg')]
        fire_to_images[fire].sort()
        
    return fire_to_images

def generate_fire_to_images(splits):
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

def generate_omit_images_list(metadata):
    """
    Description: Returns a list of images that are positive but don't have XML labels
    Args:
        - metadata (dict)
    Returns:
        - omit_images_list (list of str): list of images that are positive but don't have XML labels
    """
    omit_images_list = []
    
    for fire in metadata['fire_to_images']:
        for image in metadata['fire_to_images'][fire]:
            if metadata['ground_truth_label'][fire] != metadata['has_xml_label'][fire]:
                omit_images_list.append(image)
                
    return omit_images_list

def unpack_fire_images(fire_to_images, fires_list, omit_images_list, is_test=False):
    """
    Description: Returns images from a list of fires. If train or val, do not include images that are in 'omit_images_list'.
    Args:
        - fire_to_images (dict): maps fire to a list of images for that fire
        - fires_list (list of str): list of fire names to unpack
    Returns:
        - unpacked_images (list of str): list of images from the fires
    """
    unpacked_images = []
    
    for fire in fires_list:
        for image in fire_to_images[fire]:
            # ASSUMPTION: We want to keep all images for test set
            if is_test or image not in omit_images_list:
                unpacked_images.append(image)
                
    return unpacked_images

def shorten_time_range(fire_to_images, time_range, train_fires):
    """
    Description: From lists of images per fire, returns list of images within specified time range
    Args:
        - fire_to_images (dict): maps fire to a list of images for that fire
        - time_range (int, int): the time range of images to consider for training by time stamp
        - train_fires (list of str): list of fires in train split
    Returns:
        - fire_to_images (dict): list of images for each fire
    """
    if time_range[0] > -2400 or time_range[1] < 2400:
        for fire in train_fires:
            images_to_keep = []

            for image in fire_to_images[fire]:
                image_time_index = image_name_to_time_int(image)
                if image_time_index >= time_range[0] and image_time_index <= time_range[1]:
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
        - image_series (dict): maps image names to lists of past <series_length> images
    """
    image_series = {}
        
    for fire in fire_to_images:
        for i, img in enumerate(fire_to_images[fire]):
            image_series[img] = []
            idx = i

            while (len(image_series[img]) < series_length):
                image_series[img].append(fire_to_images[fire][idx])
                if idx != 0: idx -= 1
    
    return image_series

def xml_to_record(xml_file):
    """
    Description: Takes an XML label file and converts it to Numpy array
    Args:
        - xml_file (str): Path to XML file
    Returns:
        - all_polys (Numpy array): Numpy array with labels
    """
    
    objects_dict = {}
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

def save_labels(raw_data_path, 
                labels_path, 
                output_path):
    """
    Description: Converts XML files to Numpy arrays, with 1's for pixels with smoke and 0's for pixels without smoke
    """
    all_fires = [str(folder.stem) for folder in filter(Path.is_dir, Path(labels_path).iterdir())]
    count = 0

    for fire in all_fires:
        print('Folder ', count)
        count += 1

        for image in [get_only_image_name(str(item)) for item in Path(labels_path+'/'+fire+'/xml').glob('*.xml')]:
            image_name = fire+'/'+image

            x = cv2.imread(raw_data_path+'/'+image_name+'.jpg')
            labels = np.zeros(x.shape[:2], dtype=np.uint8) 

            label_path = labels_path+'/'+\
                get_fire_name(image_name)+'/xml/'+\
                get_only_image_name(image_name)+'.xml'

            cv2.fillPoly(labels, xml_to_record(label_path), 1)

            save_path = output_path + '/' + image_name + '.npy'

            os.makedirs(output_path + '/' + fire, exist_ok=True)
            np.save(save_path, labels.astype(np.uint8))

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


#################
## Test Metrics
#################

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