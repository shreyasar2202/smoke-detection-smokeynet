"""
Created by: Anshuman Dewangan, Yash Pande, Atman Patel
Date: 2021

Description: Converts raw wildfire images and XML label files into tiled & batched Numpy arrays
"""

import cv2
import os
import numpy as np
from pathlib import Path
import xml.etree.ElementTree as ET
import pickle
from argparse import ArgumentParser
from datetime import datetime


#####################
## Argument Parser
#####################
parser = ArgumentParser(description='Takes raw wildfire images and saves tiled images')

# Paths to relevant directories
parser.add_argument('--raw-images-path', type=str, default='/userdata/kerasData/data/new_data/raw_images',
                    help='Path to raw wildfire images')
parser.add_argument('--labels-path', type=str, default='/userdata/kerasData/data/new_data/drive_clone',
                    help='Path to XML labels for raw images')
parser.add_argument('--output-path', type=str, default='/userdata/kerasData/data/new_data/batched_tiled_data',
                    help='Path to save tiled images')

# Desired input image dimensions
parser.add_argument('--image-height', type=int, default=1536,
                    help='Desired height to resize inputted images')
parser.add_argument('--image-width', type=int, default=2048,
                    help='Desired width to resize inputted images')

# Desired tiled image dimensions
parser.add_argument('--tile-height', type=int, default=224,
                    help='Desired height of outputted tiles')
parser.add_argument('--tile-width', type=int, default=224,
                    help='Desired width of outputted tiles')

# Other hyperparameters
parser.add_argument('--smoke-threshold', type=int, default=10,
                    help='If # of smoke pixels in tile is > than threshold, tile will be labelled as 1')
parser.add_argument('--overlap-amount', type=int, default=20,
                    help='# of pixels to overlap the tiles')


#####################
## Helper Functions
#####################

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


def batch_tile_image(img_path, 
                     label_path=None, 
                     image_dimensions=(1536, 2048), 
                     tile_dimensions=(224,224), 
                     overlap_amount=20, 
                     smoke_threshold=10):
    """
    Description: Tiles an image and returns stacked image array + labels 
    Args:
        - img_path (str): path to raw image
        - label_path (str): path to XML label file
        - image_dimensions (int, int): dimensions original image should be resized to
        - tile_dimensions (int, int): desired dimensions of tiles
        - overlap_amount (int): how much the tiles should overlap in pixels
        - smoke_threshold (int): how many pixels of smoke to label tile as a positive sample? 
    Returns:
        - tiles (Numpy array): Numpy array of tiles: [num_tiles, num_channels, tile_height, tile_width]
        - labels (Numpy array): Numpy array of labels for each tile: [num_tiles]
    """
    # Read image
    original_img = cv2.imread(img_path)
    
    # Create mask_img
    mask_img = np.zeros(original_img.shape[:2], dtype=np.uint8)    
    if label_path is not None:
        cv2.fillPoly(mask_img, xml_to_record(label_path), 1)

    # Resize image
    if image_dimensions is not None:
        original_img = cv2.resize(original_img, image_dimensions)
        mask_img = cv2.resize(mask_img, image_dimensions)   
    
    # Store necessary variables
    tile_height, tile_width = tile_dimensions
    height_stride = tile_height - overlap_amount
    width_stride = tile_width - overlap_amount
    img_height, img_width = mask_img.shape
    
    tiles = []
    labels = []
    
    # Loop through coordinates of tiles
    # Make sure to add padding to the range limit so that you cover the entire image
    for x, x_0 in enumerate(range(0, img_width + width_stride -1, width_stride)):
        for y, y_0 in enumerate(range(0, img_height + height_stride -1, height_stride)):
            x_1 = x_0 + tile_width
            if x_1 > img_width:
                x_1 = img_width
                x_0 = x_1 - tile_width
            y_1 = y_0 + tile_height
            if y_1 > img_height:
                y_1 = img_height
                y_0 = y_1 - tile_height
            
            # Add tile to list
            tiles.append(np.transpose(original_img[y_0:y_1, x_0:x_1], (2,0,1)))
            
            # Add label if number of smoke pixels is > threshold
            labels.append(np.sum(mask_img[y_0:y_1, x_0:x_1]) > smoke_threshold)
    
    return np.array(tiles), np.array(labels)

def save_metadata(output_path, **kwargs):
    """
    Description: Creates a pkl file with metadata used to create batched & tiled dataset
    Args:
        - output_path: desired path of metadata.pkl (suggested to use same directory as tiled images)
        - **kwargs: any other args to save in metadata
    Saves:
        - metadata.pkl: dictionary of all associated metadata, including:
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
            - omit_images_list (list): list of images that erroneously do not have XML files for labels
    """
    # Add date & time 
    kwargs['time_created'] = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    
    # Save metadata into pickle file
    with open(f'{output_path}/metadata.pkl', 'wb') as pkl_file:
        pickle.dump(kwargs, pkl_file)

        
#####################
## Main Function
#####################
    
def save_batched_tiled_images(
    raw_images_path, 
    labels_path, 
    output_path, 
    image_dimensions=(1536, 2048), 
    tile_dimensions=(224,224), 
    overlap_amount=20, 
    smoke_threshold=10):
    """
    Description: Loops through all directories and saves tiled images
    Args:
        - raw_images_path (str): path to raw images
        - labels_path (str): path to XML labels
        - output_path (str): desired path of outputted Numpy files
        - image_dimensions (int, int): dimensions original image should be resized to
        - tile_dimensions (int, int): desired dimensions of tiles
        - overlap_amount (int): how much the tiles should overlap in pixels
        - smoke_threshold (int): how many pixels of smoke to label tile as a positive sample? 
    Saves:
        - '{output_location}_img.npy': Numpy array of tiles - [num_tiles, num_channels, tile_height, tile_width]
        - '{output_location}_lbl.npy': Numpy array of labels for each tile - [num_tiles]
    """
    image_path = Path(raw_images_path)
    label_path = Path(labels_path)
    output_path = Path(output_path)
    
    # Data structures for metadata
    fire_to_images = {}
    num_fires = 0
    num_images = 0
    ground_truth_label = {}
    has_xml_label = {}
    has_positive_tile = {}
    omit_images_list = []
    
    # Only consider folders for which there are labels
    all_folders = [folder.stem for folder in filter(Path.is_dir, label_path.iterdir())]

    for i, cur_folder in enumerate(all_folders):
        print("Processing folder ", i+1, "of ", len(all_folders))
        fire_to_images[cur_folder] = []
        num_fires += 1
        
        # Save latest path names
        cur_images_folder = image_path/cur_folder
        cur_labels_folder = label_path/cur_folder
        cur_output_folder = output_path/cur_folder
        
        # Create output directory 
        cur_output_folder.mkdir(parents=True, exist_ok=True)
        
        # Loop through all images in folder
        for cur_img in (cur_images_folder).glob('*.jpg'):
            num_images += 1
            image_name = cur_img.stem
            fire_image_name = cur_folder+"/"+image_name
            
            # Get path of XML file (if it exists)
            label_file = cur_labels_folder/"xml"/cur_img.with_suffix(".xml").name
            has_xml_label[fire_image_name] = 1
            
            # If XML file doesn't exist (ie. negative example), set label_file=None
            if not label_file.exists():
                label_file = None
                has_xml_label[fire_image_name] = 0
                
            # Tile image 
            tiles, labels = batch_tile_image(str(cur_img), label_file, image_dimensions, tile_dimensions, overlap_amount, smoke_threshold)
        
            # Save tiled image and associated labels
            output_location = str(cur_output_folder/image_name)
            np.save(f'{output_location}_img.npy', tiles)
            np.save(f'{output_location}_lbl.npy', labels)
            
            # Save metadata
            fire_to_images[cur_folder].append(cur_folder+"/"+image_name)
            ground_truth_label[fire_image_name] = 1 if "+" in image_name else 0
            has_positive_tile[fire_image_name] = 1 if labels.sum() > 0 else 0
            if ground_truth_label[fire_image_name] != has_xml_label[fire_image_name]:
                omit_images_list.append(fire_image_name)
        
        fire_to_images[cur_folder].sort()
    
        # Save all parameters in metadata.pkl every folder (just in case)       
        save_metadata(raw_images_path=raw_images_path, 
             labels_path=labels_path, 
             output_path=output_path, 
             image_dimensions=image_dimensions, 
             tile_dimensions=tile_dimensions, 
             overlap_amount=overlap_amount, 
             smoke_threshold=smoke_threshold,
             fire_to_images=fire_to_images,
             num_fires=num_fires,
             num_images=num_images,
             ground_truth_label=ground_truth_label,
             has_xml_label=has_xml_label,
             has_positive_tile=has_positive_tile,
             omit_images_list=omit_images_list)
                
if __name__ == '__main__':
    args = parser.parse_args()
    
    save_batched_tiled_images(
         raw_images_path=args.raw_images_path, 
         labels_path=args.labels_path, 
         output_path=args.output_path, 
         image_dimensions=(args.image_height, args.image_width), 
         tile_dimensions=(args.tile_height, args.tile_width), 
         overlap_amount=args.overlap_amount, 
         smoke_threshold=args.smoke_threshold)

    
