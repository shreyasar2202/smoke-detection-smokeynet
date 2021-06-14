"""
Created by: Anshuman Dewangan
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
parser.add_argument('--output-path', type=str, default='./data/batched_tiled_data',
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
parser.add_argument('--add-empty', action='store_false',
                    help='Should images without labels be tiled and stored?')

args = parser.parse_args()


#####################
## Helper Functions
#####################

def xml_to_record(xml_file):
    """
    Description: Takes an XML label file and converts it to Numpy array
    Args:
        - xml_file: Path to XML file
    Returns:
        - all_polys: Numpy array with labels
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


def batch_tile_image(img_path, polygon_mask, image_dimensions=(1536, 2048), tile_dimensions=(224,224), overlap_amount=20, smoke_threshold=10):
    """
    Description: Tiles an image and returns stacked image array + labels 
    Args:
        - img_path: path to raw image
        - polygon_mask: Numpy array of labels (use xml_to_record)
        - image_dimensions: dimensions original image should be resized to
        - tile_dimensions: desired dimensions of tiles
        - overlap_amount: how much the tiles should overlap in pixels
        - smoke_threshold: how many pixels of smoke to label tile as a positive sample? 
    Returns:
        - tiles: Numpy array of tiles: [num_tiles, num_channels, tile_height, tile_width]
        - labels: Numpy array of labels for each tile: [num_tiles]
    """
    # Read image
    original_img = cv2.imread(img_path)
    
    # Create mask_img
    mask_img = np.zeros(original_img.shape[:2], dtype=np.uint8)    
    if polygon_mask is not None:
        cv2.fillPoly(mask_img, polygon_mask, 1)

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
        - metadata.pl
    """
    # Add date & time 
    kwargs['time_created'] = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    
    # Save metadata into pickle file
    with open(f'{output_path}/metadata.pkl', 'wb') as pkl_file:
        pickle.dump(kwargs, pkl_file)

#####################
## Main Function
#####################
    
def save_tiled_images(raw_images_path, labels_path, output_path, image_dimensions=(1536, 2048), tile_dimensions=(224,224), overlap_amount=20, smoke_threshold=10, add_empty=True):
    """
    Description: Loops through all directories and saves tiled images
    Args:
        - raw_images_path: path to raw images
        - labels_path: path to XML labels
        - output_path: desired path of outputted Numpy files
        - image_dimensions: dimensions original image should be resized to
        - tile_dimensions: desired dimensions of tiles
        - overlap_amount: how much the tiles should overlap in pixels
        - smoke_threshold: how many pixels of smoke to label tile as a positive sample? 
    Saves:
        - '{output_location}_img.npy': Numpy array of tiles - [num_tiles, num_channels, tile_height, tile_width]
        - '{output_location}_lbl.npy': Numpy array of labels for each tile - [num_tiles]
    """
    image_path = Path(raw_images_path)
    label_path = Path(labels_path)
    output_path = Path(output_path)
    
    all_folders = [folder.path for folder in os.scandir(label_path) if folder.is_dir()]
    index = 0

    for cur_folder in filter(Path.is_dir, label_path.iterdir()):
        index += 1
        print("Processing folder ", index, "of ", len(all_folders))
        
        cur_images_folder = image_path/cur_folder.stem
        cur_output_folder = output_path/cur_folder.stem
        
        cur_output_folder.mkdir(parents=True, exist_ok=True)
        
        for cur_mask in (cur_folder/"xml").glob('*.xml'):
            cur_polygon_mask = xml_to_record(cur_mask)
            
            if cur_polygon_mask is not None or add_empty:
                img_name = cur_mask.with_suffix(".jpg").name
                img_file = str(cur_images_folder/img_name)
                             
                tiles, labels = batch_tile_image(img_file, cur_polygon_mask, image_dimensions, tile_dimensions, overlap_amount, smoke_threshold)
        
                output_location = str(cur_output_folder/(cur_mask.stem))
                np.save(f'{output_location}_img.npy', tiles)
                np.save(f'{output_location}_lbl.npy', labels)
    
    # Save all parameters in metadata.pkl           
    save_metadata(raw_images_path=raw_images_path, 
         labels_path=labels_path, 
         output_path=output_path, 
         image_dimensions=image_dimensions, 
         tile_dimensions=tile_dimensions, 
         overlap_amount=overlap_amount, 
         smoke_threshold=smoke_threshold,
         add_empty=add_empty)
                
if __name__ == '__main__':
    save_tiled_images(raw_images_path=args.raw_images_path, 
         labels_path=args.labels_path, 
         output_path=args.output_path, 
         image_dimensions=(args.image_height, args.image_width), 
         tile_dimensions=(args.tile_height, args.tile_width), 
         overlap_amount=args.overlap_amount, 
         smoke_threshold=args.smoke_threshold,
         add_empty=args.add_empty)

    
