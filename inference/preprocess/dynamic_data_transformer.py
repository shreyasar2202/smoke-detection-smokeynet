# reference: https://github.com/shreyasar2202/smoke-detection-smokeynet/blob/master/src/dynamic_dataloader.py

import numpy as np
import utils

class DynamicDataTransformer():
    def __init__(self, 
                 raw_data_path=None,
                 labels_path=None, 
                 optical_flow_path=None,
                 split_name='train',
                 
                 metadata=None,
                 data_split=None, 
                 omit_images_list=None,
                 is_object_detection=False,
                 is_maskrcnn=False,
                 is_background_removal=False,
                 
                 original_dimensions = (1536, 2016),
                 resize_dimensions = (1536, 2016),
                 crop_height = 1120,
                 tile_dimensions = (224,224), 
                 tile_overlap = 0,
                 pre_tile = True,
                 smoke_threshold = 250,
                 num_tile_samples = 0,
                 
                 flip_augment = True,
                 resize_crop_augment = True,
                 blur_augment = True,
                 color_augment = True,
                 brightness_contrast_augment = True):
        
        self.raw_data_path = raw_data_path
        self.labels_path = labels_path
        self.optical_flow_path = optical_flow_path
        self.split_name = split_name
        
        self.metadata = metadata
        self.data_split = data_split
        self.omit_images_list = omit_images_list
        self.is_object_detection = is_object_detection
        self.is_maskrcnn = is_maskrcnn
        self.is_background_removal = is_background_removal
        
        self.original_dimensions = original_dimensions
        self.resize_dimensions = resize_dimensions
        self.crop_height = crop_height
        self.tile_dimensions = tile_dimensions
        self.tile_overlap = tile_overlap
        self.pre_tile = pre_tile
        self.smoke_threshold = smoke_threshold
        self.num_tile_samples = num_tile_samples
        
        self.flip_augment = flip_augment
        self.resize_crop_augment = resize_crop_augment
        self.blur_augment = blur_augment
        self.color_augment = color_augment
        self.brightness_contrast_augment = brightness_contrast_augment
        
        self.num_tiles_height, self.num_tiles_width = (1,1) if self.tile_dimensions is None else utils.calculate_num_tiles(resize_dimensions, crop_height, tile_dimensions, tile_overlap)

    def transform(self, img_ndarray):
                
        data_augmentations = utils.DataAugmentations(self.original_dimensions,
                                                        self.resize_dimensions,
                                                        self.tile_dimensions,
                                                        self.crop_height,
                                                        
                                                        self.flip_augment, 
                                                        self.resize_crop_augment, 
                                                        self.blur_augment, 
                                                        self.color_augment, 
                                                        self.brightness_contrast_augment)
        
        # Apply data augmentations
        # img.shape = [crop_height, resize_dimensions[1], num_channels]
        img_ndarray = data_augmentations(img_ndarray, is_labels=False)
            
        # Tile image
        # img.shape = [num_tiles, tile_height, tile_width, num_channels]
        # print(self.num_tiles_height, self.num_tiles_width, self.resize_dimensions, self.tile_dimensions, self.tile_overlap)
        if self.pre_tile and not self.is_object_detection:
            tiled_imgs_ndarray = utils.tile_image(img_ndarray, self.num_tiles_height, self.num_tiles_width, self.resize_dimensions, self.tile_dimensions, self.tile_overlap)
        
        # Rescale and normalize
        if self.is_object_detection:
            tiled_imgs_ndarray = tiled_imgs_ndarray / 255 # torchvision object detection expects input [0,1]
        else:
            tiled_imgs_ndarray = utils.normalize_image(tiled_imgs_ndarray)
        
        # 4 dimensions: [num_tiles, 224, 224, 3]
        return tiled_imgs_ndarray


