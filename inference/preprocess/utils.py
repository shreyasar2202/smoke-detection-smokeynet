import numpy as np
import cv2

def normalize_image(img):
    """Description: Rescales and normalizes an image"""
    # Rescale to [0,1]
    img = img / 255
    # Normalize to 0.5 mean & std
    img = (img - 0.5) / 0.5
    
    return img

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

def calculate_num_tiles(resize_dimensions, crop_height, tile_dimensions, tile_overlap):
    """Description: Give image size, calculates the number of tiles along the height and width"""
    num_tiles_height = 1 + (crop_height - tile_dimensions[0]) // (tile_dimensions[0] - tile_overlap)
    num_tiles_width = 1 + (resize_dimensions[1] - tile_dimensions[1]) // (tile_dimensions[1] - tile_overlap)
    
    return num_tiles_height, num_tiles_width

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