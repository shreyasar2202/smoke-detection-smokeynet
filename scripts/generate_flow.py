"""
Created by: Anshuman Dewangan
Date: 2021

Description: Generates and saves MOG background removal and optical flow between two frames of the image.
"""

import pickle
import numpy as np
import cv2
import os


# TODO: Make sure these have correct paths
metadata = pickle.load(open('./data/metadata.pkl', 'rb'))
raw_data_path = '/userdata/kerasData/data/new_data/raw_images/'
save_path_mog = '/userdata/kerasData/data/new_data/raw_images_mog_new/'
save_path_flow = '/userdata/kerasData/data/new_data/raw_images_flow_new/'


# Loop through all the fires
for j, fire in enumerate(metadata['fire_to_images']):
    # Create directory for fire
    os.makedirs(save_path_mog + fire, exist_ok=True)
    os.makedirs(save_path_flow + fire, exist_ok=True)
    print(j, fire)
    
    # Loop through all images in fire
    for i, cur_image_name in enumerate(metadata['fire_to_images'][fire]):
        # If on the first image, just save image as current image
        if i == 0:
            cur_image_rgb = cv2.imread(raw_data_path+'/'+cur_image_name+'.jpg')
            continue

        # Replace previous image with current image
        prev_image_name = metadata['fire_to_images'][fire][i-1]
        prev_image_rgb = cur_image_rgb

        # Replace current image with newly loaded image
        cur_image_rgb = cv2.imread(raw_data_path+'/'+cur_image_name+'.jpg')

        ### MOG ###
        # Create new MOG background removal object
        mog = cv2.createBackgroundSubtractorMOG2()
        
        # Apply the previous and current image
        output = mog.apply(prev_image_rgb)
        output = mog.apply(cur_image_rgb)
        
        # Save the results
        np.save(save_path_mog + cur_image_name + '.npy', output)   
        
        ### Optical Flow ###
        cur_image = cv2.cvtColor(cur_image_rgb,cv2.COLOR_BGR2GRAY)
        prev_image = cv2.cvtColor(prev_image_rgb,cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(prev_image, cur_image, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        hsv = np.zeros_like(cur_image_rgb)
        hsv[...,1] = 255

        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        hsv[...,0] = ang*180/np.pi/2
        hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
        rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2RGB)
        
        np.save(save_path_flow + cur_image_name + '.npy', rgb)