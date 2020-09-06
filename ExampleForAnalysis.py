# ExampleForAnalysis.py
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import scipy.signal as signal
from tqdm import tqdm

plt.close('all')

#%% Reading images
folder = '6'
path = '../Data/' + folder
images = []
threshold = 1.25

#%% Read files and create two list: blur (fiber) and non-blur (stone)
for filename in tqdm(os.listdir(path)):
    img = cv2.imread(os.path.join(path , filename))
    images.append(img) 

#%% Initialize 
# define tracker
tracker = cv2.TrackerKCF_create()

# Define ROI and initialize tracker
bounding_boxes = []
first_img = images[0]
bbox = cv2.selectROI(first_img, False)
bounding_boxes.append(bbox)
ok = tracker.init(first_img, bbox)

# create folder
loc_to_save = path + '_result'
os.mkdir(loc_to_save)

# kernel for closing operation
kernel = np.ones((7,7),np.uint8)


#%% Analysis

for i,img in tqdm(enumerate(images[1:])):
    # Tracker
    ok, bbox = tracker.update(img)
    bounding_boxes.append(bbox)

    if ok:
        # Tracking success
        # Draw circle
        p_center = (int(bbox[0] + bbox[2]/2) , int(bbox[1] + bbox[3]/2))
        p_radius = int((bbox[2] + bbox[3])/4)
        cv2.circle(img,p_center,p_radius,(255,0,0),2)
    else:
        # Tracking failure
        cv2.putText(img, "Tracking failure detected", (100,80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)

    # Fiber detection
    img_blur = cv2.GaussianBlur(img, (5,5), sigmaX = 3)
    img_g = img_blur[:,:,1]
    img_g_rgb = (img_g / np.mean(img_blur,axis=2)) 
    img_thresh = np.uint8(img_g_rgb > threshold)
    img_close = cv2.morphologyEx(img_thresh, cv2.MORPH_CLOSE, kernel) * 255
    img_close_rgb = np.repeat(np.expand_dims(img_close, axis=2),3,axis=2)
    img_close_rgb[:,:,0] = 0; img_close_rgb[:,:,2] = 0; 
    cv2.circle(img_close_rgb,p_center,p_radius,(255,255,255),2)
    
    # Concate images and save
    img_save = np.concatenate((img_close_rgb, img), axis=1)
    loc_to_save_img = loc_to_save + '/' + os.listdir(path)[i]
    cv2.imwrite(loc_to_save_img,img_save)


cv2.destroyAllWindows() 

    
