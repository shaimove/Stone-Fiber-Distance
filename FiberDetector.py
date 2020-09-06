# FiberDetector.py
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import scipy.signal as signal
from tqdm import tqdm

plt.close('all')

#%% Reading images
folder = '6'
threshold = 1.2
flag_to_save = True



#%% Read files
path = '../Data/' + folder
images = []

for filename in tqdm(os.listdir(path)):
    img = plt.imread(os.path.join(path , filename))
    img = cv2.GaussianBlur(img, (5,5), sigmaX = 1)
    images.append(img)                              

#%% find fiber at any images
kernel = np.ones((7,7),np.uint8)
img_fiber = []
path_to_save = path + '_fiber'

if flag_to_save:
    os.mkdir(path_to_save)


for i,img in tqdm(enumerate(images)):
    img_g = img[:,:,1]
    img_g_rgb = (img_g / np.mean(img,axis=2)) 
    img_thresh = np.uint8(img_g_rgb > threshold)
    img_close = cv2.morphologyEx(img_thresh, cv2.MORPH_CLOSE, kernel) * 255
    img_close2 = np.repeat(np.expand_dims(img_close, axis=2),3,axis=2)
    img_fiber.append(img_close)
    loc_to_save = path + '_fiber/' + os.listdir(path)[i]
    img_save = np.concatenate((img_close2, img), axis=1)
    
    if flag_to_save:
        cv2.imwrite(loc_to_save,img_save)

#%% Plot stages in tracking fiber
plt.figure()
plt.subplot(2, 3, 1); plt.imshow(img)
plt.subplot(2, 3, 2); plt.imshow(img_g, cmap='gray')
plt.subplot(2, 3, 3); plt.imshow(img_g_rgb, cmap='gray')
plt.subplot(2, 3, 4); plt.imshow(img_thresh, cmap='gray')
plt.subplot(2, 3, 5); plt.imshow(img_thresh, cmap='gray')







