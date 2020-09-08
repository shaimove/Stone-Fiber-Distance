# TrackerMain.py
'''
    The following script calculates the distance between a tip of a laser fiber
    and sphere stone, simulating kidney stone. 
    The fiber (with green jacket) is detected using the relative value of the green 
    channel to the other blue/red channels.
    The stone is tracked using Kernelized Correlation Filters, the user has to 
    mark the stone position in the image, using ROI selector of OpenCV library. 
    The tracker was developed based on videos taken by a smartphone, and the interaction
    was imaged from above.
    Inputs: 
        A folder that contains all the frames of a video being analyzed. the names need 
        to indicate their order [for example: 'frame 1', 'frame 2', ...]
        Stone Diameter - to calibrate the image
    Output:
        Distances dictionary, with frame name and distance between the tip of the fiber 
        and stone. 
        Folder with analyzed frames, left size the original image with bounding box of 
        the stone, and left size, a binary image of the fiber position and stone position
'''
# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import scipy.signal as signal
from tqdm import tqdm


#%% Inputs 
# folder to read images from 
folder = '6'

# define threshold to detect the fiber, based on RGB channels values:
# for every pixel, if: green / (green + red + blue) > threshold: the pixel is part of the fiber 
threshold = 1.25

# Diameter of the stone for image calubration
diameter = 6 # [mm]



#%% Read images from folder
path = '../Data/' + folder # create full path, change id needed
images = [] # create empty list to store all images
list_of_frames = os.listdir(path) # create list of frames

for filename in tqdm(list_of_frames):
    img = cv2.imread(os.path.join(path , filename))
    images.append(img) 


#%% Initialize Tracker
# define tracker
tracker = cv2.TrackerKCF_create()

# Define ROI
bounding_boxes = [] # create empty list to hold bounding boxes
first_img = images[0] # take first image to initialize the tracker
bbox = cv2.selectROI(first_img, False) # selectROI of the stone

# Initialize tracker
ok = tracker.init(first_img, bbox)
bounding_boxes.append(bbox)

# Image calibration
calibration =  diameter / ((bbox[2] + bbox[3])/2) # [mm / pixel]

# Create folder to save all analyzed frames
loc_to_save = path + '_result'
os.mkdir(loc_to_save)

# create kernel for closing operation in fiber detection
kernel = np.ones((7,7),np.uint8)

# create empty dictionary of distances
Distances = {}

# function to calculate minimum distance
def minDistance(contour, contourOther):
    distanceMin = 99999999
    for i in range(contour.shape[0]):
        xA, yA = contour[i,0,:]
        for j in range(contourOther.shape[0]):
            xB, yB = contourOther[j,0,:]
            distance = ((xB-xA)**2+(yB-yA)**2)**(1/2) # distance formula
            if (distance < distanceMin):
                distanceMin = distance
                points = (xA,yA,xB,yB)
                
    return distanceMin,points

#%% Main loop for analysis

for i,img in tqdm(enumerate(images[1:])):
    #%% Update Tracker and stone position
    ok, bbox = tracker.update(img)
    bounding_boxes.append(bbox)
    img_stone = img.copy()
    
    # ok is a bollean variable indicate if the tracker succesfully detected the object
    if ok: 
        # Tracking success:
        # from: [top_right_x, top_right_y, bottom_left_x, bottom_left_y]
        # to [center_x,center_y] 
        p_center = (int(bbox[0] + bbox[2]/2) , int(bbox[1] + bbox[3]/2))
        p_radius = int((bbox[2] + bbox[3])/4) # calculate radius
        
        # Draw circle inside the rectangle bounding box to mark the stone
        cv2.circle(img_stone,p_center,p_radius,(255,0,0),2)
        
    else:
        # Tracking failure: print error message
        cv2.putText(img_stone, "Tracking failure detected", (100,80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)

    #%% Fiber detection
    # Step 1: Pre processing: Gaussian blur (LPF) to smooth the image
    img_blur = cv2.GaussianBlur(img, (5,5), sigmaX = 3)
    
    # Step 2: calculate the relative value of the green channel
    img_g = img_blur[:,:,1] # take the green channel 
    img_g_rgb = (img_g / np.mean(img_blur,axis=2)) # calculate ratio
    img_thresh = np.uint8(img_g_rgb > threshold) # threshold the image
    
    # Step 3: Post proccessing using morphological operation
    img_close = cv2.morphologyEx(img_thresh, cv2.MORPH_CLOSE, kernel) * 255 # close operation
    
    # Step 4: create BGR image to display the fiber location in green
    img_close_rgb = np.repeat(np.expand_dims(img_close, axis=2),3,axis=2) # extend to BGR
    img_close_rgb[:,:,0] = 0; img_close_rgb[:,:,2] = 0; # zero Blue and Red channels 
    cv2.circle(img_close_rgb,p_center,p_radius,(255,0,0),2) # mark the stone  
    
    # Concatenate images and save them in desired folder
    img_save = np.concatenate((img_close_rgb, img_stone), axis=1) # concatenate
    loc_to_save_img = loc_to_save + '/' + os.listdir(path)[i+1] # create path and image name
    cv2.imwrite(loc_to_save_img,img_save) # save image
    
    #%% Calculate distances using contours detection
    countour_stone = img_close_rgb[:,:,0]
    countour_fiber,contours,hierarchy = cv2.findContours(img_close_rgb[:,:,1],mode=cv2.RETR_EXTERNAL,
                                          method=cv2.CHAIN_APPROX_NONE)
    min_dist,points = minDistance(countour_stone,countour_fiber)
    Distances[list_of_frames[i+1]] = min_dist * calibration # [mm]



#%% close all open windows
plt.close('all')
cv2.destroyAllWindows() 

    
