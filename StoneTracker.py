# StoneTracker.py
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import scipy.signal as signal
from tqdm import tqdm
import sys

plt.close('all')

#%% Reading images
folder = '4'
path = '../Data/' + folder
images=[]

for filename in os.listdir(path):
    img = plt.imread(os.path.join(path , filename))
    images.append(img)                              

#%% Define tracker types
tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
tracker_type = tracker_types[2] # default

if tracker_type == 'BOOSTING':
    tracker = cv2.TrackerBoosting_create()
elif tracker_type == 'MIL':
    tracker = cv2.TrackerMIL_create()
elif tracker_type == 'KCF':
    tracker = cv2.TrackerKCF_create()
elif tracker_type == 'TLD':
    tracker = cv2.TrackerTLD_create()
elif tracker_type == 'MEDIANFLOW':
    tracker = cv2.TrackerMedianFlow_create()
elif tracker_type == 'GOTURN':
    tracker = cv2.TrackerGOTURN_create()
elif tracker_type == 'MOSSE':
    tracker = cv2.TrackerMOSSE_create()
elif tracker_type == "CSRT":
    tracker = cv2.TrackerCSRT_create()
    
#%% Define ROI and initialize tracker
bbox = cv2.selectROI(images[0], False)

# Initialize tracker with first frame and bounding box
ok = tracker.init(images[0], bbox)
video_tracker = []
video_tracker.append(cv2.rectangle(images[0], (int(bbox[0]), int(bbox[1]))
                        ,(int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                        ,(255,0,0), 2, 1))


#%% Tracking
for i,img in tqdm(enumerate(images[1:])):
    # Update tracker
    ok, bbox = tracker.update(img)
    
    # Draw bounding box
    if ok:
        # Tracking success
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(img, p1, p2, (255,0,0), 2, 1)
        video_tracker.append(img)
    else:
        # Tracking failure
        cv2.putText(img, "Tracking failure detected", (100,80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
        video_tracker.append(img)

    # save img 
    loc_to_save = path + '_stone/' + os.listdir(path)[i+1]
    cv2.imwrite(loc_to_save,img)











