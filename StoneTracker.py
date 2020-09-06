# StoneTracker.py
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from tqdm import tqdm

plt.close('all')

#%% Reading images
folder = '1'
path = '../Data/' + folder
object_to_track = 'fiber'
images = []

for filename in tqdm(os.listdir(path)):
    img = cv2.imread(os.path.join(path , filename))
    images.append(img)                              

#%% Define tracker types
tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'MOSSE', 'CSRT']
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
elif tracker_type == 'MOSSE':
    tracker = cv2.TrackerMOSSE_create()
elif tracker_type == "CSRT":
    tracker = cv2.TrackerCSRT_create()
    
#%% Define ROI and initialize tracker
bounding_boxes = []
first_img = images[0]
bbox = cv2.selectROI(first_img, False)
bounding_boxes.append(bbox)

# Initialize tracker with first frame and bounding box
ok = tracker.init(first_img, bbox)
video_tracker = []

# mark circle in first frame
p_center = (int(bbox[0] + bbox[2]/2) , int(bbox[1] + bbox[3]/2))
p_radius = int((bbox[2] + bbox[3])/4)
cv2.circle(first_img,p_center,p_radius,(255,0,0),2)

# create folder
loc_to_save = path + '_' + tracker_type + '_' + object_to_track
os.mkdir(loc_to_save)

#%% Tracking
for i,img in tqdm(enumerate(images[1:])):
    # Update tracker
    ok, bbox = tracker.update(img)
    bounding_boxes.append(bbox)

    if ok:
        # Tracking success
        # Draw rectangle
        #p1 = (int(bbox[0]), int(bbox[1])) # top left
        #p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])) # bottom right
        #cv2.rectangle(img, p1, p2, (255,0,0), 2, 1)
        
        # Draw circle
        p_center = (int(bbox[0] + bbox[2]/2) , int(bbox[1] + bbox[3]/2))
        p_radius = int((bbox[2] + bbox[3])/4)
        cv2.circle(img,p_center,p_radius,(255,0,0),2)
    else:
        # Tracking failure
        cv2.putText(img, "Tracking failure detected", (100,80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)

    # save frames
    loc_to_save_img = loc_to_save + '/' + os.listdir(path)[i+1]
    cv2.imwrite(loc_to_save_img,img)
    
    
cv2.destroyAllWindows() 

print(loc_to_save)



