# Stone-Fiber Distance
-----
The following script calculates the distance between a tip of a laser fiber and sphere stone, simulating kidney stones. 
The fiber (with green jacket) is detected using the relative value of the green channel to the other blue/red channels.
The stone is tracked using Kernelized Correlation Filters, the user has to mark the stone position in the image, using the ROI selector of the OpenCV library. 
The tracker was developed based on videos taken by a smartphone, and the interaction was imaged from above.

The repo includes the main script TrackerMain.py, StoneTracker.py to try different trackers for different datasets. FiberDetector.py is a script for trying different scripts for detecting different kinds of fibers.


## Requirements: OpenCV and standard Python packages as: Numpy, Matplotlib, Scipy

OpenCV:
```sh
conda install -c conda-forge opencv
```

## Algorithm

The algorithm includes the following steps (TrackerMain.py):
 1. Convert a video filmed by smartphone at a rate of 30fps into frames in a different folder (not included, but easy to implement)
 2. Create Tracker object and mark the first image (then the algorithm is looking for similar pixels). the tracker is using rectangles to track the stone but is displayed as a circle, using the center of the rectangle and it's side length to calculate the circle’s radius. 
 3. Stone detection using the Tracker
 4. fiber detection. we blurred the image using a Gaussian filter, and calculate the relative green channel since the fiber is green (G/R+B+G). After that, we threshold the image, and using Close morphological operation, we close the circle. Finally, saving the frame into a different t folder. 
 5. calculate the distance between the fiber and the stone, based on the known size of the stone and L2 distance function


## Results
On the right side, you can see the detection of the stone on the original image, on the left side you can see the binary image with the stone and the fiber detection.

 ![Image 11](https://github.com/shaimove/Stone-Fiber-Distance/blob/master/frame1621.jpg)
  

## License

Private Property of Lumenis LTD. 

## Contact Information
Developed by Sharon Haimov, Research Engineer at Lumenis.

Email: sharon.haimov@lumenis.com or shaimove@gmail.com
