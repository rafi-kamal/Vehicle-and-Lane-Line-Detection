# Vehicle and Lane Line Detection
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)


[image1]: ./output_images/Pipeline1.png "Lane Line Detection Pipeline"
[image2]: ./output_images/Pipeline2.png "Vehicle Detection Pipeline"

This is the term 1 final project from Udacity's Self-Driving Car Nanodegree program. In this project I've developed two pipelines, one for detecting lane lines, and the other for identifying and tracking a vehicle.
Lane Line Detection Pipeline
----------------------------
 
 - Correct the distortion introduced by the camera (`camera_cal` folder contains images used to undistorted the image).
 - Do a perspective transform on the image, so that the image represents the actual (scaled) distance between the lane lines.
 - Use color and gradient filtering to identify the portion of the image containing the lane lines
 - Divide the image into different vertical segments and identifying points on the lane lines on left and right sides of these segments
 - Fit a polynomial on left side points and another on right side points to get two continuous lines
 - Reverse the perspective transform and use these two lines to color the lane
 
![image1]

Vehicle Detection Pipeline
--------------------------

 - Train a classifier to identify car vs non-car images. I've trained my classifier in 64*64 images from [GTI vehicle image database](http://www.gti.ssr.upm.es/data/Vehicle_database.html) and [KITTI vision benchmark suite](http://www.cvlibs.net/datasets/kitti/). The trained model is saved in `model.pkl` file, so you don't need to re-train the model.
 - Use the [HOG](https://en.wikipedia.org/wiki/Histogram_of_oriented_gradients) features of the image to train the model  
 - Use sliding windows of different sizes to take out a rectangular portion of the image, resize it to 64*64 and use the trained model to predict if that portion contains a car or not
 - Merge all the windows containing cars
 
![image2]

Running the Code
----------------

Run `main.py`, it will read the video stream from `project_video.mp4` and write it to `processed_project_video.mp4`. Note that it takes quite a while to process the video stream (25 minutes in my machine). If you want to use another video, make sure the size is 1280*720.

If you want to tweak the code, it's better to start with a single image. Comment the last four lines of `main.py` and uncomment the previous line. Also change the cutoff to zero of `identify_windows_with_car()` method in `vehicle_detection.py`. Running `main.py` afterwords will read a single image from the `test_images` directory and show the output.
