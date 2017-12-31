**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/Pipeline.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in line 37-56 of `vehicle_detection.py`.  

I started by reading in all the `vehicle` and `non-vehicle` images. I tested different color spaces and tuned different parameters (pixel per block, cell per block), and tested their accuracies during model training.


#### 2. Explain how you settled on your final choice of HOG parameters.

I finally settled with grayscale image, as it is simpler and provide a good accuracy. The other parameters I've used are: orientations=9, pixel per cell=8, and cell per block=2.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using extracted hog features from grayscaled images (L59-79 in `vehicle_detection.py`). Instead of manually tuning the C value, I used a grid search with the following values of C: 0.3, 1, 3. I've saved the trained model in a file so that I don't have to train the same model in each run.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The sliding window search is implemented in L85-134. I tried windows with many different sizes in different positions, and finally settled with the following types of windows:

| Size      | Overlap | Search area (x-axis) | Search area (y-axis) |
|:---------:|:-------:|:--------------------:|:--------------------:|
| 225 * 225 | 4       | 0 - 1280             | 300 - 580            |
| 160 * 160 | 4       | 0 - 1280             | 330 - 580            |
| 128 * 128 | 4       | 0 - 1280             | 350 - 540            |
|  80 *  80 | 4       | 200 - 1080           | 400 - 500            |


#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

The following image shows the pipline in a test image.

![alt text][image1]

To optimize the performance of the classifier, I limited the sliding window search to a specific area (shown in the table above). E.g for a small 80*80 sliding window I've conducted the search only in the central portion of the image (because the cars in the bottom part of the image will look bigger, and the upper part is taken by the sky), and  also omitted the sides. 

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./processesed_project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I've done the filtering in L116-137 in `vehicle_detection.py`. I've counted how many consecutive times I've found a car in a window, and filtered all occurences with less than 10 consecutive appearances. 

I've merged the bounding boxes in L140-184 in `vehicle_detection.py`. I've iterated over the bounding boxes where a car has been identified. At each step I compared the current bounding box with all other bounding boxes in the list. If any of those bounding boxes intersects with the current bounding box, then I merged those two boxes and removed the latter box from the list. I continued this process until any more unmerged bounding boxes are left. 

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

False positives were the main problem I was facing, I've to tweak the flitering parameter a lot to ensure there are minimal false positives and the algorithm can still detect the cars. Another problem was finding out the correct sliding window sizes, which involved a lot of trial and errors.
  
The pipeline will likely fail in different weather conditions/other types of cars (e.g. for buses and trucks). Training it with a wide variety of different images will make it more robust.

