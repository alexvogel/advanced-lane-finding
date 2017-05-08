## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.
[![IMAGE ALT TEXT](https://img.youtube.com/vi/rZhzEHPJmtQ/0.jpg)](https://www.youtube.com/watch?v=rZhzEHPJmtQ "Video Title")
---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./media_readme/test1.jpg "Input Image"
[image2]: ./media_readme/result_test1.png "Result Image"
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### 2. Code And Data Setup

The code for this project is composed of 3 files:

main script:     bin/lane_line_detection.py
helper module:   lib/helper_lane_lines.py
tracking module: lib/line.py

The images for camera calibration is located in the etc directory

calibration images: etc/camera_cal

The input data is located in the inp directory

test images:     inp/img/test_images
test videos:     inp/vid

### 3. Usage
'''
usage: lane_line_detection.py [-h] [--image [PATH]] [--video [PATH]]
                              [--startTime [INT]] [--endTime [INT]]
                              [--visLog INT] [--format STRING] [--outDir PATH]
                              [--calDir PATH]

a tool for detecting lane lines in images and videos

optional arguments:
  -h, --help         show this help message and exit
  --image [PATH]     image from a front facing camera. to detect lane lines
  --video [PATH]     video from a front facing camera. to detect lane lines
  --startTime [INT]  while developing the image pipeline it can be helpful to
                     focus on the difficult parts of an video, so to start at
                     processing at a certain time. e.g. 25 for 25 seconds
                     after begin.
  --endTime [INT]    to end processing video at a certain time, use this
                     argument. e.g. 30 for end processing 30 seconds after
                     video begin.
  --visLog INT       for debugging or documentation of the pipeline.
                     1=undistorted image 2=grayscale 3=binary mask magnitude
                     sobel xy 4=hls binary mask 5=combination of binary masks
                     6=unwarped binary with polygon 7=warped binary with
                     polygon 8=warped binary 9=histogram 10=detected lines
                     11=undistorted with detected lines 12=result with text
  --format STRING    to visualize single steps of the image pipeline, use this
                     argument. --format=collage4, --format=collage9 creates a
                     collage of images instead of the result image
  --outDir PATH      directory for output data. must not exist at call time.
  --calDir PATH      directory for camera calibration images. directory must
                     only contain chessboard 9x6 calibration images.
'''

example call for processing an image:
python bin/lane_line_detection.py --image inp/img/test_images/test1.jpg
![input image][image1]![result image][image1]

example call for processing an image and output a certain step of the image pipeline instead of the end result:
python bin/lane_line_detection.py --image inp/img/test_images/test1.jpg --visLog 4
![input image][image1]![binary image][image1]

example call for processing a video:
python bin/lane_line_detection.py --image inp/vid/project_video.mp4
[![result video](https://img.youtube.com/vi/rZhzEHPJmtQ/0.jpg)](https://www.youtube.com/watch?v=rZhzEHPJmtQ "Video Title")

example call for processing only the part of a video between 38 and 45 seconds:
python bin/lane_line_detection.py --image inp/vid/project_video.mp4 --startTime 38 --endTime 45

example call for processing a video and output a certain step of the image pipeline instead of the end result:
python bin/lane_line_detection.py --image inp/vid/project_video.mp4 --visLog 4
[![result video](https://img.youtube.com/vi/rZhzEHPJmtQ/0.jpg)](https://www.youtube.com/watch?v=rZhzEHPJmtQ "Video Title")

example call for processing a video and output 4 important steps of the image pipeline instead of the end result:
python bin/lane_line_detection.py --image inp/vid/project_video.mp4 --format collage4
[![result video](https://img.youtube.com/vi/rZhzEHPJmtQ/0.jpg)](https://www.youtube.com/watch?v=rZhzEHPJmtQ "Video Title")

example call for processing a video and output 9 important steps of the image pipeline instead of the end result:
python bin/lane_line_detection.py --image inp/vid/project_video.mp4 --format collage9
[![result video](https://img.youtube.com/vi/rZhzEHPJmtQ/0.jpg)](https://www.youtube.com/watch?v=rZhzEHPJmtQ "Video Title")

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the calibrateCamera function of the file bin/helper_lane_lines.py first code cell of the IPython notebook located in "./examples/example.ipynb" (or in lines # through # of the file called `some_file.py`).  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines # through # in `another_file.py`).  Here's an example of my output for this step.  (note: this is not actually from one of the test images)

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warper()`, which appears in lines 1 through 8 in the file `example.py` (output_images/examples/example.py) (or, for example, in the 3rd code cell of the IPython notebook).  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
src = np.float32(
    [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 460      | 320, 0        | 
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 695, 460      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
