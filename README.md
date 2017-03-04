
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

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

[image1]: ./examples/undistort_output.png "Undistorted"
[image2]: ./examples/road_undistorted.jpg "Road Transformed"
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

###Camera Calibration

####1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the first code cell of the 
IPython notebook located in lines #39 through #97 of the file called `calibrate.py`).  

I start by preparing "object points", which will be the (x, y, z) coordinates of 
the chessboard corners in the world. Here I am assuming the chessboard is fixed 
on the (x, y) plane at z=0, such that the object points are the same for each 
calibration image.  Thus, `objp` is just a replicated array of coordinates, 
and `objpoints` will be appended with a copy of it every time I successfully 
detect all chessboard corners in a test image.  `imgpoints` will be appended 
with the (x, y) pixel position of each of the corners in the image plane with 
each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function. 
 I applied this distortion correction to the test image using 
 the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

###Pipeline (single images)

####1. Provide an example of a distortion-corrected image.
To demonstrate this step, I will describe how I apply the distortion correction to 
one of the test images like this one:
I used the OpenCV undistort() function to remove distortion. The function accepts 
a camera matrix and distortion coefficient retrieved from the camera calibration 
step. In this step we use OpenCV camera calibration function which accepts image points 
and object points. The image points are retrieved from found chessboard corners. 
The api call looks like this 

`ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)`.
 
 Followed by the undistort call below:
 
 `undistorted_image = cv2.undistort(img, camera_matrix, dist_coeffs, None, camera_matrix)`
 
 
![alt text][image2]
####2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.
I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines #214 through #245 in `calibrate.py`.  
I converted the image to BGR to  HLS color channel and retained the S channel and pixels within a threshold of (170, 255).
I additionally performed absolute sobel thresholding in x direction, and finally combined both 
masks.

Here's an example of my output for this step.  

![alt text][image3]

####3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warper()`, which appears 
in lines 102 through 135 in the 
file `calibrate.py` (output_images/examples/example.py).  The `warper()` 
function takes as inputs (`img, nx, ny, mtx, dist`) i.e. image, number 
 of points in x and y axis, camera matrix and distortion coefficient. 
As well as computed source (`src`) and destination (`dst`) points.  
I chose the source and destination points in the following manner:

```
src = np.float32([
    [img.shape[1] * 0.44, img.shape[0] * 0.65],
    [img.shape[1] * 0.56, img.shape[0] * 0.65],
    [img.shape[1] * 0.175, img.shape[0] * 0.95],
    [img.shape[1] * 0.825, img.shape[0] * 0.95],
])

dst = np.float32([
    [img.shape[1] * 0.2, img.shape[0] * 0.025],
    [img.shape[1] * 0.8, img.shape[0] * 0.025],
    [img.shape[1] * 0.2, img.shape[0] * 0.975],
    [img.shape[1] * 0.8, img.shape[0] * 0.975],
])

```
This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 563, 468      | 256, 18        | 
| 716, 468      | 1024, 18      |
| 224, 684      | 256, 702      |
| 1056, 684     | 1024, 702        |

I verified that my perspective transform was working as expected by drawing 
the `src` and `dst` points onto a test image and its warped counterpart to 
verify that the lines appear parallel in the warped image.

![alt text][image4]

####4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

I had to read in an image which I undistorted using the already calculated camera matrix and distortion coefficient saved in a pickle file `calibraton_pickle.p`. 
I then performed a perspective transform that gives a "bird's eye view" of the road and returns a warped 
image. With the warped image I performed thresholding steps which involved color masks such as converting 
the image to HLS color channels and using the S and L channels. I further extracted pixels with yelow and white thresholds 
 of :
 ```
    yellow_hsv_low = np.array([0, 100, 100])
    yellow_hsv_high = np.array([50, 255, 255])

    white_hsv_low = np.array([20, 0, 180])
    white_hsv_high = np.array([255, 80, 255])
```
I also performed sobel thresholded masks in x and y direction, and finally combined 
both masks.  
I detected lane lines in the first frame by detecting peaks in a histogram. 
The image height was spit into two equal parts and the bottom half was used. scipy api 
was used to detect the left and right peaks. A window of 50 pixels is then 
centered at the pic. The process is repeated as we move up the number of windows 
needed. In this case 8 times. 

I then fit my lane lines with a 
2nd order polynomial kinda like this:

![alt text][image5]

####5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

To calculate the car position, I assumed the car is at center of the image x-axis. Applied unit space convertion 
of 3.7 meters per 700 pixel in x direction. Calculated the average of the left and right x-intercepts 
`(leftmost + rightmost) / 2.0`. I then compared this average as the lane center to the car position. 
 A negative or positive value represents an offset to the left or right of the lane center

To calculate the radius of curvature, I defined the y-value where I want radius of curvature. 
I choose the maximum y-value, 
 corresponding to the bottom of the image. I then defined conversions 
 in x and y from 2D pixels space to 3D real word space in meters.
I did this in lines #204 through #219 in my code in `lane_detection.py`
Here's the code snippet used to calculate the radius of curvature
```buildoutcfg
def radius_of_curvature(yvals, left_fitx, right_fitx):
 
    y_eval = np.max(yvals)
    ym_per_pix = 30 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

    l_fit_cr = np.polyfit(yvals * ym_per_pix, left_fitx * xm_per_pix, 2)
    r_fit_cr = np.polyfit(yvals * ym_per_pix, right_fitx * xm_per_pix, 2)

    left_curve_rad = ((1 + (2 * l_fit_cr[0] * y_eval + l_fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * l_fit_cr[0])
    right_curve_rad = ((1 + (2 * r_fit_cr[0] * y_eval + r_fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * r_fit_cr[0])

    return left_curve_rad, right_curve_rad
```

####6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines #246 through #461 in my code in `lane_detection.py` in the function `pipeline_process()`.  Here is an example of my result on a test image:

![alt text][image6]

---

###Pipeline (video)

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my project_video.mp4 result](https://youtu.be/WPYp2BkG4cY)  
Here's a [link to my challenge_video.mp4 result](https://youtu.be/LZeSm4zRpZ0)

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

My project implementation and approaches did a pretty good job at detecting 
 lane lines. The project has a lot of room for improvement. One obvious challenge 
  is finding a pipeline that works for every road condition. It will likely fail 
 in more tricky conditions where there are plenty of noise in the image frames. For 
 e.g. on a rainy or snowy day. The harder challenge video exposed some key 
 weaknesses such as extreme intensity variation e.g. sunlight etc. and very sharp curves or bends. 
 This means the project will require a more robust tweaking to averaging 
 or using lanes detected from previous frames. 
 
 A way to improve on the lane detection will be to combine thresholded binary images 
 in multiple color channels, applying more robust outlier rejection techniques etc. 
 tested on a variation of road conditions. 
 
   
