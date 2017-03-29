##**Advanced Lane Finding Project**

The goals / steps of this project are the following:

Self-driving cards need to be told the correct steering angle to turn, left or right. The steering angle is calcualted by knowing speed and dynamics of the card and how much the lane is curving. The lane curvature is determined through the following steps


* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./images/undistort_output.png "Undistorted Chess Image"
[Image2]: ./images/distort.png "Undistorted Road Image"
[image3]: ./images/pipeline.png "Gradient image"
[image4]: ./images/bird_eye_view.png "Bird eye view"
[image5]: ./images/threshold_bird_view.png "Thresholded Bird eye view"
[image6]: ./images/histogram.png "Fit Visual"
[image7]: ./images/windows.png "windows"
[image8]: ./images/curvature.png "curvature"
[image9]: ./images/rcurve1.png "rcurve1"
[image10]: ./images/rcurve2.png "rcurve2"
[image11]: ./images/output.png "output"
[video1]: ./project_video_annotated.mp4 "Video"


###Camera Calibration

Camera's don't create perfect images. Some of the objects in the images, especially ones near the edges, can get stretched or skewed in various ways and we need to correct for that.

Image distortion occurs when a camera looks at 3D objects in the real world and transforms them into a 2D image; this transformation isn't perfect. Distortion actually changes what the shape and size of these 3D objects appear to be. 

Real cameras use curved lenses to form an image, and light rays often bend a little too much or tool little at the edges of these lenses. This creates an effect that distorts the edges of images, so that lines or objects appear more or less curved than they actually are. This is called `radial distortion`, and it's the most common type of distortion.

The first step in analyzing camera images, is to undo the distortion so that we can get correct and useful information out of them.


I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `pattern` is just a replicated array of coordinates, and `object_points` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `image_points` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection. 

I then used the output object_points and image_points to compute the camera calibration and distortion coefficients using the cv2.calibrateCamera() function.
  
```python
    def calibrate(self, image_files, pattern_size):
        # 3D points in real world space
        object_points = []
        # 2D points in image plane
        image_points = []

        # Prepare object points, (x, y, z) like (0,0,0), (1,0,0) ... (7,5,0)
        # The z co-ordinate is 0 because image plane is 2D and flat object.
        pattern = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
        # x, y coordinates
        pattern[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
        gray_image = None
        for i, file in enumerate(image_files):
            image = matplotlib.image.imread(file)
            gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray_image, pattern_size, None)
            # If corners are found, add objects points, image points
            if ret:
                image_points.append(corners)
                object_points.append(pattern)
                # Draw and display the corners
                image = cv2.drawChessboardCorners(image, pattern_size, corners, ret)
                plt.figure(i+1)
                plt.imshow(image)
                plt.title(file)
            else:
                print("findChessBoardCorners for {} failed".format(file))
                self._calibration_success = False

        self._calibration_success, self._camera_matrix, self._dist_coeffs, _, _ = \
            cv2.calibrateCamera(object_points,
                                image_points,
                                gray_image.shape[::-1],
                                None,
                                None)
```

```python
    def undistort(self, image):
        return cv2.undistort(image, self._camera_matrix,
                             self._dist_coeffs,
                             None,
                             self._camera_matrix)

```

I applied the distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 


![alt text][image1]

Distortion correction applied to one of the test image

![alt text][image2]


##Pipeline 

I used a combination of color and gradient thresholds to generate a binary image for detecting the lanes.

* Gradient Absolute Value

```python
def absolute_sobel_threshold(image, kernel=3, orient='x', thresh=(0, 255)):
    # Calculate x and y gradients and take the absolute value
    if orient == 'x':
        absolute = np.absolute(cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=kernel))
    elif orient == 'y':
        absolute = np.absolute(cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=kernel))
    # Rescale back to 8 bit integer
    absolute = np.uint8(255 * absolute / np.max(absolute))
    # Create a binary image of ones where threshold is met, zero otherwise
    binary_output = np.zeros_like(absolute)
    binary_output[(absolute >= thresh[0]) & (absolute <= thresh[1])] = 1
    return binary_output
```    

* Gradient Magnitude
```python
def magnitude_sobel_threshold(image, kernel=3, thresh=(0, 255)):
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=kernel)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=kernel)
    # Calculate the gradient magnitude
    magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)
    # Rescale back to 8 bit integer
    magnitude = np.uint8(255 * magnitude / np.max(magnitude))
    # Create a binary image of ones where threshold is met, zero otherwise
    binary_output = np.zeros_like(magnitude)
    binary_output[(magnitude >= thresh[0]) & (magnitude <= thresh[1])] = 1
    return binary_output
```

* Gradient Direction
```python
def direction_sobel_threshold(image, kernel=3, thresh=(0, np.pi / 2)):
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=kernel)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=kernel)
    # Take the absolute value of the gradient direction,
    direction = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    # Create a binary image of ones where threshold is met, zero otherwise
    binary_output = np.zeros_like(direction)
    binary_output[(direction >= thresh[0]) & (direction <= thresh[1])] = 1
    return binary_output

```

* Color Threshold
```python
    hls = cv2.cvtColor(np.copy(image), cv2.COLOR_RGB2HLS).astype(np.float)
    chan = hls[:, :, 2]
```

* Applying Gradient and Color Threshold
```python
class GradientPipeline(object):
    def __init__(self):
        pass
        self.kernel = 3

        self.direction_threshold = (0.7, 1.3)
        self.magnitude_threshold = (20, 100)
        self.absolute_threshold = (20, 100)
        self.color_threshold = (170, 255)

        self.magnitude_min = 20
        self.magnitude_max = 100

        self.s_channel = 2

    def __call__(self, image, stacked=False):
        hls = cv2.cvtColor(np.copy(image), cv2.COLOR_RGB2HLS).astype(np.float)
        chan = hls[:, :, self.s_channel]
        # apply sobel threshold on saturation channel
        absx = absolute_sobel_threshold(chan, kernel=self.kernel, orient='x', thresh=self.absolute_threshold)
        absy = absolute_sobel_threshold(chan, kernel=self.kernel, orient='y', thresh=self.absolute_threshold)
        magnitude = magnitude_sobel_threshold(chan, kernel=self.kernel, thresh=self.magnitude_threshold)
        direction = direction_sobel_threshold(chan, kernel=self.kernel, thresh=self.direction_threshold)
        gradient = np.zeros_like(chan)
        gradient[((absx == 1) & (absy == 1)) | ((magnitude == 1) & (direction == 1))] = 1
        # apply color threshold mask
        color = color_threshold(chan, thresh=self.color_threshold)
        if stacked:
            return np.dstack((np.zeros_like(chan), gradient, color))
        else:
            binary_output = np.zeros_like(chan)
            binary_output[(gradient == 1) | (color == 1)] = 1
            return binary_output
```

![alt text][image3]

###Prespective Transform

```python
class Warper:

    def __init__(self):
        src = np.float32([
            [580, 460],
            [700, 460],
            [1040, 680],
            [260, 680],
        ])

        dst = np.float32([
            [260, 0],
            [1040, 0],
            [1040, 720],
            [260, 720],
        ])

        self.M = cv2.getPerspectiveTransform(src, dst)
        self.M_inv = cv2.getPerspectiveTransform(dst, src)

    def __call__(self, image, unwarp=False):
        if unwarp:
            return self.unwarp(image)
        else:
            return self.warp(image)

    def warp(self, image):
        return cv2.warpPerspective(
            image,
            self.M,
            (image.shape[1], image.shape[0]),
            flags=cv2.INTER_LINEAR
        )

    def unwarp(self, image):
        return cv2.warpPerspective(
            image,
            self.M_inv,
            (image.shape[1], image.shape[0]),
            flags=cv2.INTER_LINEAR
        )
```
![alt text][image4]

###Detect lane pixels and fit to find the lane boundary.

![alt text][image5]

I now have a thresholded warped image and ready to map out the lane lines! There are many ways you could go about this, but here's one example of how you might do it:

* Line Finding Method: Peaks in a Histogram

    After applying calibration, thresholding, and a perspective transform to a road image, we have a binary image where the lane lines stand out clearly. However, we still need to decide explicitly which pixels are part of the lines and which belong to the left line and which belong to the right line.

    I first take a histogram along all the columns in the lower half of the image like this:

    ```python
    import numpy as np
    import Pipeline
    import ImagePlotter
    
    plotter = ImagePlotter.ImagePlotter(2, grid=(1,2), figsize=(24, 9))
    image = matplotlib.image.imread('test_images/test1.jpg')
    image = calibrate(image)
    plotter(image, 'original image')
    image = Pipeline.pipeline(image)[0]
    histogram = np.sum(image[image.shape[0]/2:,:], axis=0)
    plt.plot(histogram)    
    ```
![alt text][image6]

* Sliding Window

    With the histogram I add up the pixel values along each column in the image. In the thresholded binary image, pixels are either 0 or 1, so the two most prominent peaks in this histogram will be good indicators of the x-position of the base of the lane lines. I can use that as a starting point for where to search for the lines. From that point, I can use a sliding window, placed around the line centers, to find and follow the lines up to the top of the frame. We scan the frame with the windows, collecting non-zero pixels within window bounds. Once we reach the top, we try to fit a second order polynomial into collected points. This polynomial coefficients would represent a single lane boundary.
![alt_text][image7]

###Determine the curvature of the lane and vehicle position with respect to center.

We estimated which pixels belong to the left and right lane lines, and fit a polynomial to those pixel positions using the equation

![alt_text][image8]

Now the radius of curvature at any point `x` of the function `x = f(y)`  is measured using the equation

![alt_text][image9]

In the case of the second order polynomial above, the equation for radius of curvature becomes
 
![alt_text][image10] 

```python
    def measure_curvature(self):
        points = self.points()
        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 30/720  # meters per pixel in y dimension
        xm_per_pix = 3.7/700  # meters per pixel in x dimension

        x = points[:, 0]
        y = points[:, 1]

        fit_cr = np.polyfit(y * ym_per_pix, x * xm_per_pix, 2)
        curve_radius = ((1 + (2 * fit_cr[0] * 720 * ym_per_pix + fit_cr[1]) ** 2 ) ** 1.5) \
                / np.absolute(2 * fit_cr[0])
        return int(curve_radius)
```

The vehicle position with the lanes can be approximated by calculating approximate distance to a curve at the bottom of the frame.

```python
    def vehicle_position(self):
        points = self.points()
        xm_per_pix = 3.7 / 700  # meters per pixel in x dimension
        x = points[np.max(points[:, 1])][0]
        return np.absolute((self._width // 2 - x) * xm_per_pix)
```

### Processed Image

![alt_text][image11]

---

###Pipeline (video)

The pipeline and the algorithm described above is applied to a sequence of frames in the video. A line is drawn by getting getting an average of polynomial coefficients, detected over last 5 frames. We verify enough points detected in the current frame to approximate a line, and then append polynomial coefficients to recentfits collections.
```python
    def fit(self, x, y):
        if len(y) > 0 and \
           (self._current_fit is None or np.max(y) - np.min(y) > self._height * .625):
            self._current_fit = np.polyfit(y, x, 2)
            self._recent_fits.append(self._current_fit)
            self._recent_fitted_x.append(x)
```
```python
    def points(self):
        plot_y = np.linspace(0, self._height-1, self._height)
        best_fit = np.array(self._recent_fits).mean(axis=0)
        best_fit_x = best_fit[0] * plot_y ** 2 + best_fit[1] * plot_y + best_fit[2]
        return np.stack((best_fit_x, plot_y)).astype(int).T
```

I stole embedding processing overlay idea from the Alex Staravoitau blog. This is good way of debugging the video processing at runtime.   

<p align="center">
  <img src="images/project_video.gif" alt="Project Video"/>
</p>
 
Here's a [link to my video result](./project_video_annotated.mp4)



