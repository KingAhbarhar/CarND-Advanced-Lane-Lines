import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import pickle
from moviepy.editor import VideoFileClip

# Choose a Sobel kernel size
ksize = 3  # Choose a larger odd number to smooth gradient measurements


def display_img(img):
    plt.imshow(img, cmap='gray')
    plt.show()


def view_histogram(img):
    histogram = np.sum(img[img.shape[0] / 2:, :], axis=0)
    plt.plot(histogram)
    plt.show()


def find_corners(img, nx=8, ny=6):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

    # If found, draw corners
    if ret:
        # Draw and display the corners
        cv2.drawChessboardCorners(img, (nx, ny), corners, ret)

    return img


def find_corners_and_caliberate_camera():
    # If the pickle is found let's return the camera matrix and distortion coefficient
    # No need to re-run the process
    try:
        calibration = pickle.load(open("calibration_pickle.p", "rb"))
        camera_matrix = calibration["mtx"]
        dist_coeffs = calibration["dist"]
        # print("returning camera calibration mtx and dist from saved pickle")
        return camera_matrix, dist_coeffs
    except (OSError, IOError) as e:
        print("Camera calibration pickle not found. Begin processing...")

    objpoints = []  # 3D points in real world space
    imgpoints = []  # 2D points in image plane

    nx = 9  # number of inside corners in x
    ny = 6  # number of inside corners in y

    # prepare object points like (x,y,z)-> (0,0,0), (1,0,0), (2,0,0) .......(8,5,0)
    # object points will be the same for all calibration images
    # since they represent the real chessboard image
    objp = np.zeros((6 * nx, 3), np.float32)
    objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)  # x,y coordinates

    images = glob.glob("./camera_cal/calibration*.jpg")

    for index, f_name in enumerate(images):
        print("Processing", index, f_name)
        img = mpimg.imread(f_name)

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

        # If found, add image points and object points to be used for calibration.
        if ret:
            imgpoints.append(corners)
            objpoints.append(objp)

            # Draw corners for visualization
            cv2.drawChessboardCorners(img, (nx, ny), corners, ret)

        out_img = img
        # display_img(out_img)
        write_name = "./output_images/corners_found{}.jpg".format(index + 1)
        cv2.imwrite(write_name, out_img)

    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None,
                                                                        None)
    # Save camera calibration and distortion coefficient for later use
    my_pickle = dict()
    my_pickle["mtx"] = camera_matrix
    my_pickle["dist"] = dist_coeffs
    pickle.dump(my_pickle, open("calibration_pickle.p", "wb"))
    print("calibration_pickle saved for later use")

    return camera_matrix, dist_coeffs


# Define a function that takes an image, number of x and y points,
# camera matrix and distortion coefficients
def warper(img, nx, ny, mtx, dist):
    # Use the OpenCV undistort() function to remove distortion
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    # Convert undistorted image to grayscale
    gray = cv2.cvtColor(undist, cv2.COLOR_BGR2GRAY)
    # Search for corners in the grayscaled image
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

    if ret == True:
        # If we found corners, draw them! (just for fun)
        cv2.drawChessboardCorners(undist, (nx, ny), corners, ret)
        # Choose offset from image corners to plot detected corners
        # This should be chosen to present the result at the proper aspect ratio
        # My choice of 100 pixels is not exact, but close enough for our purpose here
        offset = 100  # offset for dst points
        # Grab the image shape
        img_size = (gray.shape[1], gray.shape[0])
        print("img_size", img_size)

        # For source points I'm grabbing the outer four detected corners
        src = np.float32([corners[0], corners[nx - 1], corners[-1], corners[-nx]])
        # For destination points, I'm arbitrarily choosing some points to be
        # a nice fit for displaying our warped result
        # again, not exact, but close enough for our purposes
        dst = np.float32([[offset, offset], [img_size[0] - offset, offset],
                          [img_size[0] - offset, img_size[1] - offset],
                          [offset, img_size[1] - offset]])
        # Given src and dst points, calculate the perspective transform matrix
        M = cv2.getPerspectiveTransform(src, dst)
        # Warp the image using OpenCV warpPerspective()
        warped = cv2.warpPerspective(undist, M, img_size, flags=cv2.INTER_LINEAR)

    # Return the resulting image and matrix
    return warped, M


def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    # Create a copy and apply the threshold
    binary_grad_output = np.zeros_like(scaled_sobel)
    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    # blackouts gradient in range. Note - 0 is pure white and 255 is pure black
    # Hence thresh=(0, 255) will blackout entire image
    binary_grad_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

    # Return the result
    return binary_grad_output


# Define a function that applies Sobel x and y,
# then computes the magnitude of the gradient
# and applies a threshold
def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx ** 2 + sobely ** 2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag) / 255
    gradmag = (gradmag / scale_factor).astype(np.uint8)
    # Create a binary image of ones where threshold is met, zeros otherwise
    mag_binary = np.zeros_like(gradmag)
    mag_binary[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

    # Return the binary image
    return mag_binary


# Define a function that applies Sobel x and y,
# then computes the direction of the gradient
# and applies a threshold.
def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi / 2)):
    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction,
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    dir_binary = np.zeros_like(absgraddir)
    dir_binary[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

    # Return the binary image
    return dir_binary


# Define a function that thresholds the S-channel of HLS
# Use exclusive lower bound (>) and inclusive upper (<=)
def hls_select(img, thresh=(20, 100)):
    # 1) Convert to HLS color space
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    # 2) Apply a threshold to the S channel
    s_channel = hls[:, :, 2]
    # 3) Return a binary image of threshold result
    binary_output = np.zeros_like(s_channel)
    binary_output[(s_channel > thresh[0]) & (s_channel <= thresh[1])] = 1
    return binary_output


def pipeline(img, s_thresh=(170, 255), sx_thresh=(20, 100)):
    img = np.copy(img)
    # Convert to HLS color space and separate the S channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    s_channel = hls[:, :, 2]
    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

    # Grayscale image
    # NOTE: we already saw that standard grayscaling lost color information for the lane lines
    # Explore gradients in other colors spaces / color channels to see what might work better
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Sobel x
    sobelx = cv2.Sobel(s_channel, cv2.CV_64F, 1, 0)  # Take the derivative in x
    abs_sobelx = np.absolute(sobelx)  # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))
    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1

    # Stack each channel
    # Note color_binary[:, :, 0] is all 0s, effectively an all black image. It might
    # be beneficial to replace this channel with something else.
    color_binary = np.dstack((np.zeros_like(sxbinary), sxbinary, s_binary))

    # Combine the two binary thresholds
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1

    return combined_binary


def visualize_images(original, modified, title1="Image 1", title2="Image 2", font_size=30):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(original, cmap='gray')
    ax1.set_title(title1, fontsize=font_size)
    ax2.imshow(modified, cmap='gray')
    ax2.set_title(title2, fontsize=font_size)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()


def visualize_images_3(img1, img2, img3):
    plt.figure(figsize=(10, 2))
    plt.subplot(1, 3, 1)
    plt.imshow(img1, cmap='gray')
    plt.axis('off')
    plt.title('img1')

    plt.subplot(1, 3, 2)
    plt.imshow(img2, cmap='gray')
    plt.title('img2')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(img3, cmap='gray')
    plt.title('img3')
    plt.axis('off')
    plt.show()


def perform_warp(img):
    img_size = (img.shape[1], img.shape[0])

    # Define a four sided polygon to mask
    # Polygon Points - note (x,y)->(0,0) is topmost left
    img_height = img.shape[0]
    imgWidth = img.shape[1]
    # img_height_trim_hood = img_height * 0.9
    # midx = (imgWidth / 2, img_height)
    # trapezoid_height = img_height * 0.40

    # Calculate polygon/trapezoid points based on midx point
    # E.g. leftBottom is at left of midx and rightBottom is at right side

    # offset_x_bottom = 420
    # offset_y_bottom = 18
    # left_bottom = (midx[0] - offset_x_bottom, midx[1])
    # right_bottom = (midx[0] + offset_x_bottom, midx[1])
    # left_top = (midx[0] - offset_y_bottom, midx[1] - trapezoid_height)
    # right_top = (midx[0] + offset_y_bottom, midx[1] - trapezoid_height)

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

    # cv2.fillPoly(img, np.int_([src]), (0, 255, 0))

    # Draw a polygon
    pts = np.array([src[0], src[1], src[3], src[2]], np.int32)
    pts = pts.reshape((-1, 1, 2))
    cv2.polylines(img, [pts], True, (255, 0, 0))

    # Visualize points for debugging only
    # plt.imshow(img)
    # plt.plot(src[0][0], src[0][1], "*")
    # plt.plot(src[1][0], src[1][1], "*")
    # plt.plot(src[2][0], src[2][1], "*")
    # plt.plot(src[3][0], src[3][1], "*")
    # plt.show()
    # exit("exit 777")\

    # Compute the perspective transform, M, given source and destination points:
    M = cv2.getPerspectiveTransform(src, dst)
    # Compute the inverse  perspective transform:
    Minv = cv2.getPerspectiveTransform(dst, src)

    # Warp an image using the perspective transform, M:
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)

    return warped, M, Minv


def process_testimages(mtx, dist):
    test_images = glob.glob("./test_images/test*.jpg")

    for index, f_name in enumerate(test_images):
        print("processing", index, f_name)

        # read in image
        img = cv2.imread(f_name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Undistort the image
        undist = cv2.undistort(img, mtx, dist, None, mtx)

        # perform thresholding via pipeline
        thresholded_image = pipeline(undist) * 255  # multiplication converts 1s to 255 i.e. white

        # perform warping/perspective transform to birds eye view
        warped, M, Minv = perform_warp(thresholded_image)

        warped_image_with_window, ploty, left_fitx, right_fitx = find_lane_lines_histogram_style(warped, False)
        left_curve, right_curve = radius_of_curvature(ploty, left_fitx, right_fitx)
        print("Real3D radius left: {} m, right: {} m".format(int(left_curve), int(right_curve)))

        result = draw_lines_on_road(warped, undist, ploty, left_fitx, right_fitx, Minv)

        # write_name = "./output_images/roadline/roadline{}.jpg".format(index + 1)
        # cv2.imwrite(write_name, result)

        # visualize_images(result, warped_image_with_window)

        return result


def radius_of_curvature(yvals, left_fitx, right_fitx):
    # Define y-value where we want radius of curvature
    # I'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(yvals)

    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

    l_fit_cr = np.polyfit(yvals * ym_per_pix, left_fitx * xm_per_pix, 2)
    r_fit_cr = np.polyfit(yvals * ym_per_pix, right_fitx * xm_per_pix, 2)

    left_curve_rad = ((1 + (2 * l_fit_cr[0] * y_eval + l_fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * l_fit_cr[0])
    right_curve_rad = ((1 + (2 * r_fit_cr[0] * y_eval + r_fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * r_fit_cr[0])

    return left_curve_rad, right_curve_rad


def draw_lines_on_road(warped, undist, ploty, left_fitx, right_fitx, Minv, curvature):
    # function not working at Minv
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Given src and dst points, calculate the perspective transform matrix
    # Minv = cv2.getPerspectiveTransform(warped, undist)

    # Reverse-Warp the blank back to original undistorted image space using inverse perspective matrix (Minv)
    image = np.copy(undist)
    newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0]))

    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)

    # plt.imshow(newwarp)
    # plt.show()
    # exit("yoo")

    font = cv2.FONT_HERSHEY_SIMPLEX
    text = "Radius of Curvature: {} m".format(int(curvature))
    cv2.putText(result, text, (400, 100), font, 1, (255, 255, 255), 2)

    # Get points from newwarp i.e. reverse-warped image in real world space
    pts = np.argwhere(newwarp[:, :, 1])
    # print("points", len(pts))

    position = find_position(pts, image.shape)
    direction = "left" if position < 0 else "right"
    # print("position {}, direction {}".format(position, direction))

    text = "Vehicle is {:.2f} m {} of center".format(abs(position), direction)
    cv2.putText(result, text, (400, 150), font, 1, (255, 255, 255), 2)

    # plt.imshow(result)
    # plt.show()
    # exit("yoo")

    return result


def find_position(pts, image_shape):
    '''
    :param pts: Points where reverse-warped image is not 0
    :param image_shape:
    :return: position of car from center
    Shows car is 'x' meters from the left or right
    '''
    global last_pts

    if len(pts) > 0:
        last_pts = pts
    else:
        pts = last_pts
        exit("Empty points exit point")

    position = image_shape[1] / 2  # Assume car position is at center of the image x-axis
    # left = np.min(pts[(pts[:, 1] < position) & (pts[:, 0] > 700)][:, 1])  # Get leftmost pixel's x position
    # right = np.max(pts[(pts[:, 1] > position) & (pts[:, 0] > 700)][:, 1])  # Get rightmost pixel's x position
    left = np.min(pts[(pts[:, 1] < position) & (pts[:, 0] > 700)][:, 1])  # Get leftmost pixel's x position
    right = np.max(pts[(pts[:, 1] > position)][:, 1])  # Get rightmost pixel's x position
    center = (left + right) / 2
    # Define conversions in x and y from pixels (image space) to meters (real world space)
    xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

    return (position - center) * xm_per_pix


def find_lane_lines_histogram_style(binary_warped, display=True):
    global X
    global Y

    # Assuming you have created a warped binary image called "binary_warped"
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0] / 2:, :], axis=0)
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] / 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(binary_warped.shape[0] / nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (
            nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (
            nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # -----Added modification start----------------------------------
    # Consider averaging instead of getting last value ---

    if len(righty) == 0:
        print("righty is empty retrieve from history")
        righty = Y
    else:
        Y = righty

    if len(rightx) == 0:
        print("rightx is empty retrieve from history")
        rightx = X
    else:
        X = rightx

    # -----Added modification end-----------------------------------------

    # Fit a second order polynomial to each
    # Find the coefficients of polynomials
    left_fit = np.polyfit(lefty, leftx, 2)  # left_fit.shape = (3,)
    right_fit = np.polyfit(righty, rightx, 2)  # right_fit.shape = (3,)

    # print("left_fit", left_fit.shape, left_fit)
    # print("right_fit", right_fit.shape, right_fit)
    # print(".......")

    # At this point, you're done! But here is how you can visualize the result as well

    # yvals/ploty to cover same y-range as image
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])

    # Generate x and y values for plotting
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # print("out_img", out_img.shape)
    # print("ploty", ploty.shape, ploty[:10])
    # print("left_fitx", left_fitx.shape, left_fitx[0])

    if display:
        plt.imshow(out_img)
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)
        plt.show()

    return out_img, ploty, left_fit, right_fit


def find_window_centroids(warped, window_width, window_height, margin):
    window_centroids = []  # Store the (left,right) window centroid positions per level
    window = np.ones(window_width)  # Create our window template that we will use for convolutions

    # First find the two starting positions for the left and right lane by using np.sum to get the vertical image slice
    # and then np.convolve the vertical image slice with the window template

    # Sum quarter bottom of image to get slice, could use a different ratio
    l_sum = np.sum(warped[int(3 * warped.shape[0] / 4):, :int(warped.shape[1] / 2)], axis=0)
    l_center = np.argmax(np.convolve(window, l_sum)) - window_width / 2
    r_sum = np.sum(warped[int(3 * warped.shape[0] / 4):, int(warped.shape[1] / 2):], axis=0)
    r_center = np.argmax(np.convolve(window, r_sum)) - window_width / 2 + int(warped.shape[1] / 2)

    # Add what we found for the first layer
    window_centroids.append((l_center, r_center))

    # Go through each layer looking for max pixel locations
    for level in range(1, (int)(warped.shape[0] / window_height)):
        # convolve the window into the vertical slice of the image
        image_layer = np.sum(
            warped[int(warped.shape[0] - (level + 1) * window_height):int(warped.shape[0] - level * window_height), :],
            axis=0)
        conv_signal = np.convolve(window, image_layer)
        # Find the best left centroid by using past left center as a reference
        # Use window_width/2 as offset because convolution signal reference is at right side of window,
        # not center of window
        offset = window_width / 2
        l_min_index = int(max(l_center + offset - margin, 0))
        l_max_index = int(min(l_center + offset + margin, warped.shape[1]))
        l_center = np.argmax(conv_signal[l_min_index:l_max_index]) + l_min_index - offset
        # Find the best right centroid by using past right center as a reference
        r_min_index = int(max(r_center + offset - margin, 0))
        r_max_index = int(min(r_center + offset + margin, warped.shape[1]))
        r_center = np.argmax(conv_signal[r_min_index:r_max_index]) + r_min_index - offset
        # Add what we found for that layer
        window_centroids.append((l_center, r_center))

    return window_centroids


def optimize_next_frame_lane_fine(binary_warped, left_fit, right_fit):
    # Assume you now have a new warped binary image
    # from the next frame of video (also called "binary_warped")
    # It's now much easier to find line pixels!
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100
    left_lane_inds = ((nonzerox > (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] - margin)) & (
        nonzerox < (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] + margin)))
    right_lane_inds = (
        (nonzerox > (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] - margin)) & (
            nonzerox < (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] + margin)))

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    # And you're done! But let'svisualize the result here as well
    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    window_img = np.zeros_like(out_img)
    # Color in left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + margin, ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + margin, ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

    # plt.imshow(result)
    # plt.plot(left_fitx, ploty, color='yellow')
    # plt.plot(right_fitx, ploty, color='yellow')
    # plt.xlim(0, 1280)
    # plt.ylim(720, 0)
    # plt.show()

    return result, ploty, left_fitx, right_fitx


def process_test_image(img, mtx, dist):
    # Undistort the image
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    warped_color, M, Minv = perform_warp(undist)

    # perform thresholding via pipeline
    thresholded_image = pipeline(undist) * 255  # multiplication converts 1s to 255 i.e. white

    # perform warping/perspective transform to birds eye view
    warped, M, Minv = perform_warp(thresholded_image)

    warped_image_with_window, ploty, left_fitx, right_fitx = find_lane_lines_histogram_style(warped, False)
    # outimg, ploty, left_fitx, right_fitx = optimize_next_frame_lane_fine(warped, left_fitx, right_fitx)

    left_curve, right_curve = radius_of_curvature(ploty, left_fitx, right_fitx)
    print("Real3D radius left: {} m, right: {} m".format(int(left_curve), int(right_curve)))

    result = draw_lines_on_road(warped, undist, ploty, left_fitx, right_fitx, Minv, left_curve)

    return result, ploty, left_fitx, right_fitx

# mtx, dist = find_corners_and_caliberate_camera()
# image = mpimg.imread('test_images/straight_lines1.jpg')
# process_test_image(image, mtx, dist)
