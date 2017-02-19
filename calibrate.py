import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import pickle

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
        print("returning camera calibration mtx and dist from saved pickle")
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
def corners_unwarp(img, nx, ny, mtx, dist):
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


# top_down, perspective_M = corners_unwarp(img, nx, ny, mtx, dist)



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


#  My thresholding pipeline.
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


def visualize_images(original, modified):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(original)
    ax1.set_title('Original Image', fontsize=30)
    ax2.imshow(modified, cmap='gray')
    ax2.set_title('Thresholded Image', fontsize=30)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()


def perform_warp(img):
    img_size = (img.shape[1], img.shape[0])

    # Define a four sided polygon to mask
    # Polygon Points - note (x,y)->(0,0) is topmost left
    imgHeight = img.shape[0]
    imgWidth = img.shape[1]
    img_height_trim_hood = imgHeight * 0.9
    midx = (imgWidth / 2, img_height_trim_hood)
    trapezoid_height = imgHeight * 0.28

    # Calculate polygon/trapezoid points based on midx point
    # E.g. leftBottom is at left of midx and rightBottom is at right side

    left_bottom = (midx[0] - 400, midx[1])
    right_bottom = (midx[0] + 400, midx[1])
    left_top = (midx[0] - 50, midx[1] - trapezoid_height)
    right_top = (midx[0] + 50, midx[1] - trapezoid_height)

    # # Visualize points for debugging only
    # plt.imshow(img)
    # plt.plot(leftBottom[0], leftBottom[1], "*")
    # plt.plot(leftTop[0], leftTop[1], "*")
    # plt.plot(rightTop[0], rightTop[1], "*")
    # plt.plot(rightBottom[0], rightBottom[1], "*")
    # plt.plot(midx[0], midx[1], "*")
    # plt.show()
    # exit()

    src = np.float32([left_top, right_top, right_bottom, left_bottom])

    # For destination points, I'm arbitrarily choosing a horizontal offset
    # Note that src[index] -> dst[index]
    offset = 200  # x offset for dst
    dst = np.float32([[offset, 0],
                      [imgWidth - offset, 0],
                      [imgWidth - offset, imgHeight],
                      [offset, imgHeight]])

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
        img = mpimg.imread(f_name)

        # Undistort the image
        undist = cv2.undistort(img, mtx, dist, None, mtx)

        # perform thresholding via pipeline
        thresholded_image = pipeline(undist) * 255  # multiplication converts 1s to 255 i.e. white

        # perform warping/perspective transform to birds eye view
        warped_image, perspective_m, perspective_m_inv = perform_warp(thresholded_image)
        visualize_images(img, warped_image)


# --------------------------------------------------------
# Method calls
# --------------------------------------------------------
mtx, dist = find_corners_and_caliberate_camera()
process_testimages(mtx, dist)

exit("All done")

# image = mpimg.imread('signs_vehicles_xygrad.png')
# # Apply each of the thresholding functions
# gradx = abs_sobel_thresh(image, orient='x', sobel_kernel=ksize, thresh=(20, 100))
# grady = abs_sobel_thresh(image, orient='y', sobel_kernel=ksize, thresh=(5, 100))
# mag_binary = mag_thresh(image, sobel_kernel=ksize, mag_thresh=(20, 100))
# dir_binary = dir_threshold(image, sobel_kernel=ksize, thresh=(0.7, 1.3))

# Combine results of thresholded gradients
# combined = np.zeros_like(dir_binary)
# combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1

# Visualize images
# visualize_images(image, combined)
