from scipy.signal import find_peaks_cwt
from calibrate import *


def blind_search_mask(img, window_size):
    '''
    Perform a blind search by finding peaks in a histogram.
    :param img:
    :param window_size:
    :return: mask_L_image, mask_R_image
    '''

    img = gaussian_blur(img, 5)
    img_size = np.shape(img)
    mov_filtsize = img_size[1] / 50
    mean_ln = np.mean(img[img_size[0] / 2:, :], axis=0)
    mean_ln = moving_average(mean_ln, mov_filtsize)
    mean_ln = np.array(mean_ln)

    # Get x position of peaks e.g [225, 1094]
    indexes = find_peaks_cwt(mean_ln, [100], max_distances=np.array([800]))
    indexes = np.array(indexes)

    # Get all valid indices at peaks
    valid_ind = np.array([mean_ln[indexes[i]] for i in range(len(indexes))])
    ind_sorted = np.argsort(-valid_ind)

    ind_peakL = indexes[ind_sorted[0]]
    ind_peakR = indexes[ind_sorted[1]]

    if ind_peakR < ind_peakL:
        ind_temp = ind_peakR
        ind_peakR = ind_peakL
        ind_peakL = ind_temp

    mask_L_image = np.zeros_like(img)
    mask_R_image = np.zeros_like(img)

    ind_peakR_prev = ind_peakR
    ind_peakL_prev = ind_peakL

    # Split image into 8 parts and compute histogram on each part
    for i in range(8):
        img_y1 = img_size[0] - img_size[0] * i / 8
        img_y2 = img_size[0] - img_size[0] * (i + 1) / 8

        mean_lane_y = np.mean(img[img_y2:img_y1, :], axis=0)
        mean_lane_y = moving_average(mean_lane_y, mov_filtsize)
        mean_lane_y = np.array(mean_lane_y)
        indexes = find_peaks_cwt(mean_lane_y, [100], max_distances=np.array([800]))
        indexes = np.array(indexes)

        if len(indexes) > 1.5:
            valid_ind = np.array([mean_ln[indexes[i]] for i in range(len(indexes))])
            ind_sorted = np.argsort(-valid_ind)

            ind_peakR = indexes[ind_sorted[0]]
            ind_peakL = indexes[ind_sorted[1]]
            if ind_peakR < ind_peakL:
                ind_temp = ind_peakR
                ind_peakR = ind_peakL
                ind_peakL = ind_temp

        else:
            # If no pixels are found, use previous ones.
            if len(indexes) == 1:
                if np.abs(indexes[0] - ind_peakR_prev) < np.abs(indexes[0] - ind_peakL_prev):
                    ind_peakR = indexes[0]
                    ind_peakL = ind_peakL_prev
                else:
                    ind_peakL = indexes[0]
                    ind_peakR = ind_peakR_prev
            else:
                ind_peakL = ind_peakL_prev
                ind_peakR = ind_peakR_prev

        # If new center is more than 60pixels away, use previous
        # Outlier rejection
        if np.abs(ind_peakL - ind_peakL_prev) >= 60:
            ind_peakL = ind_peakL_prev

        if np.abs(ind_peakR - ind_peakR_prev) >= 60:
            ind_peakR = ind_peakR_prev

        mask_L_image[img_y2:img_y1, ind_peakL - window_size:ind_peakL + window_size] = 1.
        mask_R_image[img_y2:img_y1, ind_peakR - window_size:ind_peakR + window_size] = 1.

        ind_peakL_prev = ind_peakL
        ind_peakR_prev = ind_peakR

    return mask_L_image, mask_R_image


def found_search_mask_poly(img, poly_fit, window_sz):
    # This function returns masks for points used in computing polynomial fit.
    mask_poly = np.zeros_like(img)
    img_size = np.shape(img)

    poly_pts = []
    pt_y_all = []

    for i in range(8):
        img_y1 = img_size[0] - img_size[0] * i / 8
        img_y2 = img_size[0] - img_size[0] * (i + 1) / 8

        pt_y = (img_y1 + img_y2) / 2
        pt_y_all.append(pt_y)
        poly_pt = np.round(poly_fit[0] * pt_y ** 2 + poly_fit[1] * pt_y + poly_fit[2])

        poly_pts.append(poly_pt)

        # draw window on blank canvas
        mask_poly[img_y2:img_y1, poly_pt - window_sz:poly_pt + window_sz] = 1.

    return mask_poly, np.array(poly_pts), np.array(pt_y_all)


def get_val(y, pol_a):
    # Returns value of a quadratic polynomial
    return pol_a[0] * y ** 2 + pol_a[1] * y + pol_a[2]


def draw_pw_lines(img, pts, color):
    # This function draws lines connecting 10 points along the polynomial
    pts = np.int_(pts)
    for i in range(10):
        x1 = pts[0][i][0]
        y1 = pts[0][i][1]
        x2 = pts[0][i + 1][0]
        y2 = pts[0][i + 1][1]
        cv2.line(img, (x1, y1), (x2, y2), color, 50)


def undistort_image(img, mtx, dist):
    # Function to undistort image
    undist_img = cv2.undistort(img, mtx, dist, None, mtx)
    return undist_img


def gaussian_blur(img, kernel=5):
    # Function to smooth image
    blur = cv2.GaussianBlur(img, (kernel, kernel), 0)
    return blur


def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    # Calculate directional gradient
    # Apply threshold
    if orient == 'x':
        img_s = cv2.Sobel(img, cv2.CV_64F, 1, 0)  # 1,0 for x-axis
    else:
        img_s = cv2.Sobel(img, cv2.CV_64F, 0, 1)  # 0,1 for y-axis

    img_abs = np.absolute(img_s)
    # Rescale back to 8 bit integer
    img_sobel = np.uint8(255 * img_abs / np.max(img_abs))
    # Create a copy and apply the threshold
    binary_output = 0 * img_sobel
    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    # blackouts gradient in range. Note - 0 is pure white and 255 is pure black
    # Hence thresh=(0, 255) will blackout entire image
    binary_output[(img_sobel >= thresh[0]) & (img_sobel <= thresh[1])] = 1

    return binary_output


def warp_image(img, src, dst, img_size):
    # Apply perspective transform
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    Minv = cv2.getPerspectiveTransform(dst, src)

    return warped, M, Minv


def binary_mask(img, low, high):
    # Takes in low and high values and returns mask
    mask = cv2.inRange(img, low, high)
    return mask


def apply_color_mask(hsv, img, low, high):
    # Takes in color mask and returns image with mask applied.
    mask = cv2.inRange(hsv, low, high)
    res = cv2.bitwise_and(img, img, mask=mask)
    return res


def moving_average(a, n=3):
    # Moving average
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def radius_of_curvature2(pol_a, y_pt):
    # Returns curvature of a quadratic
    A = pol_a[0]
    B = pol_a[1]
    R_curve = (1 + (2 * A * y_pt + B) ** 2) ** 1.5 / 2 / A
    return R_curve


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


def stack_arr(arr):
    # Stacks 1-channel array into 3-channel array to allow plotting
    return np.stack((arr, arr, arr), axis=2)


def perform_perspective_transform(image):
    # Applies bird-eye perspective transform to an image
    img_size = image.shape
    ht_window = np.uint(img_size[0] / 1.5)
    hb_window = np.uint(img_size[0])
    c_window = np.uint(img_size[1] / 2)
    ctl_window = c_window - .25 * np.uint(img_size[1] / 2)
    ctr_window = c_window + .25 * np.uint(img_size[1] / 2)
    cbl_window = c_window - 1 * np.uint(img_size[1] / 2)
    cbr_window = c_window + 1 * np.uint(img_size[1] / 2)
    src = np.float32([[cbl_window, hb_window], [cbr_window, hb_window],
                      [ctr_window, ht_window], [ctl_window, ht_window]])
    dst = np.float32([[0, img_size[0]], [img_size[1], img_size[0]],
                      [img_size[1], 0], [0, 0]])

    warped, M_warp, Minv_warp = warp_image(image, src, dst, (img_size[1], img_size[0]))  # returns birds eye image
    return warped, M_warp, Minv_warp


def pipeline_process(image):
    global left_fit_prev
    global right_fit_prev
    global col_R_prev
    global col_L_prev
    global set_flag
    global mask_poly_L
    global mask_poly_R

    # get camera matrix and distortion coefficient
    mtx, dist = find_corners_and_caliberate_camera()

    # Undistort image
    image = undistort_image(image, mtx, dist)
    image = gaussian_blur(image, kernel=5)
    img_size = np.shape(image)

    # Define window for perspective transform
    warped, M_warp, Minv_warp = perform_perspective_transform(image)

    image_HSV = cv2.cvtColor(warped, cv2.COLOR_RGB2HSV)

    # Define color ranges and apply color mask
    yellow_hsv_low = np.array([0, 100, 100])
    yellow_hsv_high = np.array([50, 255, 255])

    white_hsv_low = np.array([20, 0, 180])
    white_hsv_high = np.array([255, 80, 255])
    # get yellow and white masks
    mask_yellow = binary_mask(image_HSV, yellow_hsv_low, yellow_hsv_high)
    mask_white = binary_mask(image_HSV, white_hsv_low, white_hsv_high)
    # Combine white and yellow masks into 1
    mask_lane = cv2.bitwise_or(mask_yellow, mask_white)

    # Convert image to HLS scheme
    image_HLS = cv2.cvtColor(warped, cv2.COLOR_RGB2HLS)

    # Apply sobel filters on L channels.
    img_gs = image_HLS[:, :, 1]
    img_abs_x = abs_sobel_thresh(img_gs, 'x', 5, (50, 225))
    img_abs_y = abs_sobel_thresh(img_gs, 'y', 5, (50, 225))
    warped_L = np.copy(cv2.bitwise_or(img_abs_x, img_abs_y))

    # Apply sobel filters S channels.
    img_gs = image_HLS[:, :, 2]
    img_abs_x = abs_sobel_thresh(img_gs, 'x', 5, (50, 255))
    img_abs_y = abs_sobel_thresh(img_gs, 'y', 5, (50, 255))
    warped_S = np.copy(cv2.bitwise_or(img_abs_x, img_abs_y))

    # Combine sobel filter information from L and S channels.
    image_cmb = cv2.bitwise_or(warped_L, warped_S)
    image_cmb = gaussian_blur(image_cmb, 25)

    # Combine sobel masks and color threshold masks.
    image_cmb_final = np.zeros_like(image_cmb)
    image_cmb_final[(mask_lane >= .5) | (image_cmb >= .5)] = 1

    # If this is first frame, perform a blind search to get new mask.
    if set_flag == 0:
        mask_poly_L, mask_poly_R = blind_search_mask(image_cmb_final, 40)

    # Define all colors as white to start.
    col_R = (255, 255, 255)
    col_L = (255, 255, 255)

    # Apply mask to thresholded images and compute polynomial fit for left.
    img_L = cv2.bitwise_and(image_cmb_final, image_cmb_final, mask=mask_poly_L)
    img_L = np.array(img_L)
    vals = np.argwhere(img_L > .5)
    if len(vals) < 5:  # If less than 5 points use prev values
        left_fit = left_fit_prev
        col_L = col_L_prev
    else:
        all_x = vals.T[0]
        all_y = vals.T[1]
        left_fit = np.polyfit(all_x, all_y, 2)
        if np.sum(cv2.bitwise_and(img_L, mask_yellow)) > 1000:
            col_L = (255, 255, 0)

    # Apply mask to thresholded images and compute polynomial fit for right.
    img_R = cv2.bitwise_and(image_cmb_final, image_cmb_final, mask=mask_poly_R)
    img_R = np.array(img_R)
    vals = np.argwhere(img_R > .5)
    if len(vals) < 5:
        right_fit = right_fit_prev
        col_R = col_R_prev
    else:
        all_x = vals.T[0]
        all_y = vals.T[1]
        right_fit = np.polyfit(all_x, all_y, 2)
        if np.sum(cv2.bitwise_and(img_R, mask_yellow)) > 1000:
            col_R = (255, 255, 0)

    # save coefficient values for next frame
    if set_flag == 0:
        set_flag = 1
        right_fit_prev = right_fit
        left_fit_prev = left_fit

    # Check error between coefficient from current and previous frame
    err_p_R = np.sum((right_fit[0] - right_fit_prev[0]) ** 2)  # /np.sum(right_fit_prev[0]**2)
    err_p_R = np.sqrt(err_p_R)
    if err_p_R > .0005:
        right_fit = right_fit_prev
        col_R = col_R_prev
    else:
        right_fit = .05 * right_fit + .95 * right_fit_prev

    # Check error between current coefficient and on from previous frame
    err_p_L = np.sum((left_fit[0] - left_fit_prev[0]) ** 2)  # /np.sum(right_fit_prev[0]**2)
    err_p_L = np.sqrt(err_p_L)
    if err_p_L > .0005:
        left_fit = left_fit_prev
        col_L = col_L_prev
    else:
        left_fit = .05 * left_fit + .95 * left_fit_prev

    # Compute lane mask for future frame
    mask_poly_L, left_pts, img_pts = found_search_mask_poly(image_cmb_final, left_fit, window_size)
    mask_poly_R, right_pts, img_pts = found_search_mask_poly(image_cmb_final, right_fit, window_size)

    # Compute for lanes
    ploty = left_y = right_y = np.arange(11) * img_size[0] / 10
    right_fitx = right_fit[0] * right_y ** 2 + right_fit[1] * right_y + right_fit[2]

    # left_y = np.arange(11) * img_size[0] / 10
    left_fitx = left_fit[0] * left_y ** 2 + left_fit[1] * left_y + left_fit[2]

    warp_zero = np.zeros_like(image_cmb_final).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, left_y]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, right_y])))])
    pts = np.hstack((pts_left, pts_right))

    # Compute leftmost and rightmost intercepts
    leftmost = get_val(img_size[0], left_fit)
    rightmost = get_val(img_size[0], right_fit)

    # Compute center location
    center = (leftmost + rightmost) / 2.0

    # Compute lane offset
    xm_per_pix = 3.7 / 700  # meters per pixel (real world x dimension)
    car_position = img_size[1] / 2  # Assume car position is at center of the image x-axis
    dist_offset_meters = (car_position - center) * xm_per_pix
    dist_offset_meters = np.round(dist_offset_meters, 2)

    direction = "left" if dist_offset_meters < 0 else "right"
    text_offset = "Vehicle is {:.2f} m {} of center".format(abs(dist_offset_meters), direction)
    text_offset_no_abs = "Vehicle is {:.2f} m of center. I.e. {}".format(dist_offset_meters, direction)

    # change color if direction changes
    if dist_offset_meters > 0:
        cv2.fillPoly(color_warp, np.int_([pts]), (255, 0, 0))
    else:
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Draw the left and right lane onto the warped blank image
    draw_pw_lines(color_warp, np.int_(pts_left), col_L)
    draw_pw_lines(color_warp, np.int_(pts_right), col_R)
    # Warp the blank back to original image space using inverse perspective matrix (Minv)

    newwarp = cv2.warpPerspective(color_warp, Minv_warp, (image.shape[1], image.shape[0]))
    # Combine the result with the original image
    result = cv2.addWeighted(image, 1, newwarp, 0.5, 0)

    # left_curve = radius_of_curvature2(left_fit, img_size[0] / 2)
    # right_curve = radius_of_curvature2(right_fit, img_size[0] / 2)

    left_curve, right_curve = radius_of_curvature(ploty, left_fitx, right_fitx)

    text_curvature = "Radius of Curvature Left: {} m and Right: {} m".format(int(left_curve), int(right_curve))

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(result, text_curvature, (30, 60), font, 1, (255, 255, 255), 2)
    cv2.putText(result, text_offset, (30, 90), font, 1, (255, 255, 255), 2)

    # set previous values
    right_fit_prev = right_fit
    left_fit_prev = left_fit
    col_R_prev = col_R
    col_L_prev = col_L

    # diagnosis view for debugging only.
    if is_diagnosis == 1:
        font = cv2.FONT_HERSHEY_COMPLEX
        middlepanel = np.zeros((120, 1280, 3), dtype=np.uint8)
        cv2.putText(middlepanel, text_curvature, (30, 60), font, 1, (255, 0, 0), 2)
        cv2.putText(middlepanel, text_offset_no_abs, (30, 90), font, 1, (255, 0, 0), 2)

        # assemble the screen example
        diagScreen = np.zeros((1080, 1920, 3), dtype=np.uint8)
        diagScreen[0:720, 0:1280] = result
        diagScreen[0:240, 1280:1600] = cv2.resize(warped, (320, 240), interpolation=cv2.INTER_AREA)
        diagScreen[0:240, 1600:1920] = cv2.resize(stack_arr(mask_lane), (320, 240), interpolation=cv2.INTER_AREA)
        diagScreen[240:480, 1280:1600] = cv2.resize(
            apply_color_mask(image_HSV, warped, yellow_hsv_low, yellow_hsv_high), (320, 240),
            interpolation=cv2.INTER_AREA)
        diagScreen[240:480, 1600:1920] = cv2.resize(apply_color_mask(image_HSV, warped, white_hsv_low, white_hsv_high),
                                                    (320, 240), interpolation=cv2.INTER_AREA) * 4
        diagScreen[600:1080, 1280:1920] = cv2.resize(color_warp, (640, 480), interpolation=cv2.INTER_AREA) * 4
        diagScreen[720:840, 0:1280] = middlepanel
        diagScreen[840:1080, 0:320] = cv2.resize(newwarp, (320, 240), interpolation=cv2.INTER_AREA)

        diagScreen[840:1080, 320:640] = cv2.resize(stack_arr(255 * image_cmb_final), (320, 240),
                                                   interpolation=cv2.INTER_AREA)
        diagScreen[840:1080, 640:960] = cv2.resize(stack_arr(255 * mask_poly_L + 255 * mask_poly_R), (320, 240),
                                                   interpolation=cv2.INTER_AREA)
        diagScreen[840:1080, 960:1280] = cv2.resize(stack_arr(255 * cv2.bitwise_and(image_cmb_final, image_cmb_final,
                                                                                    mask=mask_poly_L + mask_poly_R)),
                                                    (320, 240), interpolation=cv2.INTER_AREA)
        return diagScreen
    else:
        return result


# ------------------------------------------------------------------------#
# ------------------- Begin Processing -----------------------------------#
# ------------------------------------------------------------------------#
kernel_size = 5  # kernel size for filters
window_size = 60  # size for sliding windows
is_diagnosis = 0
set_flag = 0

# image = mpimg.imread('./test_images/test1.jpg')
# img_size = np.shape(image)
# result_pipe = pipeline_process_highway(image)
# visualize_images(result_pipe, image)

# input_video = 'harder_challenge_video.mp4'
input_video = 'project_video.mp4'
output_video = './output_videos/output_video.mp4'

if is_diagnosis == 1:
    output_video = './output_videos/output_hard_debug.mp4'

# clip = VideoFileClip(input_video).subclip(0, 5)
clip = VideoFileClip(input_video)
video_clip = clip.fl_image(pipeline_process)
video_clip.write_videofile(output_video, audio=False)
