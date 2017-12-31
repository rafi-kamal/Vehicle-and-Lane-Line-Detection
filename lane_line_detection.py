import numpy as np
import cv2
import glob
import matplotlib.image as mpimg


def calibrate_camera(image_shape=(1280, 720)):
  chessboard_inner_points_x = 9
  chessboard_inner_points_y = 6

  objp = np.zeros((chessboard_inner_points_x * chessboard_inner_points_y, 3),
                  np.float32)
  objp[:, :2] = np.mgrid[0:chessboard_inner_points_x,
                0:chessboard_inner_points_y].T.reshape(-1, 2)

  imgpoints = []
  objpoints = []

  images = glob.glob('camera_cal/*')

  for idx, filename in enumerate(images):
    img = mpimg.imread(filename)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, (
      chessboard_inner_points_x, chessboard_inner_points_y))

    if ret:
      objpoints.append(objp)
      imgpoints.append(corners)

  ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints,
                                                     image_shape, None, None)
  return mtx, dist


def undistort(img):
  return cv2.undistort(img, mtx, dist, None, mtx)


def get_perspective_matrix(w=1280, h=720):
  src = np.float32(
      [[w / 2 - 100, 460], [w / 2 + 100, 460], [w, h - 35], [0, h - 35]])
  dst = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
  return cv2.getPerspectiveTransform(src, dst)


def perspective_transform(img, M):
  h = img.shape[0]
  w = img.shape[1]
  return cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_LINEAR)


mtx, dist = calibrate_camera()
M = get_perspective_matrix()
Minv = np.linalg.inv(M)


def color_gradient_filtering(img, sobel_kernel=5, sobelx_thresh=(200, 230),
    s_thresh=(100, 255), l_thresh=(50, 255)):
  gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
  sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
  sobelx = np.uint8(255 * sobelx / np.max(sobelx))
  sobelx_filtered = np.zeros_like(sobelx)
  sobelx_filtered[
    (sobelx > sobelx_thresh[0]) & (sobelx <= sobelx_thresh[1])] = 1

  hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
  s_channel = hls[:, :, 2]
  s_channel_filtered = np.zeros_like(s_channel)
  s_channel_filtered[(s_channel > s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
  l_channel = hls[:, :, 1]
  l_channel = np.uint8(255 * l_channel / np.max(l_channel))
  l_channel_filtered = np.zeros_like(l_channel)
  l_channel_filtered[(l_channel > l_thresh[0]) & (l_channel <= l_thresh[1])] = 1

  filtered_img = np.zeros_like(sobelx)
  filtered_img[((sobelx_filtered == 1) | (s_channel_filtered == 1)) & (
    l_channel_filtered == 1)] = 1
  return filtered_img


def get_weighted_avg(prev, new, weight):
  return prev * (1 - weight) + new * weight


def find_img_lower_centroid(image, window_width=51):
  h = image.shape[0]
  w = image.shape[1]
  # An array like [0, 1, 2, 3, ..., 24, 25, 24, ..., 3, 2, 1, 0]
  window = [window_width // 2 - abs(window_width // 2 - i) for i in
            range(window_width)]

  l_sum = np.sum(image[int(3 * h / 4):, :int(w / 2)], axis=0)
  l_center = np.argmax(np.convolve(window, l_sum)) - window_width / 2
  r_sum = np.sum(image[int(3 * h / 4):, int(w / 2):], axis=0)
  r_center = np.argmax(np.convolve(window, r_sum)) - window_width / 2 + int(
      w / 2)

  return (int(l_center), int(r_center))


def find_window_centroids(image, weighted_l_center, weighted_r_center,
    weight=0.67, no_of_windows=9, window_width=51, margin=140, min_points=1200):
  h = image.shape[0]
  w = image.shape[1]
  window_height = h / no_of_windows

  l_centroids = []
  r_centroids = []
  # An array like [0, 1, 2, 3, ..., 24, 25, 24, ..., 3, 2, 1, 0]
  window = [window_width // 2 - abs(window_width // 2 - i) for i in
            range(window_width)]

  # Go through each layer looking for max pixel locations
  for level in range(0, (int)(h / window_height)):
    image_layer = np.sum(image[int(h - (level + 1) * window_height):int(
        h - level * window_height), :], axis=0)
    conv_signal = np.convolve(window, image_layer)
    offset = int(window_width / 2)

    l_min_index = int(max(weighted_l_center + offset - margin, 0))
    l_max_index = int(min(weighted_l_center + offset + margin, w))
    l_center = np.argmax(
        conv_signal[l_min_index:l_max_index]) + l_min_index - offset

    r_min_index = int(max(weighted_r_center + offset - margin, 0))
    r_max_index = int(min(weighted_r_center + offset + margin, w))
    r_center = np.argmax(
        conv_signal[r_min_index:r_max_index]) + r_min_index - offset

    # We are confident enough that this is a part of the lane lines only if
    # we have found at least min_points in the convolution array
    window_center_h = int(h - (level + 0.5) * window_height)
    if conv_signal[l_center] > min_points:
      l_centroids.append((l_center, window_center_h))
      weighted_l_center = get_weighted_avg(weighted_l_center, l_center, weight)

    if conv_signal[r_center] > min_points:
      r_centroids.append((r_center, window_center_h))
      weighted_r_center = get_weighted_avg(weighted_r_center, r_center, weight)

  if len(l_centroids) > 2 and len(r_centroids) > 2:
    return True, np.array(l_centroids), np.array(
        r_centroids), weighted_l_center, weighted_r_center
  else:
    return False, None, None, None, None


def get_curvature(centroids):
  ym_per_pix = 30 / 720  # meters per pixel in y dimension
  xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

  y_eval = np.max(centroids[:, 1])
  fit_cr = np.polyfit(centroids[:, 1] * ym_per_pix,
                      centroids[:, 0] * xm_per_pix, 2)
  return ((1 + (
    2 * fit_cr[0] * y_eval * ym_per_pix + fit_cr[
      1]) ** 2) ** 1.5) / np.absolute(
      2 * fit_cr[0])


def get_position(w, left_centroids, right_centroids):
  xm_per_pix = 3.7 / 700  # meters per pixel in x dimension
  return ((left_centroids[0, 0] + right_centroids[0, 0]) - w) / 2 * xm_per_pix


def draw_lane(filtered_img, undistorted_img, Minv, left_fit, right_fit):
  ploty = np.linspace(0, filtered_img.shape[0] - 1, filtered_img.shape[0])
  left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
  right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

  # Create an image to draw the lines on
  warp_zero = np.zeros_like(filtered_img).astype(np.uint8)
  color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

  # Recast the x and y points into usable format for cv2.fillPoly()
  pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
  pts_right = np.array(
      [np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
  pts = np.hstack((pts_left, pts_right))

  # Draw the lane onto the warped blank image
  cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

  # Warp the blank back to original image space using inverse perspective matrix (Minv)
  newwarp = cv2.warpPerspective(color_warp, Minv, (
    undistorted_img.shape[1], undistorted_img.shape[0]))
  # Combine the result with the original image
  return cv2.addWeighted(undistorted_img, 1, newwarp, 0.3, 0)
