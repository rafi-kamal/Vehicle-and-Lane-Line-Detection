from lane_line_detection import *
from vehicle_detection import *
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip

class State():
  weighted_l_center = None
  weighted_r_center = None
  weight_left_fit = np.float32([0, 0, 300])
  weight_right_fit = np.float32([0, 0, 1280 - 300])

state = State()


def pipline(img, visualize_pipleine=False, poly_weight=0.3):
  undistorted_img = undistort(img)
  perspective_transformed_img = perspective_transform(undistorted_img, M)
  filtered_img = color_gradient_filtering(perspective_transformed_img)

  if state.weighted_l_center is None or state.weighted_r_center is None:
    state.weighted_l_center, state.weighted_r_center = find_img_lower_centroid(
        filtered_img)
  lane_found, l_centroids, r_centroids, state.weighted_l_center, state.weighted_r_center = find_window_centroids(
      filtered_img,
      state.weighted_l_center,
      state.weighted_r_center)

  if lane_found:
    left_fit = np.polyfit(l_centroids[:, 1], l_centroids[:, 0], 2)
    right_fit = np.polyfit(r_centroids[:, 1], r_centroids[:, 0], 2)
    state.weight_left_fit = get_weighted_avg(state.weight_left_fit, left_fit,
                                             poly_weight)
    state.weight_right_fit = get_weighted_avg(state.weight_right_fit, right_fit,
                                              poly_weight)

  output = draw_lane(filtered_img, undistorted_img, Minv, state.weight_left_fit,
                     state.weight_right_fit)
  if l_centroids is not None and r_centroids is not None:
    curvature = (get_curvature(l_centroids) + get_curvature(r_centroids)) / 2
    cv2.putText(output, 'Curvature: {:.0f}m'.format(curvature), (100, 100), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)
    position = get_position(img.shape[1], l_centroids, r_centroids)
    cv2.putText(output, 'Distance from center: {:.2f}m'.format(position), (100, 150), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)

  windows_with_car = identify_windows_with_car(undistorted_img)
  output_with_windows = paint_windows(output, windows_with_car)
  merged_windows = merge_windows(windows_with_car)
  output_with_cars = paint_windows(output, merged_windows)

  cv2.putText(output_with_cars, 'Number of cars found: {}'.format(len(merged_windows)), (100, 200), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)


  if lane_found and visualize_pipleine:
    lane_img = cv2.cvtColor(filtered_img, cv2.COLOR_GRAY2RGB) * 255
    for (x, y) in np.concatenate((l_centroids, r_centroids)):
      cv2.circle(lane_img, (x, y), 5, (255, 0, 0), thickness=10)

    f, (ax0, ax1) = plt.subplots(2, 2, figsize=(24, 9))
    f.tight_layout()

    ax0[0].set_title("Figure 1: Original")
    ax0[0].imshow(img)
    ax0[1].set_title("Figure 2: With Lane Lines")
    ax0[1].imshow(output)
    ax1[0].set_title("Figure 3: Detected Cars")
    ax1[0].imshow(filtered_img)
    ax1[0].imshow(output_with_windows)
    ax1[1].set_title("Figure 4: Final")
    ax1[1].imshow(output_with_cars)

    plt.show()

  return output_with_cars


# pipline(mpimg.imread('test_images/test10.jpg'), visualize_pipleine=True)
input_filename = 'project_video.mp4'
output_filename = 'processed_' + input_filename
clip = VideoFileClip(input_filename)
output_video = clip.fl_image(pipline)
output_video.write_videofile(output_filename, audio=False)