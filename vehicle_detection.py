import glob
import matplotlib.image as mpimg
import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.externals import joblib
from skimage.feature import hog
from sklearn import svm
from sklearn.model_selection import GridSearchCV

heat_map = {}


class Window:
  def __init__(self, point1, point2, color):
    self.point1 = point1
    self.point2 = point2
    self.color = color

  def __str__(self):
    return '{} {}'.format(self.point1, self.point2)


class WindowType:
  def __init__(self, min_y, max_y, padding_x, window_size, overlap=1,
      image_w=1280, color=(0, 255, 0)):
    self.min_y = min_y
    self.max_y = max_y
    self.min_x = padding_x
    self.max_x = image_w - padding_x
    self.window_size = window_size
    self.step_size = window_size // overlap
    self.color = color


# Returns images in GRAY
def read_images(filepattern):
  return [cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2GRAY) for filename in
          glob.iglob(filepattern, recursive=True)]


# Image: GRAY
def get_hog_feature(image, orientations=9, pixel_per_cell=8, cell_per_block=2):
  return hog(image, orientations=orientations,
             pixels_per_cell=(pixel_per_cell, pixel_per_cell),
             cells_per_block=(cell_per_block, cell_per_block), visualise=False,
             block_norm='L2-Hys', feature_vector=True)


# Images: GRAY
def get_labeled_hog_data(car_images, not_car_images):
  features = [get_hog_feature(car_image) for car_image in car_images] + [
    get_hog_feature(not_car_image) for not_car_image in not_car_images]
  print(len(features))
  labels = [1 for x in car_images] + [0 for y in not_car_images]

  return features, labels


def train(load_from_disk=True, show_accuracy=False):
  model_file_name = 'model.pkl'

  if load_from_disk:
    clf = joblib.load(model_file_name)
  else:
    cars = read_images('training_data/vehicles/**/*.png')
    notcars = read_images('training_data/non_vehicles/**/*.png')

    features, labels = get_labeled_hog_data(cars, notcars)

    parameters = {'C': [0.3, 1, 3]}
    svr = svm.SVC(kernel='linear')
    clf = GridSearchCV(svr, parameters)
    clf.fit(features, labels)
    joblib.dump(clf, model_file_name)

  if show_accuracy:
    print(clf.cv_results_)

  return clf


clf = train(load_from_disk=True, show_accuracy=True)


def get_sliding_windows():
  windows = []
  window_types = [
    WindowType(min_y=300, max_y=580, padding_x=0, window_size=225, overlap=4,
               color=(255, 255, 255)),
    WindowType(min_y=330, max_y=580, padding_x=0, window_size=160, overlap=4,
               color=(0, 0, 255)),
    WindowType(min_y=350, max_y=540, padding_x=0, window_size=128, overlap=4,
               color=(255, 0, 0)),
    WindowType(min_y=400, max_y=500, padding_x=200, window_size=80, overlap=4,
               color=(0, 255, 128)),
  ]
  for window_type in window_types:
    for x in range(window_type.min_x,
                   window_type.max_x - window_type.window_size + 1,
                   window_type.step_size):
      for y in range(window_type.max_y,
                     window_type.min_y + window_type.window_size - 1,
                     -window_type.step_size):
        windows.append(Window((x, y - window_type.window_size),
                              (x + window_type.window_size, y),
                              window_type.color))

  return windows


sliding_windows = get_sliding_windows()
for sliding_window in sliding_windows:
  heat_map[sliding_window] = 0.0


def identify_windows_with_car(img, alpha=0.5, cutoff=1.9):
  gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
  hog_features = []
  for window in sliding_windows:
    windowed_image = gray[window.point1[1]:window.point2[1],
                     window.point1[0]:window.point2[0]]
    resized_image = cv2.resize(windowed_image, (64, 64))
    hog_features.append(get_hog_feature(resized_image))

  are_cars = clf.predict(hog_features)

  windows_with_car = []
  for idx, window in enumerate(sliding_windows):
    new_value = heat_map[window] * alpha + are_cars[idx]
    heat_map[window] = new_value
    if new_value > cutoff:
      windows_with_car.append(window)

  return windows_with_car


def merge_windows(windows):
  merged_windows = []
  unmerged_windows = windows
  while len(unmerged_windows) > 0:
    new_unmerged_windows = []
    current_window = unmerged_windows[0]
    for window in unmerged_windows[1:]:
      if collides(current_window, window):
        current_window = merge(current_window, window)
      else:
        new_unmerged_windows.append(window)
    merged_windows.append(current_window)
    unmerged_windows = new_unmerged_windows
  return merged_windows


def collides(window1, window2):
  x_intersects_or_within = (window1.point1[0] <= window2.point1[0] and
                            window1.point2[0] >= window2.point1[0]) or \
                           (window1.point1[0] <= window2.point2[0] and
                            window1.point2[0] >= window2.point2[0]) or \
                           (window1.point1[0] >= window2.point1[0] and
                            window1.point2[0] <= window2.point2[0])
  y_intersects_or_within = (window1.point1[1] <= window2.point1[1] and
                            window1.point2[1] >= window2.point1[1]) or \
                           (window1.point1[1] <= window2.point2[1] and
                            window1.point2[1] >= window2.point2[1]) or \
                           (window1.point1[1] >= window2.point1[1] and
                            window1.point2[1] <= window2.point2[1])
  return x_intersects_or_within and y_intersects_or_within


def merge(window1, window2):
  return Window((min(window1.point1[0], window2.point1[0]),
                 min(window1.point1[1], window2.point1[1])), (
                max(window1.point2[0], window2.point2[0]),
                max(window1.point2[1], window2.point2[1])), (128, 255, 0))


# Img: RGB
def paint_windows(img, windows, thickness=3):
  imcopy = np.copy(img)
  for window in windows:
    cv2.rectangle(imcopy, window.point1, window.point2, window.color, thickness)
  return imcopy
