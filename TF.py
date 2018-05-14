import os

import skimage
import tensorflow as tf
from skimage import data
import numpy as np

print(os.path.dirname(os.path.realpath(__file__)))

def load_data(data_directory):
  directories = [d for d in os.listdir(data_directory)
                 if os.path.isdir(os.path.join(data_directory, d))]

  labels = []
  images = []

def load_data(data_directory):
    directories = [d for d in os.listdir(data_directory) if os.path.isdir(os.path.join(data_directory, d))]
    labels = []
    images = []
    for d in directories:
        label_directory = os.path.join(data_directory, d)
        file_names = [os.path.join(label_directory, f) for f in os.listdir(label_directory) if f.endswith(".ppm")]
        for f in file_names:
            images.append(skimage.data.imread(f))
            labels.append(int(d))
    return images, labels



TrainingData= "/Users/tbekman/PycharmProjects/PlayingWithTensorFlow/Training"
TestingData = "/Users/tbekman/PycharmProjects/PlayingWithTensorFlow/Testing"

a = load_data(TrainingData)

images = np.array(a[0])
labels = np.array(a[1])

print images.ndim
print images.size
print images[0]