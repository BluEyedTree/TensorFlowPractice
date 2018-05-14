import os

import tensorflow as tf

print(os.path.dirname(os.path.realpath(__file__)))

def load_data(data_directory):
  directories = [d for d in os.listdir(data_directory)
                 if os.path.isdir(os.path.join(data_directory, d))]

  for d in directories:
    label_directory = os.path.join(data_directory, d))

load_data("/Users/tbekman/PycharmProjects/PlayingWithTensorFlow/Training")