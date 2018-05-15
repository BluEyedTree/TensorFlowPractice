import os

import skimage
import tensorflow as tf
from skimage import data
import numpy as np
import matplotlib.pyplot as plt
from skimage import transform
from skimage.color import rgb2gray
import random

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

def turnToGreyScale(inputLIst):
    # Convert `images28` to an array
    images28 = np.array(inputLIst)
    # Convert `images28` to grayscale
    images28 = rgb2gray(images28)
    return images28


TrainingData= "/Users/tbekman/PycharmProjects/PlayingWithTensorFlow/Training"
TestingData = "/Users/tbekman/PycharmProjects/PlayingWithTensorFlow/Testing"

#KEEP
a = load_data(TrainingData)

labels = a[1] #y in the net
images = np.array(a[0])
image28 = [transform.resize(image,(28,28)) for image in images] #resizes the images so they're 28x28
image28Grey = turnToGreyScale(image28) #X in the net


def turnToGreyScale(inputLIst):
    # Convert `images28` to an array
    images28 = np.array(inputLIst)
    # Convert `images28` to grayscale
    return rgb2gray(images28)


'''
print(images[2250])


'''
def visualizeTheSings():
    unique_labels = set(labels)
    plt.figure(figsize=(12, 12))
    i = 1
    for label in unique_labels:
        image = image28Grey[labels.index(label)]
        plt.subplot(8,8,i)
        plt.axis('off')
        plt.title("Label {0} ({1})".format(label, labels.count(label)))
        plt.subplots_adjust(wspace=0.5)
        plt.subplots_adjust(hspace=0.5)
        i +=1
        plt.imshow(image, cmap="gray")
    plt.show()

visualizeTheSings()
#traffic_signs = [300, 2250, 3650, 4000]
'''
traffic_signs = [300, 2250, 3650, 4000]
for i in range(len(traffic_signs)):
    plt.subplot(1, 4, i + 1)
    plt.axis('off')
    plt.imshow(image28[traffic_signs[i]])
    plt.subplots_adjust(wspace=0.5)
    plt.show()
    print("shape: {0}, min: {1}, max: {2}".format(image28[traffic_signs[i]].shape,
                                                  image28[traffic_signs[i]].min(),
                                                 image28[traffic_signs[i]].max()))
'''

'''
print images.ndim
print images.size
print images[0]

print(labels.ndim)

# Print the number of `labels`'s elements
print(labels.size)

# Count the number of labels
print(len(set(labels)))
'''

#######################################Below we build the network#######################################

x = tf.placeholder(dtype = tf.float32, shape = [None, 28, 28])
y = tf.placeholder(dtype = tf.int32, shape = [None])

# Flatten the input data
images_flat = tf.contrib.layers.flatten(x)

# Fully connected layer
logits = tf.contrib.layers.fully_connected(images_flat, 62, tf.nn.relu)

loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y,
                                                                    logits = logits))

# Define an optimizer
train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

# Convert logits to label indexes
correct_pred = tf.argmax(logits, 1)

# Define an accuracy metric
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))



###############Now to Train the network######################################################
tf.set_random_seed(1234)
sess = tf.Session()

sess.run(tf.global_variables_initializer())

for i in range(1000):
        print('EPOCH', i)
        _, accuracy_val = sess.run([train_op, accuracy], feed_dict={x: image28Grey, y: labels})
        if i % 10 == 0:
            print("Loss: ", loss)
        print('DONE WITH EPOCH')


###########Now to see if it was accurate###################
sample_indexes = random.sample(range(len(image28Grey)), 2000)
sample_images = [image28Grey[i] for i in sample_indexes]
sample_labels = [labels[i] for i in sample_indexes]

# Run the "correct_pred" operation
predicted = sess.run([correct_pred], feed_dict={x: sample_images})[0]

# Print the real and predicted labels
print(sample_labels)
print(predicted)
correct = 0
# Display the predictions and the ground truth visually.
for i in range(len(sample_images)):
    truth = sample_labels[i]
    prediction = predicted[i]
    if truth == prediction:
        correct+=1
    print(correct/len(sample_images))
print(correct)
print(len(sample_images))
plt.show()