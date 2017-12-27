# -*- coding: UTF-8 -*-
# Testing convolutional networks, using TensorFlow.
# This lab tries to classify images of cats and dogs.
# Intended for Python 2.7.
# Magnus Lindhé, 2017

import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import math

# Path to images should be relative to current working directory.
cwd = os.getcwd()
imagePath = cwd + "/data/train/"

# Read images and resize them to constant size.
# NOTE: This now creates some distortion of images that have other proportions. So a better way would 
# be to pad each image to the correct proportions, then resize it.
fileName = imagePath + "cat.1000.jpg"
print("Reading file " + fileName)
imageFile = tf.read_file(fileName)
image = tf.image.decode_jpeg(imageFile,1) # 1 = convert to grayscale
resized_image = tf.image.resize_images(image, [200, 200])

# Start an interactive session
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)
im = sess.run(resized_image)

# Display the image
plt.figure()
plt.imshow(np.reshape(im,(200,200)))
plt.show()



