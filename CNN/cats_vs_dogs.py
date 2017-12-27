# -*- coding: UTF-8 -*-
# Testing convolutional networks, using TensorFlow.
# This lab tries to classify images of cats and dogs.
# Intended for Python 2.7.
# Magnus Lindhé, 2017

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import math

# Remove TensorFlow warnings about unused GPU instructions
os.environ['TF_CPP_MIN_LOG_LEVEL']='2' 

# Path to images should be relative to current working directory.
cwd = os.getcwd()
imageFilter = cwd + "/data/small/*.jpg"
print("File filter: " + imageFilter)

# The code for loading images is borrowed from 
# "https://gist.github.com/eerwitt/518b0c9564e500b4b50f"
# Make a queue of file names
name_list = tf.train.match_filenames_once(imageFilter)
filename_queue = tf.train.string_input_producer(name_list)

# Each image is fetched by reading a whole file.
image_reader = tf.WholeFileReader()

# Read a whole file from the queue, the first returned value in the tuple is the
# filename which we are ignoring.
_, image_file = image_reader.read(filename_queue)

# Decode the image as a JPEG file, this will turn it into a Tensor which we can
# then use in training.
image = tf.image.decode_jpeg(image_file)

# Start a new session to show example output.
with tf.Session() as sess:
    # Open a log file that we can later view by running
    # "tensorboard --logdir=/tmp/log" on the command line.
    writer = tf.summary.FileWriter("/tmp/log", sess.graph)  
    
    # Required to get the filename matching to run.
    init_op = tf.local_variables_initializer()
    sess.run(init_op)

    # Coordinate the loading of image files.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    t = sess.run(image)
    
    # Plot the file
    plt.figure()
    plt.imshow(np.squeeze(t))
    plt.show()

    # Finish off the filename queue coordinator.
    coord.request_stop()
    coord.join(threads)
    
    writer.close()

# resized_image = tf.image.resize_images(image, [200, 200])




