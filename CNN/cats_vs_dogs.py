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
filename_queue = tf.train.string_input_producer(name_list) # We can add shuffling and a limit on no of epochs

# Each image is fetched by reading a whole file.
image_reader = tf.WholeFileReader()

# Read a whole file from the queue
file_name, image_file = image_reader.read(filename_queue)

# Extract the image label from the file name
# Cat = 0, Dog = 1
sub_strings = tf.string_split([file_name],"/")
partial_file_name = sub_strings.values[-1] # File name, with path stripped off
sub_strings2 = tf.string_split([partial_file_name],".")
animal_type = sub_strings2.values[0] # First part of name "cat.X.jpg"
label = tf.where(tf.equal(animal_type,"dog"),1,0)

# Decode the image as a JPEG file, this will turn it into a Tensor which we can
# then use in training.
# NOTE: We make it a 2D tensor by decoding it into grayscale.
image = tf.image.decode_jpeg(image_file,channels=1)

# Pre-processing
float_image = tf.cast(image, tf.float32)
resized_image = tf.image.resize_images(float_image, [200, 200])
resized_image.set_shape([200, 200, 1])

# Subtract off the mean and divide by the variance of the pixels.
normalized_image = tf.image.per_image_standardization(resized_image)

# min_after_dequeue defines how big a buffer we will randomly sample
#   from -- bigger means better shuffling but slower start up and more
#   memory used.
# capacity must be larger than min_after_dequeue and the amount larger
#   determines the maximum we will prefetch.  Recommendation:
#   min_after_dequeue + (num_threads + a small safety margin) * batch_size
min_after_dequeue = 10
batch_size = 5
capacity = min_after_dequeue + 3 * batch_size
image_batch, label_batch = tf.train.shuffle_batch(
  [normalized_image, label], batch_size=batch_size, capacity=capacity,
  min_after_dequeue=min_after_dequeue)

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

    t,l = sess.run([normalized_image, label])
    print("Label: " + str(l))

    #tf.summary.image("Images",image)
  
    # Plot the file
    plt.figure()
    t_squeezed = np.squeeze(t)
    plt.imshow(t_squeezed,cmap="gray")
    plt.colorbar()
    plt.show()

    # Finish off the filename queue coordinator.
    coord.request_stop()
    coord.join(threads)
    
    writer.close()

 




