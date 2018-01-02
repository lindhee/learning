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
float_label = tf.reshape(tf.cast(label,dtype=tf.float32),[1])

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
squeezed_image = tf.reshape(normalized_image,[200, 200])

# Print the image and label types
print("Image: " + str(squeezed_image.shape) + " of type " + str(squeezed_image.dtype))
print("Label: " + str(float_label.shape) + " of type " + str(float_label.dtype))

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
  [squeezed_image, float_label], batch_size=batch_size, capacity=capacity,
  min_after_dequeue=min_after_dequeue)

x = tf.placeholder(tf.float32, (None, 200, 200), name="x")
  
# Define a single fully connected layer with a one-dimensional input vector
Nc = 200*200 # tf.shape(x)[1]*tf.shape(x)[2]
xFlat = tf.reshape(x,(tf.shape(x)[0], Nc))

W1 = tf.Variable(tf.random_normal((Nc, 1), stddev=1.0/math.sqrt(Nc)), name="W1")
b1 = tf.Variable(tf.random_normal([], stddev=1.0), name="b1")

# TensorFlow model: y = sigmoid(W1*xFlat + b1)
y = tf.sigmoid(tf.matmul(xFlat, W1) + b1)

# Correct labels, used as targets
t = tf.placeholder(tf.float32, (None, 1),name="t")

print("y: " + str(y.shape))
print("t: " + str(t.shape))

# Cross entropy cost function
loss_fcn = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=t, logits=y))
train_step = tf.train.GradientDescentOptimizer(3.0).minimize(loss_fcn)

# Start a new session to show example output.
with tf.Session() as sess:

    # Open a log file that we can later view by running
    # "tensorboard --logdir=/tmp/log" on the command line.
    writer = tf.summary.FileWriter("/tmp/log", sess.graph)  

    # Required to get the filename matching to run.
    sess.run(tf.local_variables_initializer())
    
    # Initialize the global variables (such as weights)
    tf.global_variables_initializer().run()

    # Coordinate the loading of image files.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    # Train some epochs, in each epoch we use a random subset of images from the training data
    no_of_epochs = 30
    no_of_images_per_batch = 10
    print("Starting {} epochs of stochastic gradient descent.".format(no_of_epochs))
    epoch_accuracies=np.zeros(no_of_epochs)
    epoch_indices=np.arange(0,no_of_epochs)
    for e in range(no_of_epochs):
        SGD_rounds_per_epoch = 5
        for _ in range(SGD_rounds_per_epoch):
            im_batch, l_batch = sess.run([image_batch, label_batch])
            sess.run(train_step, feed_dict={x: im_batch, t: l_batch})
            
        # Report how the model performs
        #correct_predictions = tf.equal(tf.argmax(y,1), tf.argmax(t,1))
        #accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
        #accuracy_result = sess.run(accuracy, feed_dict={x: mn.test.images, t: mn.test.labels})
        #print("Epoch {0}: {1:.1f}% correct.".format(e, 100*accuracy_result))
        #epoch_accuracies[e]=accuracy_result

    # Finish off the filename queue coordinator.
    coord.request_stop()
    coord.join(threads)

    writer.close()
 




