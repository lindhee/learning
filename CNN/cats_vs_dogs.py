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
import CNNUtils as CNNU

# Remove TensorFlow warnings about unused GPU instructions
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# We have classes cats and dogs
NUM_CLASSES = 2

trainingBatchSize = 10

# Path to images should be relative to current working directory.
cwd = os.getcwd()
(image_batch, label_batch) = CNNU.loadImages(cwd + "/data/small_train/*.jpg", trainingBatchSize)
(test_image_batch, test_label_batch) = CNNU.loadImages(cwd + "/data/test/*.jpg", 1000)

x = tf.placeholder(tf.float32, (None, 20, 20, 1), name="x")

# Conv1: 5*5 kernel with 64 output channels
kernel1 = tf.Variable(tf.random_normal([5,5,1,64], stddev=1.0/math.sqrt(5*5)),name='kernel1')
conv1 = tf.nn.conv2d(x, kernel1, strides=[1, 1, 1, 1], padding='SAME',name='conv1')
bias1 = tf.Variable(tf.random_normal([64]), name='bias1')
preActivation1 = tf.nn.bias_add(conv1, bias1)
out1 = tf.nn.relu(preActivation1, name='conv1_out')
# pool1
pool1 = tf.nn.max_pool(out1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')
# norm1
norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')

print("Norm1: " + str(norm1.shape))

# Fully connected layer 1, with 20 outputs
# Move everything into depth so we can perform a single matrix multiply.
noOfOutputs1 = 20
reshape1 = tf.reshape(norm1, [-1, 10*10*64])
dim = reshape1.get_shape()[1].value
weights1 = tf.Variable(tf.random_normal([dim, noOfOutputs1], stddev=0.04),name='a1')
loc_bias1 = tf.Variable(tf.random_normal([noOfOutputs1], mean=0.1), name='loc_bias1')
local1 = tf.nn.relu(tf.matmul(reshape1, weights1) + loc_bias1, name='local1')

print("Local1: " + str(local1.shape))

# Linear layer(WX + b),
# We don't apply softmax here because
# tf.nn.sparse_softmax_cross_entropy_with_logits accepts the unscaled logits
# and performs the softmax internally for efficiency.
linWeights = tf.Variable(tf.random_normal([noOfOutputs1, NUM_CLASSES], stddev=1/noOfOutputs1),name='linWeights')
linBias = tf.Variable(tf.random_normal([NUM_CLASSES], stddev=1/NUM_CLASSES),name='linBias')
softmax_linear = tf.add(tf.matmul(local1, linWeights), linBias, name='softmax_linear')
    
y = softmax_linear    

# Correct labels, used as targets
t = tf.placeholder(tf.float32, (None, 2),name="t")

print("y: " + str(y.shape))
print("t: " + str(t.shape))

# Cross entropy cost function
loss_fcn = tf.reduce_mean((t-y)*(t-y))
#loss_fcn = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=t, logits=y))
train_step = tf.train.GradientDescentOptimizer(3.0).minimize(loss_fcn)

# Start a new session to show example output.
with tf.Session() as sess:

    # Open a log file that we can later view by running
    # "tensorboard --logdir=/tmp/log" on the command line.
    writer = tf.summary.FileWriter("/tmp/log", sess.graph)  

    # Required to get the filename matching to run.
    sess.run(tf.local_variables_initializer())
    
    # Initialize the global variables (such as weights)
    sess.run(tf.global_variables_initializer())

    # Coordinate the loading of image files.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    # Train some epochs, in each epoch we use a random subset of images from the training data
    no_of_epochs = 30
    print("Starting {} epochs of stochastic gradient descent.".format(no_of_epochs))
    epoch_accuracies=np.zeros(no_of_epochs)
    epoch_indices=np.arange(0,no_of_epochs)
    for e in range(no_of_epochs):
        SGD_rounds_per_epoch = 5
        for _ in range(SGD_rounds_per_epoch):
            im_batch, l_batch = sess.run([image_batch, label_batch])
            sess.run(train_step, feed_dict={x: im_batch, t: l_batch})
            
        # Report how the model performs
        correct_predictions = tf.equal(tf.argmax(y,axis=1), tf.argmax(t,axis=1))
        accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
        test_im_batch, test_l_batch = sess.run([test_image_batch, test_label_batch])
        accuracy_result = sess.run(accuracy, feed_dict={x: test_im_batch, t: test_l_batch})
        print("Epoch {0}: {1:.1f}% correct.".format(e, 100*accuracy_result))
        epoch_accuracies[e]=accuracy_result

    # Finish off the filename queue coordinator.
    coord.request_stop()
    coord.join(threads)

    writer.close()
 




