# -*- coding: UTF-8 -*-
# Testing a two-layer perceptron, using TensorFlow
# Magnus Lindhé, 2017

import mnist
import numpy as np
import tensorflow as tf
import os
from tensorflow.examples.tutorials.mnist import input_data

# Path to MNIST data should be relative to current working directory.
cwd = os.getcwd()
mnist_path = cwd + "/MNIST_data"
mn = input_data.read_data_sets(mnist_path, one_hot=True)

# Display an example image
if (False):
	image_no = 17
	picture = np.array(mn.train.images[image_no]);
	picture = picture.reshape(28,28);	
	print("Image no " + str(image_no) + " has label " + str(mn.train.labels[image_no]) + ".")
	mnist.show(picture);

# Placeholder for the N*784 tensor of input data (images)
x = tf.placeholder(tf.float32, (None, 784))

# Width of second layer
M = 10

# Weights and biases for first layer
V = tf.Variable(tf.zeros((784, M)))
b = tf.Variable(tf.zeros((1, M)))

# Second layer
W = tf.Variable(tf.zeros((M, 10)))
c = tf.Variable(tf.zeros((1, 10)))

# TensorFlow model: y = softmax(W*sigmoid(V*x + b) + c)
first_layer_output = tf.matmul(x, V) + b
#y = tf.nn.softmax(first_layer_output)
y = tf.nn.softmax(tf.sigmoid(first_layer_output) + c)
#y = tf.nn.softmax(tf.matmul(tf.sigmoid(first_layer_output), W) + c)

# Correct labels, used as targets
t = tf.placeholder(tf.float32, (None, 10))

if (True):
	# Use cross entropy as loss function (loss = sum(t_i * log(y_i)) (gives about 91% accuracy)
	loss_fcn = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=t, logits=y))
else:
	# Use a quadratic loss function (gives about 89% accuracy)
	loss_fcn = tf.reduce_mean((y-t)*(y-t))	

# Train the model
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss_fcn)

# Start an interactive session
sess = tf.InteractiveSession()

# Initialize the variables
tf.global_variables_initializer().run()

# Train some epochs, in each epoch we use 100 random values from the training data
no_of_epochs = 1000
for _ in range(no_of_epochs):
  batch_xs, batch_ys = mn.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, t: batch_ys})

# Check how the model performs
correct_predictions = tf.equal(tf.argmax(y,1), tf.argmax(t,1))
accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
accuracy_result = sess.run(accuracy, feed_dict={x: mn.test.images, t: mn.test.labels})
print("Result: " + str(100*accuracy_result) + "% of the images were correctly classified.")





