# -*- coding: UTF-8 -*-
# Testing the multi-layer perceptron, using TensorFlow
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

# Weights and biases are variables
# (What if the bias is constant, and we play with its weight?)
W = tf.Variable(tf.zeros((784, 10)))
b = tf.Variable(tf.zeros((1, 10)))

# TensorFlow model: y = softmax(W*x + b)
y = tf.nn.softmax(tf.matmul(x, W) + b)

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

# Train M epochs, in each epoch we use 100 random values from the training data
M = 1000
for i in range(M):
    batch_xs, batch_ys = mn.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, t: batch_ys})
  
    if i%100==0:    
        # Report how the model performs
        correct_predictions = tf.equal(tf.argmax(y,1), tf.argmax(t,1))
        accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
        accuracy_result = sess.run(accuracy, feed_dict={x: mn.test.images, t: mn.test.labels})
        print("Round {0}: {1:.1f}% correct.".format(i, 100*accuracy_result))

# Check the resulting weights
weights = W.value().eval() # Gives a 784*10 numpy array
print("Displaying the weights for the number 5.")
w_five = np.reshape(weights[...,5],(28,28))

# Normalize to (0,1) to show weights as gray scale
w_image = (w_five-w_five.min())
if w_image.max() > 0:
	w_image = w_image/w_image.max()
mnist.show(w_image)

# Check how the model performs
correct_predictions = tf.equal(tf.argmax(y,1), tf.argmax(t,1))
accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
accuracy_result = sess.run(accuracy, feed_dict={x: mn.test.images, t: mn.test.labels})
print("Result: " + str(100*accuracy_result) + "% of the images were correctly classified.")





