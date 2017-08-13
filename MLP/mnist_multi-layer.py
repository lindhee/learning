# -*- coding: UTF-8 -*-
# Testing a two-layer perceptron, using TensorFlow
# Magnus Lindhé, 2017

import mnist
import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import math
from tensorflow.examples.tutorials.mnist import input_data

# Path to MNIST data should be relative to current working directory.
cwd = os.getcwd()
mnist_path = cwd + "/MNIST_data"
mn = input_data.read_data_sets(mnist_path, one_hot=True)
N = mn.train.num_examples # No of training images

# Display an example image
if (False):
	image_no = 17
	picture = np.array(mn.train.images[image_no]);
	picture = picture.reshape(28,28);	
	print("Image no " + str(image_no) + " has label " + str(mn.train.labels[image_no]) + ".")
	mnist.show(picture);

# Placeholder for the N*784 tensor of input data (images)
x = tf.placeholder(tf.float32, (None, 784))

# Width, weights and biases for hidden layer (Gaussian distributed)
# Scale the distribution of the initial weights, to avoid saturation in the inner layers
M = 200
W1 = tf.Variable(tf.random_normal((784, M), stddev=1.0/math.sqrt(785)))
b1 = tf.Variable(tf.random_normal((1, M), stddev=1.0/math.sqrt(785)))

# Second, output layer
W2 = tf.Variable(tf.random_normal((M, 10), stddev=1.0/math.sqrt(M+1)))
b2 = tf.Variable(tf.random_normal((1, 10), stddev=1.0/math.sqrt(M+1)))

# TensorFlow model: y = sigmoid(W2*sigmoid(W1*x + b1) + b2)
a1 = tf.sigmoid(tf.matmul(x, W1) + b1)
y = tf.sigmoid(tf.matmul(a1, W2) + b2)

# Correct labels, used as targets
t = tf.placeholder(tf.float32, (None, 10))

cost_function="regularized cross entropy"
lambda_reg = 0.1 # Regularization factor
if cost_function=="quadratic":
	loss_fcn = tf.reduce_mean((y-t)*(y-t))
elif cost_function=="regularized quadratic":    
    loss_fcn = tf.reduce_mean((y-t)*(y-t)) + lambda_reg/N*(tf.nn.l2_loss(W1) + tf.nn.l2_loss(W2))
elif cost_function=="cross entropy":
	# (loss = sum(t_i * log(y_i))
	loss_fcn = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=t, logits=y))
elif cost_function=="regularized cross entropy":
	# (loss = sum(t_i * log(y_i))
	loss_fcn = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=t, logits=y)) + lambda_reg/N*(tf.nn.l2_loss(W1) + tf.nn.l2_loss(W2))
else:
	assert("No loss function defined!")
print("Using cost function " + cost_function + ".")

# Train the model
# (The final performance seems quite robust to eta in [1, 5].)
train_step = tf.train.GradientDescentOptimizer(3.0).minimize(loss_fcn)

# Start an interactive session
sess = tf.InteractiveSession()

# Initialize the variables
tf.global_variables_initializer().run()

# Train some epochs, in each epoch we use a random subset of images from the training data
no_of_epochs = 30
no_of_images_per_batch = 10
print("Starting {} epochs of stochastic gradient descent.".format(no_of_epochs))
epoch_accuracies=np.zeros(no_of_epochs)
epoch_indices=np.arange(0,no_of_epochs)
for e in range(no_of_epochs):
    SGD_rounds_per_epoch = N/no_of_images_per_batch
    for _ in range(SGD_rounds_per_epoch):
        batch_xs, batch_ys = mn.train.next_batch(no_of_images_per_batch)
        sess.run(train_step, feed_dict={x: batch_xs, t: batch_ys})
        
    # Report how the model performs
    correct_predictions = tf.equal(tf.argmax(y,1), tf.argmax(t,1))
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
    accuracy_result = sess.run(accuracy, feed_dict={x: mn.test.images, t: mn.test.labels})
    print("Epoch {0}: {1:.1f}% correct.".format(e, 100*accuracy_result))
    epoch_accuracies[e]=accuracy_result

plt.plot(epoch_indices,100*epoch_accuracies,"bo")
plt.grid(True)
plt.axis([0, no_of_epochs, 90, 100])
plt.title("MNIST image recognition, with M={0} and cost function {1}.".format(M, cost_function))
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.show()



