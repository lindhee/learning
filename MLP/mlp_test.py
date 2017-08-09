# -*- coding: UTF-8 -*-
# Testing the multi-layer perceptron, using TensorFlow
# Magnus Lindhé, 2017

import mnist

training_data = list(mnist.read(dataset="training", path="/home/lindhe/Documents/Self studies/code/MLP/MNIST_images"))
print("Read " + str(len(training_data)) + " training images.")

# Display an example image
if (True):
	image_no = 17
	label, pixels = training_data[image_no]
	print("Image no " + str(image_no) + " has label " + str(label) + " and shape " + str(pixels.shape) +".")
	mnist.show(pixels)





