# -*- coding: UTF-8 -*-
# Classifying iris flowers, using radial basis functions (RBF)
# Magnus Lindhé, 2017

# Parts of the code are taken from:
# Code from Chapter 4 of Machine Learning: An Algorithmic Perspective (2nd Edition)
# by Stephen Marsland (http://stephenmonika.net)
# Stephen Marsland, 2008, 2014

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load the iris data (each row is a point in R^4 and then a label in {1,2,3})
iris = np.loadtxt('iris_proc.data',delimiter=',')

# Scale each dimension to make the mean 0 and the variance 1.
iris[:,0:4] = iris[:,0:4]-iris[:,0:4].mean(axis=0) # Subtract mean of columns 0-3
columnStdDev = np.sqrt(iris[:,0:4].var(axis=0));
iris[:,0:4] = iris[:,0:4]/columnStdDev # Divide columns 0-3 by their stddev

# Plot 3 of 4 dimensions
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.scatter(iris[:,0],iris[:,1],iris[:,2],c=iris[:,4])
ax.title("Iris data")
ax.set_xlabel("x_0")
ax.set_ylabel("x_1")
ax.set_zlabel("x_2")
plt.title("Iris data")
plt.show()

# Express the targets as one-hot vectors
target = np.zeros((np.shape(iris)[0],3));
indices = np.where(iris[:,4]==0) 
target[indices,0] = 1
indices = np.where(iris[:,4]==1)
target[indices,1] = 1
indices = np.where(iris[:,4]==2)
target[indices,2] = 1

# Randomly order the data
order = range(np.shape(iris)[0])
np.random.shuffle(order)
iris = iris[order,:]
target = target[order,:]

# Split into training, validation, and test sets
train = iris[::2,0:4]
traint = target[::2]
valid = iris[1::4,0:4]
validt = target[1::4]
test = iris[3::4,0:4]
testt = target[3::4]

# Do k-means clustering of each class in the training set, to get 3*k prototype vectors


# Choose variance (sigma) for the Gaussians of each prototype


# Train the neural network


# Test the performance







