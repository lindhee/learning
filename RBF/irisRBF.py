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
import kMeansClustering as km

# Find the variance of all points closest to each prototype vector
# p: k*n array of n-dimensional prototype vectors
# x: N*n array of n-dimensional data points
# Returns k*1 array of variances
def clusterVariance(p,x):
    k = p.shape[0]
    N = x.shape[0]    
    assert(k>0 and N>0)
    
    sumsOfError = np.zeros(k)
    noOfPoints = np.zeros(k)    
    for i in range(0,N):
        ci = km.closestIndex(p,x[i,:])
        sumsOfError[ci] += km.distance(p[ci,:],x[i,:])
        noOfPoints[ci] += 1
    
    # Let variance be one if cluster is empty
    variances = np.ones((k,1))
    for j in range(0,k):
        if noOfPoints[j] > 0:
            variances[j] = sumsOfError[j]/noOfPoints[j]
                
    return variances

# Returns the k activations of the first layer of the RBF network
# x: 1*n input vector
# p: k*n locations of the k n-dimensional protoype vectors
# beta: k*1 values of beta for each activation function
def networkActivation(x,p,beta):
    k = p.shape[0]
    quadraticDistances = np.sum((x-p)**2,axis=1,keepdims=True)
    activation = np.exp(-beta*quadraticDistances)
    return activation.reshape(k)

# Load the iris data (each row is a point in R^4 and then a label in {1,2,3})
iris = np.loadtxt('iris_proc.data',delimiter=',')

# Scale each dimension to make the mean 0 and the variance 1.
iris[:,0:4] = iris[:,0:4]-iris[:,0:4].mean(axis=0) # Subtract mean of columns 0-3
columnStdDev = np.sqrt(iris[:,0:4].var(axis=0));
iris[:,0:4] = iris[:,0:4]/columnStdDev # Divide columns 0-3 by their stddev

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
train = iris[::2,0:4] # Every second row
traint = target[::2]
valid = iris[1::4,0:4] # Every fourth row, starting at 1
validt = target[1::4]
test = iris[3::4,0:4] # Every fourth row, starting at 3
testt = target[3::4]

# Do k-means clustering of the training data, to get 3*kprot prototype vectors
kprot = 2
p0 = km.kmeansclustering(train[traint[:,0]==1,:],kprot)
p1 = km.kmeansclustering(train[traint[:,1]==1,:],kprot)
p2 = km.kmeansclustering(train[traint[:,2]==1,:],kprot)
p = np.vstack((p0, p1, p2))
k = p.shape[0]

# Choose beta for each prototype
# (The activation function is exp(-beta*||x-p||). )
b0 = 1/(2*clusterVariance(p0, train[traint[:,0]==1,:]))
b1 = 1/(2*clusterVariance(p1, train[traint[:,1]==1,:]))
b2 = 1/(2*clusterVariance(p2, train[traint[:,2]==1,:]))
beta = np.vstack((b0, b1, b2))
print("Beta:")
print(beta)

# Train the neural network
N = train.shape[0]
M = traint.shape[1] # No of output nodes
T = np.zeros((M,N)) # Targets
A = np.zeros((k,N)) # Activations from the first layer
for i in range(0,N):
    T[:,i] = traint[i,:].transpose()
    A[:,i] = networkActivation(train[i,:].transpose(),p,beta)

W = np.dot(T,np.linalg.pinv(A))

print("Weights: ")
print(W)

# Test the performance
Ntest = test.shape[0] # No of test points
noOfCorrectAnswers = 0
for i in range(0,Ntest):
    x = test[i,:].transpose()
    y = np.dot(W,networkActivation(x,p,beta))
    #print("y:" + str(y) + " t:" + str(testt[i,:]))
    #print("argmax(y) = " + str(np.argmax(y)) + " argmax(t) = " + str(np.argmax(testt[i,:])))
    if np.argmax(y) == np.argmax(testt[i,:]):
        noOfCorrectAnswers += 1
print("{}% of the points were correctly classified.".format(100*noOfCorrectAnswers/Ntest))

# Plot 3 of 4 dimensions
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.scatter(iris[:,0],iris[:,1],iris[:,2],c=iris[:,4])
ax.plot(p[:,0],p[:,1],p[:,2],"bo")
ax.set_xlabel("x_0")
ax.set_ylabel("x_1")
ax.set_zlabel("x_2")
plt.title("Iris data")
plt.show()





