# -*- coding: UTF-8 -*-
# K-means clustering
# Magnus Lindhé, 2017

import numpy as np
import matplotlib.pyplot as plt

# Euclidean distance between x and y
def distance(x, y):
    assert(x.shape==y.shape)
    diff = x-y
    return np.sqrt(diff.dot(diff))

# Return index of row in c that is closest to row vector x
def closestIndex(c, x):
    N = c.shape[0] # No of rows
    assert(N>0)
    minDist = distance(c[0,:],x)
    minIndex = 0
    for i in range(1,N):
        dist = distance(c[i,:],x)
        if dist < minDist:
            minDist = dist
            minIndex = i
    return minIndex

# x: An N*n matrix of N n-dimensional data points to cluster
# k: The number of clusters (k<=N)
# Returns a k*n matrix of cluster center locations
def kmeansclustering(x, k):
    N = x.shape[0] # No of rows
    n = x.shape[1] # No of columns
    
    # Random initial cluster centers
    c = np.random.rand(k,n)
    
    clusterIndex = -1*np.ones(N) # Says which cluster each point belongs to. Start with invalid index.
    while True:
        # Associate data points to cluster centers        
        clusterAllocationChanged = False
        for i in range(0,N):
            closestClusterIndex = closestIndex(c, x[i,:])
            if closestClusterIndex != clusterIndex[i]:
                clusterAllocationChanged = True
                clusterIndex[i] = closestClusterIndex
        
        # If nothing changed, end the iteration
        if not clusterAllocationChanged:
            break 
        
        # Update cluster centers
        for j in range(0,k):
            if np.any(clusterIndex==j):
                # Don't update if the cluster has no members
                c[j,:] = x[clusterIndex==j,:].mean(axis=0)
    
    return c
    
# Test code
def runTest():
    # Generate data from some normal distributions
    N = 30 # No of samples from each class
    covariance = 0.2*np.eye(2)
    x0 = np.random.multivariate_normal(np.array((0,0)),covariance,N) # Class 0
    x1 = np.random.multivariate_normal(np.array((5,5)),covariance,N) # Class 1
    x2 = np.random.multivariate_normal(np.array((5,0)),covariance,N) # Class 2
    x = np.vstack((x0,x1,x2))

    # Do clustering
    c = kmeansclustering(x,3)

    # Plot result
    plt.plot(x[:,0],x[:,1],"b.")
    plt.plot(c[:,0],c[:,1],"gx")
    plt.show()


