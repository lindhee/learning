# -*- coding: UTF-8 -*-
# A two-input perceptron, as a practice example in neural networks
# Magnus Lindhé, 2017

import numpy as np

class Perceptron2D:
	# Add static member variables here
	# =================================
	eta = 0.3 # Learning rate

	# Set up the perceptron, with its input data and targets (0 or 1)
	# points: list of triplets (x_1, x_2, t)
	def __init__(self,points):
		# Add (non-static) class member variables here
		# ===============================================
		self.noOfTrainingRounds = 0	

		# Start with random weights in [-0.05, 0.05]
		self.W = np.random.rand(3,1)*0.1-0.05
		
		# No of data points
		self.N = len(points)	
		
		P = np.asarray(points).transpose() # Each column is [x_1, x_2, y]'

		# Make a 3*N matrix X where each column is [x_0 = -1, x_1, x_2]'
		self.X = np.concatenate((-np.ones((1,self.N)), P[0:2,:]), axis=0)
		
		# Targets, a 1*N matrix
		self.T = P[2,:]
		
		print("2D perceptron constructed, with N = " + str(self.N) + " and P = \n" + str(P))

	# Scalar threshold function
	def g(self,x):
		return 0 if (x <= 0) else 1

	def doOneRoundOfTraining(self):
		self.noOfTrainingRounds += 1
		allOutputsCorrect = True
		for i in range(0,self.N):
			# Compute the output (dot is matrix multiplication...)
			y = self.g(np.dot(self.W.transpose(),self.X[:,i]))

			print("\nW = " + str(self.W.transpose()) + "\nx = " + str(self.X[:,i].transpose()) + "\nres = " + str(np.dot(self.W.transpose(),self.X[:,i])))
			
			# Update weights if the answer is wrong
			if (y != self.T[i]):			
				allOutputsCorrect = False
				# Note: X[:,i] will be a 1*3 array, but X[:,i:i+1] is 3*1.
				self.W = self.W - self.eta * (y-self.T[i]) * self.X[:,i:i+1]
				
		print("\nWeights (round " + str(self.noOfTrainingRounds) + "):\n" + str(self.W))
	
		if allOutputsCorrect:
			print("Training is done!");




	


