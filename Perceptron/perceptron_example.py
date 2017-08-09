# Test the Perceptron on a linearly separable toy dataset
# Magnus Lindhe, 2017
# Runs in Python 2.7.

import perceptron as pc
import numpy as np
import matplotlib.pyplot as plt

# List of data points (each point is a x_image,y_image,y tuple, where y = 0 for red and y = 1 for green)
points=[]

# Empty perceptron
perc = pc.Perceptron2D([(0, 0, 0)])

# When this is True, we do a training round per mouse click, rather than adding data points
trainingStarted = False

# Returns the (2*1) vector from rotating v +90 deg
def perp(v):
	return np.array([-v[1],v[0]])

# Plot the hyperplane that corresponds to the weight (column) vector W
def plotHyperplane(w):
	if (np.all(w == 0)):
		print("ERROR: A weight vector of all zeros!")
		return
	
	wTilde = w[1:,:] # Remove the weight w_0		
	u = wTilde / np.linalg.norm(wTilde) # Unit vector along wTilde

	# Let p1 be the point on the plane, closest to the origin
	k = w[0] / pow(np.linalg.norm(wTilde),2)
	p1 = k * wTilde
	
	# Let p2 be a point to the "left" of p1 along the line, and p3 to the right
	lineLength = 10
	p2 = p1 + lineLength/2 * perp(wTilde) / np.linalg.norm(wTilde)
	p3 = p1 - lineLength/2 * perp(wTilde) / np.linalg.norm(wTilde)

	plt.plot([p2[0], p3[0]],[p2[1], p3[1]],'k')

# Add a point to the set, or do a round of training
def OnClick(event):
	global trainingStarted
	global perc
	
	if not trainingStarted:
		p = event.xdata, event.ydata
		if (event.button == 1):
			print("LEFT")
			points.append((event.xdata, event.ydata, 0))
			plt.plot(p[0],p[1],"ro")
		elif (event.button == 3):
			print("RIGHT")
			points.append((event.xdata, event.ydata, 1))
			plt.plot(p[0],p[1],"go")
	else:
		perc.doOneRoundOfTraining()
		plotHyperplane(perc.W)
	
# Switch to training mode
def OnKey(event):
	global trainingStarted
	global perc
	
	if trainingStarted:
		return

	if (event.key == "enter"):
		print("Starting the training!")
		perc = pc.Perceptron2D(points)
		trainingStarted = True
	else:
		print("Press ENTER to start the training!")


# Main program
plt.ion() # Interactive mode
fig = plt.figure()
plt.gca().set_xlim(0,10)
plt.gca().set_ylim(0,10)
fig.canvas.mpl_connect('button_press_event', OnClick)
fig.canvas.mpl_connect('key_press_event', OnKey)

# Wait for input (to keep the figure window open and stay in interactive mode)
s = raw_input("Press ENTER to end the program!")

# End by going into non-interactive mode and just showing the plot until we close the window
plt.ioff()
plt.show()





