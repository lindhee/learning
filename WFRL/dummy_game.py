#!/usr/bin/python

import numpy as np

'''
Dummy game to test the effectiveness of the DQN.
The playing field is a 1*11 vector, where all cells are 0 except one, which is 1. 
Initial state is that the one is in the middle cell.
The inputs are:
(0,0,1) means "go right",
(0,1,0) means "go straight" and
(1,0,0) means "go left".
The reward is minus the distance (in cells) from the one to the middle. 
But we also add a penalty of -5 if the one is in the boundary cells and 
the player drives it out. (But it doesn't leave the field.)

Every ten steps, a new episode starts.
'''

class GameState:
    def __init__(self):
        self.n = 5 # Index of the occupied cell (0-10)
        self.steps = 0

    '''
    Given a one-hot input action vector, compute:
    screen: Resulting state: 11-element vector of zeros, except for one element that is one
    reward: Resulting reward for the action from this state 
    terminal: Bool - TRUE iff the episode ended after this step
    '''
    def frame_step(self, input_vec):
        if input_vec[0] == 1: # Go right
            self.n += 1
        elif input_vec[2] == 1: # Go left
            self.n -= 1

        # Maybe we should add some random disturbances here, that the player
        # needs to counteract?

        reward = 0
        if self.n < 0:
            self.n = 0
            reward = -5
        elif self.n > 10:
            self.n = 10
            reward = -5

        reward -= abs(self.n-5)

        self.steps += 1
        terminal = False
        if self.steps >= 10:
            self.steps = 0
            terminal = True

        # Draw the "image"
        screen = np.zeros((11))
        screen[self.n] = 1

        return screen, reward, terminal
