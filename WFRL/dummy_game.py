#!/usr/bin/python

import numpy as np

'''
Dummy game to test the effectiveness of the DQN.
The game starts out with a white screen, and will change
to black if given input 1. A black screen turns white on
input 2. Flipping the screen color gives reward 1,
but the wrong input for a given screen will give
-1 reward. Doing nothing always gives 0 reward.

ML: Removed CV dependency by making the screen a scalar,
either 0 (black) or 1 (white).

Every ten steps, a new episode starts.

Ideally the DQN learns that the best strategy is
to continually flip the screen color to maximize reward.
'''

class GameState:
    def __init__(self):
        self.screen = 255
        self.steps = 0

    '''
    The input_vec is a one-hot vector with three elements where 
    (0,0,1) means "do nothing",
    (0,1,0) means "switch to black" and
    (1,0,0) means "switch to white".
    '''
    def frame_step(self, input_vec):
        reward = -1

        if self.screen == 1:
            if input_vec[1] == 1:
                reward = 1
                self.screen = 0;
            elif input_vec[0] == 1:
                reward = 0
        else:
            if input_vec[2] == 1:
                reward = 1
                self.screen = 1;
            elif input_vec[0] == 1:
                reward = 0

        self.steps += 1
        terminal = False
        if self.steps >= 10:
            self.steps = 0
            terminal = True

        return self.screen, reward, terminal
