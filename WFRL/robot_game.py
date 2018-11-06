# Adapted by Magnus Lindhe', 2018, from pygame.examples.moveit
# Intended for Python 2.7

'''
This game represents a robot that is doing wall following in a bitmap world.
'''

import pygame as pg
import numpy as np

STEPS_PER_GAME = 10

pixelsPerMeter = 100

class RobotGameState:
    def __init__(self):
        pg.init()
        self.screen = pg.display.set_mode((640, 480))

        self._resetState()

    def _resetState(self):
        self.steps = 0

        # Robot pose (x, y, theta) (m, m, rad)
        self.x = np.array([2, 2, 0])

        self.background = pg.image.load("background.png").convert()
        self.robot = pg.image.load("robot.png").convert()
        self.screen.blit(self.background, (0, 0))

    def getState(self):

        return 5

    '''
    Given a one-hot input action vector, compute:
    screen: Resulting state: 11-element vector of zeros, except for one element that is one
    reward: Resulting reward for the action from this state
    terminal: Bool - TRUE iff the episode ended after this step
    '''
    def frame_step(self, input_vec):
        if input_vec[0] == 1: # Go left
            self.x[0] -= 1
            self.x[2] += 5
        elif input_vec[2] == 1: # Go right
            self.x[0] += 1

        # Move the robot and blit it on the background
        self.screen.blit(self.background, (0, 0))

        xPixel = self.x[0]*pixelsPerMeter
        yPixel = self.x[1]*pixelsPerMeter
        self.screen.blit(pg.transform.rotate(self.robot,self.x[2]), (xPixel, yPixel))
        pg.display.update()

        print("Blitting at " + str(xPixel) + "," + str(yPixel) + "," + str(self.x[2]))

        reward = 0

        screen = self.getState()

        self.steps += 1
        terminal = False
        if self.steps >= STEPS_PER_GAME:
            terminal = True
            self._resetState()

        return screen, reward, terminal

'''
Test fcn to exercise the code
'''
def main():
    a = RobotGameState()
    a.frame_step(np.array([0,1,0]))
    pg.display.update()

    # Keep the window alive
    running = True
    while running:
      for event in pg.event.get():
        if event.type == pg.KEYDOWN:
            if event.key == 275: # Right arrow
                a.frame_step(np.array([0,0,1]))
            elif event.key == 276: # Left arrow
                a.frame_step(np.array([1,0,0]))
            else:
                print("Key " + str(event.key) + " not recognized!")
        elif event.type == pg.QUIT:
          running = False


if __name__ == '__main__': main()