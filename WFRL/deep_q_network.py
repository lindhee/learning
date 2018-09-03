#!/usr/bin/env python

# Intended for Python 2.7

import tensorflow as tf
import dummy_game as game
import random
import numpy as np
from collections import deque

GAME = 'dummy' # the name of the game being played for log files
ACTIONS = 3 # number of valid actions
GAMMA = 0.99 # decay rate of past observations
OBSERVE = 500. # timesteps to observe before training
EXPLORE = 500. # frames over which to anneal epsilon
FINAL_EPSILON = 0.05 # final value of epsilon
INITIAL_EPSILON = 1.0 # starting value of epsilon
REPLAY_MEMORY = 5000 # number of previous transitions to remember
BATCH = 32 # size of minibatch
K = 1 # only select an action every Kth frame, repeat prev for others

'''
Definitions:
If we are in the state s_t and do action a_t, we get a reward r(s_t,a_t) (which may be stochastic) and 
end up in state s_{t+1}(s_t,a_t).

From state s_t, we want to maximize the optimal future discounted reward 
Q*(s_t,a_t) = r(s_t,a_t) + gamma * r(s_{t+1},a_{t+1}* + gamma^2 * r(s_{t+2},a_{t+2}*) + ...,
where a_t* = argmax_a Q*(s_t,a), i.e. the action that maximizes the future reward.

So if we know Q*(s,a), we can use it to find the optimal action at every step.
To compute it, we iterate with the Bellman equation:
Q(s_t,a_t) := r(s_t,a_t) + gamma * max_a Q(s_{t+1},a).
After many iterations, Q will approach Q* if:
- We train on batches of random samples (s_t, a_t, r_t, s_{t+1}) from training runs
- We avoid weight oscillations by temporarily freezing the Q function (=policy) used during training   

'''

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.01)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.01, shape = shape)
    return tf.Variable(initial)

def createNetwork():
    # network weights - guessing that the 1*ACTIONS shape of b will be broadcast to None*ACTIONS
    W_fc = weight_variable([11, ACTIONS])
    b_fc = bias_variable([1, ACTIONS])

    # input layer
    s = tf.placeholder("float", [None, 11])

    # readout layer - should get shape [None, ACTIONS]
    readout = tf.matmul(s, W_fc) + b_fc

    return s, readout

def trainNetwork(s, readout, sess):
    # Define the cost function
    # We'll let y be the right hand side of the Bellman equation above,
    # and readout_action is the left hand side.
    # We train the weights to make them equal.
    a = tf.placeholder("float", [None, ACTIONS])
    y = tf.placeholder("float", [None])
    readout_action = tf.reduce_sum(readout * a, reduction_indices = 1)
    cost = tf.reduce_mean(tf.square(y - readout_action))
    train_step = tf.train.AdamOptimizer(1e-6).minimize(cost)

    # open up a game state to communicate with emulator
    game_state = game.GameState()

    # store the previous observations in replay memory
    D = deque()

    # printing
    a_file = open("logs_" + GAME + "/readout.txt", 'w')
    h_file = open("logs_" + GAME + "/hidden.txt", 'w')

    # get the first state by going straight
    do_nothing = np.zeros(ACTIONS)
    do_nothing[1] = 1
    s_t, r_0, terminal = game_state.frame_step(do_nothing)

    # saving and loading networks
    saver = tf.train.Saver()
    tf.global_variables_initializer().run()
    checkpoint = tf.train.get_checkpoint_state("saved_networks")
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("Successfully loaded:", checkpoint.model_checkpoint_path)
    else:
        print("Could not find old network weights")

    epsilon = INITIAL_EPSILON
    t = 0
    while t<12000:
        # Choose an action epsilon greedily
        # (Feed state s_t to the current network to get output. From that we do argmax to find the best action.)
        # Should we freeze this policy for a number of iterations, to avoid oscillations?
        readout_t = readout.eval(feed_dict = {s : [s_t]})[0]
        a_t = np.zeros([ACTIONS])
        action_index = 0

        if random.random() <= epsilon or t <= OBSERVE:
            # Do random action with probability epsilon OR during the initial observation period
            action_index = random.randrange(ACTIONS)
            a_t[action_index] = 1
        else:
            # Otherwise pick the action with the highest expected reward
            action_index = np.argmax(readout_t)
            a_t[action_index] = 1

        # scale down epsilon
        if epsilon > FINAL_EPSILON and t > OBSERVE:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        for i in range(0, K):
            # run the selected action and observe next state and reward
            s_t1, r_t, terminal = game_state.frame_step(a_t)

            # store the transition in D
            D.append((s_t, a_t, r_t, s_t1, terminal))
            if len(D) > REPLAY_MEMORY:
                D.popleft()

        # only train if done observing
        if t > OBSERVE:
            # sample a minibatch to train on
            minibatch = random.sample(D, BATCH)

            # get the batch variables
            s_j_batch = [d[0] for d in minibatch]
            a_batch = [d[1] for d in minibatch]
            r_batch = [d[2] for d in minibatch]
            s_j1_batch = [d[3] for d in minibatch]

            y_batch = []
            readout_j1_batch = readout.eval(feed_dict = {s : s_j1_batch})
            for i in range(0, len(minibatch)):
                if minibatch[i][4]:
                    # If this was the last step, the future reward is simply what we got in this step
                    y_batch.append(r_batch[i])
                else:
                    # Otherwise, approximate future reward by discounted value of r(s',a*(s')),
                    # where s' is the resulting state and a*(s') is the best action.
                    y_batch.append(r_batch[i] + GAMMA * np.max(readout_j1_batch[i]))

            # perform gradient step
            train_step.run(feed_dict = {
                y : y_batch,
                a : a_batch,
                s : s_j_batch})

        # update the old values
        s_t = s_t1
        t += 1

        # save progress every 10000 iterations
        if t % 10000 == 0:
            saver.save(sess, 'saved_networks/' + GAME + '-dqn', global_step = t)

        # print info
        state = ""
        if t <= OBSERVE:
            state = "observe"
        elif t > OBSERVE and t <= OBSERVE + EXPLORE:
            state = "explore"
        else:
            state = "train"
        print("TIMESTEP", t, "/ STATE", state, "/ EPSILON", epsilon, "/ ACTION", action_index, "/ REWARD", r_t, "/ Q_MAX %e" % np.max(readout_t))

        # write info to files
        '''
        if t % 10000 <= 100:
            a_file.write(",".join([str(x) for x in readout_t]) + '\n')
            h_file.write(",".join([str(x) for x in h_fc1.eval(feed_dict={s:[s_t]})[0]]) + '\n')
            cv2.imwrite("logs_tetris/frame" + str(t) + ".png", x_t1)
        '''

def playGame():
    sess = tf.InteractiveSession()
    s, readout = createNetwork()
    trainNetwork(s, readout, sess)

def main():
    playGame()

if __name__ == "__main__":
    main()
