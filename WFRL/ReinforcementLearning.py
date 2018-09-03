# -*- coding: UTF-8 -*-
# Use reinforcement learning to let a robot learn to do wall following
# Magnus Lindhé, 2018

# See https://github.com/DanielSlater/PyGamePlayer/blob/master/examples/deep_q_pong_player.py for an example of how
# to interface this to TensorFlow.
# Or refer to its source:
# https://github.com/asrivat1/DeepLearningVideoGames

def epsilon(episodeIndex):
    """Return the probability of a random action, given the episode index."""

    # This should linearly decrease from 1.0 to 0.1 as the training progresses.
    return 0.1

# No of episodes (cleaning runs) to run
N_episodes = 1

# No of time steps (=robot moves) in each episode
T = 100

# No of experience tuples to use in each SGD step (The Atari paper uses 32.)
N_batch = 32

# Initialize Q with random weights
Q = 0
# Clear replay memory D
D = []

for episode in range(1,N_episodes):
    # Create a new Robot (with a given start pose and fresh World)

    # Every C episodes, we update the driving policy to use the current weights for its Q-value function.
    if episode % C == 0:
        Q_hat = Q

    for t in range(1,T):
        # Use the policy based on Q to decide what action to take, given the state (=world map, relative to the robot).
        # (Or, with probability epsilon, pick a random action.)

        # Perform the action on the robot and get the reward and new state

        # Save the (s,a,r,s') experience tuple to the replay memory D

        if len(D) > 50000:
            # Get a random minibatch of N_batch experiences and do a SGD step to improve the weights for Q,
            # using the semi-static value function Q_hat to compute the target.
            print('SGD step')
