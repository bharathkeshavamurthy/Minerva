# This entity describes an agent for Markov Process based Estimation (MPE) of the Occupancy States of channels in
#   the discretized spectrum of interest for our Cognitive Radio Research.
# Author: Bharath Keshavamurthy
# Organization: School of Electrical and Computer Engineering, Purdue University, West Lafayette, IN.
# Copyright (c) 2019. All Rights Reserved.

# Using the same notations as in the reference paper...
# Reference: M Gao, et. al., "Fast Spectrum Sensing: A Combination of Channel Correlation and Markov Model", 2014.

# The imports
import numpy


# This class describes the Markov Process based Estimation technique for determining the occupancy states of channels
#   in the discretized spectrum of interest. This agent will be used in tandem with the Channel Correlation based
#   sensing agents detailed in GreedyClusteringAlgorithm.py and MinimumEntropyIncrementAlgorithm.py in a Minimum
#   Entropy Merging (MEM) framework.
class MarkovProcessBasedEstimation(object):

    # The number of channels $N$ in the discretized spectrum of interest
    NUMBER_OF_CHANNELS = 18

    # The number of episodes $M$ of SU sensing/interaction with the radio environment
    NUMBER_OF_EPISODES = 1000

    # The order of the temporal Markov chain $L$ for this Markov Process based Estimation
    ORDER_OF_TEMPORAL_MARKOV_CHAIN = 1

    # The initialization sequence
    def __init__(self):
        print('[INFO] MarkovProcessBasedEstimation Initialization: Bringing things up...')
        # The start probabilities a.k.a steady-state probabilities of both the spatial and the temporal Markov chains
        self.start_probabilities = {0: 0.4,
                                    1: 0.6
                                    }
        # The transition probabilities of both the spatial and the temporal Markov chains
        self.transition_probabilities = {0: {0: 0.7,
                                             1: 0.3},
                                         1: {0: 0.2,
                                             1: 0.8
                                             }
                                         }
        # The true occupancy states of the channels according to the behavior of the incumbents in the environment
        self.true_pu_occupancy_states = self.get_true_pu_occupancy_states()

    # Simulate the occupancy behavior of the incumbents in the radio environment
    def get_true_pu_occupancy_states(self):
        # The output to be returned
        pu_occupancy_states = []
        # The occupancy state of band-0 at time-0 $X_{0}(0)$
        state = 1
        if numpy.random.random_sample() > self.start_probabilities[1]:
            state = 0
        pu_occupancy_states.append([state])
        # The occupancy state of band-0 across all episodes $X_{0}(m),\ \forall m \in \{1, 2, 3, \dots, M\}$


    # The termination sequence
    def __exit__(self, exc_type, exc_val, exc_tb):
        print('[INFO] MarkovProcessBasedEstimation Termination: Tearing things down...')
        # Nothing to do...
