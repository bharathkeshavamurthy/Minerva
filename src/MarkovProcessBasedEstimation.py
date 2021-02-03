# This entity describes an agent for Markov Process based Estimation (MPE) of the Occupancy States of channels in
#   the discretized spectrum of interest for our Cognitive Radio Research.
# Author: Bharath Keshavamurthy
# Organization: School of Electrical and Computer Engineering, Purdue University, West Lafayette, IN.
# Copyright (c) 2019. All Rights Reserved.

# Using the same notations as in the reference paper...
# Reference: M Gao, et. al., "Fast Spectrum Sensing: A Combination of Channel Correlation and Markov Model", 2014.

# This is a first order Markov process estimation algorithm.

# The imports
import numpy
from enum import Enum


# The Occupancy State Enumeration
class OccupancyState(Enum):
    # The channel is idle $X_{n}(m) = 0$
    IDLE = 0
    # The channel is occupied $X_{n}(m) = 1$
    OCCUPIED = 1


# This class describes the Markov Process based Estimation technique for determining the occupancy states of channels
#   in the discretized spectrum of interest. This agent will be used in tandem with the Channel Correlation based
#   sensing agents detailed in GreedyClusteringAlgorithm.py and MinimumEntropyIncrementAlgorithm.py in a Minimum
#   Entropy Merging (MEM) framework.
class MarkovProcessBasedEstimation(object):

    # The number of channels $N$ in the discretized spectrum of interest
    NUMBER_OF_CHANNELS = 18

    # The number of episodes $M$ of SU sensing/interaction with the radio environment
    NUMBER_OF_EPISODES = 1000

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
        # The occupancy states of band-0 across all episodes $X_{0}(m),\ \forall m \in \{1, 2, 3, \dots, M\}$
        for episode in range(1, self.NUMBER_OF_EPISODES):
            random_sample = numpy.random.random_sample()
            if state == 0 and random_sample < self.transition_probabilities[0][1]:
                state = 1
            elif state == 0 and random_sample > self.transition_probabilities[0][1]:
                state = 0
            elif state == 1 and random_sample < self.transition_probabilities[1][0]:
                state = 0
            else:
                state = 1
            pu_occupancy_states[0].append(state)
        # Choose the appropriate cell as the previous state for the upcoming analysis
        state = pu_occupancy_states[0][0]
        # The occupancy states of all channels in episode-0 $X_{n}(0),\ \forall n \in \{1, 2, 3, \dots, N\}$
        for channel in range(1, self.NUMBER_OF_CHANNELS):
            random_sample = numpy.random.random_sample()
            if state == 0 and random_sample < self.transition_probabilities[0][1]:
                state = 1
            elif state == 0 and random_sample > self.transition_probabilities[0][1]:
                state = 0
            elif state == 1 and random_sample < self.transition_probabilities[1][0]:
                state = 0
            else:
                state = 1
            pu_occupancy_states.append([state])
        # The occupancy states of the rest of the N x M matrix $X_{n}(m)\ \forall n \in \{1, 2, 3, \dots, N\}\ and\
        #   m \in \{1, 2, 3, \dots, M\}$
        for channel in range(1, self.NUMBER_OF_CHANNELS):
            for episode in range(1, self.NUMBER_OF_EPISODES):
                previous_spatial_state = pu_occupancy_states[channel - 1][episode]
                previous_temporal_state = pu_occupancy_states[channel][episode - 1]
                occupied_probability = \
                    self.transition_probabilities[previous_spatial_state][1] * \
                    self.transition_probabilities[previous_temporal_state][1]
                if numpy.random.random_sample() < occupied_probability:
                    pu_occupancy_states[channel].append(1)
                else:
                    pu_occupancy_states[channel].append(0)
        return pu_occupancy_states

    # Estimate the Occupancy States of the Incumbents using the Markov Process based Estimation Algorithm
    # \[cs_{n}^{t} = \argmax_{x \in \{0,1\}}\ \{\mathbb{P}(x|\vec{cs}_n^{h} = y) \mathbb{P}(\vec{cs}_{n}^{h} = y)\}
    def estimate_occupancy_states(self):
        # The output to be returned
        estimated_occupancy_states = []
        # The supported occupancy states
        # FIXME: This seems pretentious...
        start_probabilities_supported_states = [self.start_probabilities[member.value] for member in OccupancyState]
        for channel in range(self.NUMBER_OF_CHANNELS):
            # $cs_{n}^{0*}$
            estimated_occupancy_states.append([max(range(0, len(start_probabilities_supported_states)),
                                                   key=lambda idx: start_probabilities_supported_states[idx]
                                                   )
                                               ]
                                              )
            for episode in range(1, self.NUMBER_OF_EPISODES):
                # The order of this temporal Markov process on an individual channel basis is 1
                # $\vec{cs}_{n}^{h} = cs_{n}^{t-1} = y$ [Remember, this is a first order Markov process]
                previous_state = estimated_occupancy_states[channel][episode - 1]
                posterior_probability_members = [
                    (self.transition_probabilities[previous_state][member.value] *
                     self.start_probabilities[previous_state]
                     ) for member in OccupancyState
                ]
                # $cs_{n}^{t*} = \argmax_{x \in \{0,1\}}\ \{\mathbb{P}(x|cs_{n}^{t-1}) \mathbb{P}(cs_{n}^{t-1} = y)\}$
                estimated_occupancy_states[channel].append(max(range(0, len(posterior_probability_members)),
                                                               key=lambda idx: posterior_probability_members[idx]))
        return estimated_occupancy_states

    # The termination sequence
    def __exit__(self, exc_type, exc_val, exc_tb):
        print('[INFO] MarkovProcessBasedEstimation Termination: Tearing things down...')
        # Nothing to do...


# Run Trigger
if __name__ == '__main__':
    print('[INFO] MarkovProcessBasedEstimation main: Triggering the estimation of the occupancy behavior of the '
          'incumbents in the radio environment using a first order Markov Process based Estimation (MPE)...')
    agent = MarkovProcessBasedEstimation()
    estimated_states = agent.estimate_occupancy_states()
    print('[INFO] MarkovProcessBasedEstimation main: The estimated occupancy states (using MPE) are - {}'.format(
        str(estimated_states)
    ))
