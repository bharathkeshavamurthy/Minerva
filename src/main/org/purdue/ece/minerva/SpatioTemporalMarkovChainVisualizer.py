# A Python Script that visualizes the Double Markov Chain
# A Markov Chain across the frequency sub-bands
# A Markov Chain across time (sampling rounds or iterations)
# Author: Bharath Keshavamurthy
# School of Electrical and Computer Engineering
# Purdue University
# Copyright (c) 2018. All Rights Reserved.


class SpatioTemporalMarkovChainVisualizer(object):
    # P(Occupied|Idle) = p = 0.3 (some sample value)
    # I'm gonna use this sample for p across both time indices and channel indices
    SAMPLE_VALUE_OF_p = 0.3

    # P(Occupied) = 0.6
    # I'm gonna use this sample for PI across both time indices and channel indices
    SAMPLE_VALUE_OF_PI = 0.6

    # Initialization
    def __init__(self):
        print('[INFO] SpatioTemporalMarkovChainVisualizer Initialization: Bringing things up...')
        self.p = self.SAMPLE_VALUE_OF_p
        self.pi = self.SAMPLE_VALUE_OF_PI
        # P(0|1)
        self.q = (self.p * (1 - self.pi)) / self.pi

    # Visualization: Core method
    def visualize(self):
        temporal_transition_probability_matrix = {
            1: {1: (1 - self.q), 0: self.q},
            0: {1: self.p, 0: (1 - self.p)}
        }
