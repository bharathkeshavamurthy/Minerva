# This entity details the Minimum Entropy Merging (MEM) Algorithm for estimating the occupancy states of channels in the
#   discretized spectrum of interest for our Cognitive Radio Research.
# Author: Bharath Keshavamurthy
# Organization: School of Electrical and Computer Engineering, Purdue University, West Lafayette, IN.
# Copyright (c) 2019. All Rights Reserved.

# Using the same notations as in the reference paper...
# Reference: M Gao, et. al., "Fast Spectrum Sensing: A Combination of Channel Correlation and Markov Model", 2014.

# Visualizations:
#   1. Utility v Episodes for a Minimum Entropy Merging algorithm with Greedy Clustering as the Channel Correlation
#       technique and a First Order Temporal Markov Process based Estimation strategy
#   2. Utility v Episodes for a Minimum Entropy Merging algorithm with Minimum Entropy Increment Clustering as the
#       Channel Correlation technique and a First Order Temporal Markov Process based Estimation strategy

# The imports
import numpy
from enum import Enum
from collections import namedtuple


# This class describes the evaluation of the Minimum Entropy Merging (MEM) Algorithm which involves a combined strategy
#   incorporating both Channel Correlation based Estimation (CCE) and Markov Process based Estimation (MPE).
class MinimumEntropyMerging(object):

    # The initialization sequence
    def __init__(self):
        print('[INFO] MinimumEntropyMerging Initialization: Bringing things up...')

    # The termination sequence
    def __exit__(self, exc_type, exc_val, exc_tb):
        print('[INFO] MinimumEntropyMerging Termination: Tearing things down...')
        # Nothing to do...


# Run Trigger
if __name__ == '__main__':
    print('[INFO] MinimumEntropyMerging main: Triggering the evaluation of the Minimum Entropy Merging (MEM) Algorithm '
          'for utility maximization in our model of a Cognitive Radio Network with one SU and multiple PUs...')
    agent = MinimumEntropyMerging()
    agent.evaluate()
