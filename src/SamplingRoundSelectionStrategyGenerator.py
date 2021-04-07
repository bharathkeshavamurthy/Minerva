# This Python script deals with the generation of sampling round selection strategies for emulating episodes in which...
# ...the Multi-Armed Bandit or the RL agent chooses to sense channels in the wideband spectrum of interest.
# This is used in the "Dynamic PU with Incomplete information" extension of our model
# Author: Bharath Keshavamurthy
# Organization: School of Electrical and Computer Engineering, Purdue University
# Copyright (c) 2019. All Rights Reserved.

import random


# Sampling Round Selection Strategy Generator
class SamplingRoundSelectionStrategyGenerator(object):

    # Initialization sequence
    def __init__(self):
        print('[INFO] SamplingRoundSelectionStrategyGenerator Initialization: Bringing things up...')

    # Generic Uniform Sensing
    # Returns a collection of arrays consisting of iteratively uniformly spaced chosen sampling rounds / time indices
    # Emulates, in general, the rounds in which the MAB / RL agent is active
    # Maybe even emulates the sensing time-slot in a double POMDP formulation
    @staticmethod
    def generic_uniform_sensing(number_of_sampling_rounds):
        # Array of tuples with varying k
        channel_selection_strategies_based_on_uniform_sensing = []
        k = 0
        while k < number_of_sampling_rounds - 1:
            i = 0
            temp_array = []
            while i < number_of_sampling_rounds:
                temp_array.append(i)
                i = i + k + 1
            channel_selection_strategies_based_on_uniform_sensing.append(temp_array)
            k += 1
        return channel_selection_strategies_based_on_uniform_sensing

    # Generic Random Sensing
    # Returns a collection of arrays consisting of randomly spaced chosen sampling rounds / time indices
    # Emulates, in general, the rounds in which the MAB / RL agent is active
    # Maybe even emulates the sensing time-slot in a double POMDP formulation
    @staticmethod
    def generic_random_sensing(number_of_sampling_rounds, number_of_iterations):
        channel_selection_strategies_based_on_random_sensing = []
        for iteration in range(0, number_of_iterations):
            temp_array = []
            number_of_measurements = random.choice([k for k in range(0, number_of_sampling_rounds)])
            for i in range(0, number_of_measurements):
                temp_array.append(random.choice([k for k in range(0, number_of_sampling_rounds)]))
            channel_selection_strategies_based_on_random_sensing.append(temp_array)
        return channel_selection_strategies_based_on_random_sensing

    # Termination sequence
    def __exit__(self, exc_type, exc_val, exc_tb):
        print('[INFO] SamplingRoundSelectionStrategyGenerator Termination: Cleaning things up...')
