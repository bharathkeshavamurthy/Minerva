# This Python script emulates the operation of an RL agent / a Multi-Armed Bandit which makes recommendations on...
# ...which channels to sense in the wideband spectrum of interest based on our defined optimization objective.
# Channel Selection Strategy Generator
# Author: Bharath Keshavamurthy
# Organization: School of Electrical and Computer Engineering, Purdue University
# Copyright (c) 2019. All Rights Reserved.

# Some members in this class have been deprecated.

import PUOccupancyBehaviorEstimatorII
import random
import warnings
import functools


# This is a decorator which can be used to mark functions as deprecated.
# It will result in a warning being emitted when the function is used.
def deprecated(func):
    @functools.wraps(func)
    def new_func(*args, **kwargs):
        warnings.simplefilter('always', DeprecationWarning)
        warnings.warn("Call to deprecated function {}.".format(func.__name__),
                      category=DeprecationWarning,
                      stacklevel=2)
        warnings.simplefilter('default', DeprecationWarning)  # reset filter
        return func(*args, **kwargs)

    return new_func


# Channel Selection Strategy Generator
# Emulates an RL agent or a Multi-Armed Bandit
class ChannelSelectionStrategyGenerator(object):
    # Number of channels in the wide-band spectrum of interest
    NUMBER_OF_CHANNELS = PUOccupancyBehaviorEstimatorII.PUOccupancyBehaviorEstimatorII.NUMBER_OF_FREQUENCY_BANDS

    # Number of iterations for random sensing
    NUMBER_OF_ITERATIONS = 8

    # Initialization
    def __init__(self):
        print('[INFO] ChannelSelectionStrategyGenerator Initialization: Bringing things up...')
        self.discretized_spectrum = [k for k in range(0, self.NUMBER_OF_CHANNELS)]
        # I'm saving this in order to evaluate the duals at a later stage
        self.random_sensing_strategy = []

    # Uniform Sensing
    def uniform_sensing(self):
        # Array of tuples with varying k
        channel_selection_strategies_based_on_uniform_sensing = []
        k = 0
        while k < self.NUMBER_OF_CHANNELS - 1:
            i = 0
            temp_array = []
            while i < self.NUMBER_OF_CHANNELS:
                temp_array.append(i)
                i = i + k + 1
            channel_selection_strategies_based_on_uniform_sensing.append(temp_array)
            k += 1
        return channel_selection_strategies_based_on_uniform_sensing

    # Uniform Sensing Generic - Take in the number of channels as an argument
    @staticmethod
    def generic_uniform_sensing(number_of_channels):
        # Array of tuples with varying k
        channel_selection_strategies_based_on_uniform_sensing = []
        k = 0
        while k < number_of_channels - 1:
            i = 0
            temp_array = []
            while i < number_of_channels:
                temp_array.append(i)
                i = i + k + 1
            channel_selection_strategies_based_on_uniform_sensing.append(temp_array)
            k += 1
        return channel_selection_strategies_based_on_uniform_sensing

    # Random Sensing
    def random_sensing(self):
        channel_selection_strategies_based_on_random_sensing = []
        for iteration in range(0, self.NUMBER_OF_ITERATIONS):
            temp_array = []
            number_of_measurements = random.choice(self.discretized_spectrum)
            for i in range(0, number_of_measurements):
                temp_array.append(random.choice(self.discretized_spectrum))
            channel_selection_strategies_based_on_random_sensing.append(temp_array)
        # Setting a instance-scope variable in order to evaluate the duals at a later stage
        self.random_sensing_strategy = channel_selection_strategies_based_on_random_sensing
        return channel_selection_strategies_based_on_random_sensing

    # Random Sensing Generic - Take in the number of channels and number of iterations as arguments
    @staticmethod
    def generic_random_sensing(number_of_channels, number_of_iterations):
        channel_selection_strategies_based_on_random_sensing = []
        for iteration in range(0, number_of_iterations):
            temp_array = []
            number_of_measurements = random.choice([k for k in range(0, number_of_channels)])
            for i in range(0, number_of_measurements):
                temp_array.append(random.choice([k for k in range(0, number_of_channels)]))
            channel_selection_strategies_based_on_random_sensing.append(temp_array)
        return channel_selection_strategies_based_on_random_sensing

    # Return the duals of the channels selected by uniform sensing
    # Duals have been rendered obsolete as of 29-Dec-2018
    # A different interpretation of detection accuracies for un-sensed channels has been developed
    @deprecated
    def uniform_sensing_duals(self):
        channel_selection_strategies_based_on_uniform_sensing = self.uniform_sensing()
        channel_selection_strategies_based_on_uniform_sensing_duals = []
        for entry in channel_selection_strategies_based_on_uniform_sensing:
            temp_array = []
            for channel in self.discretized_spectrum:
                if channel not in entry:
                    temp_array.append(channel)
            channel_selection_strategies_based_on_uniform_sensing_duals.append(temp_array)
        return channel_selection_strategies_based_on_uniform_sensing_duals

    # Return the duals of the channels selected by random sensing
    # Duals have been rendered obsolete as of 29-Dec-2018
    # A different interpretation of detection accuracies for un-sensed channels has been developed
    @deprecated
    def random_sensing_duals(self):
        channel_selection_strategies_based_on_random_sensing = self.random_sensing()
        channel_selection_strategies_based_on_random_sensing_duals = []
        for entry in channel_selection_strategies_based_on_random_sensing:
            temp_array = []
            for channel in self.discretized_spectrum:
                if channel not in entry:
                    temp_array.append(channel)
            channel_selection_strategies_based_on_random_sensing_duals.append(temp_array)
        return channel_selection_strategies_based_on_random_sensing_duals

    # Uniform Sensing with their Duals
    # Duals have been rendered obsolete as of 29-Dec-2018
    # A different interpretation of detection accuracies for un-sensed channels has been developed
    @deprecated
    def uniform_sensing_with_duals(self):
        channel_selection_strategies_based_on_uniform_sensing = self.uniform_sensing()
        channel_selection_strategies_based_on_uniform_sensing_with_duals = []
        for entry in channel_selection_strategies_based_on_uniform_sensing:
            temp_array = []
            for channel in self.discretized_spectrum:
                if channel not in entry:
                    temp_array.append(channel)
            channel_selection_strategies_based_on_uniform_sensing_with_duals.append((entry, temp_array))
        return channel_selection_strategies_based_on_uniform_sensing_with_duals

    # Random Sensing with their Duals
    # Duals have been rendered obsolete as of 29-Dec-2018
    # A different interpretation of detection accuracies for un-sensed channels has been developed
    @deprecated
    def random_sensing_with_duals(self):
        channel_selection_strategies_based_on_random_sensing = self.random_sensing()
        channel_selection_strategies_based_on_random_sensing_with_duals = []
        for entry in channel_selection_strategies_based_on_random_sensing:
            temp_array = []
            for channel in self.discretized_spectrum:
                if channel not in entry:
                    temp_array.append(channel)
            channel_selection_strategies_based_on_random_sensing_with_duals.append((entry, temp_array))
        return channel_selection_strategies_based_on_random_sensing_with_duals

    # Termination
    def __exit__(self, exc_type, exc_val, exc_tb):
        print('[INFO] ChannelSelectionStrategyGenerator Termination: Cleaning things up...')
