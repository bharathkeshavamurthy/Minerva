# Channel Selection Strategy Generator
# Author: Bharath Keshavamurthy
# School of Electrical and Computer Engineering
# Purdue University
# Copyright (c) 2018. All Rights Reserved.

import PUOccupancyBehaviorEstimatorII
import random


# Channel Selection Strategy Generator
# Emulates the Multi-Armed Bandit
class ChannelSelectionStrategyGenerator(object):
    # Number of channels in the wide-band spectrum of interest
    NUMBER_OF_CHANNELS = PUOccupancyBehaviorEstimatorII.PUOccupancyBehaviorEstimatorII.NUMBER_OF_FREQUENCY_BANDS

    # Number of iterations for random sensing
    NUMBER_OF_ITERATIONS = 8

    # Initialization
    def __init__(self):
        print('[INFO] ChannelSelectionStrategyGenerator Initialization: Bringing things up...')
        self.discretized_spectrum = [k for k in range(0, self.NUMBER_OF_CHANNELS)]

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

    # Random Sensing
    def random_sensing(self):
        channel_selection_strategies_based_on_random_sensing = []
        for iteration in range(0, self.NUMBER_OF_ITERATIONS):
            temp_array = []
            number_of_measurements = random.choice(self.discretized_spectrum)
            for i in range(0, number_of_measurements):
                temp_array.append(random.choice(self.discretized_spectrum))
            channel_selection_strategies_based_on_random_sensing.append(temp_array)
        return channel_selection_strategies_based_on_random_sensing

    # Return the duals of the channels selected by uniform sensing
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
