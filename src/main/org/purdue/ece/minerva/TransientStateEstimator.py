# This Python script encapsulates the Viterbi algorithm for state estimation
# This Python script is used in the Parameter Estimation algorithm
# The state estimation is transient/fleeting/temporary in that it varies as the transition probabilities matrix varies
# The estimation method is called until convergence
# Author: Bharath Keshavamurthy
# Organization: School of Electrical and Computer Engineering, Purdue University
# Copyright (c) 2019. All Rights Reserved.

import scipy.stats
import numpy
from collections import namedtuple
from enum import Enum


# Occupancy State simple class
class OccupancyState(Enum):
    # Idle channel
    idle = 0
    # Occupied channel
    occupied = 1


# Transient/Fleeting/Dynamic Viterbi Algorithm for PU Occupancy Behavior Estimation
class TransientStateEstimator(object):
    # Band Start Probabilities named tuple
    BAND_START_PROBABILITIES = namedtuple('Band_Start_Probabilities', ['occupied', 'idle'])

    # Value function named tuple
    VALUE_FUNCTION_NAMED_TUPLE = namedtuple('ValueFunction', ['current_value', 'previous_state'])

    # Initialization sequence
    def __init__(self):
        print('[INFO] TransientStateEstimator Initialization: Bringing things up...')
        self.noise_variance = 1.0
        self.channel_impulse_response_variance = 1.0
        self.noise_mean = 0.0
        self.channel_impulse_response_mean = 0.0
        self.number_of_channels = 0
        self.number_of_samples = 0
        self.observation_samples = []
        self.transition_probabilities_matrix = {}
        self.start_probabilities = self.BAND_START_PROBABILITIES(occupied=0.0, idle=0.0)

    # Get the start probabilities from the named tuple - a simple getter utility method exclusive to this class
    def get_start_probabilities(self, state_name):
        if state_name == 'occupied':
            return self.start_probabilities.occupied
        else:
            return self.start_probabilities.idle

    # Return the transition probabilities from the transition probabilities matrix
    def get_transition_probabilities(self, row, column):
        return self.transition_probabilities_matrix[row][column]

    # Get the Emission Probabilities -> P(y|x)
    def get_emission_probabilities(self, state, observation_sample):
        # TODO: If the mean is non-zero, change this!
        return scipy.stats.norm(0, numpy.sqrt(
            (self.channel_impulse_response_variance * state) + self.noise_variance)).pdf(
            observation_sample)

    # Output the estimated state of the frequency bands in the wideband spectrum of interest
    def estimate_pu_occupancy_states(self):
        estimated_states = []
        value_function_collection = [dict() for x in range(len(self.observation_samples))]
        for state in OccupancyState:
            current_value = self.get_emission_probabilities(state.value, self.observation_samples[0]) * \
                            self.get_start_probabilities(state.name)
            value_function_collection[0][state.name] = self.VALUE_FUNCTION_NAMED_TUPLE(
                current_value=current_value,
                previous_state=None)
        for observation_index in range(1, len(self.observation_samples)):
            # Trying to find the max pointer here ...
            for state in OccupancyState:
                # Again finishing off the [0] index first
                max_pointer = self.get_transition_probabilities(OccupancyState.idle.value, state.value) * \
                              value_function_collection[observation_index - 1][
                                  OccupancyState.idle.name].current_value
                confirmed_previous_state = OccupancyState.idle.name
                for candidate_previous_state in OccupancyState:
                    if candidate_previous_state.name == OccupancyState.idle.name:
                        # Already done
                        continue
                    else:
                        pointer = self.get_transition_probabilities(candidate_previous_state.value,
                                                                    state.value) * \
                                  value_function_collection[observation_index - 1][
                                      candidate_previous_state.name].current_value
                        if pointer > max_pointer:
                            max_pointer = pointer
                            confirmed_previous_state = candidate_previous_state.name
                current_value = max_pointer * self.get_emission_probabilities(state.value,
                                                                              self.observation_samples[
                                                                                  observation_index])
                value_function_collection[observation_index][state.name] = self.VALUE_FUNCTION_NAMED_TUPLE(
                    current_value=current_value, previous_state=confirmed_previous_state)
        max_value = 0
        for value in value_function_collection[-1].values():
            if value.current_value > max_value:
                max_value = value.current_value
        for k, v in value_function_collection[-1].items():
            if v.current_value == max_value:
                estimated_states.append(self.value_from_name(k))
                previous_state = k
                break
        for i in range(len(value_function_collection) - 2, -1, -1):
            estimated_states.insert(0, self.value_from_name(
                value_function_collection[i + 1][previous_state].previous_state))
            previous_state = value_function_collection[i + 1][previous_state].previous_state
        return estimated_states

    # Get enumeration field value from name
    @staticmethod
    def value_from_name(name):
        if name == 'occupied':
            return OccupancyState.occupied.value
        else:
            return OccupancyState.idle.value

    # Termination sequence
    def __exit__(self, exc_type, exc_val, exc_tb):
        print('[INFO] TransientStateEstimator Termination: Tearing things down...')
