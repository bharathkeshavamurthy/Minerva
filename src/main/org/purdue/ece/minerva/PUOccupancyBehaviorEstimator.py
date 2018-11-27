# PU Occupancy Behavior Estimation
# Author: Bharath Keshavamurthy
# School of Electrical and Computer Engineering
# Purdue University
# Copyright (c) 2018. All Rights Reserved.

# For the math behind this algorithm, refer to:
# https://github.rcac.purdue.edu/bkeshava/Minerva/blob/master/SystemModelAndEstimator_v3_3_0.pdf

from enum import Enum
import numpy
import scipy.stats
from matplotlib import pyplot as plt
from collections import namedtuple


# Occupancy state enumeration
class OccupancyState(Enum):
    # Channel Idle
    idle = 0
    # Channel Occupied
    occupied = 1


# Main class: PU Occupancy Behavior Estimation
class PUOccupancyBehaviorEstimator(object):
    # Number of samples for this simulation
    NUMBER_OF_SAMPLES = 1000

    # Variance of the Additive White Gaussian Noise Samples
    VARIANCE_OF_AWGN = 1

    # Variance of the Channel Impulse Response which is a zero mean Gaussian
    VARIANCE_OF_CHANNEL_IMPULSE_RESPONSE = 80

    # Number of frequency bands/channels in the wideband spectrum of interest
    NUMBER_OF_FREQUENCY_BANDS = 18

    # Start probabilities of PU occupancy per frequency band
    BAND_START_PROBABILITIES = namedtuple('BandStartProbabilities', ['idle', 'occupied'])

    # Occupancy States (IDLE, OCCUPIED)
    OCCUPANCY_STATES = (OccupancyState.idle, OccupancyState.occupied)

    # Value function named tuple
    VALUE_FUNCTION_NAMED_TUPLE = namedtuple('ValueFunction', ['current_value', 'previous_state'])

    # Number of trials to smoothen the Detection Accuracy v/s P(1|0) curve
    NUMBER_OF_CYCLES = 50

    # Initialization Sequence
    def __init__(self):
        print('[INFO] PUOccupancyBehaviorEstimator Initialization: Bringing things up ...')
        # AWGN samples
        self.noise_samples = {}
        # Channel Impulse Response samples
        self.channel_impulse_response_samples = {}
        # True PU Occupancy state
        self.true_pu_occupancy_states = []
        # The parameters of observations of all the bands are stored in this dict here
        self.observations_parameters = {}
        # The observed samples at the SU receiver
        self.observation_samples = []
        # The start probabilities
        self.start_probabilities = self.BAND_START_PROBABILITIES(occupied=0.6, idle=0.4)
        # The complete state transition matrix is defined as a Python dictionary taking in named-tuples
        self.transition_probabilities_matrix = {}

    # Generate the true states using the Markovian model
    # Arguments: p -> P(1|0); q -> P(0|1); and pi -> P(1)
    def allocate_true_pu_occupancy_states(self, p_val, q_val, pi_val):
        previous = 1
        # Initial state generation -> band-0 using pi_val
        if numpy.random.random_sample() > pi_val:
            previous = 0
        self.true_pu_occupancy_states.append(previous)
        # Based on the state of band-0 and the (p_val,q_val) values, generate the states of the remaining bands
        for loop_counter in range(1, self.NUMBER_OF_FREQUENCY_BANDS):
            seed = numpy.random.random_sample()
            if previous == 1 and seed < q_val:
                previous = 0
            elif previous == 1 and seed > q_val:
                previous = 1
            elif previous == 0 and seed < p_val:
                previous = 1
            else:
                previous = 0
            self.true_pu_occupancy_states.append(previous)

    # Get the observations vector
    # Generate the observations of all the bands for a number of observation rounds or cycles
    def allocate_observations(self):
        # Each frequency band is observed ${NUMBER_OF_SAMPLES} times.
        # The sampling of each frequency band involves a noise sample corresponding to that SU receiver
        # ... and a channel impulse response sample between that SU receiver and the radio environment
        for frequency_band in range(0, self.NUMBER_OF_FREQUENCY_BANDS):
            mu_noise, std_noise = 0, numpy.sqrt(self.VARIANCE_OF_AWGN)
            self.noise_samples[frequency_band] = numpy.random.normal(mu_noise, std_noise, self.NUMBER_OF_SAMPLES)
            mu_channel_impulse_response, std_channel_impulse_response = 0, numpy.sqrt(
                self.VARIANCE_OF_CHANNEL_IMPULSE_RESPONSE)
            self.channel_impulse_response_samples[frequency_band] = numpy.random.normal(mu_channel_impulse_response,
                                                                                        std_channel_impulse_response,
                                                                                        self.NUMBER_OF_SAMPLES)
        # Re-arranging the vectors
        for band in range(0, self.NUMBER_OF_FREQUENCY_BANDS):
            obs_per_band = list()
            for count in range(0, self.NUMBER_OF_SAMPLES):
                obs_per_band.append(self.channel_impulse_response_samples[band][
                                        count] * self.true_pu_occupancy_states[band] + self.noise_samples[
                                        band][count])
            self.observation_samples.append(obs_per_band)
        return self.observation_samples

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
        return scipy.stats.norm(0, numpy.sqrt(
            (self.VARIANCE_OF_CHANNEL_IMPULSE_RESPONSE * state) + self.VARIANCE_OF_AWGN)).pdf(
            observation_sample)

    # Calculate the detection accuracy
    def get_detection_accuracy(self, estimated_states):
        accuracies = 0
        for _counter in range(0, self.NUMBER_OF_FREQUENCY_BANDS):
            if self.true_pu_occupancy_states[_counter] == estimated_states[_counter]:
                accuracies += 1
        return accuracies / self.NUMBER_OF_FREQUENCY_BANDS

    # Output the estimated state of the frequency bands in the wideband spectrum of interest
    def estimate_pu_occupancy_states(self):
        estimated_states_array = []
        for sampling_round in range(0, self.NUMBER_OF_SAMPLES):
            estimated_states = []
            reduced_observation_vector = []
            for entry in self.observation_samples:
                reduced_observation_vector.append(entry[sampling_round])
            # Now, I have to estimate the state of the ${NUMBER_OF_FREQUENCY_BANDS} based on
            # ...this reduced observation vector
            # INITIALIZATION : The array of initial probabilities is known\
            # FORWARD RECURSION
            value_function_collection = [dict() for x in range(len(reduced_observation_vector))]
            for state in OccupancyState:
                current_value = self.get_emission_probabilities(state.value, reduced_observation_vector[0]) * \
                                self.get_start_probabilities(state.name)
                value_function_collection[0][state.name] = self.VALUE_FUNCTION_NAMED_TUPLE(current_value=current_value,
                                                                                           previous_state=None)
            # For each observation after the first one (I can't apply Markovian to [0])
            for observation_index in range(1, len(reduced_observation_vector)):
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
                            pointer = self.get_transition_probabilities(candidate_previous_state.value, state.value) * \
                                      value_function_collection[observation_index - 1][
                                          candidate_previous_state.name].current_value
                            if pointer > max_pointer:
                                max_pointer = pointer
                                confirmed_previous_state = candidate_previous_state.name
                    current_value = max_pointer * self.get_emission_probabilities(state.value,
                                                                                  reduced_observation_vector[
                                                                                      observation_index])
                    value_function_collection[observation_index][state.name] = self.VALUE_FUNCTION_NAMED_TUPLE(
                        current_value=current_value, previous_state=confirmed_previous_state)
            max_value = 0
            # Finding the max value among the named tuples
            for value in value_function_collection[-1].values():
                if value.current_value > max_value:
                    max_value = value.current_value
            # Finding the state corresponding to this max_value and using this as the final confirmed state to ...
            # ...backtrack and find the previous states
            for k, v in value_function_collection[-1].items():
                if v.current_value == max_value:
                    estimated_states.append(self.value_from_name(k))
                    previous_state = k
                    break
            # Backtracking
            for i in range(len(value_function_collection) - 2, -1, -1):
                estimated_states.insert(0, self.value_from_name(
                    value_function_collection[i + 1][previous_state].previous_state))
                previous_state = value_function_collection[i + 1][previous_state].previous_state
            estimated_states_array.append(estimated_states)
        detection_accuracies = [
            self.get_detection_accuracy(estimated_states_per_iteration) for
            estimated_states_per_iteration in estimated_states_array]
        sum_for_average = 0
        for accuracy_entry in detection_accuracies:
            sum_for_average += accuracy_entry
        return sum_for_average / self.NUMBER_OF_SAMPLES

    # Get enumeration field value from name
    @staticmethod
    def value_from_name(name):
        if name == 'occupied':
            return OccupancyState.occupied.value
        else:
            return OccupancyState.idle.value

    # Reset every collection for the next run
    def reset(self):
        self.true_pu_occupancy_states.clear()
        self.transition_probabilities_matrix.clear()
        self.noise_samples.clear()
        self.channel_impulse_response_samples.clear()
        self.observation_samples.clear()

    # Exit strategy
    def __exit__(self, exc_type, exc_val, exc_tb):
        print('[INFO] PUOccupancyBehaviorEstimator Clean-up: Cleaning things up ...')


# Run Trigger
if __name__ == '__main__':
    # Increment p by this value
    increment_value_for_p = 0.030
    # X-Axis for the P(1|0) versus Detection Accuracy plot
    p_values_overall = []
    # Y-Axis for the P(1|0) versus Detection Accuracy plot
    detection_accuracies_across_p_values_overall = []
    print(
        '[INFO] PUOccupancyBehaviorEstimator main: Creating an instance and starting the initialization process ...')
    puOccupancyBehaviorEstimation = PUOccupancyBehaviorEstimator()
    # P(1)
    pi = puOccupancyBehaviorEstimation.start_probabilities.occupied
    final_frontier = int(pi / increment_value_for_p)
    for iteration_cycle in range(0, puOccupancyBehaviorEstimation.NUMBER_OF_CYCLES):
        # P(1|0) - Let's start small and move towards independence
        p = increment_value_for_p
        p_values = []
        detection_accuracies_across_p_values = []
        for increment_counter in range(0, final_frontier):
            print('[INFO] PUOccupancyBehaviorEstimator main: Pass [', increment_counter, ']')
            p_values.append(p)
            # P(0|1)
            q = (p * puOccupancyBehaviorEstimation.start_probabilities.idle) \
                / puOccupancyBehaviorEstimation.start_probabilities.occupied
            puOccupancyBehaviorEstimation.transition_probabilities_matrix = {
                1: {1: (1 - q), 0: q},
                0: {1: p, 0: (1 - p)}
            }
            # True PU Occupancy State
            puOccupancyBehaviorEstimation.allocate_true_pu_occupancy_states(p, q, pi)
            puOccupancyBehaviorEstimation.allocate_observations()
            print(
                '[INFO] PUOccupancyBehaviorEstimator main: Now, '
                'let us estimate the PU occupancy states in these frequency bands for p = ', p)
            detection_accuracy_per_p = puOccupancyBehaviorEstimation.estimate_pu_occupancy_states()
            print('[INFO] PUOccupancyBehaviorEstimator main: p = ', p, ' Detection Accuracy: ',
                  detection_accuracy_per_p)
            detection_accuracies_across_p_values.append(detection_accuracy_per_p)
            p += increment_value_for_p
            print('[INFO] PUOccupancyBehaviorEstimator main: Reset everything and Re-initialize for the next pass ...')
            puOccupancyBehaviorEstimation.reset()
        p_values_overall = p_values
        detection_accuracies_across_p_values_overall.append(detection_accuracies_across_p_values)
    final_detection_accuracy_array_for_averaging = []
    # I've run multiple passes of the logic for smoothening the curve
    # Now, average over them and plot it
    for pass_counter in range(0, final_frontier):
        _sum = 0
        for _entry in detection_accuracies_across_p_values_overall:
            _sum += _entry[pass_counter]
        final_detection_accuracy_array_for_averaging.append(_sum / puOccupancyBehaviorEstimation.NUMBER_OF_CYCLES)
    fig, ax = plt.subplots()
    ax.plot(p_values_overall, final_detection_accuracy_array_for_averaging, linestyle='--', linewidth=1.0, marker='o',
            color='r')
    fig.suptitle('Detection Accuracy v/s P(Occupied | Idle) at P( Xi = 1 ) = 0.6', fontsize=20)
    ax.set_xlabel('P(Occupied | Idle)', fontsize=12)
    ax.set_ylabel('Detection Accuracy', fontsize=12)
    plt.show()
