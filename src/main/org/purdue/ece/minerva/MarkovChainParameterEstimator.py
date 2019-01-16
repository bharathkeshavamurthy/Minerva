# This Python script encapsulates an algorithm to estimate the parameters of Markov chains used in our research
# Static PU with Markovian Correlation across the channel indices
# Author: Bharath Keshavamurthy
# Organization: School of Electrical and Computer Engineering, Purdue University
# Copyright (c) 2019. All Rights Reserved
# For the complete math behind this algorithm, refer to the following link.
# https://github.rcac.purdue.edu/bkeshava/Minerva/tree/master/latex
# This link is subject to change. Please contact the author at bkeshava@purdue.edu for more details.

# Imports - Additional Dependencies needed for this to work
from enum import Enum
from collections import namedtuple
import numpy
import scipy.stats
import TransientStateEstimator


# OccupancyState Enumeration
# Based on Energy Detection, E[|X_k(i)|^2] = 1, if Occupied ; else, E[|X_k(i)|^2] = 0
class OccupancyState(Enum):
    # Occupancy state IDLE
    idle = 0
    # Occupancy state OCCUPIED:
    occupied = 1


# Markov Chain Parameter Estimation Algorithm
class MarkovChainParameterEstimator(object):
    # Number of channels in the discretized spectrum of interest
    NUMBER_OF_FREQUENCY_BANDS = 18

    # Number of observations made by the SU during the simulation period
    NUMBER_OF_SAMPLES = 1000

    # Variance of the Additive White Gaussian Noise Samples
    VARIANCE_OF_AWGN = 1

    # Variance of the Channel Impulse Response which is a zero mean Gaussian
    # This channel and noise model gives an SNR ~ 19 dB
    VARIANCE_OF_CHANNEL_IMPULSE_RESPONSE = 80

    # Start probabilities of PU occupancy per frequency band
    BAND_START_PROBABILITIES = namedtuple('BandStartProbabilities', ['idle', 'occupied'])

    # Confidence bound for termination
    CONFIDENCE_BOUND = 100

    # Initialization sequence
    def __init__(self):
        print('[INFO] MarkovChainParameterEstimator Initialization: Bringing things up...')
        # AWGN samples
        self.noise_samples = {}
        # Channel Impulse Response samples
        self.channel_impulse_response_samples = {}
        # True PU Occupancy state
        self.true_pu_occupancy_states = []
        # The observed samples at the SU receiver
        self.observation_samples = []
        # The start probabilities
        self.start_probabilities = self.BAND_START_PROBABILITIES(occupied=0.6, idle=0.4)
        # The transition probabilities
        self.transition_probabilities = {0: {0: list(), 1: list()}, 1: {0: list(), 1: list()}}
        # The forward probabilities collection across simulation time
        self.forward_probabilities = [dict() for k in range(0, self.NUMBER_OF_FREQUENCY_BANDS)]
        # The backward probabilities collection across simulation time
        self.backward_probabilities = [dict() for k in range(0, self.NUMBER_OF_FREQUENCY_BANDS)]
        # Convergence check epsilon - a very small value > 0
        # Distance Metric
        self.epsilon = 0.001
        # Initial P(Occupied|Idle)
        self.p_initial = 0.0
        # Initial Forward Probabilities dict
        self.initial_forward_probabilities = None
        # Initial Backward Probabilities dict
        self.initial_backward_probabilities = None

    # Generate the true states using the Markovian model
    # Arguments: p -> P(1|0); q -> P(0|1); and pi -> P(1)
    def allocate_true_pu_occupancy_states(self, p_val, q_val, pi_val):
        print('[DEBUG] MarkovChainParameterEstimator Allocate_True_PU_Occupancy: Number of channels = ',
              str(self.NUMBER_OF_FREQUENCY_BANDS))
        print('[DEBUG] MarkovChainParameterEstimator Allocate_True_PU_Occupancy: P(Occupied) = ', str(pi_val))
        print('[DEBUG] MarkovChainParameterEstimator Allocate_True_PU_Occupancy: P(Occupied|Idle) = ', str(p_val))
        print('[DEBUG] MarkovChainParameterEstimator Allocate_True_PU_Occupancy: P(Idle|Occupied) = ', str(q_val))
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
        print('[INFO] MarkovChainParameterEstimator Allocate_True_PU_Occupancy: The true PU occupancy states are - ',
              str(self.true_pu_occupancy_states))

    # Get the observations vector
    # Generate the observations of all the bands for a number of observation rounds or cycles
    def allocate_observations(self):
        print('[DEBUG] MarkovChainParameterEstimator Allocate_Observations: Number of observation rounds = ',
              str(self.NUMBER_OF_SAMPLES))
        print('[DEBUG] MarkovChainParameterEstimator Allocate_Observations: Variance of AWGN = ',
              str(self.VARIANCE_OF_AWGN))
        print('[DEBUG] MarkovChainParameterEstimator Allocate_Observations: Variance of Channel Impulse Response = ',
              str(self.VARIANCE_OF_CHANNEL_IMPULSE_RESPONSE))
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
        print('[INFO] MarkovChainParameterEstimator Allocate_Observations: The allocated observations are - ',
              str(self.observation_samples))
        return self.observation_samples

    # Get the Emission Probabilities -> P(y|x)
    def get_emission_probabilities(self, _state, observation_sample):
        emission_probability = scipy.stats.norm(0, numpy.sqrt(
            (self.VARIANCE_OF_CHANNEL_IMPULSE_RESPONSE * _state) + self.VARIANCE_OF_AWGN)).pdf(
            observation_sample)
        print('[TRACE] MarkovChainParameterEstimator Emission_Probability: Emission Probability of observation [',
              str(observation_sample), '] given state [', str(_state), '] is [', str(emission_probability), ']')
        return emission_probability

    # Convergence check
    def has_it_converged(self, iteration):
        print('[DEBUG] MarkovChainParameterEstimator Convergence_Check: Using Distance Metric [', self.epsilon,
              '] for Iteration [', str(iteration), ']')
        # Obviously, not enough data to check for convergence
        if iteration == 1:
            return False
        for key, value in self.transition_probabilities.items():
            # I get 0: dict() and 1:dict()
            for k, v in value.items():
                # I get 0:list() and 1:list()
                if abs(v[iteration - 1] - v[iteration - 2]) > self.epsilon:
                    return False
        # All have converged
        return True

    # Construct the transition probabilities matrix in order to pass it to the state estimator
    def construct_transition_probabilities_matrix(self, iteration):
        # A constructed transition probabilities matrix
        interim_transition_probabilities_matrix = {
            0: {0: self.transition_probabilities[0][0][iteration - 1],
                1: self.transition_probabilities[0][1][iteration - 1]},
            1: {0: self.transition_probabilities[1][0][iteration - 1],
                1: self.transition_probabilities[1][1][iteration - 1]}
        }
        print(
            '[DEBUG] MarkovChainParameterEstimator Construct_Transition_Probabilities_Matrix: Passing the transition '
            'probability matrix for state estimation - ', str(interim_transition_probabilities_matrix))
        return interim_transition_probabilities_matrix

    # In-situ update of Forward Probabilities
    def fill_up_forward_probabilities(self, iteration, observation_vector):
        print('[DEBUG] MarkovChainParameterEstimator Calculate_Forward_Probabilities: Iteration - [', str(iteration),
              '] with observation vector - ', str(observation_vector))
        # j
        for channel_index in range(1, self.NUMBER_OF_FREQUENCY_BANDS):
            # state l for channel j
            for current_occupancy_state in OccupancyState:
                occupancy_state_sum = 0
                # for all states r of channel i
                for previous_occupancy_state in OccupancyState:
                    occupancy_state_sum += self.forward_probabilities[channel_index - 1][
                                               previous_occupancy_state.value] * \
                                           self.transition_probabilities[previous_occupancy_state.value][
                                               current_occupancy_state.value][
                                               iteration - 1] * self.get_emission_probabilities(
                        current_occupancy_state.value, observation_vector[channel_index])
                self.forward_probabilities[channel_index][current_occupancy_state.value] = occupancy_state_sum
        print('[INFO] MarkovChainParameterEstimator Calculate_Forward_Probabilities: New Forward Probabilities '
              'collection - ', str(self.forward_probabilities))

    # In-situ update of Backward Probabilities
    def fill_up_backward_probabilities(self, iteration, observation_vector):
        print('[DEBUG] MarkovChainParameterEstimator Calculate_Backward_Probabilities: Iteration - [', str(iteration),
              '] with observation vector - ', str(observation_vector))
        # j
        for channel_index in range(self.NUMBER_OF_FREQUENCY_BANDS - 2, -1, -1):
            # state r for channel i
            for previous_occupancy_state in OccupancyState:
                occupancy_state_sum = 0
                # state l for channel j+1
                for next_occupancy_state in OccupancyState:
                    occupancy_state_sum += self.backward_probabilities[channel_index + 1][next_occupancy_state.value] \
                                           * self.transition_probabilities[previous_occupancy_state.value]
                    [next_occupancy_state.value][iteration - 1] * self.get_emission_probabilities(next_occupancy_state.
                                                                                                  value,
                                                                                                  observation_vector[
                                                                                                      channel_index])
                self.backward_probabilities[channel_index][previous_occupancy_state.value] = occupancy_state_sum
        print('[INFO] MarkovChainParameterEstimator Calculate_Backward_Probabilities: New Backward Probabilities '
              'collection - ', str(self.backward_probabilities))

    # Get the numerator of the fraction in the Algorithm (refer to the LaTeX document for more information)
    def get_numerator(self, previous_state, next_state, iteration, _estimated_state_sequence):
        print('[INFO] MarkovChainParameterEstimator Calculate_Numerator: Previous_State - [', str(previous_state),
              '], Next_State - [', str(next_state), '], and Iteration - [', str(iteration), ']')
        print(
            '[INFO] MarkovChainParameterEstimator Calculate_Numerator: Estimated State Sequence from the '
            'Viterbi Algorithm - ', str(_estimated_state_sequence))
        transition_probability_of_the_previous_iteration = self.transition_probabilities[previous_state][next_state][
            iteration - 1]
        transition_sum = 0
        for _index in range(0, len(_estimated_state_sequence)):
            if _index == 0:
                past_state = _estimated_state_sequence[_index]
                continue
            else:
                if _estimated_state_sequence[_index] == next_state and past_state == previous_state:
                    if _index == self.NUMBER_OF_FREQUENCY_BANDS - 1:
                        transition_sum += self.forward_probabilities[_index - 1][previous_state]
                    else:
                        transition_sum += self.forward_probabilities[_index - 1][previous_state] * \
                                          self.backward_probabilities[_index + 1][next_state]
        return transition_probability_of_the_previous_iteration * transition_sum

    # Get the denominator of the fraction in the Algorithm (refer to the LaTeX document for more information)
    def get_denominator(self, previous_state, iteration, _estimated_state_sequence, observation_vector):
        denominator = 0
        for sample_state in OccupancyState:
            for observation in observation_vector:
                combined_product_value = self.get_emission_probabilities(sample_state.value, observation) * \
                                         self.transition_probabilities[previous_state][sample_state.value][
                                             iteration - 1]
                transition_sum = 0
                for _index in range(0, len(_estimated_state_sequence)):
                    if _index == 0:
                        past_state = _estimated_state_sequence[_index]
                        continue
                    else:
                        if _estimated_state_sequence[_index] == sample_state.value and past_state == previous_state:
                            if _index == self.NUMBER_OF_FREQUENCY_BANDS - 1:
                                transition_sum += self.forward_probabilities[_index - 1][previous_state]
                            else:
                                transition_sum += self.forward_probabilities[_index - 1][previous_state] * \
                                                  self.backward_probabilities[_index + 1][sample_state.value]
                denominator += combined_product_value * transition_sum
        return denominator

    # A controlled reset of all the collections for the next set of observations
    def controlled_reset(self):
        # Controlled reset of forward probabilities
        self.forward_probabilities = [dict() for k in range(0, self.NUMBER_OF_FREQUENCY_BANDS)]
        self.forward_probabilities.append(self.initial_forward_probabilities)
        # Controlled reset of backward probabilities
        self.backward_probabilities = [dict() for k in range(0, self.NUMBER_OF_FREQUENCY_BANDS)]
        self.backward_probabilities.append(self.initial_backward_probabilities)
        # Controlled reset of the transition probabilities matrix
        for key, value in self.transition_probabilities.items():
            for k, v in value.items():
                _new_list = list()
                _new_list.append(v[0])
                self.transition_probabilities[key][k] = _new_list

    # Get the converged transition probabilities matrix
    def get_converged_transition_matrix(self):
        _converged_transition_matrix = {0: dict(), 1: dict()}
        for key, value in self.transition_probabilities.items():
            for k, v in value.items():
                _converged_transition_matrix[key][k] = v[len(v) - 1]
        return _converged_transition_matrix

    # Core method
    # Estimate the Markov Chain State Transition Probabilities matrix
    def estimate_parameters(self, _state_estimator):
        # The initial values have been set for transition probabilities and for the forward and backward probabilities
        # A collection to hold all the estimated state transition probabilities matrices for averaging later
        collection_of_estimates = []
        # Let us first construct a reduced observation vector like we did for the Viterbi algorithm
        for _sampling_round in range(0, self.NUMBER_OF_SAMPLES):
            observation_vector = []
            for _channel in range(0, self.NUMBER_OF_FREQUENCY_BANDS):
                observation_vector.append(self.observation_samples[_channel][_sampling_round][0])
            _state_estimator.observation_samples = observation_vector
            # Confidence variable to determine when the algorithm has converged
            confidence = 0
            # Let us start estimating now that we have the reduced observation vector
            iteration = 1
            # Until convergence
            while self.has_it_converged(iteration) is False or confidence < self.CONFIDENCE_BOUND:
                # A confidence check
                convergence = self.has_it_converged(iteration)
                if convergence is True:
                    # It has converged. Doing this to ensure that the convergence is permanent.
                    confidence += 1
                else:
                    confidence = 0
                # Fill up the remaining elements of the forward probabilities array
                self.fill_up_forward_probabilities(iteration, observation_vector)
                # Fill up the remaining elements of the backward probabilities array
                self.fill_up_backward_probabilities(iteration, observation_vector)
                # Set the transition probabilities matrix
                _state_estimator.transition_probabilities_matrix = self.construct_transition_probabilities_matrix(
                    iteration)
                # Get the estimated state sequence based on the estimated transition probabilities
                _estimated_state_sequence = _state_estimator.estimate_pu_occupancy_states()
                # state r in {0, 1}
                # Total 4 combinations arise from this double loop
                for previous_state in OccupancyState:
                    # state l in {0, 1}
                    for next_state in OccupancyState:
                        numerator = self.get_numerator(previous_state.value, next_state.value, iteration,
                                                       _estimated_state_sequence)
                        denominator = self.get_denominator(previous_state.value, iteration, _estimated_state_sequence,
                                                           observation_vector)
                        self.transition_probabilities[previous_state.value][next_state.value].append(
                            numerator / denominator)
                iteration += 1
            # After convergence
            collection_of_estimates.append(self.get_converged_transition_matrix())
            # A controlled reset for the next set of observations
            self.controlled_reset()
        final_estimated_parameters = {0: dict(), 1: dict()}
        for prev_state in OccupancyState:
            for nxt_state in OccupancyState:
                sum_for_averaging = 0
                for estimate in collection_of_estimates:
                    sum_for_averaging += estimate[prev_state.value][nxt_state.value]
                final_estimated_parameters[prev_state.value][nxt_state.value] = sum_for_averaging / len(
                    collection_of_estimates)
        return final_estimated_parameters

    # Termination sequence
    def __exit__(self, exc_type, exc_val, exc_tb):
        print('[INFO] MarkovChainParameterEstimator Termination: Tearing things down...')


# Run Trigger
if __name__ == '__main__':
    print('[INFO] MarkovChainParameterEstimator main: Creating instance and beginning the process of parameter '
          'estimation')
    # Viterbi Algorithm - State Estimator
    stateEstimator = TransientStateEstimator.TransientStateEstimator()
    # Create the instance
    markovChainParameterEstimator = MarkovChainParameterEstimator()
    # Number of Channels in the discretized spectrum of interest
    number_of_channels = markovChainParameterEstimator.NUMBER_OF_FREQUENCY_BANDS
    stateEstimator.number_of_channels = number_of_channels
    # Number of samples (observation rounds)
    # NOTE: The PU is static in this case
    number_of_samples = markovChainParameterEstimator.NUMBER_OF_SAMPLES
    stateEstimator.number_of_samples = number_of_samples
    # Channel information
    stateEstimator.noise_variance = markovChainParameterEstimator.VARIANCE_OF_AWGN
    stateEstimator.channel_impulse_response_variance = markovChainParameterEstimator. \
        VARIANCE_OF_CHANNEL_IMPULSE_RESPONSE
    # Steady State Probability P(Occupied) = P(X_i = 1)
    pi = markovChainParameterEstimator.start_probabilities.occupied
    stateEstimator.start_probabilities = markovChainParameterEstimator.start_probabilities
    # P(Occupied | Idle)
    p = 0.3
    # P(Idle|Occupied)
    q = (p * (1 - pi)) / pi
    # Actual State Transition Probabilities Matrix
    actual_transition_probabilities_matrix = {
        1: {1: (1 - q), 0: q},
        0: {1: p, 0: (1 - p)}
    }
    # Generate the true PU Occupancy states based on the Markov chain parameters described above
    markovChainParameterEstimator.allocate_true_pu_occupancy_states(p, q, pi)
    # Allocate observations based on given Markov Model and Channel Model
    markovChainParameterEstimator.allocate_observations()
    # Before, we go ahead and estimate the state transition probabilities matrix, let's set them to some initial values
    # P(Occupied|Idle) initial assumption
    p_initial = 0.45
    markovChainParameterEstimator.p_initial = p_initial
    # P(Idle|Occupied) initial assumption
    q_initial = (p_initial * (1 - pi)) / pi
    # Set the initial values to the transition probabilities matrix
    markovChainParameterEstimator.transition_probabilities[0][0].append(1 - p_initial)
    markovChainParameterEstimator.transition_probabilities[0][1].append(p_initial)
    markovChainParameterEstimator.transition_probabilities[1][0].append(q_initial)
    markovChainParameterEstimator.transition_probabilities[1][1].append(1 - q_initial)
    # TODO: Extract the forward probabilities boundary condition evaluation to a separate method
    # Setting the Boundary conditions for Forward Probabilities
    # Occupied
    forward_boundary_condition_occupied = markovChainParameterEstimator.get_emission_probabilities(
        OccupancyState.occupied.value, markovChainParameterEstimator.observation_samples[0][
            0]) * markovChainParameterEstimator.start_probabilities.occupied
    # Idle
    forward_boundary_condition_idle = markovChainParameterEstimator.get_emission_probabilities(
        OccupancyState.idle.value, markovChainParameterEstimator.observation_samples[0][
            0]) * markovChainParameterEstimator.start_probabilities.idle
    temp_forward_probabilities_dict = {0: forward_boundary_condition_idle, 1: forward_boundary_condition_occupied}
    markovChainParameterEstimator.forward_probabilities[0] = temp_forward_probabilities_dict
    # Assign the evaluated boundary condition to initial_forward_probabilities
    markovChainParameterEstimator.initial_forward_probabilities = temp_forward_probabilities_dict
    # TODO: Extract the backward probabilities boundary condition evaluation to a separate method
    # Setting the Boundary conditions for Backward Probabilities
    # Occupied
    state_sum = 0
    # outside summation - refer to the definition of backward probability
    for state in OccupancyState:
        state_sum += markovChainParameterEstimator.transition_probabilities[1][state.value][0] * \
                     markovChainParameterEstimator.get_emission_probabilities(state.value,
                                                                              markovChainParameterEstimator.
                                                                              observation_samples[number_of_channels
                                                                                                  - 1][number_of_samples
                                                                                                       - 1])
    backward_boundary_condition_occupied = state_sum
    # Idle
    state_sum = 0
    # outside summation - refer to the definition of backward probability
    for state in OccupancyState:
        state_sum += markovChainParameterEstimator.transition_probabilities[0][state.value][
                         0] * markovChainParameterEstimator.get_emission_probabilities(state.value,
                                                                                       markovChainParameterEstimator.
                                                                                       observation_samples[
                                                                                           number_of_channels - 1][
                                                                                           number_of_samples - 1])
    backward_boundary_condition_idle = state_sum
    temp_backward_probabilities_dict = {0: backward_boundary_condition_idle,
                                        1: backward_boundary_condition_occupied}
    markovChainParameterEstimator.backward_probabilities[number_of_channels - 1] = temp_backward_probabilities_dict
    # Assign the evaluated boundary condition to the initial_backward_probabilities
    markovChainParameterEstimator.initial_backward_probabilities = temp_backward_probabilities_dict
    # Estimate the parameters of the Markov chain
    try:
        estimated_markov_chain_parameters = markovChainParameterEstimator.estimate_parameters(stateEstimator)
        print('[INFO] MarkovChainParameterEstimator main: Estimated Transition Probabilities Matrix is ',
              estimated_markov_chain_parameters)
    except Exception as e:
        print('[ERROR] MarkovChainParameterEstimation main: Exception caught in the core method - [', e, ']')
