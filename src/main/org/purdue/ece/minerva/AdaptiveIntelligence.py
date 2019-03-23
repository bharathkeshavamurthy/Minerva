# This entity encapsulates the overall framework essential for this research
# This entity includes an Oracle (knows everything about everything and hence can choose the most optimal action at...
# ...any given time)
# This entity also brings in the EM Parameter Estimation algorithm, the Viterbi State Estimation algorithm, and the...
# ...PERSEUS Approximate Value Iteration algorithm for POMDPs
# Author: Bharath Keshavamurthy
# Organization: School of Electrical and Computer Engineering, Purdue University
# Copyright (c) 2019. All Rights Reserved.

from enum import Enum
from collections import namedtuple
import numpy
import random
import itertools
import scipy.stats


# Markovian Correlation Class Enumeration
class MarkovianCorrelationClass(Enum):
    # Markovian Correlation across channel indices
    spatial = 0
    # Markovian Correlation across time indices
    temporal = 1
    # Invalid
    invalid = 2


# OccupancyState Enumeration
# Based on Energy Detection, E[|X_k(i)|^2] = 1, if Occupied ; else, E[|X_k(i)|^2] = 0
class OccupancyState(Enum):
    # Occupancy state IDLE
    idle = 0
    # Occupancy state OCCUPIED:
    occupied = 1


# The Markov Chain object that can be used via extension or replication in order to imply Markovian correlation...
# ...across either the channel indices or the time indices
class MarkovChain(object):

    # Initialization sequence
    def __init__(self):
        print('[INFO] MarkovChain Initialization: Initializing the Markov Chain...')
        # The steady-state probabilities (a.k.a start probabilities for each channel / each sampling round...
        # ...independently in the Markov Chain)
        self.start_probabilities = dict()
        # The state transition probabilities matrix - Two states [Occupied and Idle]
        self.transition_probabilities = {0: dict(), 1: dict()}
        # Markovian Correlation Class member
        self.markovian_correlation_class = None

    # External exposition to set the Markovian Correlation Class
    def set_markovian_correlation_class(self, _markovian_correlation_class):
        if isinstance(_markovian_correlation_class, MarkovianCorrelationClass):
            self.markovian_correlation_class = _markovian_correlation_class
        else:
            print('[ERROR] MarkovChain set_markovian_correlation_class: Invalid Enumeration object! Proceeding with '
                  'the INVALID Enumeration member...')
            self.markovian_correlation_class = MarkovianCorrelationClass.invalid
        print('[INFO] MarkovChain set_markovian_correlation_class: Markovian Correlation Class - ',
              self.markovian_correlation_class)

    # External exposition to set the start probability parameter 'pi'
    def set_start_probability_parameter(self, pi):
        if 0 <= pi <= 1:
            self.start_probabilities[1] = pi
            self.start_probabilities[0] = 1 - pi
        else:
            print('[ERROR] MarkovChain set_start_probability_parameter: Invalid probability entry used! '
                  'Proceeding with default values...')
            # Default Values...
            self.start_probabilities = {0: 0.4, 1: 0.6}
        print('[INFO] MarkovChain set_start_probability_parameter: Start Probabilities - ',
              str(self.start_probabilities))

    # External exposition to set the transition probability parameter 'p'
    def set_transition_probability_parameter(self, p):
        if 0 <= p <= 1 and self.start_probabilities[1] is not None:
            # P(Occupied|Idle) = p
            self.transition_probabilities[0][1] = p
            # P(Idle|Idle) = 1 - p
            self.transition_probabilities[0][0] = 1 - p
            # P(Idle|Occupied) = {p(1-pi)}/pi
            self.transition_probabilities[1][0] = (p * (1 - self.start_probabilities[1])) / self.start_probabilities[1]
            # P(Occupied|Occupied) = 1 - P(Idle|Occupied)
            self.transition_probabilities[1][1] = 1 - self.transition_probabilities[1][0]
        else:
            print(
                '[ERROR] MarkovChain set_transition_probability_parameter: Error while populating the state transition '
                'probabilities matrix! Proceeding with default values...')
            # Default Values...
            self.transition_probabilities = {0: {1: 0.3, 0: 0.7}, 1: {0: 0.2, 1: 0.8}}
        print('[INFO] MarkovChain set_transition_probability_parameter: State Transition Probabilities Matrix - ',
              str(self.transition_probabilities))

    # Termination sequence
    def __exit__(self, exc_type, exc_val, exc_tb):
        print('[INFO] MarkovChain Termination: Tearing things down...')


# A Utility class for all to use...
class Util(object):

    # Initialization sequence
    def __init__(self):
        print('[INFO] Util Initialization: Bringing things up...')

    # Construct the complete start probabilities dictionary
    @staticmethod
    def construct_start_probabilities_dict(pi):
        return {0: (1 - pi), 1: pi}

    # Construct the complete transition probability matrix from P(Occupied|Idle), i.e. 'p' and P(Occupied), i.e. 'pi'
    @staticmethod
    def construct_transition_probability_matrix(p, pi):
        # P(Idle|Occupied)
        q = (p * (1 - pi)) / pi
        return {0: {1: p, 0: 1 - p}, 1: {0: q, 1: 1 - q}}

    # Generate the action set based on the SU sensing limitations and the Number of Channels in the discretized...
    # ...spectrum of interest
    @staticmethod
    def get_action_set(number_of_channels, limitation):
        # Return this
        action_set = []
        # Discretize the spectrum
        discretized_spectrum_of_interest = [k for k in range(0, number_of_channels)]
        # Combinatorial Analysis for #limitation channels
        combinations = itertools.combinations(discretized_spectrum_of_interest, limitation)
        for combination in combinations:
            action = [k - k for k in range(0, number_of_channels)]
            # Extract the digits (channels) from the consolidated combination extracted in the previous step
            channels = [int(digit) for digit in str(combination)]
            for channel in discretized_spectrum_of_interest:
                if channel in channels:
                    action[channel] = 1
            action_set.append(action)
        return action_set

    # Termination sequence
    def __exit__(self, exc_type, exc_val, exc_tb):
        print('[INFO] Util Termination: Tearing things down...')


# This entity encapsulates the Channel object - simulates the Channel, i.e. Complex AWGN and Complex Impulse Response
class Channel(object):

    # Initialization sequence
    def __init__(self, _number_of_channels, _number_of_sampling_rounds, _number_of_episodes, _noise_mean,
                 _noise_variance, _impulse_response_mean, _impulse_response_variance):
        print('[INFO] Channel Initialization: Bringing things up...')
        # Noise Statistics
        self.noise_mean = _noise_mean
        if self.noise_mean is not 0:
            print('[WARN] Channel Initialization: The system assumes Zero-Mean, Complex, Additive, White, Gaussian '
                  'Noise...')
            self.noise_mean = 0
        self.noise_variance = _noise_variance
        # Channel Impulse Response Statistics
        self.impulse_response_mean = _impulse_response_mean
        if self.impulse_response_mean is not 0:
            print('[WARN] Channel Initialization: The system assumes Zero-Mean, Complex, Gaussian Impulse Response...')
            self.impulse_response_mean = 0
        self.impulse_response_variance = _impulse_response_variance
        # Number of channels in the discretized spectrum of interest
        self.number_of_channels = _number_of_channels
        # Number of sampling rounds undertaken by the SU per episode
        self.number_of_sampling_rounds = _number_of_sampling_rounds
        # Number of episodes of in which the POMDP agent interacts with the radio environment
        self.number_of_episodes = _number_of_episodes
        # Channel Impulse Response used in the Observation Model
        self.impulse_response = self.get_impulse_response()
        # The Complex AWGN used in the Observation Model
        self.noise = self.get_noise()

    # Generate the Channel Impulse Response samples
    def get_impulse_response(self):
        channel_impulse_response_samples = []
        for k in range(0, self.number_of_episodes):
            channel_impulse_response_samples.append(dict())
        for episode in range(0, self.number_of_episodes):
            for frequency_band in range(0, self.number_of_channels):
                mu_channel_impulse_response, std_channel_impulse_response = self.impulse_response_mean, numpy.sqrt(
                    self.impulse_response_variance)
                # The Re and Im parts of the channel impulse response samples are IID and distributed as...
                # ...N(0,\sigma_H^2/2)
                real_channel_impulse_response_samples = numpy.random.normal(mu_channel_impulse_response,
                                                                            (std_channel_impulse_response /
                                                                             numpy.sqrt(2)),
                                                                            self.number_of_sampling_rounds)
                img_channel_impulse_response_samples = numpy.random.normal(mu_channel_impulse_response,
                                                                           (std_channel_impulse_response /
                                                                            numpy.sqrt(2)),
                                                                           self.number_of_sampling_rounds)
                channel_impulse_response_samples[episode][frequency_band] = real_channel_impulse_response_samples + (
                        1j * img_channel_impulse_response_samples)
        return channel_impulse_response_samples

    # Generate the Complex AWGN samples
    def get_noise(self):
        noise_samples = []
        for k in range(0, self.number_of_episodes):
            noise_samples.append(dict())
        for episode in range(0, self.number_of_episodes):
            for frequency_band in range(0, self.number_of_channels):
                mu_noise, std_noise = self.noise_mean, numpy.sqrt(self.noise_variance)
                # The Re and Im parts of the noise samples are IID and distributed as N(0,\sigma_V^2/2)
                real_noise_samples = numpy.random.normal(mu_noise, (std_noise / numpy.sqrt(2)),
                                                         self.number_of_sampling_rounds)
                img_noise_samples = numpy.random.normal(mu_noise, (std_noise / numpy.sqrt(2)),
                                                        self.number_of_sampling_rounds)
                noise_samples[episode][frequency_band] = real_noise_samples + (1j * img_noise_samples)
        return noise_samples

    # Termination sequence
    def __exit__(self, exc_type, exc_val, exc_tb):
        print('[INFO] Channel Termination: Tearing things down...')


# This class encapsulates the Licensed User dynamically occupying the discretized spectrum under analysis
class PrimaryUser(object):

    # Initialization sequence
    def __init__(self, _number_of_channels, _number_of_episodes, _spatial_markov_chain, _temporal_markov_chain):
        print('[INFO] PrimaryUser Initialization: Bringing things up...')
        # The number of channels in the discretized spectrum of interest
        self.number_of_channels = _number_of_channels
        # The number of sampling rounds undertaken by the Secondary User during its operation
        self.number_of_episodes = _number_of_episodes
        # The Markov Chain across channels
        self.spatial_markov_chain = _spatial_markov_chain
        # The Markov Chain across rounds
        self.temporal_markov_chain = _temporal_markov_chain
        # Occupancy Behavior Collection
        self.occupancy_behavior_collection = []

    # Generate the initial states for k = 0 across time
    # Note here that I'm using the Temporal Correlation Statistics
    def get_initial_states_temporal_variation(self, p_val, q_val, pi_val):
        initial_state_vector = []
        previous = 1
        # Initial state generation -> band-0 at time-0 using pi_val
        if numpy.random.random_sample() > pi_val:
            previous = 0
        initial_state_vector.append(previous)
        # Based on the state of band-0 at time-0 and the (p_val, q_val) values, generate the states of the remaining...
        # ...bands
        for _loop_iterator in range(1, self.number_of_episodes):
            seed = numpy.random.random_sample()
            if previous == 1 and seed < q_val:
                previous = 0
            elif previous == 1 and seed > q_val:
                previous = 1
            elif previous == 0 and seed < p_val:
                previous = 1
            else:
                previous = 0
            initial_state_vector.append(previous)
        return initial_state_vector

    # Get the spatial and temporal occupancy behavior of the Primary User based on the statistics shared during the...
    # ...creation of the Spatial Markov Chain and the Temporal Markov Chain
    # TODO: This seems to be a weird way to generate the true_pu_occupancy_states - use a dict() with channels as...
    # ...keys and [] as values
    def simulate_occupancy_behavior(self):
        # Extracting the statistics from the objects for easy use in this method
        spatial_transition_probabilities_matrix = self.spatial_markov_chain.transition_probabilities
        spatial_start_probabilities = self.spatial_markov_chain.start_probabilities
        temporal_transition_probabilities_matrix = self.temporal_markov_chain.transition_probabilities
        temporal_start_probabilities = self.temporal_markov_chain.start_probabilities
        # Global System Steady-State Analysis - What if it's wrong?
        if spatial_start_probabilities[1] != temporal_start_probabilities[1]:
            print(
                '[ERROR] PrimaryUser get_occupancy_behavior: Looks like the start probabilities are different across...'
                'the Spatial and the Temporal Markov Chains. This is inaccurate! Proceeding with defaults...')
            # Default Values
            spatial_start_probabilities = {1: 0.6, 0: 0.4}
            temporal_start_probabilities = {1: 0.6, 0: 0.4}
            print('[WARN] PrimaryUser get_occupancy_behavior: Modified System Steady State Probabilities - ',
                  str(temporal_start_probabilities))
        # Everything's alright with the system steady-state statistics - Start simulating the PU Occupancy Behavior
        # This is global and system-specific. So, it doesn't matter which chain's steady-state probabilities is used...
        pi_val = spatial_start_probabilities[1]
        # Get the Initial state vector to get things going - row 0
        self.occupancy_behavior_collection.append(
            self.get_initial_states_temporal_variation(temporal_transition_probabilities_matrix[0][1],
                                                       temporal_transition_probabilities_matrix[1][0], pi_val))
        previous_state = self.occupancy_behavior_collection[0][0]
        # Start filling things based on spatial correlation (i.e. across rows for column-0)
        for channel_index in range(1, self.number_of_channels):
            seed = numpy.random.random_sample()
            if previous_state == 1 and seed < spatial_transition_probabilities_matrix[1][0]:
                previous_state = 0
            elif previous_state == 1 and seed > spatial_transition_probabilities_matrix[1][0]:
                previous_state = 1
            elif previous_state == 0 and seed < spatial_transition_probabilities_matrix[0][1]:
                previous_state = 1
            else:
                previous_state = 0
            self.occupancy_behavior_collection.append([previous_state])
        # Go on and fill in the remaining cells in the Occupancy Behavior Matrix
        # Use the definitions of Conditional Probabilities to realize the math - P(A|B,C)
        for channel_index in range(1, self.number_of_channels):
            for round_index in range(1, self.number_of_episodes):
                previous_temporal_state = self.occupancy_behavior_collection[channel_index][round_index - 1]
                previous_spatial_state = self.occupancy_behavior_collection[channel_index - 1][round_index]
                probability_occupied_temporal = temporal_transition_probabilities_matrix[previous_temporal_state][1]
                probability_occupied_spatial = spatial_transition_probabilities_matrix[previous_spatial_state][1]
                probability_occupied = (probability_occupied_spatial * probability_occupied_temporal) / pi_val
                seed = numpy.random.random_sample()
                if seed < probability_occupied:
                    previous_state = 1
                else:
                    previous_state = 0
                self.occupancy_behavior_collection[channel_index].append(previous_state)

    # Termination sequence
    def __exit__(self, exc_type, exc_val, exc_tb):
        print('[INFO] PrimaryUser Termination: Tearing things down...')


# This entity emulates a Secondary User (SU) intelligently accessing the spectrum un-occupied by the licensed user (PU)
class SecondaryUser(object):

    # Initialization sequence
    def __init__(self, _number_of_channels, _number_of_sampling_rounds, _channel, _true_pu_occupancy_states,
                 _limitation):
        print('[INFO] SecondaryUser Initialization: Bringing things up...')
        # The channel passed down by the Adaptive Intelligence top-level wrapper
        self.channel = _channel
        # Number of channels in the discretized spectrum of interest
        self.number_of_channels = _number_of_channels
        # Number of sampling rounds undertaken by the SU per episode
        self.number_of_sampling_rounds = _number_of_sampling_rounds
        # Occupancy Status of the cells based on simulated PU behavior - needed to simulate SU observations
        self.true_pu_occupancy_states = _true_pu_occupancy_states
        # A limit on the number of channels that can be sensed by the SU due to physical design constraints
        self.limitation = _limitation

    # The Secondary User making observations of the channels in the spectrum of interest
    def make_observations(self, episode, channel_selection_strategy):
        observation_samples = []
        for band in range(0, self.number_of_channels):
            obs_per_band = [k - k for k in range(0, self.number_of_sampling_rounds)]
            if channel_selection_strategy[band] == 1:
                obs_per_band = (self.channel.impulse_response[episode][band] *
                                self.true_pu_occupancy_states[band][episode]) + \
                               self.channel.noise[episode][band]
            observation_samples.append(obs_per_band)
        # The observation_samples member is a kxt matrix
        return observation_samples

    # Termination sequence
    def __exit__(self, exc_type, exc_val, exc_tb):
        print('[INFO] SecondaryUser Termination: Tearing things down...')


# This entity evaluates the emission probabilities, i.e. P(y|x)
class EmissionEvaluator(object):

    # Initialization sequence
    def __init__(self, _impulse_response_variance, _noise_variance):
        print('[INFO] EmissionEvaluator Initialization: Bringing things up...')
        # Variance of the Channel Impulse Response
        self.impulse_response_variance = _impulse_response_variance
        # Variance of the Complex AWGN
        self.noise_variance = _noise_variance

    # Get the Emission Probabilities -> P(y|x)
    def get_emission_probabilities(self, state, observation_sample):
        # If the channel is not observed, i.e. if the observation is [phi] or [0], report m_r(y_i) as 1
        # The Empty Place-Holder value is 0
        if observation_sample == 0:
            return 1
        # Normal Emission Estimation using the distribution of the observations given the state
        else:
            # Idea: P(A, B|C) = P(A|B, C)P(B|C) = P(A|C)P(B|C) because the real and imaginary components are independent
            # P(\Re(y_k(i)), \Im(y_k(i))|x_k) = P(\Re(y_k(i))|x_k)P(\Im(y_k(i)|x_k)
            # The emission probability from the real component
            emission_real_gaussian_component = scipy.stats.norm(0, numpy.sqrt(
                ((self.impulse_response_variance / 2) * state) + (self.noise_variance / 2))).pdf(
                observation_sample.real)
            # The emission probability from the imaginary component
            emission_imaginary_gaussian_component = scipy.stats.norm(0, numpy.sqrt(
                ((self.impulse_response_variance / 2) * state) + (self.noise_variance / 2))).pdf(
                observation_sample.imag)
            return emission_real_gaussian_component * emission_imaginary_gaussian_component

    # Termination sequence
    def __exit__(self, exc_type, exc_val, exc_tb):
        print('[INFO] EmissionEvaluator Termination: Tearing things down...')


# The Markov Chain Parameter Estimator Algorithm (modified EM - Baum-Welch)
class ParameterEstimator(object):

    # Initialization sequence
    def __init__(self, _number_of_chain_links, _number_of_repetitions, _transition_probabilities, _start_probabilities,
                 _observation_samples, _emission_evaluator, _epsilon, _confidence_bound, _util):
        print('[INFO] ParameterEstimator Initialization: Bringing things up...')
        # Transition Statistics of the Chain under analysis
        self.transition_probabilities = {0: {0: [_transition_probabilities[0][0]],
                                             1: [_transition_probabilities[0][1]]},
                                         1: {0: [_transition_probabilities[1][0]],
                                             1: [_transition_probabilities[1][1]]}}
        # Threshold for convergence
        self.epsilon = _epsilon
        # Number of links in the Chain under analysis
        self.number_of_chain_links = _number_of_chain_links
        # Number of repetitions to fine tune the estimation
        self.number_of_repetitions = _number_of_repetitions
        # Start Probabilities - Steady State Statistics of individual cells in the grid
        self.start_probabilities = _start_probabilities
        # Observation Samples
        self.observation_samples = _observation_samples
        # Emission Evaluator
        self.emission_evaluator = _emission_evaluator
        # Confidence Bound
        self.confidence_bound = _confidence_bound
        # The forward probabilities collection across simulation time
        self.forward_probabilities = []
        for k in range(0, self.number_of_chain_links):
            self.forward_probabilities.append(dict())
        # The backward probabilities collection across simulation time
        self.backward_probabilities = []
        for k in range(0, self.number_of_chain_links):
            self.backward_probabilities.append(dict())
        # The Utility instance for some general tasks
        self.util = _util

    # Convergence check
    def has_it_converged(self, iteration):
        # Obviously, not enough data to check for convergence
        if iteration == 1:
            return False
        else:
            p_list = self.transition_probabilities[0][1]
            # No convergence
            if abs(p_list[iteration - 1] - p_list[iteration - 2]) > self.epsilon:
                return False
        # Convergence
        return True

    # Get the Emission Probabilities -> P(y|x)
    def get_emission_probabilities(self, state, observation_sample):
        return self.emission_evaluator.get_emission_probabilities(state, observation_sample)

    # Construct the transition probabilities matrix in order to pass it to the state estimator
    def construct_transition_probabilities_matrix(self, iteration):
        # A constructed transition probabilities matrix
        interim_transition_probabilities_matrix = {
            0: {0: self.transition_probabilities[0][0][iteration - 1],
                1: self.transition_probabilities[0][1][iteration - 1]},
            1: {0: self.transition_probabilities[1][0][iteration - 1],
                1: self.transition_probabilities[1][1][iteration - 1]}
        }
        return interim_transition_probabilities_matrix

    # In-situ update of Forward Probabilities
    def fill_up_forward_probabilities(self, iteration, observation_vector):
        # Setting the Boundary conditions for Forward Probabilities - again for coherence
        # Occupied
        _forward_boundary_condition_occupied = self.get_emission_probabilities(
            OccupancyState.occupied.value,
            observation_vector[0]) * self.start_probabilities[1]
        # Idle
        _forward_boundary_condition_idle = self.get_emission_probabilities(
            OccupancyState.idle.value, observation_vector[0]) * self.start_probabilities[0]
        _temp_forward_probabilities_dict = {0: _forward_boundary_condition_idle,
                                            1: _forward_boundary_condition_occupied}
        self.forward_probabilities[0] = _temp_forward_probabilities_dict
        # j
        for channel_index in range(1, self.number_of_chain_links):
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

    # In-situ update of Backward Probabilities
    def fill_up_backward_probabilities(self, iteration, observation_vector):
        # Setting the Boundary conditions for Backward Probabilities - again for coherence
        # Occupied
        _state_sum = 0
        # outside summation - refer to the definition of backward probability
        for _state in OccupancyState:
            _state_sum += self.transition_probabilities[1][_state.value][iteration - 1] * \
                          self.get_emission_probabilities(_state.value,
                                                          observation_vector[self.number_of_chain_links - 1])
        _backward_boundary_condition_occupied = _state_sum
        # Idle
        _state_sum = 0
        # outside summation - refer to the definition of backward probability
        for _state in OccupancyState:
            _state_sum += self.transition_probabilities[0][_state.value][iteration - 1] * \
                          self.get_emission_probabilities(_state.value,
                                                          observation_vector[self.number_of_chain_links - 1])
        _backward_boundary_condition_idle = _state_sum
        _temp_backward_probabilities_dict = {0: _backward_boundary_condition_idle,
                                             1: _backward_boundary_condition_occupied}
        self.backward_probabilities[self.number_of_chain_links - 1] = _temp_backward_probabilities_dict
        # j
        for channel_index in range(self.number_of_chain_links - 2, -1, -1):
            # state r for channel i
            for previous_occupancy_state in OccupancyState:
                occupancy_state_sum = 0
                # state l for channel j+1
                for next_occupancy_state in OccupancyState:
                    occupancy_state_sum += self.backward_probabilities[channel_index + 1][
                                               next_occupancy_state.value] * self.transition_probabilities[
                                               previous_occupancy_state.value][next_occupancy_state.value][
                                               iteration - 1] * \
                                           self.get_emission_probabilities(next_occupancy_state.value,
                                                                           observation_vector[channel_index])
                self.backward_probabilities[channel_index][previous_occupancy_state.value] = occupancy_state_sum

    # Get the numerator of the fraction in the Algorithm (refer to the LaTeX document for more information)
    def get_numerator(self, previous_state, next_state, iteration, observation_vector):
        numerator_sum = self.get_emission_probabilities(next_state, observation_vector[0]) * \
                        self.transition_probabilities[previous_state][next_state][iteration - 1] * \
                        self.backward_probabilities[1][next_state]
        for spatial_index in range(1, self.number_of_chain_links - 1):
            numerator_sum += self.forward_probabilities[spatial_index - 1][previous_state] * \
                             self.get_emission_probabilities(next_state, observation_vector[spatial_index]) * \
                             self.transition_probabilities[previous_state][next_state][iteration - 1] * \
                             self.backward_probabilities[spatial_index + 1][next_state]
        numerator_sum += self.forward_probabilities[self.number_of_chain_links - 2][previous_state] * \
            self.get_emission_probabilities(next_state, observation_vector[self.number_of_chain_links - 1]) * \
            self.transition_probabilities[previous_state][next_state][iteration - 1]
        return numerator_sum

    # Get the denominator of the fraction in the Algorithm (refer to the LaTeX document for more information)
    def get_denominator(self, previous_state, iteration, observation_vector):
        denominator_sum = 0
        for nxt_state in OccupancyState:
            denominator_sum_internal = self.get_emission_probabilities(nxt_state.value, observation_vector[0]) * \
                                       self.transition_probabilities[previous_state][nxt_state.value][
                                           iteration - 1] * \
                                       self.backward_probabilities[1][nxt_state.value]
            for _spatial_index in range(1, self.number_of_chain_links - 1):
                denominator_sum_internal += self.forward_probabilities[_spatial_index - 1][previous_state] * \
                                            self.get_emission_probabilities(nxt_state.value,
                                                                            observation_vector[_spatial_index]) * \
                                            self.transition_probabilities[previous_state][nxt_state.value][
                                                iteration - 1] * \
                                            self.backward_probabilities[_spatial_index + 1][nxt_state.value]
            denominator_sum_internal += self.forward_probabilities[self.number_of_chain_links - 2][
                                            previous_state] \
                * self.get_emission_probabilities(nxt_state.value, observation_vector[self.number_of_chain_links - 1]) \
                * self.transition_probabilities[previous_state][nxt_state.value][iteration - 1]
            denominator_sum += denominator_sum_internal
        return denominator_sum

    # Core method
    # Estimate the Markov Chain State Transition Probabilities matrix
    def estimate_parameters(self):
        # Confidence variable to determine when the algorithm has converged
        confidence = 0
        # Iteration counter
        iteration = 1
        # Until convergence - I'm only checking for the convergence of 'p' in order to speed up the convergence process
        while self.has_it_converged(iteration) is False or confidence < self.confidence_bound:
            # A confidence check
            convergence = self.has_it_converged(iteration)
            if convergence is True:
                # It has converged. Doing this to ensure that the convergence is permanent.
                confidence += 1
            else:
                confidence = 0
            # Numerators collection
            numerators_collection = {0: {0: list(), 1: list()}, 1: {0: list(), 1: list()}}
            # Denominators collection
            denominators_collection = {0: {0: list(), 1: list()}, 1: {0: list(), 1: list()}}
            # Let us first construct a reduced observation vector like we did for the Viterbi algorithm
            for _sampling_round in range(0, self.number_of_repetitions):
                observation_vector = []
                for _channel in range(0, self.number_of_chain_links):
                    observation_vector.append(self.observation_samples[_channel][_sampling_round])
                # Fill up the remaining elements of the forward probabilities array
                self.fill_up_forward_probabilities(iteration, observation_vector)
                # Fill up the remaining elements of the backward probabilities array
                self.fill_up_backward_probabilities(iteration, observation_vector)
                # state r in {0, 1}
                # Total 4 combinations arise from this double loop
                for previous_state in OccupancyState:
                    # state l in {0, 1}
                    for next_state in OccupancyState:
                        numerators_collection[previous_state.value][next_state.value].append(self.get_numerator(
                            previous_state.value, next_state.value, iteration, observation_vector))
                        denominators_collection[previous_state.value][next_state.value].append(
                            self.get_denominator(
                                previous_state.value, iteration, observation_vector))
            for previous_state in OccupancyState:
                for next_state in OccupancyState:
                    numerator_sum = 0
                    denominator_sum = 0
                    for numerator in numerators_collection[previous_state.value][next_state.value]:
                        numerator_sum += numerator
                    for denominator in denominators_collection[previous_state.value][next_state.value]:
                        denominator_sum += denominator
                    self.transition_probabilities[previous_state.value][next_state.value].append(
                        numerator_sum / denominator_sum)
            iteration += 1
        final_length = len(self.transition_probabilities[0][1])
        # Return 'p'
        return self.transition_probabilities[0][1][final_length - 1]

    # Termination sequence
    def __exit__(self, exc_type, exc_val, exc_tb):
        print('[INFO] ParameterEstimator Termination: Tearing things down...')


# The Markov Chain State Estimator algorithm - Viterbi Algorithm
class StateEstimator(object):
    # Value function named tuple
    VALUE_FUNCTION_NAMED_TUPLE = namedtuple('ValueFunction', ['current_value', 'previous_state'])

    # Initialization sequence
    def __init__(self, _number_of_channels, _number_of_sampling_rounds, _observation_samples, _start_probabilities,
                 _transition_probabilities, _emission_evaluator):
        print('[INFO] StateEstimator Initialization: Bringing things up...')
        # Number of channels in the discretized spectrum of interest
        self.number_of_channels = _number_of_channels
        # Number of sampling rounds undertaken by the SU per episode
        self.number_of_sampling_rounds = _number_of_sampling_rounds
        # The Observation Samples
        self.observation_samples = _observation_samples
        # The Steady State Probabilities
        self.start_probabilities = _start_probabilities
        # The Transition Statistics
        self.transition_probabilities = _transition_probabilities
        # The Emission Evaluator
        self.emission_evaluator = _emission_evaluator

    # Safe entry access using indices from a collection object
    @staticmethod
    def get_entry(collection, index):
        if len(collection) is not 0 and collection[index] is not None:
            return collection[index]
        else:
            # Empty Place-Holder value is 0
            return 0

    # Get enumeration field value from name
    @staticmethod
    def value_from_name(name):
        if name == 'occupied':
            return OccupancyState.occupied.value
        else:
            return OccupancyState.idle.value

    # Get the start probabilities from the named tuple - a simple getter utility method exclusive to this class
    def get_start_probabilities(self, state_name):
        if state_name == 'occupied':
            return self.start_probabilities[1]
        else:
            return self.start_probabilities[0]

    # Return the transition probabilities from the transition probabilities matrix
    def get_transition_probabilities(self, row, column):
        return self.transition_probabilities[row][column]

    # Get the Emission Probabilities -> P(y|x)
    def get_emission_probabilities(self, state, observation_sample):
        return self.emission_evaluator.get_emission_probabilities(state, observation_sample)

    # Output the estimated state of the frequency bands in the wideband spectrum of interest
    # Output a collection consisting of the required parameters of interest
    def estimate_pu_occupancy_states(self):
        # Variable reference to get rid of the idiotic "prior-referenced usage' warning
        previous_state = None
        estimated_states_array = []
        for sampling_round in range(0, self.number_of_sampling_rounds):
            estimated_states = []
            reduced_observation_vector = []
            for entry in self.observation_samples:
                reduced_observation_vector.append(self.get_entry(entry, sampling_round))
            # Now, I have to estimate the state of the ${NUMBER_OF_FREQUENCY_BANDS} based on
            # ...this reduced observation vector
            # INITIALIZATION : The array of initial probabilities is known
            # FORWARD RECURSION
            value_function_collection = []
            for x in range(0, len(reduced_observation_vector)):
                value_function_collection.append(dict())
            for state in OccupancyState:
                current_value = self.get_emission_probabilities(state.value, reduced_observation_vector[0]) * \
                                self.get_start_probabilities(state.name)
                value_function_collection[0][state.name] = self.VALUE_FUNCTION_NAMED_TUPLE(
                    current_value=current_value,
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
                            pointer = self.get_transition_probabilities(candidate_previous_state.value,
                                                                        state.value) * \
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
            # Doesn't happen but adding this to remove the warnings
            if previous_state is None or max_value is None:
                print('[ERROR] StateEstimator estimate_pu_occupancy_states: previous_state AND/OR max_value members '
                      'are invalid! Returning junk results! Please look into this!')
                # Return junk
                return [[k - k for k in range(0, self.number_of_channels)], 0]
            # Backtracking
            for i in range(len(value_function_collection) - 2, -1, -1):
                estimated_states.insert(0, self.value_from_name(
                    value_function_collection[i + 1][previous_state].previous_state))
                previous_state = value_function_collection[i + 1][previous_state].previous_state
            estimated_states_array.append(estimated_states)
        return estimated_states_array[self.number_of_sampling_rounds - 1]

    # Termination sequence
    def __exit__(self, exc_type, exc_val, exc_tb):
        print('[INFO] StateEstimator Termination: Tearing things down...')


# This entity encapsulates the agent which provides rewards based on the state of the system and the action taken by...
# ...the POMDP agent
class Sweepstakes(object):

    # Initialization sequence
    # _nu = Missed_Detection_Cost
    def __init__(self, _primary_user, _mu):
        print('[INFO] Sweepstakes Initialization: Bringing things up...')
        # The Primary User
        self.primary_user = _primary_user
        # Missed_Detection_Cost
        self.mu = _mu

    # Get Reward based on the system state and the action taken by the POMDP agent
    # The system_belief member is created by the action taken by the POMDP agent
    # Reward = (1-Probability_of_false_alarm) + Missed_Detection_Cost*Probability_of_missed_detection
    def roll(self, system_belief, system_state):
        # Used to evaluate Missed Detection Probability
        correct_detection_count = 0
        actual_occupancy_count = 0
        # Used to evaluate False Alarm Probability
        false_alarm_count = 0
        actual_idle_count = 0
        for i in range(0, len(system_belief)):
            if system_state[i] == 1:
                actual_occupancy_count += 1
                if system_belief[i] == 1:
                    correct_detection_count += 1
            else:
                actual_idle_count += 1
                if system_belief[i] == 1:
                    false_alarm_count += 1
        # Reward = (1-Probability_of_false_alarm) + Missed_Detection_Cost*Probability_of_missed_detection
        return (1 - (false_alarm_count / actual_idle_count)) + \
               (self.mu * (1 - (correct_detection_count / actual_occupancy_count)))

    # Termination sequence
    def __exit__(self, exc_type, exc_val, exc_tb):
        print('[INFO] Sweepstakes Termination: Tearing things down...')


# This entity encapsulates an Oracle which knows the best possible channels to use in each episode.
# Hence, the policy followed by this Oracle is the most optimal policy
# The action policy achieved by the POMDP agent will be evaluated/benchmarked against the Oracle's policy thereby...
# ...giving us a regret metric.
class Oracle(object):

    # Initialization sequence
    def __init__(self, _number_of_episodes):
        print('[INFO] Oracle Initialization: Bringing things up...')
        # Number of relevant time steps of POMDP agent interaction with the radio environment
        self.number_of_episodes = _number_of_episodes

    # Get the long term reward by following the optimal policy - This is the Oracle. She knows everything.
    # P_FA = 0
    # P_MD = 0
    # Return = Sum_{1}^{number_of_episodes}\ [1] = number_of_episodes
    # Episodic Reward = 1
    def get_return(self):
        return self.number_of_episodes

    # Windowed return for regret analysis
    @staticmethod
    def get_windowed_return(window_size):
        return window_size

    # Termination sequence
    def __exit__(self, exc_type, exc_val, exc_tb):
        print('[INFO] Oracle Termination: Tearing things down...')


# Top-level Executive class
# This entity encapsulates the POMDP Approximate Point-based Value Iteration algorithm named The PERSEUS algorithm
# Training to fine-tune the belief space -> Perform backup until convergence -> Re-sample using the most recent policy
# References to the ParameterEstimation algorithm and the StateEstimation algorithm in the belief analysis phase
class AdaptiveIntelligence(object):
    # Number of channels in the discretized spectrum of interest
    NUMBER_OF_CHANNELS = 5

    # Number of sampling rounds undertaken by the Secondary User per episode
    NUMBER_OF_SAMPLING_ROUNDS = 10

    # Number of episodes during which the SU interacts with the radio environment
    NUMBER_OF_EPISODES = 100

    # Exploration period of the POMDP agent to find a set of reachable beliefs
    EXPLORATION_PERIOD = 20

    # Mean of the Complex AWGN
    NOISE_MEAN = 0

    # Variance of the Complex AWGN
    NOISE_VARIANCE = 1

    # Mean of the Complex Channel Impulse Response
    IMPULSE_RESPONSE_MEAN = 0

    # Variance of the Complex Channel Impulse Response
    IMPULSE_RESPONSE_VARIANCE = 80

    # False Alarm Cost
    MU = -1

    # SU Sensing Limitation
    LIMITATION = 2

    # Parameter Estimation Convergence Threshold
    EPSILON = 0.001

    # Convergence Confidence Metric for the Parameter Estimation algorithm
    CONFIDENCE_BOUND = 1

    # Discount Factor
    GAMMA = 0.9

    # Initialization sequence
    def __init__(self):
        print('[INFO] AdaptiveIntelligence Initialization: Bringing things up...')
        # P(Occupied) - The steady state probability of being occupied
        self.pi = 0.6
        # P(Occupied|Idle) - The transition probability parameter
        self.p = 0.3
        # Initial Estimate of P(Occupied|Idle) a.k.a 'p'
        self.initial_p = 0.000005
        # The Utility object
        self.util = Util()
        # Initial Transition Probabilities_Matrix
        self.initial_transition_probabilities_matrix = self.util.construct_transition_probability_matrix(self.initial_p,
                                                                                                         self.pi)
        # Start Probabilities Dict
        self.start_probabilities_dict = self.util.construct_start_probabilities_dict(self.pi)
        # Setup the Spatial Markov Chain
        self.spatial_markov_chain = self.setup_markov_chain(self.pi, self.p, MarkovianCorrelationClass.spatial)
        # Setup the Temporal Markov Chain
        self.temporal_markov_chain = self.setup_markov_chain(self.pi, self.p,
                                                             MarkovianCorrelationClass.temporal)
        # Channel
        self.channel = Channel(self.NUMBER_OF_CHANNELS, self.NUMBER_OF_SAMPLING_ROUNDS, self.NUMBER_OF_EPISODES,
                               self.NOISE_MEAN, self.NOISE_VARIANCE, self.IMPULSE_RESPONSE_MEAN,
                               self.IMPULSE_RESPONSE_VARIANCE)
        # The Emission Evaluator
        self.emission_evaluator = EmissionEvaluator(self.IMPULSE_RESPONSE_VARIANCE, self.NOISE_VARIANCE)
        # Primary User
        self.primary_user = PrimaryUser(self.NUMBER_OF_CHANNELS, self.NUMBER_OF_EPISODES,
                                        self.spatial_markov_chain, self.temporal_markov_chain)
        self.primary_user.simulate_occupancy_behavior()
        # Secondary User
        self.secondary_user = SecondaryUser(self.NUMBER_OF_CHANNELS, self.NUMBER_OF_SAMPLING_ROUNDS, self.channel,
                                            self.primary_user.occupancy_behavior_collection, self.LIMITATION)
        # Sweepstakes - The Reward Analyzer
        self.sweepstakes = Sweepstakes(self.primary_user, self.MU)
        # The Oracle
        self.oracle = Oracle(self.NUMBER_OF_EPISODES)
        # Parameter Estimator - Modified Baum-Welch / Modified EM
        self.parameter_estimator = ParameterEstimator(self.NUMBER_OF_CHANNELS, self.NUMBER_OF_SAMPLING_ROUNDS,
                                                      self.initial_transition_probabilities_matrix,
                                                      self.start_probabilities_dict, None,
                                                      self.emission_evaluator, self.EPSILON,
                                                      self.CONFIDENCE_BOUND, self.util)
        # State Estimator
        self.state_estimator = StateEstimator(self.NUMBER_OF_CHANNELS, self.NUMBER_OF_SAMPLING_ROUNDS, None,
                                              self.start_probabilities_dict,
                                              self.initial_transition_probabilities_matrix,
                                              self.emission_evaluator)

    # Randomly explore the environment and collect a set of beliefs B of reachable belief points
    # ...strategy as a part of the modified PERSEUS strategy
    def random_exploration(self, policy):
        reachable_beliefs = dict()
        discretized_spectrum = [k for k in range(0, self.NUMBER_OF_CHANNELS)]
        state_space_size = 2 ** self.NUMBER_OF_CHANNELS
        # All possible states
        all_possible_states = list(map(list, itertools.product([0, 1], repeat=self.NUMBER_OF_CHANNELS)))
        # Uniform belief assignment to all states in the state space
        initial_belief_vector = dict()
        for state in all_possible_states:
            state_key = ''.join(str(k) for k in state)
            initial_belief_vector[state_key] = 1 / state_space_size
        previous_belief_vector = initial_belief_vector
        reachable_beliefs['0'] = initial_belief_vector
        if policy is not None:
            # TODO: An "ordered random" exploration strategy
            print(
                '[WARN] AdaptiveIntelligence random_exploration: Specific policy exploration is yet to be implemented')
        print('[INFO] AdaptiveIntelligence random_exploration: Using a truly random exploration strategy')
        # Start exploring
        for episode in range(0, self.EXPLORATION_PERIOD):
            # Choose an action
            action = [k - k for k in range(0, self.NUMBER_OF_CHANNELS)]
            # SU has limited sensing capabilities
            for capability in range(0, self.LIMITATION):
                action[random.choice(discretized_spectrum)] = 1
            # Perform the sensing action and make the observations
            observations = self.secondary_user.make_observations(episode, action)
            self.parameter_estimator.observation_samples = observations
            transition_probabilities_matrix = self.util.construct_transition_probability_matrix(
                self.parameter_estimator.estimate_parameters(), self.pi)
            # Perform the Belief Update
            updated_belief_vector = dict()
            for state in all_possible_states:
                state_key = ''.join(str(k) for k in state)
                updated_belief_vector[state_key] = self.belief_update(observations, previous_belief_vector, state,
                                                                      transition_probabilities_matrix)
            # Add the new belief vector to the reachable beliefs set
            reachable_beliefs[str(episode)] = updated_belief_vector
        return reachable_beliefs

    # Belief Update sequence
    def belief_update(self, observations, previous_belief_vector, new_state, transition_probabilities_matrix):
        multiplier = 0
        denominator = 0
        # All possible states
        all_possible_states = list(map(list, itertools.product([0, 1], repeat=self.NUMBER_OF_CHANNELS)))
        for prev_state in all_possible_states:
            multiplier += self.get_transition_probability(prev_state, new_state, transition_probabilities_matrix) * \
                          previous_belief_vector[''.join(str(k) for k in prev_state)]
        numerator = self.get_emission_probability(observations, new_state) * multiplier
        for next_state in all_possible_states:
            emission_probability = self.get_emission_probability(observations, next_state)
            multiplier = 0
            for prev_state in all_possible_states:
                multiplier += self.get_transition_probability(prev_state, next_state, transition_probabilities_matrix) \
                              * previous_belief_vector[''.join(str(k) for k in prev_state)]
            denominator += emission_probability * multiplier
        return numerator / denominator

    # Get Emission Probabilities for the Belief Update sequence
    def get_emission_probability(self, observations, state):
        emission_probability = 1
        for index in range(0, self.NUMBER_OF_CHANNELS):
            emission_probability = emission_probability * self.emission_evaluator.get_emission_probabilities(
                state[index], observations[index][0])
        return emission_probability

    # Get State Transition Probabilities for the Belief Update sequence
    def get_transition_probability(self, prev_state, next_state, transition_probabilities_matrix):
        transition_probability = transition_probabilities_matrix[next_state[0]][prev_state[0]]
        for index in range(1, self.NUMBER_OF_CHANNELS):
            state_probability = (lambda: 1 - self.pi, lambda: self.pi)[next_state[index] == 1]()
            transition_probability = transition_probability * ((transition_probabilities_matrix[next_state[index]][
                                                                    prev_state[index]] *
                                                                transition_probabilities_matrix[next_state[index]][
                                                                    next_state[index - 1]]) / state_probability)
        return transition_probability

    # Initialization
    @staticmethod
    def initialize(reachable_beliefs):
        # V_0 for the reachable beliefs
        value_function_collection = dict()
        for belief_key in reachable_beliefs.keys():
            value_function_collection[belief_key] = (-10, None)
        return value_function_collection

    # The Backup stage
    # TODO: The same reachable beliefs are used throughout...Can we re-sample at regular intervals instead?
    def backup(self, stage_number, reachable_beliefs, previous_stage_value_function_collection):
        discretized_spectrum = [k for k in range(0, self.NUMBER_OF_CHANNELS)]
        unimproved_belief_points = reachable_beliefs
        next_stage_value_function_collection = dict()
        # All possible actions
        action_set = list(map(list, itertools.product(discretized_spectrum, repeat=self.LIMITATION)))
        # All possible states
        all_possible_states = list(map(list, itertools.product([0, 1], repeat=self.NUMBER_OF_CHANNELS)))
        number_of_belief_changes = 0
        # While there are still some un-improved belief points
        while len(unimproved_belief_points) is not 0:
            belief_sample_key = random.choice(unimproved_belief_points.keys)
            belief_sample = unimproved_belief_points[belief_sample_key]
            max_value_function = -100
            max_action = None
            for action in action_set:
                # Make observations based on the chosen action
                observation_samples = self.secondary_user.make_observations(stage_number, action)
                # Estimate the Model Parameters
                self.parameter_estimator.observation_samples = observation_samples
                transition_probability_matrix = self.util.construct_transition_probability_matrix(
                    self.parameter_estimator.estimate_parameters(), self.pi)
                # Estimate the System State
                self.state_estimator.observation_samples = observation_samples
                self.state_estimator.transition_probabilities = transition_probability_matrix
                estimated_system_state = self.state_estimator.estimate_pu_occupancy_states()
                reward_sum = 0
                normalization_constant = 0
                for state in all_possible_states:
                    emission_probability = self.get_emission_probability(observation_samples, state)
                    multiplier = 0
                    for prev_state in all_possible_states:
                        multiplier += self.get_transition_probability(prev_state, state,
                                                                      transition_probability_matrix) * \
                                      belief_sample[''.join(str(k) for k in prev_state)]
                    normalization_constant += emission_probability * multiplier
                    reward_sum += self.sweepstakes.roll(estimated_system_state, state) * belief_sample[
                        ''.join(str(k) for k in state)]
                internal_term = reward_sum + (self.GAMMA * normalization_constant * -10)
                if internal_term > max_value_function:
                    max_value_function = internal_term
                    max_action = action
            if max_value_function > previous_stage_value_function_collection[belief_sample_key]:
                del unimproved_belief_points[belief_sample_key]
                next_stage_value_function_collection[belief_sample_key] = (max_value_function, max_action)
                number_of_belief_changes += 1
            else:
                next_stage_value_function_collection[belief_sample_key] = previous_stage_value_function_collection[
                    belief_sample_key]
                del unimproved_belief_points[belief_sample_key]
                max_action = previous_stage_value_function_collection[belief_sample_key][1]
            for belief_point_key, belief_point in unimproved_belief_points:
                normalization_constant = 0
                reward_sum = 0
                observation_samples = self.secondary_user.make_observations(stage_number, max_action)
                # Estimate the Model Parameters
                self.parameter_estimator.observation_samples = observation_samples
                transition_probability_matrix = self.util.construct_transition_probability_matrix(
                    self.parameter_estimator.estimate_parameters(), self.pi)
                # Estimate the System State
                self.state_estimator.observation_samples = observation_samples
                self.state_estimator.transition_probabilities = transition_probability_matrix
                estimated_system_state = self.state_estimator.estimate_pu_occupancy_states()
                for state in all_possible_states:
                    emission_probability = self.get_emission_probability(observation_samples, state)
                    multiplier = 0
                    for prev_state in all_possible_states:
                        multiplier += self.get_transition_probability(prev_state, state,
                                                                      transition_probability_matrix) * \
                                      belief_point[''.join(str(k) for k in prev_state)]
                    normalization_constant += emission_probability * multiplier
                    reward_sum += self.sweepstakes.roll(estimated_system_state, state) * belief_point[
                        ''.join(str(k) for k in state)]
                internal_term = reward_sum + (self.GAMMA * normalization_constant * -10)
                if internal_term > previous_stage_value_function_collection[belief_point_key]:
                    del unimproved_belief_points[belief_point_key]
                    next_stage_value_function_collection[belief_point_key] = internal_term
                    number_of_belief_changes += 1
        return [next_stage_value_function_collection, number_of_belief_changes]

    # The PERSEUS algorithm
    # Calls to Random Exploration, Initialization, and Backup stages
    def run_perseus(self):
        # Random Exploration
        reachable_beliefs = self.random_exploration(None)
        # Initialization
        initial_value_function_collection = self.initialize(reachable_beliefs)
        # Relevant collections
        previous_value_function_collection = initial_value_function_collection
        stage_number = self.EXPLORATION_PERIOD
        belief_changes = -1
        # Check for termination condition here...
        while belief_changes is not 0:
            stage_number += 1
            if stage_number == self.NUMBER_OF_EPISODES:
                return 0
            # Backup to find \alpha -> Get V_{n+1} and #BeliefChanges
            backup_results = self.backup(stage_number, reachable_beliefs, previous_value_function_collection)
            next_value_function_collection = backup_results[0]
            belief_changes = backup_results[1]
            previous_value_function_collection = next_value_function_collection
        return 0

    # Setup the Markov Chain
    @staticmethod
    def setup_markov_chain(_pi, _p, _correlation_class):
        print('[INFO] AdaptiveIntelligence setup_markov_chain: Setting up the Markov Chain...')
        transient_markov_chain_object = MarkovChain()
        transient_markov_chain_object.set_markovian_correlation_class(_correlation_class)
        transient_markov_chain_object.set_start_probability_parameter(_pi)
        transient_markov_chain_object.set_transition_probability_parameter(_p)
        return transient_markov_chain_object

    # Regret Analysis
    def get_regret(self):
        optimal_reward = self.oracle.get_return()
        pomdp_reward_upon_convergence = self.run_perseus()
        return abs(optimal_reward - pomdp_reward_upon_convergence)

    # Termination sequence
    def __exit__(self, exc_type, exc_val, exc_tb):
        print('[INFO] AdaptiveIntelligence Termination: Tearing things down...')


# Run Trigger
if __name__ == '__main__':
    print('[INFO] AdaptiveIntelligence main: Starting system simulation...')
    adaptive_intelligent_agent = AdaptiveIntelligence()
    print('[INFO] AdaptiveIntelligence main: Regret: ', adaptive_intelligent_agent.get_regret())
