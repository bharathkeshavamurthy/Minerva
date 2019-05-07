# This entity encapsulates the overall evaluation of our adaptive, intelligent, hierarchical framework for...
# ...Cognitive Radio System developed as a part of this research.
# The evaluation includes the following:
# Visualization - Utility over the entire interaction time of the agent with the radio environment v/s Episodes
# Metrics: Utility - Captures both SU network throughput rewards and PU interference penalties
# Variants:
# a. Viterbi algorithm with complete observations
# b. Viterbi algorithm with incomplete observations (use different channel selection heuristics - 1-gap/2-gap...)
# c. The correlation coefficient based algorithm in the state-of-the-art
# d. PERSEUS algorithm with concurrent model learning: 6 channels per process: 18 channels in the spectrum cumulatively
# e. PERSEUS algorithm with model foresight: 6 channels per process: 18 channels in the spectrum cumulatively
# f. PERSEUS algorithm with model foresight: 6 channels per process: 18 channels in the spectrum cumulatively
# ...and simplified belief update procedure (up to a Hamming distance of 2 in each process - 22 allowed transitions)
# g. Independent channels with complete information - Use Neyman-Pearson detection to estimate the occupancy states
# Author: Bharath Keshavamurthy
# Organization: School of Electrical & Computer Engineering, Purdue University
# Copyright (c) 2019. All Rights Reserved.

import math
import numpy
import random
import warnings
import functools
import itertools
import scipy.stats
import multiprocessing
from enum import Enum
from collections import namedtuple
from matplotlib import pyplot as plt


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

    # Get the steady state probability for the state passed as argument
    @staticmethod
    def get_state_probability(state, pi):
        return (lambda: 1 - pi, lambda: pi)[state == 1]()

    # Construct the complete transition probability matrix from P(Occupied|Idle), i.e. 'p' and P(Occupied), i.e. 'pi'
    @staticmethod
    def construct_transition_probability_matrix(p, pi):
        # P(Idle|Occupied)
        q = (p * (1 - pi)) / pi
        return {0: {1: p, 0: 1 - p}, 1: {0: q, 1: 1 - q}}

    # Generate the action set based on the SU sensing limitations and the Number of Channels in the discretized...
    # ...spectrum of interest
    @staticmethod
    @deprecated
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

    #  Normalize the belief, if necessary
    def normalize(self, belief, belief_sum):
        normalized_sum = 0
        for key in belief.keys():
            belief[key] /= belief_sum
            normalized_sum += belief[key]
        return belief, self.validate_belief(normalized_sum)

    # Perform belief validation
    @staticmethod
    def validate_belief(normalized_sum):
        # Interval to account for precision errors
        if 0.90 <= normalized_sum <= 1.10:
            return True
        return False

    # Termination sequence
    def __exit__(self, exc_type, exc_val, exc_tb):
        print('[INFO] Util Termination: Tearing things down...')


# This entity encapsulates the Channel object - simulates the Channel
class Channel(object):

    # Initialization sequence
    def __init__(self, _number_of_channels, _number_of_sampling_rounds, _number_of_episodes, _noise_mean,
                 _noise_variance, _impulse_response_mean, _impulse_response_variance):
        print('[INFO] Channel Initialization: Bringing things up...')
        # Noise Statistics
        self.noise_mean = _noise_mean
        if self.noise_mean is not 0:
            print('[WARN] Channel Initialization: The system assumes Zero-Mean, Additive, White, Gaussian '
                  'Noise...')
            self.noise_mean = 0
        self.noise_variance = _noise_variance
        # Channel Impulse Response Statistics
        self.impulse_response_mean = _impulse_response_mean
        if self.impulse_response_mean is not 0:
            print('[WARN] Channel Initialization: The system assumes Zero-Mean, Gaussian Impulse Response...')
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
        # The AWGN used in the Observation Model
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
                channel_impulse_response_samples[episode][frequency_band] = numpy.random.normal(
                    mu_channel_impulse_response, std_channel_impulse_response, self.number_of_sampling_rounds)
        return channel_impulse_response_samples

    # Generate the AWGN samples
    def get_noise(self):
        noise_samples = []
        for k in range(0, self.number_of_episodes):
            noise_samples.append(dict())
        for episode in range(0, self.number_of_episodes):
            for frequency_band in range(0, self.number_of_channels):
                mu_noise, std_noise = self.noise_mean, numpy.sqrt(self.noise_variance)
                noise_samples[episode][frequency_band] = numpy.random.normal(mu_noise, std_noise,
                                                                             self.number_of_sampling_rounds)
        return noise_samples

    # Termination sequence
    def __exit__(self, exc_type, exc_val, exc_tb):
        print('[INFO] Channel Termination: Tearing things down...')


# This class encapsulates the Licensed User dynamically occupying the discretized spectrum under analysis
class PrimaryUser(object):

    # Initialization sequence
    def __init__(self, _number_of_channels, _number_of_episodes, _spatial_markov_chain, _temporal_markov_chain, _util):
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
        # The Utility class
        self.util = _util

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
        # P(A=a|B=b) = \sum_{c\in\{0,1\}}\ P(A=a|B=b,C=c)P(C=c)
        # Using the definition of Marginal Probability in discrete distributions
        for channel_index in range(1, self.number_of_channels):
            for round_index in range(1, self.number_of_episodes):
                occupied_spatial_transition = spatial_transition_probabilities_matrix[
                    self.occupancy_behavior_collection[channel_index - 1][round_index]][1]
                seed = numpy.random.random_sample()
                if seed < occupied_spatial_transition:
                    self.occupancy_behavior_collection[channel_index].append(1)
                else:
                    self.occupancy_behavior_collection[channel_index].append(0)

    # Termination sequence
    def __exit__(self, exc_type, exc_val, exc_tb):
        print('[INFO] PrimaryUser Termination: Tearing things down...')


# This entity emulates a Secondary User (SU) heuristically accessing the spectrum un-occupied by the licensed user (PU)
class SecondaryUser(object):

    # Initialization sequence
    def __init__(self, _number_of_channels, _number_of_sampling_rounds, _number_of_episodes, _channel,
                 _true_pu_occupancy_states, _spatial_limitation):
        print('[INFO] SecondaryUser Initialization: Bringing things up...')
        # The channel passed down by the Adaptive Intelligence top-level wrapper
        self.channel = _channel
        # Number of channels in the discretized spectrum of interest
        self.number_of_channels = _number_of_channels
        # Number of sampling rounds undertaken by the SU per episode
        self.number_of_sampling_rounds = _number_of_sampling_rounds
        # Number of episodes of interaction with the radio environment [encapsulates #sampling_rounds]
        self.number_of_episodes = _number_of_episodes
        # Occupancy Status of the cells based on simulated PU behavior - needed to simulate SU observations
        self.true_pu_occupancy_states = _true_pu_occupancy_states
        # A limit on the number of channels that can be sensed by the SU due to physical design constraints
        self.spatial_limitation = _spatial_limitation

    # Observe everything from a global perspective for the unconstrained non-POMDP agent
    def observe_everything_unconstrained(self):
        observation_samples = []
        for band in range(0, self.number_of_channels):
            obs_per_band = []
            for episode in range(0, self.number_of_episodes):
                obs_per_band.append((self.channel.impulse_response[episode][band][0] * self.
                                     true_pu_occupancy_states[band][episode]) + self.channel.noise[episode][band][0])
            observation_samples.append(obs_per_band)
        # The observation_samples member is a kxt matrix
        return observation_samples

    # Make observations from a global perspective for the constrained non-POMDP agent
    # The same subset of channels are observed in each episode
    def observe_everything_with_spatial_constraints(self, channel_selection_heuristic):
        observation_samples = []
        for band in range(0, self.number_of_channels):
            obs_per_band = [k - k for k in range(0, self.number_of_episodes)]
            if band in channel_selection_heuristic:
                obs_per_band = []
                for episode in range(0, self.number_of_episodes):
                    obs_per_band.append((self.channel.impulse_response[episode][band][0] *
                                         self.true_pu_occupancy_states[band][episode]) +
                                        self.channel.noise[episode][band][0])
            observation_samples.append(obs_per_band)
        # The observation_samples member is a kxt matrix
        return observation_samples

    # Make observations from a global perspective for the constrained non-POMDP agent
    # Different subsets of channels are sensed in each episode - patterned or randomized heuristic
    def observe_everything_with_spatio_temporal_constraints(self, channel_selection_heuristic,
                                                            episode_selection_heuristic):
        raise NotImplementedError('This functionality is yet to be implemented! Please check back later!')

    # The Secondary User making observations of the channels in the spectrum of interest
    def make_observations(self, episode, channel_selection_strategy):
        observation_samples = []
        for band in range(0, self.number_of_channels):
            obs_per_band = [k - k for k in range(0, self.number_of_sampling_rounds)]
            if channel_selection_strategy[band] == 1:
                obs_per_band = list((numpy.array(
                    self.channel.impulse_response[episode][band]) * self.true_pu_occupancy_states[band][episode]) +
                                     numpy.array(self.channel.noise[episode][band]))
            observation_samples.append(obs_per_band)
        # The observation_samples member is a kxt matrix
        return observation_samples

    # The observation procedure required for the Neyman-Pearson detector
    def make_sampled_observations_across_all_episodes(self):
        observation_samples = []
        for band in range(0, self.number_of_channels):
            obs_per_band = []
            for episode in range(0, self.number_of_episodes):
                obs_per_band.append(list((numpy.array(self.channel.impulse_response[episode][band]) * self.
                                          true_pu_occupancy_states[band][episode]) +
                                         numpy.array(self.channel.noise[episode][band])))
            observation_samples.append(obs_per_band)
        return observation_samples

    # Termination sequence
    def __exit__(self, exc_type, exc_val, exc_tb):
        print('[INFO] SecondaryUser Termination: Tearing things down...')


# Channel Selection Strategy Generator
# Emulates an RL agent or a Multi-Armed Bandit
class ChannelSelectionStrategyGenerator(object):

    # Initialization
    def __init__(self, _number_of_channels, _number_of_iterations):
        print('[INFO] ChannelSelectionStrategyGenerator Initialization: Bringing things up...')
        # The number of channels in the discretized spectrum of interest
        self.number_of_channels = _number_of_channels
        # The number of iterations to perform in random sensing
        self.number_of_iterations = _number_of_iterations
        # The discretized spectrum of interest
        self.discretized_spectrum = [k for k in range(0, self.number_of_channels)]
        # I'm saving this in order to evaluate the duals at a later stage
        self.random_sensing_strategy = []

    # Uniform Sensing
    def uniform_sensing(self):
        # Array of tuples with varying k
        channel_selection_strategies_based_on_uniform_sensing = []
        k = 0
        while k < self.number_of_channels - 1:
            i = 0
            temp_array = []
            while i < self.number_of_channels:
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
        for iteration in range(0, self.number_of_iterations):
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


# This entity evaluates the emission probabilities, i.e. P(y|x)
class EmissionEvaluator(object):

    # Initialization sequence
    def __init__(self, _noise_variance, _impulse_response_variance):
        print('[INFO] EmissionEvaluator Initialization: Bringing things up...')
        # Variance of the AWGN samples
        self.noise_variance = _noise_variance
        # Variance of the Channel Impulse Response samples
        self.impulse_response_variance = _impulse_response_variance

    # Get the Emission Probabilities -> P(y|x)
    def get_emission_probabilities(self, state, observation_sample):
        # If the channel is not observed, i.e. if the observation is [phi] or [0], report m_r(y_i) as 1
        # The Empty Place-Holder value is 0
        if observation_sample == 0:
            return 1
        # Normal Emission Estimation using the distribution of the observations given the state
        else:
            emission_probability = scipy.stats.norm(0, numpy.sqrt(
                (self.impulse_response_variance * state) + self.noise_variance)).pdf(observation_sample)
            return emission_probability

    # Termination sequence
    def __exit__(self, exc_type, exc_val, exc_tb):
        print('[INFO] EmissionEvaluator Termination: Tearing things down...')


# The constrained/unconstrained non-POMDP agent
# The double Markov chain Viterbi algorithm with/without complete observations
class DoubleMarkovChainViterbiAlgorithm(object):
    # Start probabilities of PU occupancy per frequency band
    BAND_START_PROBABILITIES = namedtuple('BandStartProbabilities', ['idle', 'occupied'])

    # Occupancy States (IDLE, OCCUPIED)
    OCCUPANCY_STATES = (OccupancyState.idle, OccupancyState.occupied)

    # Value function named tuple
    VALUE_FUNCTION_NAMED_TUPLE = namedtuple('ValueFunction',
                                            ['current_value', 'previous_temporal_state', 'previous_spatial_state'])

    # Initialization Sequence
    def __init__(self, _number_of_channels, _number_of_episodes, _emission_evaluator, _true_pu_occupancy_states,
                 _observation_samples, _spatial_start_probabilities, _temporal_start_probabilities,
                 _spatial_transition_probabilities_matrix, _temporal_transition_probabilities_matrix, _mu, _agent_id):
        print('[INFO] DoubleMarkovChainViterbiAlgorithm Initialization: Bringing things up ...')
        # The number of channels in the discretized spectrum of interest
        self.number_of_channels = _number_of_channels
        # The number of episodes of interaction of this constrained/unconstrained non-POMDP agent with the environment
        self.number_of_episodes = _number_of_episodes
        # The emission evaluator
        self.emission_evaluator = _emission_evaluator
        # True PU Occupancy state
        self.true_pu_occupancy_states = _true_pu_occupancy_states
        # The observed samples at the SU receiver
        self.observation_samples = _observation_samples
        # The start probabilities
        # For now, the same steady-state model across both chains
        self.start_probabilities = self.BAND_START_PROBABILITIES(idle=_spatial_start_probabilities[0],
                                                                 occupied=_spatial_start_probabilities[1])
        # The transition probability matrix
        # For now, the same transition model across both chains
        self.transition_probabilities_matrix = _spatial_transition_probabilities_matrix
        # The missed detections penalty term
        self.mu = _mu
        # The agent ID
        self.agent_id = _agent_id

    # Get the start probabilities from the named tuple - a simple getter utility method exclusive to this class
    def get_start_probabilities(self, state):
        # name or value
        if state == 'occupied' or state == '1':
            return self.start_probabilities.occupied
        else:
            return self.start_probabilities.idle

    # Return the transition probabilities from the transition probabilities matrix - two dimensions
    # Using the concept of marginal probability here as I did for simulating the PrimaryUser behavior
    def get_transition_probabilities(self, temporal_prev_state, spatial_prev_state, current_state):
        # temporal_prev_state is unused here...
        print('[DEBUG] DoubleMarkovChainViterbiAlgorithm get_transition_probabilities: current_state - {}, '
              'spatial_previous_state - {}, temporal_previous_state - {}'.format(current_state, spatial_prev_state,
                                                                                 temporal_prev_state))
        return self.transition_probabilities_matrix[spatial_prev_state][current_state]

    # Return the transition probabilities from the transition probabilities matrix - single dimension
    def get_transition_probabilities_single(self, row, column):
        return self.transition_probabilities_matrix[row][column]

    # Evaluate the Probability of False Alarm P_FA
    # P(\hat{X_k} = 1 | X_k = 0) \forall k \in \{0, 1, 2, ...., K-1\}
    # Relative Frequency approach to estimate this parameter
    # A global perspective is no longer needed. I need an episodic perspective.
    @deprecated
    def get_probability_of_false_alarm(self, estimated_states):
        idle_count = 0
        number_of_false_alarms = 0
        for channel_index in range(0, self.number_of_channels):
            for time_index in range(0, self.number_of_episodes):
                pu_state = self.true_pu_occupancy_states[channel_index][time_index]
                if pu_state == 0:
                    idle_count += 1
                    if estimated_states[channel_index][time_index] == 1:
                        number_of_false_alarms += 1
        if idle_count == 0:
            return 0
        return number_of_false_alarms / idle_count

    # Evaluate the Probability of Missed Detection P_MD
    # P(\hat{X_k} = 0 | X_k = 1) \forall k \in \{0, 1, 2, ...., K-1\}
    # Relative Frequency approach to estimate this parameter
    # A global perspective no longer needed. I need an episodic perspective.
    @deprecated
    def get_probability_of_missed_detection(self, estimated_states):
        occupancies = 0
        number_of_missed_detections = 0
        for channel_index in range(0, self.number_of_channels):
            for time_index in range(0, self.number_of_episodes):
                pu_state = self.true_pu_occupancy_states[channel_index][time_index]
                if pu_state == 1:
                    occupancies += 1
                    if estimated_states[channel_index][time_index] == 0:
                        number_of_missed_detections += 1
        if occupancies == 0:
            return 0
        return number_of_missed_detections / occupancies

    # Get the utility obtained by this non-POMDP agent
    def get_episodic_utility(self, estimated_state_vector, episode):
        idle_count = 0
        occupancies = 0
        false_alarms = 0
        missed_detections = 0
        for channel in range(0, self.number_of_channels):
            if self.true_pu_occupancy_states[channel][episode] == 0:
                idle_count += 1
                if estimated_state_vector[channel] == 1:
                    false_alarms += 1
            if self.true_pu_occupancy_states[channel][episode] == 1:
                occupancies += 1
                if estimated_state_vector[channel] == 0:
                    missed_detections += 1
        return ((lambda: (1 - (false_alarms / idle_count)), lambda: 1)[idle_count == 0]()) + (self.mu * (
            (lambda: missed_detections / occupancies, lambda: 0)[occupancies == 0]()))

    # Output the estimated state of the frequency bands in the wideband spectrum of interest
    def estimate_pu_occupancy_states(self):
        previous_state_spatial = None
        previous_state_temporal = None
        # Estimated states - kxt matrix
        estimated_states = []
        for x in range(0, self.number_of_channels):
            estimated_states[x] = []
        value_function_collection = []
        # A value function collection to store and index the calculated value functions across t and k
        for k in range(0, self.number_of_channels):
            row = []
            for x in range(self.number_of_episodes):
                row[x] = dict()
            value_function_collection[k] = row
        # t = 0 and k = 0 - No previous state to base the Markovian Correlation on in either dimension
        for state in OccupancyState:
            current_value = self.emission_evaluator.get_emission_probabilities(state.value,
                                                                               self.observation_samples[0][0]) * \
                            self.get_start_probabilities(state.name)
            value_function_collection[0][0][state.name] = self.VALUE_FUNCTION_NAMED_TUPLE(current_value=current_value,
                                                                                          previous_temporal_state=None,
                                                                                          previous_spatial_state=None)
        # First row - Only temporal correlation
        i = 0
        for j in range(1, self.number_of_episodes):
            # Trying to find the max pointer here ...
            for state in OccupancyState:
                # Again finishing off the [0] index first
                max_pointer = self.get_transition_probabilities_single(OccupancyState.idle.value,
                                                                       state.value) * \
                              value_function_collection[i][j - 1][OccupancyState.idle.name].current_value
                # Using IDLE as the confirmed previous state
                confirmed_previous_state = OccupancyState.idle.name
                for candidate_previous_state in OccupancyState:
                    if candidate_previous_state.name == OccupancyState.idle.name:
                        # Already done
                        continue
                    else:
                        pointer = self.get_transition_probabilities_single(candidate_previous_state.value,
                                                                           state.value) * \
                                  value_function_collection[i][j - 1][
                                      candidate_previous_state.name].current_value
                        if pointer > max_pointer:
                            max_pointer = pointer
                            confirmed_previous_state = candidate_previous_state.name
                current_value = max_pointer * self.emission_evaluator.get_emission_probabilities(
                    state.value, self.observation_samples[i][j])
                value_function_collection[i][j][state.name] = self.VALUE_FUNCTION_NAMED_TUPLE(
                    current_value=current_value, previous_temporal_state=confirmed_previous_state,
                    previous_spatial_state=None)
        # First column - Only spatial correlation
        j = 0
        for i in range(1, self.number_of_channels):
            # Trying to find the max pointer here ...
            for state in OccupancyState:
                # Again finishing off the [0] index first
                max_pointer = self.get_transition_probabilities_single(OccupancyState.idle.value,
                                                                       state.value) * \
                              value_function_collection[i - 1][j][
                                  OccupancyState.idle.name].current_value
                confirmed_previous_state = OccupancyState.idle.name
                for candidate_previous_state in OccupancyState:
                    if candidate_previous_state.name == OccupancyState.idle.name:
                        # Already done
                        continue
                    else:
                        pointer = self.get_transition_probabilities_single(candidate_previous_state.value,
                                                                           state.value) * \
                                  value_function_collection[i - 1][j][
                                      candidate_previous_state.name].current_value
                        if pointer > max_pointer:
                            max_pointer = pointer
                            confirmed_previous_state = candidate_previous_state.name
                current_value = max_pointer * self.emission_evaluator.get_emission_probabilities(
                    state.value, self.observation_samples[i][j])
                value_function_collection[i][j][state.name] = self.VALUE_FUNCTION_NAMED_TUPLE(
                    current_value=current_value, previous_temporal_state=None,
                    previous_spatial_state=confirmed_previous_state)
        # I'm done with the first row and first column
        # Moving on to the other rows and columns
        for i in range(1, self.number_of_channels):
            # For every row, I'm going across laterally (across columns) and populating the value_function_collection
            for j in range(1, self.number_of_episodes):
                for state in OccupancyState:
                    # Again finishing off the [0] index first
                    max_pointer = self.get_transition_probabilities(OccupancyState.idle.value,
                                                                    OccupancyState.idle.value, state.value) * \
                                  value_function_collection[i][j - 1][OccupancyState.idle.name].current_value * \
                                  value_function_collection[i - 1][j][OccupancyState.idle.name].current_value
                    confirmed_previous_state_temporal = OccupancyState.idle.name
                    confirmed_previous_state_spatial = OccupancyState.idle.name
                    for candidate_previous_state_temporal in OccupancyState:
                        for candidate_previous_state_spatial in OccupancyState:
                            if candidate_previous_state_temporal.name == OccupancyState.idle.name and \
                                    candidate_previous_state_spatial.name == OccupancyState.idle.name:
                                # Already done
                                continue
                            else:
                                pointer = self.get_transition_probabilities(candidate_previous_state_temporal.value,
                                                                            candidate_previous_state_spatial.value,
                                                                            state.value) * \
                                          value_function_collection[i][j - 1][
                                              candidate_previous_state_temporal.name].current_value * \
                                          value_function_collection[i - 1][j][
                                              candidate_previous_state_spatial.name].current_value
                                if pointer > max_pointer:
                                    max_pointer = pointer
                                    confirmed_previous_state_temporal = candidate_previous_state_temporal.name
                                    confirmed_previous_state_spatial = candidate_previous_state_spatial.name
                    # Now, I have the value function for this i and this j
                    # Populate the value function collection with this value
                    # I found maximum of Double Markov Chain value functions from the past and now I'm multiplying it...
                    # ...with the emission probability of this particular observation
                    current_value = max_pointer * self.emission_evaluator.get_emission_probabilities(
                        state.value,
                        self.observation_samples[i][j])
                    value_function_collection[i][j][state.name] = self.VALUE_FUNCTION_NAMED_TUPLE(
                        current_value=current_value, previous_temporal_state=confirmed_previous_state_temporal,
                        previous_spatial_state=confirmed_previous_state_spatial)
        # I think the forward path is perfect
        # I have doubts in the backtrack path
        max_value = 0
        # Finding the max value among the named tuples
        for _value in value_function_collection[-1][-1].values():
            if _value.current_value > max_value:
                max_value = _value.current_value
        # Finding the state corresponding to this max_value and using this as the final confirmed state to ...
        # ...backtrack and find the previous states
        for k, v in value_function_collection[-1][-1].items():
            if v.current_value == max_value:
                estimated_states[self.number_of_channels - 1].append(self.value_from_name(k))
                previous_state_temporal = k
                previous_state_spatial = k
                break
        # Backtracking
        for i in range(self.number_of_channels - 1, -1, -1):
            for j in range(self.number_of_episodes - 1, -1, -1):
                if len(estimated_states[i]) == 0:
                    estimated_states[i].insert(0, self.value_from_name(
                        value_function_collection[i + 1][j][previous_state_spatial].previous_spatial_state))
                    previous_state_temporal = value_function_collection[i][j][
                        previous_state_spatial].previous_temporal_state
                    continue
                estimated_states[i].insert(0, self.value_from_name(
                    value_function_collection[i][j][previous_state_temporal].previous_temporal_state))
                previous_state_temporal = value_function_collection[i][j][
                    previous_state_temporal].previous_temporal_state
            previous_state_spatial = value_function_collection[i][self.number_of_episodes - 1][
                previous_state_spatial].previous_spatial_state
        utilities = []
        for time_index in range(0, self.number_of_episodes):
            utilities.append(self.get_episodic_utility([estimated_states[k][time_index] for k in range(
                0, self.number_of_channels)], time_index))
        return {'id': self.agent_id, 'utilities': utilities}

    # Get enumeration field value from name
    @staticmethod
    def value_from_name(name):
        if name == 'occupied':
            return OccupancyState.occupied.value
        else:
            return OccupancyState.idle.value

    # Exit strategy
    def __exit__(self, exc_type, exc_val, exc_tb):
        print('[INFO] DoubleMarkovChainViterbiAlgorithm Clean-up: Cleaning things up ...')


# The Markov Chain Parameter Estimator Algorithm (modified EM - Baum-Welch)
# This entity is employed by the POMDP agent in order to learn the model concurrently
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
                                            previous_state] * \
                self.get_emission_probabilities(nxt_state.value, observation_vector[self.number_of_chain_links - 1]) \
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
# This entity is employed during the operation of the POMDP agent
# For non-POMDP agents, I employ the DoubleMarkovChainViterbiAlgorithm
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
    # The estimated_state member is created by the action taken by the POMDP agent
    # Reward = (1-Probability_of_false_alarm) + Missed_Detection_Cost*Probability_of_missed_detection
    def roll(self, estimated_state, system_state):
        # Used to evaluate Missed Detection Probability
        correct_detection_count = 0
        actual_occupancy_count = 0
        # Used to evaluate False Alarm Probability
        false_alarm_count = 0
        actual_idle_count = 0
        for i in range(0, len(estimated_state)):
            if system_state[i] == 1:
                actual_occupancy_count += 1
                if estimated_state[i] == 1:
                    correct_detection_count += 1
            else:
                actual_idle_count += 1
                if estimated_state[i] == 1:
                    false_alarm_count += 1
        false_alarm_probability = (lambda: 0,
                                   lambda: (false_alarm_count / actual_idle_count))[actual_idle_count is
                                                                                    not 0]()
        missed_detection_probability = (lambda: 0,
                                        lambda: 1 - (correct_detection_count / actual_occupancy_count))[
            actual_occupancy_count is not 0]()
        # Reward = (1-Probability_of_false_alarm) + Missed_Detection_Cost*Probability_of_missed_detection
        return (1 - false_alarm_probability) + (self.mu * missed_detection_probability)

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


# This entity encapsulates the POMDP Approximate Point-based Value Iteration algorithm named The PERSEUS algorithm
# Training to fine-tune the belief space -> Perform backup until convergence -> Re-sample using the most recent policy
# References to the StateEstimation algorithm in the belief analysis phase
# POMDP agent with Model Foresight and no belief simplification
class AdaptiveIntelligenceWithModelForesight(object):

    # Setup the Markov Chain
    @staticmethod
    def setup_markov_chain(_pi, _p, _correlation_class):
        print(
            '[INFO] AdaptiveIntelligenceWithModelForesight setup_markov_chain: Setting up the Markov Chain...')
        transient_markov_chain_object = MarkovChain()
        transient_markov_chain_object.set_markovian_correlation_class(_correlation_class)
        transient_markov_chain_object.set_start_probability_parameter(_pi)
        transient_markov_chain_object.set_transition_probability_parameter(_p)
        return transient_markov_chain_object

    # Initialization sequence
    def __init__(self, _number_of_channels, _number_of_sampling_rounds, _number_of_episodes, _exploration_period,
                 _noise_mean, _noise_variance, _impulse_response_mean, _impulse_response_variance, _penalty,
                 _limitation, _confidence_bound, _gamma, _agent_id):
        print('[INFO] AdaptiveIntelligenceWithModelForesight Initialization: Bringing things up...')
        # The number of channels in the discretized spectrum of interest
        self.number_of_channels = _number_of_channels
        # The number of sampling rounds in each episode
        self.number_of_sampling_rounds = _number_of_sampling_rounds
        # The number of time slots of interaction of the POMDP agent with the radio environment
        self.number_of_episodes = _number_of_episodes
        # The exploration period of the PERSEUS algorithm
        self.exploration_period = _exploration_period
        # The mean of the AWGN samples
        self.noise_mean = _noise_mean
        # The variance of the AWGN samples
        self.noise_variance = _noise_variance
        # The mean of the impulse response samples
        self.impulse_response_mean = _impulse_response_mean
        # The variance of the impulse response samples
        self.impulse_response_variance = _impulse_response_variance
        # The penalty for missed detection, i.e. PU interference
        self.penalty = _penalty
        # The SU sensing limitation
        self.limitation = _limitation
        # The confidence bound for convergence analysis
        self.confidence_bound = _confidence_bound
        # The discount factor employed in the Bellman equation
        self.gamma = _gamma
        # P(Occupied) - The steady state probability of being occupied
        self.pi = 0.6
        # P(Occupied|Idle) - The transition probability parameter
        self.p = 0.3
        # These have been learnt prior to triggering PERSEUS
        self.transition_probabilities_matrix = {0: {1: 0.3, 0: 0.7}, 1: {0: 0.2, 1: 0.8}}
        # The Utility object
        self.util = Util()
        # Start Probabilities Dict
        self.start_probabilities_dict = self.util.construct_start_probabilities_dict(self.pi)
        # Setup the Spatial Markov Chain
        self.spatial_markov_chain = self.setup_markov_chain(self.pi, self.p, MarkovianCorrelationClass.spatial)
        # Setup the Temporal Markov Chain
        self.temporal_markov_chain = self.setup_markov_chain(self.pi, self.p,
                                                             MarkovianCorrelationClass.temporal)
        # Channel
        self.channel = Channel(self.number_of_channels, self.number_of_sampling_rounds, self.number_of_episodes,
                               self.noise_mean, self.noise_variance, self.impulse_response_mean,
                               self.impulse_response_variance)
        # The Emission Evaluator
        self.emission_evaluator = EmissionEvaluator(self.noise_variance, self.impulse_response_variance)
        # Primary User
        self.primary_user = PrimaryUser(self.number_of_channels, self.number_of_episodes,
                                        self.spatial_markov_chain, self.temporal_markov_chain, self.util)
        self.primary_user.simulate_occupancy_behavior()
        # Secondary User
        self.secondary_user = SecondaryUser(self.number_of_channels, self.number_of_sampling_rounds,
                                            self.number_of_episodes, self.channel,
                                            self.primary_user.occupancy_behavior_collection, self.limitation)
        # Sweepstakes - The Reward Analyzer
        self.sweepstakes = Sweepstakes(self.primary_user, self.penalty)
        # The Oracle
        self.oracle = Oracle(self.number_of_episodes)
        # State Estimator
        self.state_estimator = StateEstimator(self.number_of_channels, self.number_of_sampling_rounds, None,
                                              self.start_probabilities_dict,
                                              self.transition_probabilities_matrix,
                                              self.emission_evaluator)
        # Utilities as the algorithm progresses towards optimality
        self.utilities = []
        # The number of policy changes as the algorithm progresses towards optimality
        self.policy_changes = []
        # Belief choice for value function tracking
        self.belief_choice = None
        # The progression of the value function as the algorithm progresses towards optimality
        self.value_function_changes_array = []
        # All possible states
        self.all_possible_states = list(map(list, itertools.product([0, 1], repeat=self.number_of_channels)))
        # All possible actions based on my SU sensing limitations
        self.all_possible_actions = []
        for state in self.all_possible_states:
            if sum(state) == self.limitation:
                self.all_possible_actions.append(state)
        # The agent id
        self.agent_id = _agent_id

    # Get Emission Probabilities for the Belief Update sequence
    def get_emission_probability(self, observations, state):
        emission_probability = 1
        # round_choice to introduce some randomness into the emission evaluations
        round_choice = random.choice([k for k in range(0, self.number_of_sampling_rounds)])
        # Given the system state, the emissions are independent of each other
        for index in range(0, self.number_of_channels):
            emission_probability = emission_probability * self.emission_evaluator.get_emission_probabilities(
                state[index], observations[index][round_choice])
        return emission_probability

    # Get State Transition Probabilities for the Belief Update sequence
    def get_transition_probability(self, prev_state, next_state, transition_probabilities_matrix):
        # Temporal/Episodic change for the first channel
        transition_probability = transition_probabilities_matrix[prev_state[0]][next_state[0]]
        for index in range(1, self.number_of_channels):
            transition_probability = transition_probability * transition_probabilities_matrix[next_state[index - 1]][
                next_state[index]]
        return transition_probability

    # Get the normalization constant
    def get_normalization_constant(self, previous_belief_vector, observations):
        # The normalization constant
        normalization_constant = 0
        # Calculate the normalization constant in the belief update formula
        for _next_state in self.all_possible_states:
            multiplier = 0
            for _prev_state in self.all_possible_states:
                multiplier += self.get_transition_probability(_prev_state, _next_state,
                                                              self.transition_probabilities_matrix) * \
                              previous_belief_vector[''.join(str(k) for k in _prev_state)]
            normalization_constant += self.get_emission_probability(observations, _next_state) * multiplier
        return normalization_constant

    # Belief Update sequence
    def belief_update(self, observations, previous_belief_vector, new_state,
                      transition_probabilities_matrix, normalization_constant):
        multiplier = 0
        # Calculate the numerator in the belief update formula
        # \vec{x} \in \mathcal{X} in your belief update rule
        for prev_state in self.all_possible_states:
            multiplier += self.get_transition_probability(prev_state,
                                                          new_state,
                                                          transition_probabilities_matrix) * previous_belief_vector[
                              ''.join(str(k) for k in prev_state)]
        numerator = self.get_emission_probability(observations, new_state) * multiplier
        return numerator / normalization_constant

    # Randomly explore the environment and collect a set of beliefs B of reachable belief points
    # ...strategy as a part of the modified PERSEUS strategy
    def random_exploration(self, policy):
        reachable_beliefs = dict()
        state_space_size = 2 ** self.number_of_channels
        # Uniform belief assignment to all states in the state space
        initial_belief_vector = dict()
        for state in self.all_possible_states:
            state_key = ''.join(str(k) for k in state)
            initial_belief_vector[state_key] = 1 / state_space_size
        previous_belief_vector = initial_belief_vector
        reachable_beliefs['0'] = initial_belief_vector
        if policy is not None:
            print('[WARN] AdaptiveIntelligenceWithModelForesight random_exploration: '
                  'Specific policy exploration is yet to be implemented')
        print('[INFO] AdaptiveIntelligenceWithModelForesight random_exploration: '
              'Using a truly random exploration strategy')
        # Start exploring
        for episode in range(1, self.exploration_period):
            # Perform the sensing action and make the observations
            # Making observations by choosing a random sensing action
            observations = self.secondary_user.make_observations(episode, random.choice(self.all_possible_actions))
            # Perform the Belief Update
            updated_belief_vector = dict()
            # Belief sum for this updated belief vector
            belief_sum = 0
            # Calculate the denominator which is nothing but the normalization constant
            # Calculate normalization constant only once...
            denominator = self.get_normalization_constant(previous_belief_vector, observations)
            # Possible next states to update the belief, i.e. b(\vec{x}')
            for state in self.all_possible_states:
                state_key = ''.join(str(k) for k in state)
                belief_val = self.belief_update(observations, previous_belief_vector, state,
                                                self.transition_probabilities_matrix, denominator)
                # Belief sum for Kolmogorov validation
                belief_sum += belief_val
                updated_belief_vector[state_key] = belief_val
            # Normalization to get valid belief vectors (satisfying axioms of probability measures)
            updated_belief_information = self.util.normalize(updated_belief_vector, belief_sum)
            if updated_belief_information[1] is False:
                raise ArithmeticError('The belief is a probability distribution over the state space. It should sum '
                                      'to one!')
            # Add the new belief vector to the reachable beliefs set
            reachable_beliefs[str(episode)] = updated_belief_information[0]
            print('[INFO] AdaptiveIntelligenceWithModelForesight random_exploration: {}% Exploration '
                  'completed'.format(int(((episode + 1) / self.exploration_period) * 100)))
        return reachable_beliefs

    # Initialization
    def initialize(self, reachable_beliefs):
        # V_0 for the reachable beliefs
        value_function_collection = dict()
        # Default action is not sensing anything - a blind guess from just the noise
        default_action = [k - k for k in range(0, self.number_of_channels)]
        for belief_key in reachable_beliefs.keys():
            value_function_collection[belief_key] = (-10, default_action)
        return value_function_collection

    # Calculate Utility
    def calculate_utility(self, policy_collection):
        utility = 0
        for key, value in policy_collection.items():
            system_state = []
            for channel in range(0, self.number_of_channels):
                # int(key) refers to the episode number
                system_state.append(self.primary_user.occupancy_behavior_collection[channel][int(key)])
            # In that episode, the POMDP agent believed that the system is in a certain probabilistic distribution...
            # over the state vector, i.e. the belief, and based on this it recommended the best possible action to be...
            # taken in that state - obtained by going through the backup procedure...
            observation_samples = self.secondary_user.make_observations(int(key), value[1])
            self.state_estimator.observation_samples = observation_samples
            estimated_state = self.state_estimator.estimate_pu_occupancy_states()
            print('[DEBUG] AdaptiveIntelligenceWithModelForesight calculate_utility: '
                  'Estimated PU Occupancy states - {}'.format(str(estimated_state)))
            utility += self.sweepstakes.roll(estimated_state, system_state)
        return utility

    # The Backup stage
    def backup(self, stage_number, reachable_beliefs, previous_stage_value_function_collection):
        # Just a reassignment because the reachable_beliefs turns out to be a mutable collection
        unimproved_belief_points = {}
        for key, value in reachable_beliefs.items():
            unimproved_belief_points[key] = value
        # This assignment will help me in the utility calculation within each backup stage
        next_stage_value_function_collection = dict()
        # A simple dict copy - just to be safe from mutations
        for k, v in previous_stage_value_function_collection.items():
            next_stage_value_function_collection[k] = v
        number_of_belief_changes = 0
        # While there are still some un-improved belief points
        while len(unimproved_belief_points) is not 0:
            print('[INFO] AdaptiveIntelligenceWithModelForesight backup: '
                  'Size of unimproved belief set = {}'.format(len(unimproved_belief_points)))
            # Sample a belief point uniformly at random from \tilde{B}
            belief_sample_key = random.choice(list(unimproved_belief_points.keys()))
            belief_sample = unimproved_belief_points[belief_sample_key]
            max_value_function = -10 ** 9
            max_action = None
            for action in self.all_possible_actions:
                new_belief_vector = {}
                new_belief_sum = 0
                # Make observations based on the chosen action
                observation_samples = self.secondary_user.make_observations(stage_number, action)
                # Estimate the System State
                self.state_estimator.observation_samples = observation_samples
                estimated_system_state = self.state_estimator.estimate_pu_occupancy_states()
                print('[DEBUG] AdaptiveIntelligenceWithModelForesight backup: '
                      'Estimated PU Occupancy states - {}'.format(str(estimated_system_state)))
                reward_sum = 0
                normalization_constant = 0
                for state in self.all_possible_states:
                    emission_probability = self.get_emission_probability(observation_samples, state)
                    multiplier = 0
                    for prev_state in self.all_possible_states:
                        multiplier += self.get_transition_probability(prev_state, state,
                                                                      self.transition_probabilities_matrix) * \
                                      belief_sample[''.join(str(k) for k in prev_state)]
                    normalization_constant += emission_probability * multiplier
                    reward_sum += self.sweepstakes.roll(estimated_system_state, state) * belief_sample[
                        ''.join(str(k) for k in state)]
                # Belief Update
                for state in self.all_possible_states:
                    state_key = ''.join(str(k) for k in state)
                    value_of_belief = self.belief_update(observation_samples, belief_sample, state,
                                                         self.transition_probabilities_matrix,
                                                         normalization_constant)
                    new_belief_sum += value_of_belief
                    new_belief_vector[state_key] = value_of_belief
                # Normalization to get valid belief vectors (satisfying axioms of probability measures)
                new_normalized_belief_information = self.util.normalize(new_belief_vector, new_belief_sum)
                if new_normalized_belief_information[1] is False:
                    raise ArithmeticError('The belief is a probability distribution over the state space. It should '
                                          'sum to one!')
                # Updated re-assignment
                new_belief_vector = new_normalized_belief_information[0]
                highest_belief_key = max(new_belief_vector, key=new_belief_vector.get)
                # You could've used an OrderedDict here to simplify operations
                # Find the closest pilot belief and its associated value function
                relevant_data = {episode_key: belief[highest_belief_key] for episode_key, belief in
                                 reachable_beliefs.items()}
                pilot_belief_key = max(relevant_data, key=relevant_data.get)
                internal_term = reward_sum + (self.gamma * normalization_constant *
                                              previous_stage_value_function_collection[pilot_belief_key][0])
                if internal_term > max_value_function:
                    max_value_function = internal_term
                    max_action = action
            if round(max_value_function, 3) > round(previous_stage_value_function_collection[belief_sample_key][0], 3):
                print('[DEBUG] AdaptiveIntelligenceWithModelForesight backup: '
                      '[Action: {} and New_Value_Function: {}] pair improves the existing policy - '
                      'Corresponding sequence triggered!'.format(str(max_action), str(max_value_function) + ' > ' +
                                                                 str(previous_stage_value_function_collection[
                                                                         belief_sample_key][0])))
                # Note here that del mutates the contents of the dict for everyone who has a reference to it
                del unimproved_belief_points[belief_sample_key]
                next_stage_value_function_collection[belief_sample_key] = (max_value_function, max_action)
                number_of_belief_changes += 1
            else:
                next_stage_value_function_collection[belief_sample_key] = previous_stage_value_function_collection[
                    belief_sample_key]
                # Note here that del mutates the contents of the dict for everyone who has a reference to it
                del unimproved_belief_points[belief_sample_key]
                max_action = previous_stage_value_function_collection[belief_sample_key][1]
            for belief_point_key in list(unimproved_belief_points.keys()):
                print('[DEBUG] AdaptiveIntelligenceWithModelForesight backup: '
                      'Improving the other belief points...')
                normalization_constant = 0
                reward_sum = 0
                observation_samples = self.secondary_user.make_observations(stage_number, max_action)
                # Estimate the System State
                self.state_estimator.observation_samples = observation_samples
                estimated_system_state = self.state_estimator.estimate_pu_occupancy_states()
                print('[DEBUG] AdaptiveIntelligenceWithModelForesight backup: '
                      'Estimated PU Occupancy states - {}'.format(str(estimated_system_state)))
                for state in self.all_possible_states:
                    emission_probability = self.get_emission_probability(observation_samples, state)
                    multiplier = 0
                    for prev_state in self.all_possible_states:
                        multiplier += self.get_transition_probability(prev_state, state,
                                                                      self.transition_probabilities_matrix) * \
                                      unimproved_belief_points[belief_point_key][''.join(str(k) for k in prev_state)]
                    normalization_constant += emission_probability * multiplier
                    reward_sum += self.sweepstakes.roll(estimated_system_state, state) * unimproved_belief_points[
                        belief_point_key][''.join(str(k) for k in state)]
                new_aux_belief_vector = {}
                new_aux_belief_sum = 0
                aux_belief_sample = unimproved_belief_points[belief_point_key]
                # Belief Update
                for state in self.all_possible_states:
                    state_key = ''.join(str(k) for k in state)
                    new_aux_belief_val = self.belief_update(observation_samples, aux_belief_sample, state,
                                                            self.transition_probabilities_matrix,
                                                            normalization_constant)
                    new_aux_belief_sum += new_aux_belief_val
                    new_aux_belief_vector[state_key] = new_aux_belief_val
                # Normalization to get valid belief vectors (satisfying axioms of probability measures)
                new_aux_normalized_belief_information = self.util.normalize(new_aux_belief_vector, new_aux_belief_sum)
                if new_aux_normalized_belief_information[1] is False:
                    raise ArithmeticError('The belief is a probability distribution over the state space. It should '
                                          'sum to one!')
                # Updated re-assignment
                new_aux_belief_vector = new_aux_normalized_belief_information[0]
                highest_belief_key = max(new_aux_belief_vector, key=new_aux_belief_vector.get)
                # You could've used an OrderedDict here to simplify operations
                # Find the closest pilot belief and its associated value function
                relevant_data = {episode_key: belief[highest_belief_key] for episode_key, belief in
                                 reachable_beliefs.items()}
                aux_pilot_belief_key = max(relevant_data, key=relevant_data.get)
                internal_term = reward_sum + (self.gamma * normalization_constant *
                                              previous_stage_value_function_collection[aux_pilot_belief_key][0])
                if round(internal_term, 3) > round(previous_stage_value_function_collection[belief_point_key][0], 3):
                    # Note here that del mutates the contents of the dict for everyone who has a reference to it
                    print('[DEBUG] AdaptiveIntelligenceWithModelForesight backup: Auxiliary points improved '
                          'by action {} with {}'.format(str(max_action), str(internal_term) + ' > ' +
                                                        str(previous_stage_value_function_collection[
                                                                belief_point_key][0])))
                    del unimproved_belief_points[belief_point_key]
                    next_stage_value_function_collection[belief_point_key] = (internal_term, max_action)
                    number_of_belief_changes += 1
            utility = self.calculate_utility(next_stage_value_function_collection)
            print('[INFO] AdaptiveIntelligenceWithModelForesight backup: '
                  'Logging all the relevant metrics within this Backup stage - [Utility: {}, #policy_changes: {}, '
                  'sampled_value_function: {}]'.format(utility, number_of_belief_changes,
                                                       next_stage_value_function_collection[self.belief_choice][0]))
            self.utilities.append(utility)
        return [next_stage_value_function_collection, number_of_belief_changes]

    # The PERSEUS algorithm
    # Calls to Random Exploration, Initialization, and Backup stages
    # Signed off by bkeshava on 01-May-2019
    def run_perseus(self):
        # Random Exploration - Get the set of reachable beliefs by randomly interacting with the radio environment
        reachable_beliefs = self.random_exploration(None)
        # Belief choice for value function tracking (visualization component)
        self.belief_choice = (lambda: self.belief_choice,
                              lambda: random.choice([
                                  k for k in reachable_beliefs.keys()]))[self.belief_choice is None]()
        # Initialization - Initializing to -10 for all beliefs in the reachable beliefs set
        initial_value_function_collection = self.initialize(reachable_beliefs)
        # Relevant collections
        previous_value_function_collection = initial_value_function_collection
        stage_number = self.exploration_period - 1
        # Utility addition for the initial value function
        utility = self.calculate_utility(previous_value_function_collection)
        print('[DEBUG] AdaptiveIntelligenceWithModelForesight run_perseus: '
              'Adding the utility metric for the initial value function - {}'.format(utility))
        self.utilities.append(utility)
        # Local confidence check for modelling policy convergence
        confidence = 0
        # Check for termination condition here...
        while confidence < self.confidence_bound:
            self.value_function_changes_array.append(previous_value_function_collection[self.belief_choice][0])
            stage_number += 1
            # We've reached the end of our allowed interaction time with the radio environment
            if stage_number == self.number_of_episodes:
                print('[WARN] AdaptiveIntelligenceWithModelForesight run_perseus: '
                      'We have reached the end of our allowed interaction time with the radio environment!')
                return
            # Backup to find \alpha -> Get V_{n+1} and #BeliefChanges
            backup_results = self.backup(stage_number, reachable_beliefs, previous_value_function_collection)
            print(
                '[DEBUG] AdaptiveIntelligenceWithModelForesight run_perseus: '
                'Backup for stage {} completed...'.format(stage_number - self.exploration_period + 1))
            next_value_function_collection = backup_results[0]
            belief_changes = backup_results[1]
            self.policy_changes.append(belief_changes)
            if len(next_value_function_collection) is not 0:
                previous_value_function_collection = next_value_function_collection
            if belief_changes is 0:
                print('[DEBUG] AdaptiveIntelligenceWithModelForesight run_perseus: '
                      'Confidence Update - {}'.format(confidence))
                confidence += 1
            else:
                confidence = 0
                print('[DEBUG] AdaptiveIntelligenceWithModelForesight run_perseus: '
                      'Confidence Stagnation/Fallback - {}'.format(confidence))
        optimal_utilities = []
        for episode_number, results_tuple in previous_value_function_collection.items():
            system_state = []
            for channel in range(0, self.number_of_channels):
                system_state.append(self.primary_user.occupancy_behavior_collection[channel][int(episode_number)])
            optimal_action = results_tuple[1]
            observation_samples = self.secondary_user.make_observations(int(episode_number), optimal_action)
            self.state_estimator.observation_samples = observation_samples
            estimated_states = self.state_estimator.estimate_pu_occupancy_states()
            optimal_utilities.append(self.sweepstakes.roll(estimated_states, system_state))
        return {'id': self.agent_id, 'utilities': optimal_utilities}

    # Termination sequence
    # Signed off by bkeshava on 01-May-2019
    def __exit__(self, exc_type, exc_val, exc_tb):
        print('[INFO] AdaptiveIntelligenceWithModelForesight Termination: Tearing things down...')


# This entity encapsulates the POMDP Approximate Point-based Value Iteration algorithm named The PERSEUS algorithm
# Training to fine-tune the belief space -> Perform backup until convergence -> Re-sample using the most recent policy
# References to the ParameterEstimation algorithm and the StateEstimation algorithm in the belief analysis phase
# Concurrent Model Learning with no simplification whatsoever
class ModelFreeAdaptiveIntelligence(object):

    # Setup the Markov Chain
    @staticmethod
    def setup_markov_chain(_pi, _p, _correlation_class):
        print('[INFO] ModelFreeAdaptiveIntelligence setup_markov_chain: Setting up the Markov Chain...')
        transient_markov_chain_object = MarkovChain()
        transient_markov_chain_object.set_markovian_correlation_class(_correlation_class)
        transient_markov_chain_object.set_start_probability_parameter(_pi)
        transient_markov_chain_object.set_transition_probability_parameter(_p)
        return transient_markov_chain_object

    # Initialization sequence
    # Initialization sequence
    def __init__(self, _number_of_channels, _number_of_sampling_rounds, _number_of_episodes, _exploration_period,
                 _noise_mean, _noise_variance, _impulse_response_mean, _impulse_response_variance, _penalty,
                 _limitation, _confidence_bound, _gamma, _epsilon, _agent_id):
        print('[INFO] ModelFreeAdaptiveIntelligence Initialization: Bringing things up...')
        # The Utility object
        self.util = Util()
        # The number of channels in the discretized spectrum of interest
        self.number_of_channels = _number_of_channels
        # The number of sampling rounds in each episode
        self.number_of_sampling_rounds = _number_of_sampling_rounds
        # The number of time slots of interaction of the POMDP agent with the radio environment
        self.number_of_episodes = _number_of_episodes
        # The exploration period of the PERSEUS algorithm
        self.exploration_period = _exploration_period
        # The mean of the AWGN samples
        self.noise_mean = _noise_mean
        # The variance of the AWGN samples
        self.noise_variance = _noise_variance
        # The mean of the impulse response samples
        self.impulse_response_mean = _impulse_response_mean
        # The variance of the impulse response samples
        self.impulse_response_variance = _impulse_response_variance
        # The penalty for missed detection, i.e. PU interference
        self.penalty = _penalty
        # The SU sensing limitation
        self.limitation = _limitation
        # The confidence bound for convergence analysis
        self.confidence_bound = _confidence_bound
        # The discount factor employed in the Bellman equation
        self.gamma = _gamma
        # P(Occupied) - The steady state probability of being occupied
        self.pi = 0.6
        # P(Occupied|Idle) - The transition probability parameter
        self.p = 0.3
        # The initial estimate of P(Occupied|Idle) a.k.a 'p'
        self.initial_p = 0.000005
        # Initial Transition Probabilities_Matrix
        self.initial_transition_probabilities_matrix = self.util.construct_transition_probability_matrix(self.initial_p,
                                                                                                         self.pi)
        # The convergence threshold for the parameter estimator
        self.epsilon = _epsilon
        # Start Probabilities Dict
        self.start_probabilities_dict = self.util.construct_start_probabilities_dict(self.pi)
        # Setup the Spatial Markov Chain
        self.spatial_markov_chain = self.setup_markov_chain(self.pi, self.p, MarkovianCorrelationClass.spatial)
        # Setup the Temporal Markov Chain
        self.temporal_markov_chain = self.setup_markov_chain(self.pi, self.p,
                                                             MarkovianCorrelationClass.temporal)
        # Channel
        self.channel = Channel(self.number_of_channels, self.number_of_sampling_rounds, self.number_of_episodes,
                               self.noise_mean, self.noise_variance, self.impulse_response_mean,
                               self.impulse_response_variance)
        # The Emission Evaluator
        self.emission_evaluator = EmissionEvaluator(self.noise_variance, self.impulse_response_variance)
        # Primary User
        self.primary_user = PrimaryUser(self.number_of_channels, self.number_of_episodes,
                                        self.spatial_markov_chain, self.temporal_markov_chain, self.util)
        self.primary_user.simulate_occupancy_behavior()
        # Secondary User
        self.secondary_user = SecondaryUser(self.number_of_channels, self.number_of_sampling_rounds,
                                            self.number_of_episodes, self.channel,
                                            self.primary_user.occupancy_behavior_collection, self.limitation)
        # Sweepstakes - The Reward Analyzer
        self.sweepstakes = Sweepstakes(self.primary_user, self.penalty)
        # The Oracle
        self.oracle = Oracle(self.number_of_episodes)
        # Parameter Estimator
        # Parameter Estimator - Modified Baum-Welch / Modified EM
        self.parameter_estimator = ParameterEstimator(self.number_of_channels, self.number_of_sampling_rounds,
                                                      self.initial_transition_probabilities_matrix,
                                                      self.start_probabilities_dict, None,
                                                      self.emission_evaluator, self.epsilon,
                                                      self.confidence_bound, self.util)
        # State Estimator
        self.state_estimator = StateEstimator(self.number_of_channels, self.number_of_sampling_rounds, None,
                                              self.start_probabilities_dict,
                                              self.initial_transition_probabilities_matrix,
                                              self.emission_evaluator)
        # Utilities as the algorithm progresses towards optimality
        self.utilities = []
        # The number of policy changes as the algorithm progresses towards optimality
        self.policy_changes = []
        # Belief choice for value function tracking
        self.belief_choice = None
        # The progression of the value function as the algorithm progresses towards optimality
        self.value_function_changes_array = []
        # All possible states
        self.all_possible_states = list(map(list, itertools.product([0, 1], repeat=self.number_of_channels)))
        # All possible actions based on my SU sensing limitations
        self.all_possible_actions = []
        for state in self.all_possible_states:
            if sum(state) == self.limitation:
                self.all_possible_actions.append(state)
        # The agent id
        self.agent_id = _agent_id

    # Get Emission Probabilities for the Belief Update sequence
    def get_emission_probability(self, observations, state):
        emission_probability = 1
        # round_choice to introduce some randomness into the emission evaluations
        round_choice = random.choice([k for k in range(0, self.number_of_sampling_rounds)])
        # Given the system state, the emissions are independent of each other
        for index in range(0, self.number_of_channels):
            emission_probability = emission_probability * self.emission_evaluator.get_emission_probabilities(
                state[index], observations[index][round_choice])
        return emission_probability

    # Get State Transition Probabilities for the Belief Update sequence
    def get_transition_probability(self, prev_state, next_state, transition_probabilities_matrix):
        # Temporal/Episodic change for the first channel
        transition_probability = transition_probabilities_matrix[prev_state[0]][next_state[0]]
        for index in range(1, self.number_of_channels):
            transition_probability = transition_probability * transition_probabilities_matrix[next_state[index - 1]][
                next_state[index]]
        return transition_probability

    # Get the normalization constant
    def get_normalization_constant(self, previous_belief_vector, observations, transition_probabilities_matrix):
        # The normalization constant
        normalization_constant = 0
        # Calculate the normalization constant in the belief update formula
        for _next_state in self.all_possible_states:
            multiplier = 0
            for _prev_state in self.all_possible_states:
                multiplier += self.get_transition_probability(_prev_state, _next_state,
                                                              transition_probabilities_matrix) * \
                              previous_belief_vector[''.join(str(k) for k in _prev_state)]
            normalization_constant += self.get_emission_probability(observations, _next_state) * multiplier
        return normalization_constant

    # Belief Update sequence
    def belief_update(self, observations, previous_belief_vector, new_state,
                      transition_probabilities_matrix, normalization_constant):
        multiplier = 0
        # Calculate the numerator in the belief update formula
        # \vec{x} \in \mathcal{X} in your belief update rule
        for prev_state in self.all_possible_states:
            multiplier += self.get_transition_probability(prev_state,
                                                          new_state,
                                                          transition_probabilities_matrix) * previous_belief_vector[
                              ''.join(str(k) for k in prev_state)]
        numerator = self.get_emission_probability(observations, new_state) * multiplier
        return numerator / normalization_constant

    # Randomly explore the environment and collect a set of beliefs B of reachable belief points
    # ...strategy as a part of the modified PERSEUS strategy
    def random_exploration(self, policy):
        reachable_beliefs = dict()
        state_space_size = 2 ** self.number_of_channels
        # Uniform belief assignment to all states in the state space
        initial_belief_vector = dict()
        for state in self.all_possible_states:
            state_key = ''.join(str(k) for k in state)
            initial_belief_vector[state_key] = 1 / state_space_size
        previous_belief_vector = initial_belief_vector
        reachable_beliefs['0'] = initial_belief_vector
        if policy is not None:
            print('[WARN] ModelFreeAdaptiveIntelligence random_exploration: '
                  'Specific policy exploration is yet to be implemented')
        print('[INFO] ModelFreeAdaptiveIntelligence random_exploration: '
              'Using a truly random exploration strategy')
        # Start exploring
        for episode in range(1, self.exploration_period):
            # Perform the sensing action and make the observations
            # Making observations by choosing a random sensing action
            observations = self.secondary_user.make_observations(episode, random.choice(self.all_possible_actions))
            # Set the observations in the ParameterEstimator
            self.parameter_estimator.observation_samples = observations
            # Estimate the transition probabilities matrix
            transition_probabilities_matrix = self.util.construct_transition_probability_matrix(
                self.parameter_estimator.estimate_parameters(), self.pi)
            # Perform the Belief Update
            updated_belief_vector = dict()
            # Belief sum for this updated belief vector
            belief_sum = 0
            # Calculate the denominator which is nothing but the normalization constant
            # Calculate normalization constant only once...
            denominator = self.get_normalization_constant(previous_belief_vector, observations,
                                                          transition_probabilities_matrix)
            # Possible next states to update the belief, i.e. b(\vec{x}')
            for state in self.all_possible_states:
                state_key = ''.join(str(k) for k in state)
                belief_val = self.belief_update(observations, previous_belief_vector, state,
                                                transition_probabilities_matrix, denominator)
                # Belief sum for Kolmogorov validation
                belief_sum += belief_val
                updated_belief_vector[state_key] = belief_val
            # Normalization to get valid belief vectors (satisfying axioms of probability measures)
            updated_belief_information = self.util.normalize(updated_belief_vector, belief_sum)
            if updated_belief_information[1] is False:
                raise ArithmeticError('The belief is a probability distribution over the state space. It should sum '
                                      'to one!')
            # Add the new belief vector to the reachable beliefs set
            reachable_beliefs[str(episode)] = updated_belief_information[0]
            print('[INFO] ModelFreeAdaptiveIntelligence random_exploration: {}% Exploration '
                  'completed'.format(int(((episode + 1) / self.exploration_period) * 100)))
        return reachable_beliefs

    # Initialization
    def initialize(self, reachable_beliefs):
        # V_0 for the reachable beliefs
        value_function_collection = dict()
        # Default action is not sensing anything - a blind guess from just the noise
        default_action = [k - k for k in range(0, self.number_of_channels)]
        for belief_key in reachable_beliefs.keys():
            value_function_collection[belief_key] = (-10, default_action)
        return value_function_collection

    # Calculate Utility
    def calculate_utility(self, policy_collection):
        utility = 0
        for key, value in policy_collection.items():
            system_state = []
            for channel in range(0, self.number_of_channels):
                # int(key) refers to the episode number
                system_state.append(self.primary_user.occupancy_behavior_collection[channel][int(key)])
            # In that episode, the POMDP agent believed that the system is in a certain probabilistic distribution...
            # over the state vector, i.e. the belief, and based on this it recommended the best possible action to be...
            # taken in that state - obtained by going through the backup procedure...
            observation_samples = self.secondary_user.make_observations(int(key), value[1])
            self.parameter_estimator.observation_samples = observation_samples
            transition_probabilities_matrix = self.util.construct_transition_probability_matrix(
                self.parameter_estimator.estimate_parameters(), self.pi)
            self.state_estimator.transition_probabilities = transition_probabilities_matrix
            self.state_estimator.observation_samples = observation_samples
            estimated_state = self.state_estimator.estimate_pu_occupancy_states()
            print('[DEBUG] ModelFreeAdaptiveIntelligence calculate_utility: '
                  'Estimated PU Occupancy states - {}'.format(str(estimated_state)))
            utility += self.sweepstakes.roll(estimated_state, system_state)
        return utility

    # The Backup stage
    def backup(self, stage_number, reachable_beliefs, previous_stage_value_function_collection):
        # Just a reassignment because the reachable_beliefs turns out to be a mutable collection
        unimproved_belief_points = {}
        for key, value in reachable_beliefs.items():
            unimproved_belief_points[key] = value
        # This assignment will help me in the utility calculation within each backup stage
        next_stage_value_function_collection = dict()
        # A simple dict copy - just to be safe from mutations
        for k, v in previous_stage_value_function_collection.items():
            next_stage_value_function_collection[k] = v
        number_of_belief_changes = 0
        # While there are still some un-improved belief points
        while len(unimproved_belief_points) is not 0:
            print('[INFO] ModelFreeAdaptiveIntelligence backup: '
                  'Size of unimproved belief set = {}'.format(len(unimproved_belief_points)))
            # Sample a belief point uniformly at random from \tilde{B}
            belief_sample_key = random.choice(list(unimproved_belief_points.keys()))
            belief_sample = unimproved_belief_points[belief_sample_key]
            max_value_function = -10 ** 9
            max_action = None
            for action in self.all_possible_actions:
                new_belief_vector = {}
                new_belief_sum = 0
                # Make observations based on the chosen action
                observation_samples = self.secondary_user.make_observations(stage_number, action)
                self.parameter_estimator.observation_samples = observation_samples
                transition_probabilities_matrix = self.util.construct_transition_probability_matrix(
                    self.parameter_estimator.estimate_parameters(), self.pi)
                # Estimate the System State
                self.state_estimator.transition_probabilities = transition_probabilities_matrix
                self.state_estimator.observation_samples = observation_samples
                estimated_system_state = self.state_estimator.estimate_pu_occupancy_states()
                print('[DEBUG] ModelFreeAdaptiveIntelligence backup: '
                      'Estimated PU Occupancy states - {}'.format(str(estimated_system_state)))
                reward_sum = 0
                normalization_constant = 0
                for state in self.all_possible_states:
                    emission_probability = self.get_emission_probability(observation_samples, state)
                    multiplier = 0
                    for prev_state in self.all_possible_states:
                        multiplier += self.get_transition_probability(prev_state, state,
                                                                      transition_probabilities_matrix) * \
                                      belief_sample[''.join(str(k) for k in prev_state)]
                    normalization_constant += emission_probability * multiplier
                    reward_sum += self.sweepstakes.roll(estimated_system_state, state) * belief_sample[
                        ''.join(str(k) for k in state)]
                # Belief Update
                for state in self.all_possible_states:
                    state_key = ''.join(str(k) for k in state)
                    value_of_belief = self.belief_update(observation_samples, belief_sample, state,
                                                         transition_probabilities_matrix,
                                                         normalization_constant)
                    new_belief_sum += value_of_belief
                    new_belief_vector[state_key] = value_of_belief
                # Normalization to get valid belief vectors (satisfying axioms of probability measures)
                new_normalized_belief_information = self.util.normalize(new_belief_vector, new_belief_sum)
                if new_normalized_belief_information[1] is False:
                    raise ArithmeticError('The belief is a probability distribution over the state space. It should '
                                          'sum to one!')
                # Updated re-assignment
                new_belief_vector = new_normalized_belief_information[0]
                highest_belief_key = max(new_belief_vector, key=new_belief_vector.get)
                # You could've used an OrderedDict here to simplify operations
                # Find the closest pilot belief and its associated value function
                relevant_data = {episode_key: belief[highest_belief_key] for episode_key, belief in
                                 reachable_beliefs.items()}
                pilot_belief_key = max(relevant_data, key=relevant_data.get)
                internal_term = reward_sum + (self.gamma * normalization_constant *
                                              previous_stage_value_function_collection[pilot_belief_key][0])
                if internal_term > max_value_function:
                    max_value_function = internal_term
                    max_action = action
            if round(max_value_function, 3) > round(previous_stage_value_function_collection[belief_sample_key][0], 3):
                print('[DEBUG] ModelFreeAdaptiveIntelligence backup: '
                      '[Action: {} and New_Value_Function: {}] pair improves the existing policy - '
                      'Corresponding sequence triggered!'.format(str(max_action), str(max_value_function) + ' > ' +
                                                                 str(previous_stage_value_function_collection[
                                                                         belief_sample_key][0])))
                # Note here that del mutates the contents of the dict for everyone who has a reference to it
                del unimproved_belief_points[belief_sample_key]
                next_stage_value_function_collection[belief_sample_key] = (max_value_function, max_action)
                number_of_belief_changes += 1
            else:
                next_stage_value_function_collection[belief_sample_key] = previous_stage_value_function_collection[
                    belief_sample_key]
                # Note here that del mutates the contents of the dict for everyone who has a reference to it
                del unimproved_belief_points[belief_sample_key]
                max_action = previous_stage_value_function_collection[belief_sample_key][1]
            for belief_point_key in list(unimproved_belief_points.keys()):
                print('[DEBUG] ModelFreeAdaptiveIntelligence backup: '
                      'Improving the other belief points...')
                normalization_constant = 0
                reward_sum = 0
                observation_samples = self.secondary_user.make_observations(stage_number, max_action)
                self.parameter_estimator.observation_samples = observation_samples
                transition_probabilities_matrix = self.util.construct_transition_probability_matrix(
                    self.parameter_estimator.estimate_parameters(), self.pi)
                # Estimate the System State
                self.state_estimator.transition_probabilities = transition_probabilities_matrix
                self.state_estimator.observation_samples = observation_samples
                estimated_system_state = self.state_estimator.estimate_pu_occupancy_states()
                print('[DEBUG] ModelFreeAdaptiveIntelligence backup: '
                      'Estimated PU Occupancy states - {}'.format(str(estimated_system_state)))
                for state in self.all_possible_states:
                    emission_probability = self.get_emission_probability(observation_samples, state)
                    multiplier = 0
                    for prev_state in self.all_possible_states:
                        multiplier += self.get_transition_probability(prev_state, state,
                                                                      transition_probabilities_matrix) * \
                                      unimproved_belief_points[belief_point_key][''.join(str(k) for k in prev_state)]
                    normalization_constant += emission_probability * multiplier
                    reward_sum += self.sweepstakes.roll(estimated_system_state, state) * unimproved_belief_points[
                        belief_point_key][''.join(str(k) for k in state)]
                new_aux_belief_vector = {}
                new_aux_belief_sum = 0
                aux_belief_sample = unimproved_belief_points[belief_point_key]
                # Belief Update
                for state in self.all_possible_states:
                    state_key = ''.join(str(k) for k in state)
                    new_aux_belief_val = self.belief_update(observation_samples, aux_belief_sample, state,
                                                            transition_probabilities_matrix,
                                                            normalization_constant)
                    new_aux_belief_sum += new_aux_belief_val
                    new_aux_belief_vector[state_key] = new_aux_belief_val
                # Normalization to get valid belief vectors (satisfying axioms of probability measures)
                new_aux_normalized_belief_information = self.util.normalize(new_aux_belief_vector, new_aux_belief_sum)
                if new_aux_normalized_belief_information[1] is False:
                    raise ArithmeticError('The belief is a probability distribution over the state space. It should '
                                          'sum to one!')
                # Updated re-assignment
                new_aux_belief_vector = new_aux_normalized_belief_information[0]
                highest_belief_key = max(new_aux_belief_vector, key=new_aux_belief_vector.get)
                # You could've used an OrderedDict here to simplify operations
                # Find the closest pilot belief and its associated value function
                relevant_data = {episode_key: belief[highest_belief_key] for episode_key, belief in
                                 reachable_beliefs.items()}
                aux_pilot_belief_key = max(relevant_data, key=relevant_data.get)
                internal_term = reward_sum + (self.gamma * normalization_constant *
                                              previous_stage_value_function_collection[aux_pilot_belief_key][0])
                if round(internal_term, 3) > round(previous_stage_value_function_collection[belief_point_key][0], 3):
                    # Note here that del mutates the contents of the dict for everyone who has a reference to it
                    print('[DEBUG] ModelFreeAdaptiveIntelligence backup: Auxiliary points improved '
                          'by action {} with {}'.format(str(max_action), str(internal_term) + ' > ' +
                                                        str(previous_stage_value_function_collection[
                                                                belief_point_key][0])))
                    del unimproved_belief_points[belief_point_key]
                    next_stage_value_function_collection[belief_point_key] = (internal_term, max_action)
                    number_of_belief_changes += 1
            utility = self.calculate_utility(next_stage_value_function_collection)
            print('[INFO] ModelFreeAdaptiveIntelligence backup: '
                  'Logging all the relevant metrics within this Backup stage - [Utility: {}, #policy_changes: {}, '
                  'sampled_value_function: {}]'.format(utility, number_of_belief_changes,
                                                       next_stage_value_function_collection[self.belief_choice][0]))
            self.utilities.append(utility)
        return [next_stage_value_function_collection, number_of_belief_changes]

    # The PERSEUS algorithm
    # Calls to Random Exploration, Initialization, and Backup stages
    def run_perseus(self):
        # Random Exploration - Get the set of reachable beliefs by randomly interacting with the radio environment
        reachable_beliefs = self.random_exploration(None)
        # Belief choice for value function tracking (visualization component)
        self.belief_choice = (lambda: self.belief_choice,
                              lambda: random.choice([
                                  k for k in reachable_beliefs.keys()]))[self.belief_choice is None]()
        # Initialization - Initializing to -10 for all beliefs in the reachable beliefs set
        initial_value_function_collection = self.initialize(reachable_beliefs)
        # Relevant collections
        previous_value_function_collection = initial_value_function_collection
        stage_number = self.exploration_period - 1
        # Utility addition for the initial value function
        utility = self.calculate_utility(previous_value_function_collection)
        print('[DEBUG] ModelFreeAdaptiveIntelligence run_perseus: '
              'Adding the utility metric for the initial value function - {}'.format(utility))
        self.utilities.append(utility)
        # Local confidence check for modelling policy convergence
        confidence = 0
        # Check for termination condition here...
        while confidence < self.confidence_bound:
            self.value_function_changes_array.append(previous_value_function_collection[self.belief_choice][0])
            stage_number += 1
            # We've reached the end of our allowed interaction time with the radio environment
            if stage_number == self.number_of_episodes:
                print('[WARN] ModelFreeAdaptiveIntelligence run_perseus: '
                      'We have reached the end of our allowed interaction time with the radio environment!')
                return
            # Backup to find \alpha -> Get V_{n+1} and #BeliefChanges
            backup_results = self.backup(stage_number, reachable_beliefs, previous_value_function_collection)
            print(
                '[DEBUG] ModelFreeAdaptiveIntelligence run_perseus: '
                'Backup for stage {} completed...'.format(stage_number - self.exploration_period + 1))
            next_value_function_collection = backup_results[0]
            belief_changes = backup_results[1]
            self.policy_changes.append(belief_changes)
            if len(next_value_function_collection) is not 0:
                previous_value_function_collection = next_value_function_collection
            if belief_changes is 0:
                print('[DEBUG] ModelFreeAdaptiveIntelligence run_perseus: '
                      'Confidence Update - {}'.format(confidence))
                confidence += 1
            else:
                confidence = 0
                print('[DEBUG] ModelFreeAdaptiveIntelligence run_perseus: '
                      'Confidence Stagnation/Fallback - {}'.format(confidence))
        optimal_utilities = []
        for episode_number, results_tuple in previous_value_function_collection.items():
            system_state = []
            for channel in range(0, self.number_of_channels):
                system_state.append(self.primary_user.occupancy_behavior_collection[channel][int(episode_number)])
            optimal_action = results_tuple[1]
            observation_samples = self.secondary_user.make_observations(int(episode_number), optimal_action)
            self.parameter_estimator.observation_samples = observation_samples
            transition_probabilities_matrix = self.util.construct_transition_probability_matrix(
                self.parameter_estimator.estimate_parameters(), self.pi)
            self.state_estimator.transition_probabilities = transition_probabilities_matrix
            self.state_estimator.observation_samples = observation_samples
            estimated_states = self.state_estimator.estimate_pu_occupancy_states()
            optimal_utilities.append(self.sweepstakes.roll(estimated_states, system_state))
        return {'id': self.agent_id, 'utilities': optimal_utilities}

    # Termination sequence
    # Signed off by bkeshava on 01-May-2019
    def __exit__(self, exc_type, exc_val, exc_tb):
        print('[INFO] ModelFreeAdaptiveIntelligence Termination: Tearing things down...')


# This entity encapsulates the POMDP Approximate Point-based Value Iteration algorithm named The PERSEUS algorithm
# Training to fine-tune the belief space -> Perform backup until convergence -> Re-sample using the most recent policy
# References to the ParameterEstimation algorithm and the StateEstimation algorithm in the belief analysis phase
# Simplified Belief update procedure incorporated in this version
class AdaptiveIntelligenceWithModelForesightSimplified(object):

    # Setup the Markov Chain
    @staticmethod
    def setup_markov_chain(_pi, _p, _correlation_class):
        print(
            '[INFO] AdaptiveIntelligenceWithModelForesightSimplified setup_markov_chain: '
            'Setting up the Markov Chain...')
        transient_markov_chain_object = MarkovChain()
        transient_markov_chain_object.set_markovian_correlation_class(_correlation_class)
        transient_markov_chain_object.set_start_probability_parameter(_pi)
        transient_markov_chain_object.set_transition_probability_parameter(_p)
        return transient_markov_chain_object

    # Initialization sequence
    # Initialization sequence
    def __init__(self, _number_of_channels, _number_of_sampling_rounds, _number_of_episodes, _exploration_period,
                 _noise_mean, _noise_variance, _impulse_response_mean, _impulse_response_variance, _penalty,
                 _limitation, _confidence_bound, _gamma, _transition_threshold, _agent_id):
        print('[INFO] AdaptiveIntelligenceWithModelForesightSimplified Initialization: Bringing things up...')
        # The number of channels in the discretized spectrum of interest
        self.number_of_channels = _number_of_channels
        # The number of sampling rounds in each episode
        self.number_of_sampling_rounds = _number_of_sampling_rounds
        # The number of time slots of interaction of the POMDP agent with the radio environment
        self.number_of_episodes = _number_of_episodes
        # The exploration period of the PERSEUS algorithm
        self.exploration_period = _exploration_period
        # The mean of the AWGN samples
        self.noise_mean = _noise_mean
        # The variance of the AWGN samples
        self.noise_variance = _noise_variance
        # The mean of the impulse response samples
        self.impulse_response_mean = _impulse_response_mean
        # The variance of the impulse response samples
        self.impulse_response_variance = _impulse_response_variance
        # The penalty for missed detection, i.e. PU interference
        self.penalty = _penalty
        # The SU sensing limitation
        self.limitation = _limitation
        # The confidence bound for convergence analysis
        self.confidence_bound = _confidence_bound
        # The discount factor employed in the Bellman equation
        self.gamma = _gamma
        # The allowed transition threshold
        self.transition_threshold = _transition_threshold
        # P(Occupied) - The steady state probability of being occupied
        self.pi = 0.6
        # P(Occupied|Idle) - The transition probability parameter
        self.p = 0.3
        # These have been learnt prior to triggering PERSEUS
        self.transition_probabilities_matrix = {0: {1: 0.3, 0: 0.7}, 1: {0: 0.2, 1: 0.8}}
        # The Utility object
        self.util = Util()
        # Start Probabilities Dict
        self.start_probabilities_dict = self.util.construct_start_probabilities_dict(self.pi)
        # Setup the Spatial Markov Chain
        self.spatial_markov_chain = self.setup_markov_chain(self.pi, self.p, MarkovianCorrelationClass.spatial)
        # Setup the Temporal Markov Chain
        self.temporal_markov_chain = self.setup_markov_chain(self.pi, self.p,
                                                             MarkovianCorrelationClass.temporal)
        # Channel
        self.channel = Channel(self.number_of_channels, self.number_of_sampling_rounds, self.number_of_episodes,
                               self.noise_mean, self.noise_variance, self.impulse_response_mean,
                               self.impulse_response_variance)
        # The Emission Evaluator
        self.emission_evaluator = EmissionEvaluator(self.noise_variance, self.impulse_response_variance)
        # Primary User
        self.primary_user = PrimaryUser(self.number_of_channels, self.number_of_episodes,
                                        self.spatial_markov_chain, self.temporal_markov_chain, self.util)
        self.primary_user.simulate_occupancy_behavior()
        # Secondary User
        self.secondary_user = SecondaryUser(self.number_of_channels, self.number_of_sampling_rounds,
                                            self.number_of_episodes, self.channel,
                                            self.primary_user.occupancy_behavior_collection, self.limitation)
        # Sweepstakes - The Reward Analyzer
        self.sweepstakes = Sweepstakes(self.primary_user, self.penalty)
        # The Oracle
        self.oracle = Oracle(self.number_of_episodes)
        # State Estimator
        self.state_estimator = StateEstimator(self.number_of_channels, self.number_of_sampling_rounds, None,
                                              self.start_probabilities_dict,
                                              self.transition_probabilities_matrix,
                                              self.emission_evaluator)
        # Utilities as the algorithm progresses towards optimality
        self.utilities = []
        # The number of policy changes as the algorithm progresses towards optimality
        self.policy_changes = []
        # Belief choice for value function tracking
        self.belief_choice = None
        # The progression of the value function as the algorithm progresses towards optimality
        self.value_function_changes_array = []
        # All possible states
        self.all_possible_states = list(map(list, itertools.product([0, 1], repeat=self.number_of_channels)))
        # All possible actions based on my SU sensing limitations
        self.all_possible_actions = []
        for state in self.all_possible_states:
            if sum(state) == self.limitation:
                self.all_possible_actions.append(state)
        # Maximum number of channel changes allowed
        self.max_allowed_transitions = math.ceil(self.transition_threshold * self.number_of_channels)
        # The agent id
        self.agent_id = _agent_id

    # Get Emission Probabilities for the Belief Update sequence
    def get_emission_probability(self, observations, state):
        emission_probability = 1
        # round_choice to introduce some randomness into the emission evaluations
        round_choice = random.choice([k for k in range(0, self.number_of_sampling_rounds)])
        # Given the system state, the emissions are independent of each other
        for index in range(0, self.number_of_channels):
            emission_probability = emission_probability * self.emission_evaluator.get_emission_probabilities(
                state[index], observations[index][round_choice])
        return emission_probability

    # Get State Transition Probabilities for the Belief Update sequence
    def get_transition_probability(self, prev_state, next_state, transition_probabilities_matrix):
        # Temporal/Episodic change for the first channel
        transition_probability = transition_probabilities_matrix[prev_state[0]][next_state[0]]
        for index in range(1, self.number_of_channels):
            transition_probability = transition_probability * transition_probabilities_matrix[next_state[index - 1]][
                next_state[index]]
        return transition_probability

    # Get the allowed state transitions previous_states -> next_states
    def get_allowed_state_transitions(self, state):
        allowed_state_transitions = list()
        combinations_array = dict()
        for transition_allowance in range(1, self.max_allowed_transitions + 1):
            combinations_array[transition_allowance] = list(itertools.combinations([k for k in range(0, len(state))],
                                                                                   transition_allowance))
        for combination in combinations_array.values():
            for group in combination:
                new_state = [k for k in state]
                for entry in group:
                    new_state[entry] = (lambda: 1, lambda: 0)[new_state[entry] == 1]()
                allowed_state_transitions.append(new_state)
        return allowed_state_transitions

    # Get the normalization constant
    def get_normalization_constant(self, previous_belief_vector, observations):
        # The normalization constant
        normalization_constant = 0
        # Calculate the normalization constant in the belief update formula
        for _next_state in self.all_possible_states:
            multiplier = 0
            allowed_previous_states = self.get_allowed_state_transitions(_next_state)
            for _prev_state in allowed_previous_states:
                multiplier += self.get_transition_probability(_prev_state, _next_state,
                                                              self.transition_probabilities_matrix) * \
                              previous_belief_vector[''.join(str(k) for k in _prev_state)]
            normalization_constant += self.get_emission_probability(observations, _next_state) * multiplier
        return normalization_constant

    # Belief Update sequence
    def belief_update(self, observations, previous_belief_vector, new_state,
                      transition_probabilities_matrix, normalization_constant):
        multiplier = 0
        allowed_previous_states = self.get_allowed_state_transitions(new_state)
        # Calculate the numerator in the belief update formula
        # \vec{x} \in \mathcal{X} in your belief update rule
        for prev_state in allowed_previous_states:
            multiplier += self.get_transition_probability(prev_state,
                                                          new_state,
                                                          transition_probabilities_matrix) * previous_belief_vector[
                              ''.join(str(k) for k in prev_state)]
        numerator = self.get_emission_probability(observations, new_state) * multiplier
        return numerator / normalization_constant

    # Randomly explore the environment and collect a set of beliefs B of reachable belief points
    # ...strategy as a part of the modified PERSEUS strategy
    def random_exploration(self, policy):
        reachable_beliefs = dict()
        state_space_size = 2 ** self.number_of_channels
        # Uniform belief assignment to all states in the state space
        initial_belief_vector = dict()
        for state in self.all_possible_states:
            state_key = ''.join(str(k) for k in state)
            initial_belief_vector[state_key] = 1 / state_space_size
        previous_belief_vector = initial_belief_vector
        reachable_beliefs['0'] = initial_belief_vector
        if policy is not None:
            print('[WARN] AdaptiveIntelligenceWithModelForesightSimplified random_exploration: '
                  'Specific policy exploration is yet to be implemented')
        print('[INFO] AdaptiveIntelligenceWithModelForesightSimplified random_exploration: '
              'Using a truly random exploration strategy')
        # Start exploring
        for episode in range(1, self.exploration_period):
            # Perform the sensing action and make the observations
            # Making observations by choosing a random sensing action
            observations = self.secondary_user.make_observations(episode, random.choice(self.all_possible_actions))
            # Perform the Belief Update
            updated_belief_vector = dict()
            # Belief sum for this updated belief vector
            belief_sum = 0
            # Calculate the denominator which is nothing but the normalization constant
            # Calculate normalization constant only once...
            denominator = self.get_normalization_constant(previous_belief_vector, observations)
            # Possible next states to update the belief, i.e. b(\vec{x}')
            for state in self.all_possible_states:
                state_key = ''.join(str(k) for k in state)
                belief_val = self.belief_update(observations, previous_belief_vector, state,
                                                self.transition_probabilities_matrix, denominator)
                # Belief sum for Kolmogorov validation
                belief_sum += belief_val
                updated_belief_vector[state_key] = belief_val
            # Normalization to get valid belief vectors (satisfying axioms of probability measures)
            updated_belief_information = self.util.normalize(updated_belief_vector, belief_sum)
            if updated_belief_information[1] is False:
                raise ArithmeticError('The belief is a probability distribution over the state space. It should sum '
                                      'to one!')
            # Add the new belief vector to the reachable beliefs set
            reachable_beliefs[str(episode)] = updated_belief_information[0]
            print('[INFO] AdaptiveIntelligenceWithModelForesightSimplified random_exploration: {}% Exploration '
                  'completed'.format(int(((episode + 1) / self.exploration_period) * 100)))
        return reachable_beliefs

    # Initialization
    def initialize(self, reachable_beliefs):
        # V_0 for the reachable beliefs
        value_function_collection = dict()
        # Default action is not sensing anything - a blind guess from just the noise
        default_action = [k - k for k in range(0, self.number_of_channels)]
        for belief_key in reachable_beliefs.keys():
            value_function_collection[belief_key] = (-10, default_action)
        return value_function_collection

    # Calculate Utility
    def calculate_utility(self, policy_collection):
        utility = 0
        for key, value in policy_collection.items():
            system_state = []
            for channel in range(0, self.number_of_channels):
                # int(key) refers to the episode number
                system_state.append(self.primary_user.occupancy_behavior_collection[channel][int(key)])
            # In that episode, the POMDP agent believed that the system is in a certain probabilistic distribution...
            # over the state vector, i.e. the belief, and based on this it recommended the best possible action to be...
            # taken in that state - obtained by going through the backup procedure...
            observation_samples = self.secondary_user.make_observations(int(key), value[1])
            self.state_estimator.observation_samples = observation_samples
            estimated_state = self.state_estimator.estimate_pu_occupancy_states()
            print('[DEBUG] AdaptiveIntelligenceWithModelForesightSimplified calculate_utility: '
                  'Estimated PU Occupancy states - {}'.format(str(estimated_state)))
            utility += self.sweepstakes.roll(estimated_state, system_state)
        return utility

    # The Backup stage
    def backup(self, stage_number, reachable_beliefs, previous_stage_value_function_collection):
        # Just a reassignment because the reachable_beliefs turns out to be a mutable collection
        unimproved_belief_points = {}
        for key, value in reachable_beliefs.items():
            unimproved_belief_points[key] = value
        # This assignment will help me in the utility calculation within each backup stage
        next_stage_value_function_collection = dict()
        # A simple dict copy - just to be safe from mutations
        for k, v in previous_stage_value_function_collection.items():
            next_stage_value_function_collection[k] = v
        number_of_belief_changes = 0
        # While there are still some un-improved belief points
        while len(unimproved_belief_points) is not 0:
            print('[INFO] AdaptiveIntelligenceWithModelForesightSimplified backup: '
                  'Size of unimproved belief set = {}'.format(len(unimproved_belief_points)))
            # Sample a belief point uniformly at random from \tilde{B}
            belief_sample_key = random.choice(list(unimproved_belief_points.keys()))
            belief_sample = unimproved_belief_points[belief_sample_key]
            max_value_function = -10 ** 9
            max_action = None
            for action in self.all_possible_actions:
                new_belief_vector = {}
                new_belief_sum = 0
                # Make observations based on the chosen action
                observation_samples = self.secondary_user.make_observations(stage_number, action)
                # Estimate the System State
                self.state_estimator.observation_samples = observation_samples
                estimated_system_state = self.state_estimator.estimate_pu_occupancy_states()
                print('[DEBUG] AdaptiveIntelligenceWithModelForesightSimplified backup: '
                      'Estimated PU Occupancy states - {}'.format(str(estimated_system_state)))
                reward_sum = 0
                normalization_constant = 0
                for state in self.all_possible_states:
                    emission_probability = self.get_emission_probability(observation_samples, state)
                    multiplier = 0
                    allowed_previous_states = self.get_allowed_state_transitions(state)
                    for prev_state in allowed_previous_states:
                        multiplier += self.get_transition_probability(prev_state, state,
                                                                      self.transition_probabilities_matrix) * \
                                      belief_sample[''.join(str(k) for k in prev_state)]
                    normalization_constant += emission_probability * multiplier
                    reward_sum += self.sweepstakes.roll(estimated_system_state, state) * belief_sample[
                        ''.join(str(k) for k in state)]
                # Belief Update
                for state in self.all_possible_states:
                    state_key = ''.join(str(k) for k in state)
                    value_of_belief = self.belief_update(observation_samples, belief_sample, state,
                                                         self.transition_probabilities_matrix,
                                                         normalization_constant)
                    new_belief_sum += value_of_belief
                    new_belief_vector[state_key] = value_of_belief
                # Normalization to get valid belief vectors (satisfying axioms of probability measures)
                new_normalized_belief_information = self.util.normalize(new_belief_vector, new_belief_sum)
                if new_normalized_belief_information[1] is False:
                    raise ArithmeticError('The belief is a probability distribution over the state space. It should '
                                          'sum to one!')
                # Updated re-assignment
                new_belief_vector = new_normalized_belief_information[0]
                highest_belief_key = max(new_belief_vector, key=new_belief_vector.get)
                # You could've used an OrderedDict here to simplify operations
                # Find the closest pilot belief and its associated value function
                relevant_data = {episode_key: belief[highest_belief_key] for episode_key, belief in
                                 reachable_beliefs.items()}
                pilot_belief_key = max(relevant_data, key=relevant_data.get)
                internal_term = reward_sum + (self.gamma * normalization_constant *
                                              previous_stage_value_function_collection[pilot_belief_key][0])
                if internal_term > max_value_function:
                    max_value_function = internal_term
                    max_action = action
            if round(max_value_function, 3) > round(previous_stage_value_function_collection[belief_sample_key][0], 3):
                print('[DEBUG] AdaptiveIntelligenceWithModelForesightSimplified backup: '
                      '[Action: {} and New_Value_Function: {}] pair improves the existing policy - '
                      'Corresponding sequence triggered!'.format(str(max_action), str(max_value_function) + ' > ' +
                                                                 str(previous_stage_value_function_collection[
                                                                         belief_sample_key][0])))
                # Note here that del mutates the contents of the dict for everyone who has a reference to it
                del unimproved_belief_points[belief_sample_key]
                next_stage_value_function_collection[belief_sample_key] = (max_value_function, max_action)
                number_of_belief_changes += 1
            else:
                next_stage_value_function_collection[belief_sample_key] = previous_stage_value_function_collection[
                    belief_sample_key]
                # Note here that del mutates the contents of the dict for everyone who has a reference to it
                del unimproved_belief_points[belief_sample_key]
                max_action = previous_stage_value_function_collection[belief_sample_key][1]
            for belief_point_key in list(unimproved_belief_points.keys()):
                print('[DEBUG] AdaptiveIntelligenceWithModelForesightSimplified backup: '
                      'Improving the other belief points...')
                normalization_constant = 0
                reward_sum = 0
                observation_samples = self.secondary_user.make_observations(stage_number, max_action)
                # Estimate the System State
                self.state_estimator.observation_samples = observation_samples
                estimated_system_state = self.state_estimator.estimate_pu_occupancy_states()
                print('[DEBUG] AdaptiveIntelligenceWithModelForesightSimplified backup: '
                      'Estimated PU Occupancy states - {}'.format(str(estimated_system_state)))
                for state in self.all_possible_states:
                    emission_probability = self.get_emission_probability(observation_samples, state)
                    multiplier = 0
                    allowed_previous_states = self.get_allowed_state_transitions(state)
                    for prev_state in allowed_previous_states:
                        multiplier += self.get_transition_probability(prev_state, state,
                                                                      self.transition_probabilities_matrix) * \
                                      unimproved_belief_points[belief_point_key][''.join(str(k) for k in prev_state)]
                    normalization_constant += emission_probability * multiplier
                    reward_sum += self.sweepstakes.roll(estimated_system_state, state) * unimproved_belief_points[
                        belief_point_key][''.join(str(k) for k in state)]
                new_aux_belief_vector = {}
                new_aux_belief_sum = 0
                aux_belief_sample = unimproved_belief_points[belief_point_key]
                # Belief Update
                for state in self.all_possible_states:
                    state_key = ''.join(str(k) for k in state)
                    new_aux_belief_val = self.belief_update(observation_samples, aux_belief_sample, state,
                                                            self.transition_probabilities_matrix,
                                                            normalization_constant)
                    new_aux_belief_sum += new_aux_belief_val
                    new_aux_belief_vector[state_key] = new_aux_belief_val
                # Normalization to get valid belief vectors (satisfying axioms of probability measures)
                new_aux_normalized_belief_information = self.util.normalize(new_aux_belief_vector, new_aux_belief_sum)
                if new_aux_normalized_belief_information[1] is False:
                    raise ArithmeticError('The belief is a probability distribution over the state space. It should '
                                          'sum to one!')
                # Updated re-assignment
                new_aux_belief_vector = new_aux_normalized_belief_information[0]
                highest_belief_key = max(new_aux_belief_vector, key=new_aux_belief_vector.get)
                # You could've used an OrderedDict here to simplify operations
                # Find the closest pilot belief and its associated value function
                relevant_data = {episode_key: belief[highest_belief_key] for episode_key, belief in
                                 reachable_beliefs.items()}
                aux_pilot_belief_key = max(relevant_data, key=relevant_data.get)
                internal_term = reward_sum + (self.gamma * normalization_constant *
                                              previous_stage_value_function_collection[aux_pilot_belief_key][0])
                if round(internal_term, 3) > round(previous_stage_value_function_collection[belief_point_key][0], 3):
                    # Note here that del mutates the contents of the dict for everyone who has a reference to it
                    print('[DEBUG] AdaptiveIntelligenceWithModelForesightSimplified backup: Auxiliary points improved '
                          'by action {} with {}'.format(str(max_action), str(internal_term) + ' > ' +
                                                        str(previous_stage_value_function_collection[
                                                                belief_point_key][0])))
                    del unimproved_belief_points[belief_point_key]
                    next_stage_value_function_collection[belief_point_key] = (internal_term, max_action)
                    number_of_belief_changes += 1
            utility = self.calculate_utility(next_stage_value_function_collection)
            print('[INFO] AdaptiveIntelligenceWithModelForesightSimplified backup: '
                  'Logging all the relevant metrics within this Backup stage - [Utility: {}, #policy_changes: {}, '
                  'sampled_value_function: {}]'.format(utility, number_of_belief_changes,
                                                       next_stage_value_function_collection[self.belief_choice][0]))
            self.utilities.append(utility)
        return [next_stage_value_function_collection, number_of_belief_changes]

    # The PERSEUS algorithm
    # Calls to Random Exploration, Initialization, and Backup stages
    def run_perseus(self):
        # Random Exploration - Get the set of reachable beliefs by randomly interacting with the radio environment
        reachable_beliefs = self.random_exploration(None)
        # Belief choice for value function tracking (visualization component)
        self.belief_choice = (lambda: self.belief_choice,
                              lambda: random.choice([
                                  k for k in reachable_beliefs.keys()]))[self.belief_choice is None]()
        # Initialization - Initializing to -10 for all beliefs in the reachable beliefs set
        initial_value_function_collection = self.initialize(reachable_beliefs)
        # Relevant collections
        previous_value_function_collection = initial_value_function_collection
        stage_number = self.exploration_period - 1
        # Utility addition for the initial value function
        utility = self.calculate_utility(previous_value_function_collection)
        print('[DEBUG] AdaptiveIntelligenceWithModelForesightSimplified run_perseus: '
              'Adding the utility metric for the initial value function - {}'.format(utility))
        self.utilities.append(utility)
        # Local confidence check for modelling policy convergence
        confidence = 0
        # Check for termination condition here...
        while confidence < self.confidence_bound:
            self.value_function_changes_array.append(previous_value_function_collection[self.belief_choice][0])
            stage_number += 1
            # We've reached the end of our allowed interaction time with the radio environment
            if stage_number == self.number_of_episodes:
                print('[WARN] AdaptiveIntelligenceWithModelForesightSimplified run_perseus: '
                      'We have reached the end of our allowed interaction time with the radio environment!')
                return
            # Backup to find \alpha -> Get V_{n+1} and #BeliefChanges
            backup_results = self.backup(stage_number, reachable_beliefs, previous_value_function_collection)
            print(
                '[DEBUG] AdaptiveIntelligenceWithModelForesightSimplified run_perseus: '
                'Backup for stage {} completed...'.format(stage_number - self.exploration_period + 1))
            next_value_function_collection = backup_results[0]
            belief_changes = backup_results[1]
            self.policy_changes.append(belief_changes)
            if len(next_value_function_collection) is not 0:
                previous_value_function_collection = next_value_function_collection
            if belief_changes is 0:
                print('[DEBUG] AdaptiveIntelligenceWithModelForesightSimplified run_perseus: '
                      'Confidence Update - {}'.format(confidence))
                confidence += 1
            else:
                confidence = 0
                print('[DEBUG] AdaptiveIntelligenceWithModelForesightSimplified run_perseus: '
                      'Confidence Stagnation/Fallback - {}'.format(confidence))
        optimal_utilities = []
        for episode_number, results_tuple in previous_value_function_collection.items():
            system_state = []
            for channel in range(0, self.number_of_channels):
                system_state.append(self.primary_user.occupancy_behavior_collection[channel][int(episode_number)])
            optimal_action = results_tuple[1]
            observation_samples = self.secondary_user.make_observations(int(episode_number), optimal_action)
            self.state_estimator.observation_samples = observation_samples
            estimated_states = self.state_estimator.estimate_pu_occupancy_states()
            optimal_utilities.append(self.sweepstakes.roll(estimated_states, system_state))
        return {'id': self.agent_id, 'utilities': optimal_utilities}

    # Termination sequence
    # Signed off by bkeshava on 01-May-2019
    def __exit__(self, exc_type, exc_val, exc_tb):
        print('[INFO] AdaptiveIntelligenceWithModelForesightSimplified Termination: Tearing things down...')


# A Neyman-Pearson Detector assuming independence among channels
# The channels are actually Markov correlated
# But, this class helps me evaluate the performance if I assume independence
# The signal model is s[n] = A (occupancy)
# The observation model is x[n] = w[n], null hypothesis, and
#                          x[n] = A + w[n], alternative hypothesis
class NeymanPearsonDetector(object):

    # The initialization sequence
    def __init__(self, _number_of_channels, _number_of_sampling_rounds, _number_of_episodes, _false_alarm_probability,
                 _noise_mean, _noise_variance, _observations, _true_pu_occupancy_states, _penalty, _agent_id):
        print('[INFO] NeymanPearsonDetector Initialization: Bringing things up...')
        # The number of channels in the discretized spectrum of interest
        self.number_of_channels = _number_of_channels
        # The number of sampling rounds per episode
        self.number_of_sampling_rounds = _number_of_sampling_rounds
        # The number of episodes of interaction of this agent with the radio environment
        self.number_of_episodes = _number_of_episodes
        # The false alarm probability constraint
        self.false_alarm_probability = _false_alarm_probability
        # The mean of the AWGN samples
        self.noise_mean = _noise_mean
        if self.noise_mean is not 0:
            print('[WARN] NeymanPearsonDetector Initialization: The observation model assumes zero mean additive white '
                  'Gaussian noise samples...')
            self.noise_mean = 0
        # The variance of the AWGN samples
        self.noise_variance = _noise_variance
        # The observations made by the Secondary User
        self.observations = _observations
        # The true PU occupancy states - Markov correlation is left unexploited here...
        self.true_pu_occupancy_states = _true_pu_occupancy_states
        # The threshold for the Likelihood Ratio Test (LRT)
        self.threshold = math.sqrt(self.noise_variance / self.number_of_sampling_rounds) * scipy.stats.norm.ppf(
            1 - self.false_alarm_probability, loc=self.noise_mean, scale=math.sqrt(self.noise_variance))
        # The penalty for missed detections
        self.penalty = _penalty
        # The agent id
        self.agent_id = _agent_id

    # Detect occupancy across episodes by averaging out noisy observations over numerous sampling rounds
    # Likelihood Ratio Test (LRT) based on a test statistic with the threshold determined from the P_FA constraint
    def get_utilities(self):
        utilities = []
        for episode in range(0, self.number_of_episodes):
            estimated_states = []
            occupancies = 0
            idle_count = 0
            false_alarms = 0
            missed_detections = 0
            for channel in range(0, self.number_of_channels):
                test_statistic = sum(self.observations[channel][episode]) / self.number_of_sampling_rounds
                estimated_states[channel] = (lambda: 0, lambda: 1)[test_statistic >= self.threshold]()
                if self.true_pu_occupancy_states[channel][episode] == 1:
                    occupancies += 1
                    if estimated_states[channel] == 0:
                        missed_detections += 1
                if self.true_pu_occupancy_states[channel][episode] == 0:
                    idle_count += 1
                    if estimated_states[channel] == 1:
                        false_alarms += 1
            episodic_false_alarm_probability = (lambda: 0, lambda: false_alarms/idle_count)[idle_count is not 0]()
            episodic_missed_detection_probability = (lambda: 0, lambda: missed_detections/occupancies)[
                occupancies is not 0]()
            utilities.append((1 - episodic_false_alarm_probability) +
                             (self.penalty * episodic_missed_detection_probability))
        return {'id': self.agent_id, 'utilities': utilities}


# This class encapsulates the complete evaluation framework detailed in the header of this script.
class EvaluationFramework(object):
    # The number of channels in the discretized spectrum of interest
    NUMBER_OF_CHANNELS = 18

    # The number of sampling rounds per episode
    NUMBER_OF_SAMPLING_ROUNDS = 250

    # The number of periods of interaction of the agent with the radio environment
    NUMBER_OF_EPISODES = 1000

    # The mean of the AWGN samples
    NOISE_MEAN = 0

    # The mean of the frequency domain channel
    IMPULSE_RESPONSE_MEAN = 0

    # The variance of the AWGN samples
    NOISE_VARIANCE = 1

    # The variance of the frequency domain channel
    IMPULSE_RESPONSE_VARIANCE = 80

    # The Secondary User's sensing limitation w.r.t the number of channels it can sense simultaneously in an episode
    SPATIAL_SENSING_LIMITATION = 9

    # Limitation per fragment
    FRAGMENTED_SPATIAL_SENSING_LIMITATION = 3

    # The exploration period of the PERSEUS algorithm
    EXPLORATION_PERIOD = 100

    # The discount factor employed in the Bellman update
    DISCOUNT_FACTOR = 0.9

    # The confidence bound for convergence analysis
    CONFIDENCE_BOUND = 10

    # The transition threshold for the simplified belief update procedure
    TRANSITION_THRESHOLD = 0.1

    # The size of each agent-assigned individual fragment of the spectrum which is independent from the other fragments
    # I assume the same Markovian correlation within each fragment
    FRAGMENT_SIZE = 6

    # The choice of heuristic for the constrained non-POMDP agent
    HEURISTIC_CHOICE = 3

    # The penalty for missed detections
    PENALTY = -1

    # The convergence threshold for the parameter estimation algorithm
    CONVERGENCE_THRESHOLD = 0.00001

    # The constraint on false alarm probability for the Neyman-Pearson detector
    FALSE_ALARM_PROBABILITY_CONSTRAINT = 0.7

    # The number of agents being evaluated in this script
    NUMBER_OF_AGENTS = 7

    # Setup the Markov Chain
    # Signed off by bkeshava on 01-May-2019
    @staticmethod
    def setup_markov_chain(_pi, _p, _correlation_class):
        print('[INFO] EvaluationFramework setup_markov_chain: Setting up the Markov Chain...')
        transient_markov_chain_object = MarkovChain()
        transient_markov_chain_object.set_markovian_correlation_class(_correlation_class)
        transient_markov_chain_object.set_start_probability_parameter(_pi)
        transient_markov_chain_object.set_transition_probability_parameter(_p)
        return transient_markov_chain_object

    # A worker bee with task delegations based on the job-id
    def worker(self, job_id):
        if job_id == 0:
            print('[INFO] EvaluationFramework worker: Starting job thread for the unconstrained non-POMDP agent!')
            # The unconstrained global observation samples
            unconstrained_global_observations = self.secondary_user.observe_everything_unconstrained()
            # The unconstrained double Markov chain state estimator
            unconstrained_non_pomdp_agent = DoubleMarkovChainViterbiAlgorithm(
                self.NUMBER_OF_CHANNELS, self.NUMBER_OF_EPISODES, self.emission_evaluator,
                self.primary_user.occupancy_behavior_collection, unconstrained_global_observations,
                self.spatial_start_probabilities, self.temporal_start_probabilities,
                self.spatial_transition_probability_matrix,
                self.temporal_transition_probability_matrix, self.PENALTY, job_id)
            return unconstrained_non_pomdp_agent.estimate_pu_occupancy_states()
        elif job_id == 1:
            print('[INFO] EvaluationFramework worker: Starting job thread for the constrained POMDP agent!')
            # The channel selection heuristic generator
            # Setting the number of iterations for random sensing to 1
            # I don't want to employ random sensing just yet...
            # Let's stick to patterned heuristics
            channel_selection_heuristic_generator = ChannelSelectionStrategyGenerator(self.NUMBER_OF_CHANNELS, 1)
            # Get the channel sensing strategy
            sensing_heuristic = channel_selection_heuristic_generator.uniform_sensing()[self.HEURISTIC_CHOICE]
            # The constrained channel sensing heuristics based observation samples
            constrained_global_observations = self.secondary_user.observe_everything_with_spatial_constraints(
                sensing_heuristic)
            # The constrained double Markov chain state estimator
            constrained_non_pomdp_agent = DoubleMarkovChainViterbiAlgorithm(
                self.NUMBER_OF_CHANNELS, self.NUMBER_OF_EPISODES, self.emission_evaluator,
                self.primary_user.occupancy_behavior_collection, constrained_global_observations,
                self.spatial_start_probabilities, self.temporal_start_probabilities,
                self.spatial_transition_probability_matrix,
                self.temporal_transition_probability_matrix, self.PENALTY, job_id)
            return constrained_non_pomdp_agent.estimate_pu_occupancy_states()
        elif job_id == 2:
            print('[INFO] EvaluationFramework worker: Starting job thread for the channel correlation based clustering '
                  'and MAP estimation algorithm in the state-of-the-art!')
            raise NotImplementedError('This agent is yet to be implemented. Please check back later!')
        elif job_id == 3:
            print('[INFO] EvaluationFramework worker: Starting job thread for the model-free PERSEUS POMDP agent!')
            model_free_perseus = ModelFreeAdaptiveIntelligence(self.FRAGMENT_SIZE, self.NUMBER_OF_SAMPLING_ROUNDS,
                                                               self.NUMBER_OF_EPISODES, self.EXPLORATION_PERIOD,
                                                               self.NOISE_MEAN, self.NOISE_VARIANCE,
                                                               self.IMPULSE_RESPONSE_MEAN,
                                                               self.IMPULSE_RESPONSE_VARIANCE, self.PENALTY,
                                                               self.FRAGMENTED_SPATIAL_SENSING_LIMITATION,
                                                               self.CONFIDENCE_BOUND, self.DISCOUNT_FACTOR,
                                                               self.CONVERGENCE_THRESHOLD, job_id)
            return model_free_perseus.run_perseus()
        elif job_id == 4:
            print('[INFO] EvaluationFramework worker: Starting job thread for the PERSEUS POMDP agent with model '
                  'foresight')
            perseus_with_model_foresight = AdaptiveIntelligenceWithModelForesight(self.FRAGMENT_SIZE,
                                                                                  self.NUMBER_OF_SAMPLING_ROUNDS,
                                                                                  self.NUMBER_OF_EPISODES,
                                                                                  self.EXPLORATION_PERIOD,
                                                                                  self.NOISE_MEAN, self.NOISE_VARIANCE,
                                                                                  self.IMPULSE_RESPONSE_MEAN,
                                                                                  self.IMPULSE_RESPONSE_VARIANCE,
                                                                                  self.PENALTY,
                                                                                  self.
                                                                                  FRAGMENTED_SPATIAL_SENSING_LIMITATION,
                                                                                  self.CONFIDENCE_BOUND,
                                                                                  self.DISCOUNT_FACTOR, job_id)
            return perseus_with_model_foresight.run_perseus()
        elif job_id == 5:
            print('[INFO] EvaluationFramework worker: Starting job thread for the PERSEUS POMDP agent with model '
                  'foresight and belief update simplification')
            perseus_with_model_foresight_simplified = AdaptiveIntelligenceWithModelForesightSimplified(
                self.FRAGMENT_SIZE, self.NUMBER_OF_SAMPLING_ROUNDS, self.NUMBER_OF_EPISODES,
                self.EXPLORATION_PERIOD, self.NOISE_MEAN, self.NOISE_VARIANCE, self.IMPULSE_RESPONSE_MEAN,
                self.IMPULSE_RESPONSE_VARIANCE, self.PENALTY, self.FRAGMENTED_SPATIAL_SENSING_LIMITATION,
                self.CONFIDENCE_BOUND, self.DISCOUNT_FACTOR, self.TRANSITION_THRESHOLD, job_id)
            return perseus_with_model_foresight_simplified.run_perseus()
        elif job_id == 6:
            print('[INFO] EvaluationFramework worker: Starting job thread for the Neyman-Pearson detector!')
            sampled_observations_across_all_episodes = self.secondary_user.\
                make_sampled_observations_across_all_episodes()
            neyman_pearson_detector = NeymanPearsonDetector(self.NUMBER_OF_CHANNELS, self.NUMBER_OF_SAMPLING_ROUNDS,
                                                            self.NUMBER_OF_EPISODES,
                                                            self.FALSE_ALARM_PROBABILITY_CONSTRAINT,
                                                            self.NOISE_MEAN, self.NOISE_VARIANCE,
                                                            sampled_observations_across_all_episodes,
                                                            self.primary_user.occupancy_behavior_collection,
                                                            self.PENALTY, job_id)
            return neyman_pearson_detector.get_utilities()
        else:
            print('[INFO] EvaluationFramework worker: Invalid job_id - {}'.format(job_id))
        return None

    # Initialize the Evaluation Process
    # Context Manager method for Instance Creation
    def __init__(self):
        print('[INFO] EvaluationFramework Initialization: Bringing things up...')
        # The start probabilities for the spatial Markov chain
        self.spatial_start_probabilities = {0: 0.4, 1: 0.6}
        # The start probabilities for the temporal Markov chain
        self.temporal_start_probabilities = {0: 0.4, 1: 0.6}
        # The transition model of the spatial Markov chain
        self.spatial_transition_probability_matrix = {0: {0: 0.7, 1: 0.3}, 1: {0: 0.2, 1: 0.8}}
        # The transition model of the temporal Markov chain
        self.temporal_transition_probability_matrix = {0: {0: 0.7, 1: 0.3}, 1: {0: 0.2, 1: 0.8}}
        # The spatial Markov chain
        self.spatial_markov_chain = self.setup_markov_chain(self.spatial_start_probabilities[1],
                                                            self.spatial_transition_probability_matrix[0][1],
                                                            MarkovianCorrelationClass.spatial)
        # The temporal Markov chain
        self.temporal_markov_chain = self.setup_markov_chain(self.temporal_start_probabilities[1],
                                                             self.temporal_transition_probability_matrix[0][1],
                                                             MarkovianCorrelationClass.temporal)
        # The utilities instance
        self.util = Util()
        # The channel instance
        self.channel = Channel(self.NUMBER_OF_CHANNELS, self.NUMBER_OF_SAMPLING_ROUNDS, self.NUMBER_OF_EPISODES,
                               self.NOISE_MEAN, self.NOISE_VARIANCE, self.IMPULSE_RESPONSE_MEAN,
                               self.IMPULSE_RESPONSE_VARIANCE)
        # The Primary User
        self.primary_user = PrimaryUser(self.NUMBER_OF_CHANNELS, self.NUMBER_OF_EPISODES, self.spatial_markov_chain,
                                        self.temporal_markov_chain, self.util)
        # Simulate the Primary User behavior
        self.primary_user.simulate_occupancy_behavior()

        # The Secondary User
        self.secondary_user = SecondaryUser(self.NUMBER_OF_CHANNELS, self.NUMBER_OF_SAMPLING_ROUNDS,
                                            self.NUMBER_OF_EPISODES, self.channel,
                                            self.primary_user.occupancy_behavior_collection,
                                            self.SPATIAL_SENSING_LIMITATION)
        # The emission evaluator
        self.emission_evaluator = EmissionEvaluator(self.NOISE_VARIANCE, self.IMPULSE_RESPONSE_VARIANCE)
        # Colors
        self.colors = ['b', 'r', 'y', 'k', 'g', 'c', 'm']
        # Job-ID: Label Map
        self.job_id_map = {0: 'Unconstrained Non-POMDP Agent',
                           1: 'Constrained POMDP Agent',
                           2: 'State-of-the-art',
                           3: 'Model-Free PERSEUS',
                           4: 'PERSEUS with Model-Foresight',
                           5: 'PERSEUS with Model-Foresight and Belief-Update Simplification',
                           6: 'Neyman-Pearson Detector with Independence Assumptions'}

    # Start the evaluation
    def evaluate(self):
        print('[INFO] EvaluationFramework evaluate: Beginning the evaluation of the Minerva framework!')
        # The x-axis corresponds to the episodes of interaction
        x_axis = [k + 1 for k in range(0, self.NUMBER_OF_EPISODES)]
        # The job_ids/tokens for the various agents
        job_ids = [k for k in range(0, self.NUMBER_OF_AGENTS)]
        # Create a job pool for all the agents under evaluation
        pool = multiprocessing.Pool(self.NUMBER_OF_AGENTS)
        # Run the workers and gather the results
        result_collection = pool.map(self.worker, (job_id for job_id in job_ids))
        # Close the job pool
        pool.close()
        # Join with the main thread
        pool.join()
        # The color index
        color_index = 0
        # The figure
        fig, ax = plt.subplots()
        for result in result_collection:
            color_index += 1
            for job_id, utilities in result.items():
                if job_id == 3 or job_id == 4 or job_id == 5:
                    # Joining the utilities of the fragments for the POMDP agents
                    utilities = list(numpy.array(utilities) * math.ceil(self.NUMBER_OF_CHANNELS / self.FRAGMENT_SIZE))
                ax.plot(x_axis, utilities, linewidth=1.0, marker='o', color=self.colors[color_index],
                        label=self.job_id_map[job_id])
        fig.suptitle('Evaluation of the obtained utilities per episode for various agents', fontsize=14)
        ax.set_xlabel('Episode Number -->', fontsize=14)
        ax.set_ylabel('Obtained Episodic Utility -->', fontsize=14)
        ax.legend()
        plt.show()

    # Exit the Evaluation Process
    # Context Manager method for tearing things down
    def __exit__(self, exc_type, exc_val, exc_tb):
        print('[INFO] EvaluationFramework Termination: Tearing things down...')


# Run Trigger
if __name__ == '__main__':
    print('[INFO] EvaluationFramework main: Starting the system simulation!')
    evaluationFramework = EvaluationFramework()
    evaluationFramework.evaluate()
