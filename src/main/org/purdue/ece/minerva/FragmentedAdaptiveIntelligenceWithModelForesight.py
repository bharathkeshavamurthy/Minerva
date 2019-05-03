# This entity encapsulates the simplified overall framework essential for this research
# This version of the AdaptiveIntelligenceWithModelForesight implementation fragments the spectrum occupancy model...
# ...into independent subsets consisting of correlated channels
# Each independent subset will then have a POMDP agent to look after it!
# This entity includes an Oracle (knows everything about everything and hence can choose the most optimal action at...
# ...any given time)
# This entity also brings in the EM Parameter Estimation algorithm, the Viterbi State Estimation algorithm, and the...
# ...fragmented PERSEUS Approximate Value Iteration algorithm for POMDPs
# Author: Bharath Keshavamurthy
# Organization: School of Electrical & Computer Engineering, Purdue University
# Copyright (c) 2019. All Rights Reserved.

# Imports - Signed off by bkeshava on 01-May-2019
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
# Signed off by bkeshava on 01-May-2019
class MarkovianCorrelationClass(Enum):
    # Markovian Correlation across channel indices
    spatial = 0
    # Markovian Correlation across time indices
    temporal = 1
    # Invalid
    invalid = 2


# OccupancyState Enumeration
# Based on Energy Detection, E[|X_k(i)|^2] = 1, if Occupied ; else, E[|X_k(i)|^2] = 0
# Signed off by bkeshava on 01-May-2019
class OccupancyState(Enum):
    # Occupancy state IDLE
    idle = 0
    # Occupancy state OCCUPIED:
    occupied = 1


# The Markov Chain object that can be used via extension or replication in order to imply Markovian correlation...
# ...across either the channel indices or the time indices
# Signed off by bkeshava on 01-May-2019
class MarkovChain(object):

    # Initialization sequence
    # Signed off by bkeshava on 01-May-2019
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
    # Signed off by bkeshava on 01-May-2019
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
    # Signed off by bkeshava on 01-May-2019
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
    # Signed off by bkeshava on 01-May-2019
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
    # Signed off by bkeshava on 01-May-2019
    def __exit__(self, exc_type, exc_val, exc_tb):
        print('[INFO] MarkovChain Termination: Tearing things down...')


# A Utility class for all to use...
# Signed off by bkeshava on 01-May-2019
class Util(object):

    # Initialization sequence
    # Signed off by bkeshava on 01-May-2019
    def __init__(self):
        print('[INFO] Util Initialization: Bringing things up...')

    # Construct the complete start probabilities dictionary
    # Signed off by bkeshava on 01-May-2019
    @staticmethod
    def construct_start_probabilities_dict(pi):
        return {0: (1 - pi), 1: pi}

    # Get the steady state probability for the state passed as argument
    # Signed off by bkeshava on 01-May-2019
    @staticmethod
    def get_state_probability(state, pi):
        return (lambda: 1 - pi, lambda: pi)[state == 1]()

    # Construct the complete transition probability matrix from P(Occupied|Idle), i.e. 'p' and P(Occupied), i.e. 'pi'
    # Signed off by bkeshava on 01-May-2019
    @staticmethod
    def construct_transition_probability_matrix(p, pi):
        # P(Idle|Occupied)
        q = (p * (1 - pi)) / pi
        return {0: {1: p, 0: 1 - p}, 1: {0: q, 1: 1 - q}}

    # Generate the action set based on the SU sensing limitations and the Number of Channels in the discretized...
    # ...spectrum of interest
    @staticmethod
    @deprecated
    # Deprecated off by bkeshava on 01-May-2019
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
    # Signed off by bkeshava on 01-May-2019
    def normalize(self, belief, belief_sum):
        normalized_sum = 0
        for key in belief.keys():
            belief[key] /= belief_sum
            normalized_sum += belief[key]
        return belief, self.validate_belief(normalized_sum)

    # Perform belief validation
    # Signed off by bkeshava on 01-May-2019
    @staticmethod
    def validate_belief(normalized_sum):
        # Interval to account for precision errors
        if 0.90 <= normalized_sum <= 1.10:
            return True
        return False

    # Termination sequence
    # Signed off by bkeshava on 01-May-2019
    def __exit__(self, exc_type, exc_val, exc_tb):
        print('[INFO] Util Termination: Tearing things down...')


# This entity encapsulates the Channel object - simulates the Channel, i.e. Complex AWGN and Complex Impulse Response
# Signed off by bkeshava on 01-May-2019
class Channel(object):

    # Initialization sequence
    # Signed off by bkeshava on 01-May-2019
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
    # Signed off by bkeshava on 01-May-2019
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

    # Generate the Complex AWGN samples
    # Signed off by bkeshava on 01-May-2019
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
    # Signed off by bkeshava on 01-May-2019
    def __exit__(self, exc_type, exc_val, exc_tb):
        print('[INFO] Channel Termination: Tearing things down...')


# This class encapsulates the Licensed User dynamically occupying the discretized spectrum under analysis
# Signed off by bkeshava on 01-May-2019
class PrimaryUser(object):

    # Initialization sequence
    # Signed off by bkeshava on 01-May-2019
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
    # Signed off by bkeshava on 01-May-2019
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
    # Signed off by bkeshava on 01-May-2019
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
    # Signed off by bkeshava on 01-May-2019
    def __exit__(self, exc_type, exc_val, exc_tb):
        print('[INFO] PrimaryUser Termination: Tearing things down...')


# This entity emulates a Secondary User (SU) intelligently accessing the spectrum un-occupied by the licensed user (PU)
# Signed off by bkeshava on 01-May-2019
class SecondaryUser(object):

    # Initialization sequence
    # Signed off by bkeshava on 01-May-2019
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
    # Signed off by bkeshava on 01-May-2019
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
    # Signed off by bkeshava on 01-May-2019
    def __exit__(self, exc_type, exc_val, exc_tb):
        print('[INFO] SecondaryUser Termination: Tearing things down...')


# This entity evaluates the emission probabilities, i.e. P(y|x)
# Signed off by bkeshava on 01-May-2019
class EmissionEvaluator(object):

    # Initialization sequence
    # Signed off by bkeshava on 01-May-2019
    def __init__(self, _impulse_response_variance, _noise_variance):
        print('[INFO] EmissionEvaluator Initialization: Bringing things up...')
        # Variance of the Channel Impulse Response
        self.impulse_response_variance = _impulse_response_variance
        # Variance of the Complex AWGN
        self.noise_variance = _noise_variance

    # Get the Emission Probabilities -> P(y|x)
    # Signed off by bkeshava on 01-May-2019
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
    # Signed off by bkeshava on 01-May-2019
    def __exit__(self, exc_type, exc_val, exc_tb):
        print('[INFO] EmissionEvaluator Termination: Tearing things down...')


# The Markov Chain State Estimator algorithm - Viterbi Algorithm
# Signed off by bkeshava on 01-May-2019
class StateEstimator(object):
    # Value function named tuple
    VALUE_FUNCTION_NAMED_TUPLE = namedtuple('ValueFunction', ['current_value', 'previous_state'])

    # Initialization sequence
    # Signed off by bkeshava on 01-May-2019
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
    # Signed off by bkeshava on 01-May-2019
    @staticmethod
    def get_entry(collection, index):
        if len(collection) is not 0 and collection[index] is not None:
            return collection[index]
        else:
            # Empty Place-Holder value is 0
            return 0

    # Get enumeration field value from name
    # Signed off by bkeshava on 01-May-2019
    @staticmethod
    def value_from_name(name):
        if name == 'occupied':
            return OccupancyState.occupied.value
        else:
            return OccupancyState.idle.value

    # Get the start probabilities from the named tuple - a simple getter utility method exclusive to this class
    # Signed off by bkeshava on 01-May-2019
    def get_start_probabilities(self, state_name):
        if state_name == 'occupied':
            return self.start_probabilities[1]
        else:
            return self.start_probabilities[0]

    # Return the transition probabilities from the transition probabilities matrix
    # Signed off by bkeshava on 01-May-2019
    def get_transition_probabilities(self, row, column):
        return self.transition_probabilities[row][column]

    # Get the Emission Probabilities -> P(y|x)
    # Signed off by bkeshava on 01-May-2019
    def get_emission_probabilities(self, state, observation_sample):
        return self.emission_evaluator.get_emission_probabilities(state, observation_sample)

    # Output the estimated state of the frequency bands in the wideband spectrum of interest
    # Output a collection consisting of the required parameters of interest
    # Signed off by bkeshava on 01-May-2019
    def estimate_pu_occupancy_states(self):
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
        resultant_estimated_state = []
        # Average the error out of multiple iterations of this state estimation sub-routine
        for channel in range(0, self.number_of_channels):
            _list = [entry[channel] for entry in estimated_states_array]
            resultant_estimated_state.append(max(_list, key=_list.count))
        return resultant_estimated_state

    # Termination sequence
    # Signed off by bkeshava on 01-May-2019
    def __exit__(self, exc_type, exc_val, exc_tb):
        print('[INFO] StateEstimator Termination: Tearing things down...')


# This entity encapsulates the agent which provides rewards based on the state of the system and the action taken by...
# ...the POMDP agent
# Signed off by bkeshava on 01-May-2019
class Sweepstakes(object):

    # Initialization sequence
    # _mu = Missed_Detection_Cost
    # Signed off by bkeshava on 01-May-2019
    def __init__(self, _primary_user, _mu):
        print('[INFO] Sweepstakes Initialization: Bringing things up...')
        # The Primary User
        self.primary_user = _primary_user
        # Missed_Detection_Cost
        self.mu = _mu

    # Get Reward based on the system state and the action taken by the POMDP agent
    # The estimated_state member is created by the action taken by the POMDP agent
    # Reward = (1-Probability_of_false_alarm) + Missed_Detection_Cost*Probability_of_missed_detection
    # Signed off by bkeshava on 01-May-2019
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
    # Signed off by bkeshava on 01-May-2019
    def __exit__(self, exc_type, exc_val, exc_tb):
        print('[INFO] Sweepstakes Termination: Tearing things down...')


# This entity encapsulates an Oracle which knows the best possible channels to use in each episode.
# Hence, the policy followed by this Oracle is the most optimal policy
# The action policy achieved by the POMDP agent will be evaluated/benchmarked against the Oracle's policy thereby...
# ...giving us a regret metric.
# Signed off by bkeshava on 01-May-2019
class Oracle(object):

    # Initialization sequence
    # Signed off by bkeshava on 01-May-2019
    def __init__(self, _number_of_episodes):
        print('[INFO] Oracle Initialization: Bringing things up...')
        # Number of relevant time steps of POMDP agent interaction with the radio environment
        self.number_of_episodes = _number_of_episodes

    # Get the long term reward by following the optimal policy - This is the Oracle. She knows everything.
    # P_FA = 0
    # P_MD = 0
    # Return = Sum_{1}^{number_of_episodes}\ [1] = number_of_episodes
    # Episodic Reward = 1
    # Signed off by bkeshava on 01-May-2019
    def get_return(self):
        return self.number_of_episodes

    # Windowed return for regret analysis
    # Signed off by bkeshava on 01-May-2019
    @staticmethod
    def get_windowed_return(window_size):
        return window_size

    # Termination sequence
    # Signed off by bkeshava on 01-May-2019
    def __exit__(self, exc_type, exc_val, exc_tb):
        print('[INFO] Oracle Termination: Tearing things down...')


# Top-level Executive class
# This entity encapsulates the POMDP Approximate Point-based Value Iteration algorithm named The PERSEUS algorithm
# Training to fine-tune the belief space -> Perform backup until convergence -> Re-sample using the most recent policy
# References to the ParameterEstimation algorithm and the StateEstimation algorithm in the belief analysis phase
# Independent Fragmentation based POMDP Approximate Value Iteration method
# Signed off by bkeshava on 01-May-2019
# This will serve as a single agent working on a single fragment of correlated channels from a set of independent...
# ...fragments as allocated by the Multiprocessor
class FragmentedAdaptiveIntelligenceWithModelForesight(object):

    # Setup the Markov Chain
    # Signed off by bkeshava on 01-May-2019
    @staticmethod
    def setup_markov_chain(_pi, _p, _correlation_class):
        print('[INFO] AdaptiveIntelligence setup_markov_chain: Setting up the Markov Chain...')
        transient_markov_chain_object = MarkovChain()
        transient_markov_chain_object.set_markovian_correlation_class(_correlation_class)
        transient_markov_chain_object.set_start_probability_parameter(_pi)
        transient_markov_chain_object.set_transition_probability_parameter(_p)
        return transient_markov_chain_object

    # Initialization sequence
    # Signed off by bkeshava on 01-May-2019
    def __init__(self, _id, _channel_count, _round_count, _episode_count, _exploration_window, _limitation_metric,
                 _confidence_bound, _discount_factor, _awgn_mean, _awgn_variance, _frequency_domain_channel_mean,
                 _frequency_domain_channel_variance, _penalty):
        print('[INFO] AdaptiveIntelligence Initialization: Bringing things up...')
        # Identification tag
        self.id = _id
        # Agent-specific - The number of channels in the discretized spectrum of interest
        self.channel_count = _channel_count
        # Agent-specific - The number of sampling rounds in each episode
        self.round_count = _round_count
        # Agent-specific - The number of interactions of the POMDP with the radio environment
        self.episode_count = _episode_count
        # Agent-specific - The exploration period of the PERSEUS algorithm
        self.exploration_window = _exploration_window
        # Agent-specific - The maximum number of channels that can be sensed by the SU simultaneously
        self.limitation_metric = _limitation_metric
        # Agent-specific - The confidence bound for convergence analysis
        self.confidence_bound = _confidence_bound
        # Agent-specific - The discount factor employed in the Bellman equation
        self.discount_factor = _discount_factor
        # Agent-specific - The mean of the AWGN samples
        self.awgn_mean = _awgn_mean
        # Agent-specific - The variance of the AWGN samples
        self.awgn_variance = _awgn_variance
        # Agent-specific - The mean of the channel impulse response samples
        self.frequency_domain_channel_mean = _frequency_domain_channel_mean
        # Agent-specific - The variance of the channel impulse response samples
        self.frequency_domain_channel_variance = _frequency_domain_channel_variance
        # Agent-specific - The cost for missed detections
        self.penalty = _penalty
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
        self.channel = Channel(self.channel_count, self.round_count, self.episode_count,
                               self.awgn_mean, self.awgn_variance, self.frequency_domain_channel_mean,
                               self.frequency_domain_channel_variance)
        # The Emission Evaluator
        self.emission_evaluator = EmissionEvaluator(self.frequency_domain_channel_variance, self.awgn_variance)
        # Primary User
        self.primary_user = PrimaryUser(self.channel_count, self.episode_count,
                                        self.spatial_markov_chain, self.temporal_markov_chain, self.util)
        self.primary_user.simulate_occupancy_behavior()
        # Secondary User
        self.secondary_user = SecondaryUser(self.channel_count, self.round_count, self.channel,
                                            self.primary_user.occupancy_behavior_collection, self.limitation_metric)
        # Sweepstakes - The Reward Analyzer
        self.sweepstakes = Sweepstakes(self.primary_user, self.penalty)
        # The Oracle
        self.oracle = Oracle(self.episode_count)
        # State Estimator
        self.state_estimator = StateEstimator(self.channel_count, self.round_count, None,
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
        self.all_possible_states = list(map(list, itertools.product([0, 1], repeat=self.channel_count)))
        # All possible actions based on my SU sensing limitations
        self.all_possible_actions = []
        for state in self.all_possible_states:
            if sum(state) == self.limitation_metric:
                self.all_possible_actions.append(state)

    # Get Emission Probabilities for the Belief Update sequence
    # Signed off by bkeshava on 01-May-2019
    def get_emission_probability(self, observations, state):
        emission_probability = 1
        # round_choice to introduce some randomness into the emission evaluations
        round_choice = random.choice([k for k in range(0, self.round_count)])
        # Given the system state, the emissions are independent of each other
        for index in range(0, self.channel_count):
            emission_probability = emission_probability * self.emission_evaluator.get_emission_probabilities(
                state[index], observations[index][round_choice])
        return emission_probability

    # Get State Transition Probabilities for the Belief Update sequence
    # Signed off by bkeshava on 01-May-2019
    def get_transition_probability(self, prev_state, next_state, transition_probabilities_matrix):
        # Temporal/Episodic change for the first channel
        transition_probability = transition_probabilities_matrix[prev_state[0]][next_state[0]]
        for index in range(1, self.channel_count):
            transition_probability = transition_probability * transition_probabilities_matrix[next_state[index - 1]][
                next_state[index]]
        return transition_probability

    # Get the normalization constant
    # Signed off by bkeshava on 01-May-2019
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
    # Signed off by bkeshava on 01-May-2019
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
    # Signed off by bkeshava on 01-May-2019
    def random_exploration(self, policy):
        reachable_beliefs = dict()
        state_space_size = 2 ** self.channel_count
        # Uniform belief assignment to all states in the state space
        initial_belief_vector = dict()
        for state in self.all_possible_states:
            state_key = ''.join(str(k) for k in state)
            initial_belief_vector[state_key] = 1 / state_space_size
        previous_belief_vector = initial_belief_vector
        reachable_beliefs['0'] = initial_belief_vector
        if policy is not None:
            print(
                '[WARN] AdaptiveIntelligence random_exploration: Specific policy exploration is yet to be implemented')
        print('[INFO] AdaptiveIntelligence random_exploration: Using a truly random exploration strategy')
        # Start exploring
        for episode in range(1, self.exploration_window):
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
            print('[INFO] AdaptiveIntelligence random_exploration: {}% Exploration '
                  'completed'.format(int(((episode + 1) / self.exploration_window) * 100)))
        return reachable_beliefs

    # Initialization
    # Signed off by bkeshava on 01-May-2019
    def initialize(self, reachable_beliefs):
        # V_0 for the reachable beliefs
        value_function_collection = dict()
        # Default action is not sensing anything - a blind guess from just the noise
        default_action = [k - k for k in range(0, self.channel_count)]
        for belief_key in reachable_beliefs.keys():
            value_function_collection[belief_key] = (-10, default_action)
        return value_function_collection

    # Calculate Utility
    # Signed off by bkeshava on 01-May-2019
    def calculate_utility(self, policy_collection):
        utility = 0
        for key, value in policy_collection.items():
            system_state = []
            for channel in range(0, self.channel_count):
                # int(key) refers to the episode number
                system_state.append(self.primary_user.occupancy_behavior_collection[channel][int(key)])
            # In that episode, the POMDP agent believed that the system is in a certain probabilistic distribution...
            # over the state vector, i.e. the belief, and based on this it recommended the best possible action to be...
            # taken in that state - obtained by going through the backup procedure...
            observation_samples = self.secondary_user.make_observations(int(key), value[1])
            self.state_estimator.observation_samples = observation_samples
            estimated_state = self.state_estimator.estimate_pu_occupancy_states()
            print('[DEBUG] AdaptiveIntelligence calculate_utility: Estimated PU Occupancy states - {}'.format(
                str(estimated_state)))
            utility += self.sweepstakes.roll(estimated_state, system_state)
        return utility

    # The Backup stage
    # Signed off by bkeshava on 01-May-2019
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
            print('[INFO] AdaptiveIntelligence backup: Size of unimproved belief set = {}'.format(
                len(unimproved_belief_points)))
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
                print('[DEBUG] AdaptiveIntelligence backup: Estimated PU Occupancy states - {}'.format(
                    str(estimated_system_state)))
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
                internal_term = reward_sum + (self.discount_factor * normalization_constant *
                                              previous_stage_value_function_collection[pilot_belief_key][0])
                if internal_term > max_value_function:
                    max_value_function = internal_term
                    max_action = action
            if round(max_value_function, 3) > round(previous_stage_value_function_collection[belief_sample_key][0], 3):
                print('[DEBUG] AdaptiveIntelligence backup: [Action: {} and New_Value_Function: {}] pair improves '
                      'the existing policy - '
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
                print('[DEBUG] AdaptiveIntelligence backup: Improving the other belief points...')
                normalization_constant = 0
                reward_sum = 0
                observation_samples = self.secondary_user.make_observations(stage_number, max_action)
                # Estimate the System State
                self.state_estimator.observation_samples = observation_samples
                estimated_system_state = self.state_estimator.estimate_pu_occupancy_states()
                print('[DEBUG] AdaptiveIntelligence backup: Estimated PU Occupancy states - {}'.format(
                    str(estimated_system_state)))
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
                internal_term = reward_sum + (self.discount_factor * normalization_constant *
                                              previous_stage_value_function_collection[aux_pilot_belief_key][0])
                if round(internal_term, 3) > round(previous_stage_value_function_collection[belief_point_key][0], 3):
                    # Note here that del mutates the contents of the dict for everyone who has a reference to it
                    print('[DEBUG] AdaptiveIntelligence backup: Auxiliary points improved by action {} with {}'.format(
                        str(max_action),
                        str(internal_term) + ' > ' + str(previous_stage_value_function_collection[
                                                             belief_point_key][0])))
                    del unimproved_belief_points[belief_point_key]
                    next_stage_value_function_collection[belief_point_key] = (internal_term, max_action)
                    number_of_belief_changes += 1
            utility = self.calculate_utility(next_stage_value_function_collection)
            print('[INFO] AdaptiveIntelligence backup: Logging all the relevant metrics within this Backup '
                  'stage - [Utility: {}, '
                  '#policy_changes: {}, sampled_value_function: {}]'.format(utility,
                                                                            number_of_belief_changes,
                                                                            next_stage_value_function_collection[
                                                                                self.belief_choice][0]
                                                                            ))
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
        stage_number = self.exploration_window - 1
        # Utility addition for the initial value function
        utility = self.calculate_utility(previous_value_function_collection)
        print('[DEBUG] AdaptiveIntelligence run_perseus: Adding the utility metric for the initial value '
              'function - {}'.format(utility))
        self.utilities.append(utility)
        # Local confidence check for modelling policy convergence
        confidence = 0
        # Check for termination condition here...
        while confidence < self.confidence_bound:
            self.value_function_changes_array.append(previous_value_function_collection[self.belief_choice][0])
            stage_number += 1
            # We've reached the end of our allowed interaction time with the radio environment
            if stage_number == self.episode_count:
                print('[WARN] AdaptiveIntelligence run_perseus: We have reached the end of our allowed interaction '
                      'time with the radio environment!')
                return
            # Backup to find \alpha -> Get V_{n+1} and #BeliefChanges
            backup_results = self.backup(stage_number, reachable_beliefs, previous_value_function_collection)
            print('[DEBUG] AdaptiveIntelligence run_perseus: Backup for stage {} completed...'.format(
                stage_number - self.exploration_window + 1))
            next_value_function_collection = backup_results[0]
            belief_changes = backup_results[1]
            self.policy_changes.append(belief_changes)
            if len(next_value_function_collection) is not 0:
                previous_value_function_collection = next_value_function_collection
            if belief_changes is 0:
                print('[DEBUG] AdaptiveIntelligence run_perseus: Confidence Update - {}'.format(confidence))
                confidence += 1
            else:
                confidence = 0
                print(
                    '[DEBUG] AdaptiveIntelligence run_perseus: Confidence Stagnation/Fallback - {}'.format(confidence))

    # Termination sequence
    # Signed off by bkeshava on 01-May-2019
    def __exit__(self, exc_type, exc_val, exc_tb):
        print('[INFO] AdaptiveIntelligence Termination: Tearing things down...')


# Regret Analysis
# Signed off by bkeshava on 01-May-2019
def analyze_regret(_utilities, _episodes, _exploration_window):
    print('[INFO] AdaptiveIntelligence analyze_regret: Plotting the Regret Convergence Plot for the PERSEUS '
          'algorithm employed for PU Behavioral Analysis with Double Markov Chain assumptions...')
    oracle = Oracle(_episodes)
    utilities_array = numpy.array(_utilities[0])
    for i in range(1, len(_utilities)):
        utilities_array += numpy.array(_utilities[i])
    x_axis = []
    y_axis = []
    for k in range(0, len(utilities_array)):
        x_axis.append(k)
        y_axis.append(oracle.get_windowed_return(_exploration_window) - utilities_array[k])
    fig, ax = plt.subplots()
    ax.plot(x_axis, y_axis, linewidth=1.0, marker='o', color='r')
    fig.suptitle(
        'Regret convergence plot of the PERSEUS algorithm for a Double Markov Chain PU Behavioral Model',
        fontsize=12)
    ax.set_xlabel('Stages -->', fontsize=14)
    ax.set_ylabel('Regret', fontsize=14)
    plt.show()


# Visualize the progression of policy changes
# Signed off by bkeshava on 01-May-2019
def visualize_progression_of_policy_changes(_policy_changes):
    print('[INFO] AdaptiveIntelligence visualize_progression_of_policy_changes: Plotting the #policy_changes for '
          'the PERSEUS algorithm employed for PU Behavioral Analysis with Double Markov Chain assumptions...')
    policy_changes_array = numpy.array(_policy_changes[0])
    for i in range(1, len(_policy_changes)):
        policy_changes_array += numpy.array(_policy_changes[i])
    x_axis = []
    y_axis = []
    for k in range(0, len(policy_changes_array)):
        x_axis.append(k)
        y_axis.append(policy_changes_array[k])
    fig, ax = plt.subplots()
    ax.plot(x_axis, y_axis, linewidth=1.0, marker='o', color='b')
    fig.suptitle(
        '#policy_changes during the course of the PERSEUS algorithm for a Double Markov Chain PU Behavioral Model',
        fontsize=12)
    ax.set_xlabel('Stages -->', fontsize=14)
    ax.set_ylabel('#policy_changes', fontsize=14)
    plt.show()


# Visualize the progression of the value function for a random choice of belief as the algorithm moves towards...
# ...optimality
# Signed off by bkeshava on 01-May-2019
def visualize_progression_of_the_value_function(_value_function_changes):
    print(
        '[INFO] AdaptiveIntelligence visualize_progression_of_the_value_function: Plotting the progression of the '
        'value function for the PERSEUS algorithm employed for PU Behavioral Analysis '
        'with Double Markov Chain assumptions...')
    value_function_changes_array = numpy.array(_value_function_changes[0])
    for i in range(1, len(_value_function_changes)):
        value_function_changes_array += numpy.array(_value_function_changes[i])
    x_axis = []
    y_axis = []
    for k in range(0, len(value_function_changes_array)):
        x_axis.append(k)
        y_axis.append(value_function_changes_array[k])
    fig, ax = plt.subplots()
    ax.plot(x_axis, y_axis, linewidth=1.0, marker='o', color='k')
    fig.suptitle(
        'Progression of the value function during the course of the PERSEUS algorithm for a Double Markov Chain '
        'PU Behavioral Model',
        fontsize=12)
    ax.set_xlabel('Stages -->', fontsize=14)
    ax.set_ylabel('Stage Specific Value Function', fontsize=14)
    plt.show()


@deprecated
# Group the channels in the discretized spectrum of interest into fragments
def group(spectrum, _fragment_size):
    args = [iter(spectrum)] * _fragment_size
    return ([k for k in t if k is not None] for t in itertools.zip_longest(*args))


# Worker bee
def worker(obj):
    # Waiting for run from the queen
    obj.run_perseus()
    return {'UTILITIES': obj.utilities,
            'VALUE_FUNCTION_CHANGES': obj.value_function_changes_array,
            'POLICY_CHANGES': obj.policy_changes}


# Post-processing of the results to visualize the regret convergence, #policy_changes, and value function progression
def post_processor(results, episodes, exploration):
    utilities_collection = [k['UTILITIES'] for k in results]
    policy_changes_collection = [k['POLICY_CHANGES'] for k in results]
    value_function_progression_collection = [k['VALUE_FUNCTION_CHANGES'] for k in results]
    utilities_resolution = min(utilities_collection, key=len)
    policy_changes_resolution = min(policy_changes_collection, key=len)
    value_function_progression_resolution = min(value_function_progression_collection, key=len)
    trimmed_utilities_collection = [k[0:utilities_resolution] for k in utilities_collection]
    trimmed_policy_changes_collection = [k[0:policy_changes_resolution] for k in policy_changes_collection]
    trimmed_value_function_progression_collection = [k[0:value_function_progression_resolution] for k in
                                                     value_function_progression_collection]
    analyze_regret(trimmed_utilities_collection, episodes, exploration)
    visualize_progression_of_policy_changes(trimmed_policy_changes_collection)
    visualize_progression_of_the_value_function(trimmed_value_function_progression_collection)


# Run Trigger
# Signed off by bkeshava on 01-May-2019
if __name__ == '__main__':
    print('[INFO] AdaptiveIntelligence main: Starting system simulation...')
    # The number of channels in the discretized spectrum of interest - global
    number_of_channels_global = 18
    # The size of each fragment under analysis by individual POMDP agents - per agent
    fragment_size = 6
    # The number of sampling rounds in each episode - per agent
    sampling_rounds_count = 250
    # The number of interactions of the POMDP with the radio environment - per agent
    episodes_count = 1000
    # The exploration period of the PERSEUS algorithm - per agent
    exploration_duration = 30
    # The mean of the AWGN samples - global
    noise_mean_value = 0
    # The mean of the channel impulse response samples - global
    impulse_response_mean_value = 0
    # The variance of the AWGN samples - global
    noise_variance_value = 1
    # The variance of the channel impulse response samples - global
    impulse_response_variance_value = 80
    # The confidence bound for convergence analysis - per agent
    confidence_threshold = 10
    # The maximum number of channels that can be sensed by the SU simultaneously - per agent
    su_sensing_limitation_per_agent = 3
    # The discount factor employed in the Bellman equation - per agent/global
    discount_factor_value = 0.9
    # The cost for missed detections - per agent/global
    penalty_value = -1
    global_discretized_spectrum_of_interest = [k for k in range(0, number_of_channels_global)]
    jobs = []
    number_of_fragments = math.ceil(number_of_channels_global / fragment_size)
    agents = [FragmentedAdaptiveIntelligenceWithModelForesight(fragment_index, fragment_size, sampling_rounds_count,
                                                               episodes_count, exploration_duration,
                                                               su_sensing_limitation_per_agent, confidence_threshold,
                                                               discount_factor_value, noise_mean_value,
                                                               noise_variance_value, impulse_response_mean_value,
                                                               impulse_response_variance_value,
                                                               penalty_value) for fragment_index in range(
        1, number_of_fragments + 1)]
    # Create the multi-processor pool
    # Keep in mind the number of cores you have on your machine
    pool = multiprocessing.Pool(number_of_fragments)
    # Map objs to worker bees
    result_collection = pool.map(worker, (agent for agent in agents))
    # Close the pool
    pool.close()
    # Join with the main thread - termination
    pool.join()
    # Post-Processing
    post_processor(result_collection, episodes_count, exploration_duration)
    print('[INFO] AdaptiveIntelligence main: System Simulation terminated...')
