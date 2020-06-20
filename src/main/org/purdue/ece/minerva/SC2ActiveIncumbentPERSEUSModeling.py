# This entity describes the POMDP Approximate Point-Based Value Iteration Algorithm, i.e. the PERSEUS algorithm
#   applied to PU Occupancy Behavior Estimation in Cognitive Radio Networks.
# This is the improved version modified to handle the changes to the correlation model underlying the occupancy
#   behavior of incumbents in the network along with some additional changes to the analytics engine.
# Author: Bharath Keshavamurthy
# Organization: School of Electrical and Computer Engineering, Purdue University, West Lafayette, IN.
# Copyright (c) 2020. All Rights Reserved.

# WIKI
#   This entity describes the PERSEUS Algorithm for solving the optimal Spectrum Sensing and Access Problem in Cognitive
#   Radio Networks wherein a SecondaryUser (SU) needs to sense a subset of channels and access channels in the radio
#   environment that it deems to be idle, i.e. un-occupied by the Primary Users (PUs - Incumbents).

# PREMISE OF THE FRAGMENTED PERSEUS ALGORITHM
#   The entire premise behind the fragmented PERSEUS algorithm in a radio environment where there is more than one PU
#   is that the PUs generally are restricted to certain segments of the radio spectrum by design and by bureaucracy.
#   Therefore, in order to speed up the PERSEUS algorithm over a significantly large number of channels, we set out to
#   partition the radio spectrum into fragments, each corresponding to the spectral constraints of incumbents, as set
#   by governments or regulatory agencies. So, we run instances of the PERSEUS agent (SU with PERSEUS as its
#   Channel Sensing and Access strategy/heuristic) over these fragments and then, aggregate the utilities obtained
#   by these individual PERSEUS instances in order to get the overall utility obtained by the SU over numerous
#   episodes of interaction with the radio environment.

# PERSEUS ALGORITHM WITH MODEL FORESIGHT AND SIMPLIFIED BELIEF UPDATE PRACTICES

# %%%%% IMPROVED %%%%%

# Additionally, this entity evaluates the performance of this PERSEUS engine in the DARPA SC2 Active Incumbent scenario.

# The imports
import math
import numpy
import random
import itertools
import scipy.stats
from enum import Enum
from collections import namedtuple
import DARPASC2ActiveIncumbentAnalysis as Analyser


# Occupancy State Enumeration
# Based on Energy Detection, \mathbb{E}[|X_k(i)|^2] = 1, if Occupied; else, \mathbb{E}[|X_k(i)|^2] = 0
class OccupancyState(Enum):
    # Occupancy state IDLE
    IDLE = 0
    # Occupancy state OCCUPIED
    OCCUPIED = 1


# Delegate
# A Utility class for all to use...
class Util(object):

    # The initialization sequence
    def __init__(self):
        print('[INFO] Util Initialization: Bringing things up...')
        # Nothing to do...

    # Construct the complete start probabilities dictionary
    @staticmethod
    def construct_start_probabilities_dict(pi):
        return {0: (1 - pi), 1: pi}

    # Get the steady state probability for the state passed as the argument
    # The "state" member is an enum instance of OccupancyState.
    @staticmethod
    def get_state_probability(state, pi):
        return (lambda: 1 - pi, lambda: pi)[state == OccupancyState.OCCUPIED]()

    # Construct the complete transition probability matrix from \mathbb{P}(Occupied|Idle), i.e., 'p' and
    #   \mathbb{P}(Occupied), i.e., 'pi'
    @staticmethod
    def construct_transition_probability_matrix(p, pi):
        # \mathbb{P}(Idle|Occupied)
        q = (p * (1 - pi)) / pi
        return {0: {0: 1 - p, 1: p}, 1: {0: q, 1: 1 - q}}

    # Perform belief validation
    @staticmethod
    def validate_belief(normalized_sum):
        # An interval to account for precision errors
        if 0.90 <= normalized_sum <= 1.10:
            return True
        return False

    # Normalize the belief, if necessary
    def normalize(self, belief, belief_sum):
        normalized_sum = 0
        for key in belief.keys():
            belief[key] /= belief_sum
            normalized_sum += belief[key]
        return belief, self.validate_belief(normalized_sum)

    # The termination sequence
    def __exit__(self, exc_type, exc_val, exc_tb):
        print('[INFO] Util Termination: Tearing things down...')
        # Nothing to do...


# Delegate
# This entity encapsulates the Channel object - simulates the Channel
class Channel(object):

    # The initialization sequence
    def __init__(self, _number_of_channels, _number_of_sampling_rounds, _number_of_episodes, _noise_mean,
                 _noise_variance, _impulse_response_mean, _impulse_response_variance):
        print('[INFO] Channel Initialization: Bringing things up...')
        # Noise Statistics
        self.noise_mean = _noise_mean
        if self.noise_mean != 0:
            print('[WARN] Channel Initialization: The system assumes Zero-Mean, Additive, White, Gaussian '
                  'Noise...')
            self.noise_mean = 0
        self.noise_variance = _noise_variance
        # Channel Impulse Response Statistics
        self.impulse_response_mean = _impulse_response_mean
        if self.impulse_response_mean != 0:
            print('[WARN] Channel Initialization: The system assumes Zero-Mean, Gaussian Channel Impulse Response...')
            self.impulse_response_mean = 0
        self.impulse_response_variance = _impulse_response_variance
        # Number of channels in the discretized spectrum of interest
        self.number_of_channels = _number_of_channels
        # Number of sampling rounds undertaken by the SU per episode
        self.number_of_sampling_rounds = _number_of_sampling_rounds
        # Number of episodes in which the PERSEUS-III agent interacts with the radio environment
        self.number_of_episodes = _number_of_episodes
        # The Channel Impulse Response samples used in the Observation Model
        self.impulse_response = self.get_impulse_response()
        # The AWGN samples used in the Observation Model
        self.noise = self.get_noise()

    # Generate the Channel Impulse Response samples
    def get_impulse_response(self):
        # The metrics to be passed to numpy.random.normal(mu, std, n)
        mu_channel_impulse_response = self.impulse_response_mean
        std_channel_impulse_response = numpy.sqrt(self.impulse_response_variance)
        n_channel_impulse_response = self.number_of_sampling_rounds
        # The output
        channel_impulse_response_samples = []
        # Logic
        for k in range(0, self.number_of_channels):
            channel_impulse_response_samples.append([])
        for channel in range(0, self.number_of_channels):
            for episode in range(0, self.number_of_episodes):
                channel_impulse_response_samples[channel].append(numpy.random.normal(mu_channel_impulse_response,
                                                                                     std_channel_impulse_response,
                                                                                     n_channel_impulse_response))
        return channel_impulse_response_samples

    # Generate the AWGN samples
    def get_noise(self):
        # The metrics to be passed to numpy.random.normal(mu, std, n)
        mu_noise = self.noise_mean
        std_noise = numpy.sqrt(self.noise_variance)
        n_noise = self.number_of_sampling_rounds
        # The output
        noise_samples = []
        # Logic
        for k in range(0, self.number_of_channels):
            noise_samples.append([])
        for channel in range(0, self.number_of_channels):
            for episode in range(0, self.number_of_episodes):
                noise_samples[channel].append(numpy.random.normal(mu_noise, std_noise, n_noise))
        return noise_samples

    # The termination sequence
    def __exit__(self, exc_type, exc_val, exc_tb):
        print('[INFO] Channel Termination: Tearing things down...')
        # Nothing to do...


# Delegate
# This entity emulates a Secondary User (SU) making observations of the required channels in the discretized spectrum
class SecondaryUser(object):

    # The initialization sequence
    def __init__(self, _number_of_channels, _number_of_sampling_rounds, _number_of_episodes, _channel,
                 _true_pu_occupancy_states):
        print('[INFO] SecondaryUser Initialization: Bringing things up...')
        # Channel instance
        self.channel = _channel
        # Number of channels in the discretized spectrum of interest
        self.number_of_channels = _number_of_channels
        # Number of sampling rounds undertaken by the SU per episode
        self.number_of_sampling_rounds = _number_of_sampling_rounds
        # Number of episodes of interaction with the radio environment [encapsulates #sampling_rounds]
        self.number_of_episodes = _number_of_episodes
        # Occupancy Status of the cells based on emulated PU behavior - needed to simulate SU observations
        # DARPA SC2 Active Incumbent
        self.true_pu_occupancy_states = _true_pu_occupancy_states

    # The Secondary User making observations of the channels in the spectrum of interest
    def make_observations(self, episode, channel_selection_strategy):
        observation_samples = []
        for band in range(0, self.number_of_channels):
            obs_per_band = [k - k for k in range(0, self.number_of_sampling_rounds)]
            if channel_selection_strategy[band] == 1:
                obs_per_band = list((numpy.array(
                    self.channel.impulse_response[band][episode]) * self.true_pu_occupancy_states[band][episode]) +
                                    numpy.array(self.channel.noise[band][episode]))
            observation_samples.append(obs_per_band)
        # The observation_samples member is a kxn [channels x sampling_rounds] matrix w.r.t a specific episode
        return observation_samples

    # The termination sequence
    def __exit__(self, exc_type, exc_val, exc_tb):
        print('[INFO] SecondaryUser Termination: Tearing things down...')
        # Nothing to do...


# Delegate
# This entity evaluates the emission probabilities, i.e. \mathbb{P}(y|x)
class EmissionEvaluator(object):

    # The initialization sequence
    def __init__(self, _noise_variance, _impulse_response_variance):
        print('[INFO] EmissionEvaluator Initialization: Bringing things up...')
        # Variance of the AWGN samples
        self.noise_variance = _noise_variance
        # Variance of the Channel Impulse Response samples
        self.impulse_response_variance = _impulse_response_variance

    # Get the Emission Probabilities -> \mathbb{P}(y|x)
    # The "state" member is an enum instance of OccupancyState.
    def get_emission_probabilities(self, state, observation_sample):
        # If the channel is not observed, i.e., if the observation is [$\phi$] or [$0$], report $m_r(y_i)$ as $1$.
        # The Empty Place-Holder value is 0.
        if observation_sample == 0:
            return 1
        # Normal Emission Estimation using the distribution of the observations given the state
        else:
            emission_probability = scipy.stats.norm(0,
                                                    numpy.sqrt(
                                                        (self.impulse_response_variance * state.value) +
                                                        self.noise_variance)
                                                    ).pdf(observation_sample)
            return emission_probability

    # The termination sequence
    def __exit__(self, exc_type, exc_val, exc_tb):
        print('[INFO] EmissionEvaluator Termination: Tearing things down...')
        # Nothing to do...


# Delegate
# This entity encapsulates the routine which provides rewards based on the state of the system and the action taken by
#   the PERSEUS-III agent
class Sweepstakes(object):

    # The initialization sequence
    def __init__(self, _mu):
        print('[INFO] Sweepstakes Initialization: Bringing things up...')
        # The penalty for missed detections
        self.mu = _mu

    # Get Reward based on the system state and the action taken by the PERSEUS-III agent
    def roll(self, estimated_state, system_state):
        if len(estimated_state) != len(system_state):
            print('[ERROR] Sweepstakes roll: The estimated_state arg and the system_state arg need to be of the same'
                  'dimension!')
            return 0
        # Let B_k(i) denote the actual true occupancy status of the channel in this 'episode.'
        # Let \hat{B}_k(i) denote the estimated occupancy status of the channel in this 'episode.'
        # Utility = R = \sum_{k=1}^{K}\ (1 - B_k(i)) (1 - \hat{B}_k(i)) + \mu B_k(i) (1 - \hat{B}_k(i))
        reward = 0
        throughput = 0
        interference = 0
        for k in range(0, len(system_state)):
            throughput += (1 - estimated_state[k]) * (1 - system_state[k])
            interference += system_state[k] * (1 - estimated_state[k])
            reward += ((1 - estimated_state[k]) * (1 - system_state[k])) + \
                      (self.mu * (system_state[k] * (1 - estimated_state[k])))
        return reward, throughput, interference

    @staticmethod
    # Get the SU throughput and the PU interference metrics for this episode
    def get_analytics(estimated_state, true_occupancy_state):
        if len(estimated_state) != len(true_occupancy_state):
            print('[ERROR] Sweepstakes get_analytics: The estimated_state arg and the true_occupancy_state arg need to '
                  'be of the same dimension!')
            return 0
        su_throughput = 0
        pu_interference = 0
        for k in range(len(true_occupancy_state)):
            su_throughput += (1 - estimated_state[k]) * (1 - true_occupancy_state[k])
            pu_interference += (1 - estimated_state[k]) * true_occupancy_state[k]
        return su_throughput, pu_interference

    # The termination sequence
    def __exit__(self, exc_type, exc_val, exc_tb):
        print('[INFO] Sweepstakes Termination: Tearing things down...')
        # Nothing to do...


# Delegate
# The Markov Chain State Estimator algorithm - Viterbi Algorithm
# This entity is employed during the operation of the PERSEUS-III agent
class StateEstimator(object):
    # Value function named tuple
    VALUE_FUNCTION_NAMED_TUPLE = namedtuple('ValueFunction', ['current_value', 'previous_state'])

    # The initialization sequence
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
        if collection is not None and len(collection) != 0 and collection[index] is not None:
            return collection[index]
        else:
            # Empty Place-Holder value is 0
            return 0

    # Get enumeration field value from name
    @staticmethod
    def value_from_name(name):
        if name == OccupancyState.OCCUPIED.name:
            return OccupancyState.OCCUPIED.value
        else:
            return OccupancyState.IDLE.value

    # Get the start probabilities from the named tuple - a simple getter utility method exclusive to this class
    # The "state" arg is an instance of the OccupancyState enumeration.
    def get_start_probabilities(self, state):
        if state == OccupancyState.OCCUPIED:
            return self.start_probabilities[1]
        else:
            return self.start_probabilities[0]

    # Return the transition probabilities from the transition probabilities matrix
    # The "row" arg and the "column" arg are instances of the OccupancyState enumeration.
    def get_transition_probabilities(self, row, column):
        return self.transition_probabilities[row.value][column.value]

    # Get the Emission Probabilities -> \mathbb{P}(y|x)
    # The "state" arg is an instance of the OccupancyState enumeration.
    def get_emission_probabilities(self, state, observation_sample):
        return self.emission_evaluator.get_emission_probabilities(state, observation_sample)

    # Output the estimated state of the frequency bands in the discretized spectrum of interest
    def estimate_pu_occupancy_states(self):
        previous_state = None
        estimated_states_array = []
        for sampling_round in range(0, self.number_of_sampling_rounds):
            estimated_states = []
            reduced_observation_vector = []
            for entry in self.observation_samples:
                reduced_observation_vector.append(self.get_entry(entry, sampling_round))
            # Now, I have to estimate the state of the ${NUMBER_OF_FREQUENCY_BANDS} based on
            #   this reduced observation vector
            # INITIALIZATION : The array of initial probabilities is known
            # FORWARD RECURSION
            value_function_collection = []
            for x in range(0, len(reduced_observation_vector)):
                value_function_collection.append(dict())
            for state in OccupancyState:
                current_value = self.get_emission_probabilities(state, reduced_observation_vector[0]) * \
                                self.get_start_probabilities(state)
                value_function_collection[0][state.name] = self.VALUE_FUNCTION_NAMED_TUPLE(
                    current_value=current_value,
                    previous_state=None)
            # For each observation after the first one (I can't apply Markovian to [0])
            for observation_index in range(1, len(reduced_observation_vector)):
                # Trying to find the max pointer here ...
                for state in OccupancyState:
                    max_pointer = self.get_transition_probabilities(OccupancyState.IDLE, state) * \
                                  value_function_collection[observation_index-1][
                                      OccupancyState.IDLE.name].current_value
                    confirmed_previous_state = OccupancyState.IDLE.name
                    for candidate_previous_state in OccupancyState:
                        if candidate_previous_state == OccupancyState.IDLE:
                            # Already done
                            continue
                        else:
                            pointer = self.get_transition_probabilities(candidate_previous_state,
                                                                        state) * \
                                      value_function_collection[observation_index-1][
                                          candidate_previous_state.name].current_value
                            if pointer > max_pointer:
                                max_pointer = pointer
                                confirmed_previous_state = candidate_previous_state.name
                    current_value = max_pointer * self.get_emission_probabilities(state,
                                                                                  reduced_observation_vector[
                                                                                      observation_index])
                    value_function_collection[observation_index][state.name] = self.VALUE_FUNCTION_NAMED_TUPLE(
                        current_value=current_value, previous_state=confirmed_previous_state)
            max_value = 0
            # Finding the max value among the named tuples
            for value in value_function_collection[-1].values():
                if value.current_value > max_value:
                    max_value = value.current_value
            # Finding the state corresponding to this max_value and using this as the final confirmed state to
            #   backtrack and find the previous states
            for k, v in value_function_collection[-1].items():
                if v.current_value == max_value:
                    # FIXME: Using the 'name' member to reference & deference the value function collection is not safe.
                    estimated_states.append(self.value_from_name(k))
                    previous_state = k
                    break
            # BACKTRACKING
            for i in range(len(value_function_collection) - 2, -1, -1):
                estimated_states.insert(0, self.value_from_name(
                    value_function_collection[i+1][previous_state].previous_state))
                previous_state = value_function_collection[i+1][previous_state].previous_state
            estimated_states_array.append(estimated_states)
        # FIXME: Come up with some averaging/thresholding logic here instead of picking the final estimated vector...
        return estimated_states_array[self.number_of_sampling_rounds - 1]

    # The termination sequence
    def __exit__(self, exc_type, exc_val, exc_tb):
        print('[INFO] StateEstimator Termination: Tearing things down...')
        # Nothing to do...


# POMDP agent under analysis (PERSEUS-III)
# This entity encapsulates a POMDP Approximate Point-based Value Iteration Algorithm known as the PERSEUS Algorithm.
# Training to fine-tune the belief space -> Perform backup until convergence -> Re-sample using the most recent policy
# References to the State Estimation Algorithm in the belief analysis phase
# Model Foresight with Belief Simplification
class PERSEUS(object):

    # Rendered delegate behavior
    # Simulate the incumbent occupancy behavior in the spectrum of interest according to the true correlation model
    # DARPA SC2 Active Incumbent scenario
    def simulate_pu_occupancy(self):
        return self.analyser.get_occupancy_behavior()

    # The initialization sequence
    def __init__(self, _number_of_channels, _number_of_sampling_rounds, _number_of_episodes, _exploration_period,
                 _noise_mean, _noise_variance, _impulse_response_mean, _impulse_response_variance, _penalty,
                 _limitation, _confidence_bound, _gamma, _utility_multiplication_factor,
                 _transition_threshold, _exploration_period_divisor, _db):
        print('[INFO] PERSEUS Initialization: Bringing things up...')
        # The Utility object
        self.util = Util()
        # The database file containing the SC2 run information
        self.db = _db
        # The divisor for exploration period determination
        self.exploration_period_divisor = _exploration_period_divisor
        # The DARPA SC2 Active Incumbent scenario emulation analyser
        self.analyser = Analyser.DARPASC2ActiveIncumbentAnalysis(self.db)
        # The number of channels in the discretized spectrum of interest (fragment size)
        self.number_of_channels = _number_of_channels
        # The number of sampling rounds in each episode
        self.number_of_sampling_rounds = _number_of_sampling_rounds
        # The number of time slots of interaction of the PERSEUS-III agent with the radio environment
        self.number_of_episodes = _number_of_episodes
        # The exploration period of the PERSEUS algorithm
        self.exploration_period = _exploration_period
        # The mean of the AWGN samples
        self.noise_mean = _noise_mean
        # The variance of the AWGN samples
        self.noise_variance = _noise_variance
        # The mean of the channel impulse response samples
        self.impulse_response_mean = _impulse_response_mean
        # The variance of the channel impulse response samples
        self.impulse_response_variance = _impulse_response_variance
        # The penalty for missed detections, i.e. PU interference
        self.penalty = _penalty
        # The SU sensing limitation
        self.limitation = _limitation
        # The confidence bound for convergence analysis
        self.confidence_bound = _confidence_bound
        # The discount factor employed in the Bellman equation
        self.gamma = _gamma
        # The true occupancy states of the incumbents in the network
        # Simulate incumbent occupancy behavior to get the occupancy behavior collection...
        self.true_occupancy_states = self.simulate_pu_occupancy()
        # \mathbb{P}(Occupied) - The steady state probability of being occupied
        self.pi = self.analyser.get_steady_state_occupancy_probability(self.true_occupancy_states)
        # The correlation model parameters, i.e., $\vec{\theta}$ which have been learnt prior to triggering PERSEUS...
        # Obtain this by running SC2ActiveIncumbentCorrelationModelEstimator.py -- time-frequency correlation structure
        # Off-script
        self.correlation_model = {
            '0': 0.5,   # q0
            '1': 0.5,   # q1
            '00': 0.5,  # p00
            '01': 0.5,  # p01
            '10': 0.5,  # p10
            '11': 0.5   # p11
        }
        # The spatial chain parameters
        # Obtain this by running SC2ActiveIncumbentCorrelationModelEstimator.py -- frequency correlation only
        # Off-script
        self.spatial_correlation_model = {'0': 0.5, '1': 0.5}
        # The single chain correlation model - temporal
        self.single_chain_transition_model_temporal = {0: {0: 1 - self.correlation_model['0'],
                                                           1: self.correlation_model['0']},
                                                       1: {0: 1 - self.correlation_model['1'],
                                                           1: self.correlation_model['1']}
                                                       }
        # The single chain correlation model - spatial
        self.single_chain_transition_model_spatial = {0: {0: 1 - self.spatial_correlation_model['0'],
                                                          1: self.spatial_correlation_model['0']},
                                                      1: {0: 1 - self.spatial_correlation_model['1'],
                                                          1: self.spatial_correlation_model['1']}
                                                      }
        # The double chain correlation model
        self.double_chain_transition_model = {'00': {0: 1 - self.correlation_model['00'],
                                                     1: self.correlation_model['00']},
                                              '01': {0: 1 - self.correlation_model['01'],
                                                     1: self.correlation_model['01']},
                                              '10': {0: 1 - self.correlation_model['10'],
                                                     1: self.correlation_model['10']},
                                              '11': {0: 1 - self.correlation_model['11'],
                                                     1: self.correlation_model['11']}
                                              }
        # The start probabilities dict
        self.start_probabilities_dict = self.util.construct_start_probabilities_dict(self.pi)
        # The Channel
        self.channel = Channel(self.number_of_channels, self.number_of_sampling_rounds, self.number_of_episodes,
                               self.noise_mean, self.noise_variance, self.impulse_response_mean,
                               self.impulse_response_variance)
        # The Emission Evaluator
        self.emission_evaluator = EmissionEvaluator(self.noise_variance, self.impulse_response_variance)
        # The Secondary User
        self.secondary_user = SecondaryUser(self.number_of_channels, self.number_of_sampling_rounds,
                                            self.number_of_episodes, self.channel,
                                            self.true_occupancy_states)
        # Sweepstakes - The Reward Analyzer
        self.sweepstakes = Sweepstakes(self.penalty)
        # The State Estimator - Single Markov Chain Viterbi Algorithm
        self.state_estimator = StateEstimator(self.number_of_channels, self.number_of_sampling_rounds, None,
                                              self.start_probabilities_dict,
                                              self.single_chain_transition_model_spatial,
                                              self.emission_evaluator)
        # All possible states
        self.all_possible_states = list(map(list, itertools.product([0, 1], repeat=self.number_of_channels)))
        # All possible actions based on the given SU sensing limitation
        self.all_possible_actions = []
        for state in self.all_possible_states:
            if sum(state) == self.limitation:
                self.all_possible_actions.append(state)
        # The utility multiplication factor
        self.utility_multiplication_factor = _utility_multiplication_factor
        # The allowed transition threshold
        self.transition_threshold = _transition_threshold
        # Maximum number of channel changes allowed
        self.max_allowed_transitions = math.ceil(self.transition_threshold * self.number_of_channels)
        # The analytics returned by this agent
        self.analytics = namedtuple('ANALYTICS',
                                    ['su_throughput', 'pu_interference'])

    # Get the enumeration instance based on the value passed as an argument in order to ensure compliance with the
    #   'state' communication APIs
    @staticmethod
    def get_enum_from_value(state_value):
        if state_value == OccupancyState.OCCUPIED.value:
            return OccupancyState.OCCUPIED
        else:
            return OccupancyState.IDLE

    # Get Emission Probabilities for the Belief Update sequence
    def get_emission_probability(self, observations, state):
        emission_probability = 1
        # The 'round_choice' parameter to introduce some randomness into the emission evaluation process
        round_choice = random.choice([k for k in range(0, self.number_of_sampling_rounds)])
        # Given the system state, the emissions are independent of each other
        for index in range(0, self.number_of_channels):
            emission_probability *= \
                self.emission_evaluator.get_emission_probabilities(self.get_enum_from_value(state[index]),
                                                                   observations[index][round_choice])
        return emission_probability

    # Get State Transition Probabilities for the Belief Update sequence
    def get_transition_probability(self, prev_state, next_state):
        # Temporal/Episodic change for the first channel
        transition_probability = self.single_chain_transition_model_temporal[prev_state[0]][next_state[0]]
        for index in range(1, self.number_of_channels):
            # Spatial and Temporal change for the rest of 'em
            transition_probability *= self.double_chain_transition_model[''.join(
                [str(next_state[index-1], str(prev_state[index]))])][next_state[index]]
        return transition_probability

    # Get the allowed state transitions previous_states -> next_states
    def get_allowed_state_transitions(self, state):
        allowed_state_transitions = list()
        combinations_array = dict()
        for transition_allowance in range(1, self.max_allowed_transitions + 1):
            combinations_array[transition_allowance] = list(
                itertools.combinations([k for k in range(0, len(state))],
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
                multiplier += self.get_transition_probability(_prev_state, _next_state) * \
                              previous_belief_vector[''.join(str(k) for k in _prev_state)]
            normalization_constant += self.get_emission_probability(observations, _next_state) * multiplier
        return normalization_constant

    # Belief Update sequence
    def belief_update(self, observations, previous_belief_vector, new_state, normalization_constant):
        multiplier = 0
        allowed_previous_states = self.get_allowed_state_transitions(new_state)
        # Calculate the numerator in the belief update formula
        # \vec{x} \in \mathcal{X} in your belief update rule
        for prev_state in allowed_previous_states:
            multiplier += self.get_transition_probability(prev_state, new_state) * previous_belief_vector[
                              ''.join(str(k) for k in prev_state)]
        numerator = self.get_emission_probability(observations, new_state) * multiplier
        return numerator / normalization_constant

    # Randomly explore the environment and collect a set of beliefs 'B' of reachable belief points
    def random_exploration(self):
        reachable_beliefs = dict()
        state_space_size = 2 ** self.number_of_channels
        # Uniform belief assignment to all states in the state space
        initial_belief_vector = dict()
        for state in self.all_possible_states:
            state_key = ''.join(str(k) for k in state)
            initial_belief_vector[state_key] = 1 / state_space_size
        previous_belief_vector = initial_belief_vector
        reachable_beliefs['0'] = initial_belief_vector
        # Start exploring
        for episode in range(1, self.exploration_period):
            # Making observations by choosing a random sensing action
            observations = self.secondary_user.make_observations(episode, random.choice(self.all_possible_actions))
            # Perform the Belief Update
            updated_belief_vector = dict()
            # Belief sum for this updated belief vector
            belief_sum = 0
            # Calculate the denominator which is nothing but the normalization constant
            denominator = self.get_normalization_constant(previous_belief_vector,
                                                          observations)
            # Possible next states to update the belief, i.e. b(\vec{x}')
            for state in self.all_possible_states:
                state_key = ''.join(str(k) for k in state)
                belief_val = self.belief_update(observations, previous_belief_vector, state, denominator)
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
            print('[INFO] PERSEUS random_exploration: {}% Exploration completed'.format(
                int(((episode + 1) / self.exploration_period) * 100)
            ))
        return reachable_beliefs

    # The value function initialization
    def initialize(self, reachable_beliefs):
        # V_0 for the reachable beliefs
        value_function_collection = dict()
        # Default action is not sensing anything - a blind guess from just the noise
        default_action = [k - k for k in range(0, self.number_of_channels)]
        for belief_key in reachable_beliefs.keys():
            # FIXME: Is -10 the right initial value for the beliefs in the reachable beliefs set
            value_function_collection[belief_key] = (-10, default_action)
        return value_function_collection

    # Calculate Utility
    def calculate_utility(self, policy_collection):
        utility = 0
        throughput = 0
        interference = 0
        for key, value in policy_collection.items():
            system_state = []
            for channel in range(0, self.number_of_channels):
                # int(key) refers to the episode number
                system_state.append(self.true_occupancy_states[channel][int(key)])
            observation_samples = self.secondary_user.make_observations(int(key), value[1])
            self.state_estimator.observation_samples = observation_samples
            estimated_state = self.state_estimator.estimate_pu_occupancy_states()
            print('[DEBUG] PERSEUS calculate_utility: Estimated PU Occupancy states - {}'.format(str(estimated_state)))
            sweepstakes = self.sweepstakes.roll(estimated_state, system_state)
            utility += sweepstakes[0]
            throughput += sweepstakes[1]
            interference += sweepstakes[2]
        throughput /= len(policy_collection.keys())
        interference /= len(policy_collection.keys())
        throughput *= self.utility_multiplication_factor
        interference *= self.utility_multiplication_factor
        print('[INFO] PERSEUS calculate_utility: SU Network Throughput = {} | PU Interference = {}'.format(throughput,
                                                                                                           interference
                                                                                                           ))
        return utility

    # The Backup stage
    def backup(self, stage_number, reachable_beliefs, previous_stage_value_function_collection):
        # Just a re-assignment because the reachable_beliefs turns out to be a mutable collection
        unimproved_belief_points = {}
        for key, value in reachable_beliefs.items():
            unimproved_belief_points[key] = value
        # This assignment will help me in the utility calculation within each backup stage
        next_stage_value_function_collection = dict()
        # A simple dict copy - just to be safe from mutations
        for k, v in previous_stage_value_function_collection.items():
            next_stage_value_function_collection[k] = v
        number_of_belief_changes = 0
        # While there are still some un-improved belief points...
        while len(unimproved_belief_points) is not 0:
            print('[INFO] PERSEUS backup: Size of unimproved belief set = {}'.format(len(unimproved_belief_points)))
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
                reward_sum = 0
                normalization_constant = 0
                for state in self.all_possible_states:
                    emission_probability = self.get_emission_probability(observation_samples, state)
                    multiplier = 0
                    allowed_previous_states = self.get_allowed_state_transitions(state)
                    for prev_state in allowed_previous_states:
                        multiplier += self.get_transition_probability(prev_state, state) * \
                                      belief_sample[''.join(str(k) for k in prev_state)]
                    normalization_constant += emission_probability * multiplier
                    reward_sum += self.sweepstakes.roll(estimated_system_state, state)[0] * belief_sample[
                        ''.join(str(k) for k in state)]
                # Belief Update
                for state in self.all_possible_states:
                    state_key = ''.join(str(k) for k in state)
                    value_of_belief = self.belief_update(observation_samples, belief_sample, state,
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
                # FIXME: You could've used an OrderedDict here to simplify operations
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
                print('[DEBUG] PERSEUS backup: '
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
                print('[DEBUG] PERSEUS backup: Improving the other belief points...')
                normalization_constant = 0
                reward_sum = 0
                observation_samples = self.secondary_user.make_observations(stage_number, max_action)
                # Estimate the System State
                self.state_estimator.observation_samples = observation_samples
                estimated_system_state = self.state_estimator.estimate_pu_occupancy_states()
                for state in self.all_possible_states:
                    emission_probability = self.get_emission_probability(observation_samples, state)
                    multiplier = 0
                    allowed_previous_states = self.get_allowed_state_transitions(state)
                    for prev_state in allowed_previous_states:
                        multiplier += self.get_transition_probability(prev_state, state) * \
                                      unimproved_belief_points[belief_point_key][''.join(str(k) for k in prev_state)]
                    normalization_constant += emission_probability * multiplier
                    reward_sum += self.sweepstakes.roll(estimated_system_state, state)[0] * unimproved_belief_points[
                        belief_point_key][''.join(str(k) for k in state)]
                new_aux_belief_vector = {}
                new_aux_belief_sum = 0
                aux_belief_sample = unimproved_belief_points[belief_point_key]
                # Belief Update
                for state in self.all_possible_states:
                    state_key = ''.join(str(k) for k in state)
                    new_aux_belief_val = self.belief_update(observation_samples, aux_belief_sample, state,
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
                # FIXME: You could've used an OrderedDict here to simplify operations
                # Find the closest pilot belief and its associated value function
                relevant_data = {episode_key: belief[highest_belief_key] for episode_key, belief in
                                 reachable_beliefs.items()}
                aux_pilot_belief_key = max(relevant_data, key=relevant_data.get)
                internal_term = reward_sum + (self.gamma * normalization_constant *
                                              previous_stage_value_function_collection[aux_pilot_belief_key][0])
                if round(internal_term, 3) > round(previous_stage_value_function_collection[belief_point_key][0], 3):
                    print('[DEBUG] PERSEUS backup: Auxiliary points improved by action {} with {}'.format(
                        str(max_action),
                        str(internal_term) + ' > ' + str(previous_stage_value_function_collection[belief_point_key][0]))
                    )
                    # Note here that del mutates the contents of the dict for everyone who has a reference to it
                    del unimproved_belief_points[belief_point_key]
                    next_stage_value_function_collection[belief_point_key] = (internal_term, max_action)
                    number_of_belief_changes += 1
        return [next_stage_value_function_collection, number_of_belief_changes]

    # The PERSEUS algorithm
    # Calls to Random Exploration, Initialization, and Backup stages
    def run_perseus(self):
        # Random Exploration - Get the set of reachable beliefs by randomly interacting with the radio environment
        reachable_beliefs = self.random_exploration()
        # Initialization - Initializing to -10 for all beliefs in the reachable beliefs set
        # FIXME: Is -10 the right initial value for the beliefs in the reachable beliefs set
        initial_value_function_collection = self.initialize(reachable_beliefs)
        # Relevant collections
        previous_value_function_collection = initial_value_function_collection
        stage_number = self.exploration_period - 1
        # Local confidence check for modeling policy convergence
        confidence = 0
        # Check for termination condition here...
        while confidence < self.confidence_bound:
            stage_number += 1
            # We've reached the end of our allowed interaction time with the radio environment
            if stage_number == self.number_of_episodes:
                print('[WARN] PERSEUS run_perseus: '
                      'We have reached the end of our allowed interaction time with the radio environment!')
                return
            # Backup to find \alpha -> Get V_{n+1} and #BeliefChanges
            backup_results = self.backup(stage_number, reachable_beliefs, previous_value_function_collection)
            print(
                '[DEBUG] PERSEUS run_perseus: '
                'Backup for stage {} completed...'.format(stage_number - self.exploration_period + 1))
            next_value_function_collection = backup_results[0]
            belief_changes = backup_results[1]
            if len(next_value_function_collection) is not 0:
                previous_value_function_collection = next_value_function_collection
            if belief_changes is 0:
                print('[DEBUG] PERSEUS run_perseus: Confidence Update - {}'.format(confidence))
                confidence += 1
            else:
                confidence = 0
                print('[DEBUG] PERSEUS run_perseus: Confidence Stagnation/Fallback - {}'.format(confidence))
        su_throughputs = []
        pu_interferences = []
        for episode_number, results_tuple in previous_value_function_collection.items():
            system_state = []
            for channel in range(0, self.number_of_channels):
                system_state.append(self.true_occupancy_states[channel][int(episode_number)])
            optimal_action = results_tuple[1]
            observation_samples = self.secondary_user.make_observations(int(episode_number), optimal_action)
            self.state_estimator.observation_samples = observation_samples
            estimated_states = self.state_estimator.estimate_pu_occupancy_states()
            analytics = self.sweepstakes.get_analytics(estimated_states, system_state)
            su_throughputs.append(analytics[0])
            pu_interferences.append(analytics[1])
        return self.analytics(su_throughput=sum(su_throughputs) / self.number_of_episodes,
                              pu_interference=sum(pu_interferences) / self.number_of_episodes
                              )

    # The termination sequence
    def __exit__(self, exc_type, exc_val, exc_tb):
        print('[INFO] PERSEUS Termination: Tearing things down...')
        # Nothing to do...


# The evaluation engine
# This is the improved version modified to handle changes to the transition model underlying the occupancy behavior of
#   incumbents in the network along with some changes to the analytics engine.
# This class encapsulates the evaluation framework for the PERSEUS-III agent detailed in the rest of this script.
class FragmentedSimplifiedPERSEUSModeling(object):
    # The number of channels in the discretized spectrum of interest
    NUMBER_OF_CHANNELS = 20

    # The number of sampling rounds per episode
    NUMBER_OF_SAMPLING_ROUNDS = 360

    # The number of periods of interaction of the agent with the radio environment
    NUMBER_OF_EPISODES = 3588

    # The mean of the AWGN samples
    NOISE_MEAN = 0

    # The mean of the frequency domain channel
    IMPULSE_RESPONSE_MEAN = 0

    # The variance of the AWGN samples
    NOISE_VARIANCE = 1

    # The variance of the frequency domain channel
    IMPULSE_RESPONSE_VARIANCE = 80

    # The Secondary User's sensing limitation w.r.t the number of channels it can sense simultaneously in an episode
    SPATIAL_SENSING_LIMITATION = 10

    # Limitation per fragment
    # DARPA SC2 Active Incumbent: I have 10 nodes in the network--I can sense 1 channel per node in a given episode
    FRAGMENTED_SPATIAL_SENSING_LIMITATION = 2

    # The exploration period of the PERSEUS algorithm
    EXPLORATION_PERIOD = 100

    # The discount factor employed in the Bellman update
    DISCOUNT_FACTOR = 0.9

    # The confidence bound for convergence analysis
    CONFIDENCE_BOUND = 10

    # The size of each agent-assigned individual fragment of the spectrum which is independent from the other fragments
    # I assume the same Markovian correlation within each fragment
    # There are 5 PUs (4 competitors + 1 active incumbent)--fragmented into 5 fragments (Markovian within each fragment,
    #   but independent across fragments)--4 channels in each fragment, for a total of 20 channels in the spectrum
    FRAGMENT_SIZE = 4

    # The penalty for missed detections
    PENALTY = -1

    # The transition threshold for the simplified belief update procedure
    # Within each 4-channel fragment, I allow candidate states with a Hamming distance of (0.5 * 4 = 2) to be
    #   considered to be viable next states
    TRANSITION_THRESHOLD = 0.5

    # The divisor for the exploration index determination
    EXPLORATION_INDEX_DIVISOR = 33

    # The initialization sequence
    def __init__(self, db):
        print('[INFO] FragmentedSimplifiedPERSEUSModeling Initialization: Bringing things up...')
        self.perseus_with_model_foresight_and_simplified_belief_update = \
            PERSEUS(self.FRAGMENT_SIZE, self.NUMBER_OF_SAMPLING_ROUNDS,
                    self.NUMBER_OF_EPISODES, self.EXPLORATION_PERIOD,
                    self.NOISE_MEAN, self.NOISE_VARIANCE,
                    self.IMPULSE_RESPONSE_MEAN,
                    self.IMPULSE_RESPONSE_VARIANCE, self.PENALTY,
                    self.FRAGMENTED_SPATIAL_SENSING_LIMITATION,
                    self.CONFIDENCE_BOUND, self.DISCOUNT_FACTOR,
                    math.ceil(self.NUMBER_OF_CHANNELS / self.FRAGMENT_SIZE),
                    self.TRANSITION_THRESHOLD, self.EXPLORATION_INDEX_DIVISOR, db)

    # The evaluation routine
    def evaluate(self):
        agent_analytics = self.perseus_with_model_foresight_and_simplified_belief_update.run_perseus()
        print('[INFO] FragmentedSimplifiedPERSEUSModeling evaluate: Fragmented PERSEUS with belief simplification - '
              'Average Episodic SU Throughput = {} | '
              'Average Episodic PU Interference = {}\n'.format(agent_analytics.su_throughput,
                                                               agent_analytics.pu_interference))

    # The termination sequence
    def __exit__(self, exc_type, exc_val, exc_tb):
        print('[INFO] FragmentedSimplifiedPERSEUSModeling Termination: Tearing things down...')
        # Nothing to do...


# Run Trigger
if __name__ == '__main__':
    # The default DB file (Active Incumbent Scenario-8342)
    _db = 'data/active_incumbent_scenario8342.db'
    print('[INFO] FragmentedSimplifiedPERSEUSModeling main: Triggering the evaluation of the PERSEUS-III agent, '
          'i.e., the PERSEUS Algorithm with Model Foresight and with a Simplified Belief Update procedure...')
    _agent = FragmentedSimplifiedPERSEUSModeling(_db)
    _agent.evaluate()
