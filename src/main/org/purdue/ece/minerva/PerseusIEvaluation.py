# This entity describes the POMDP Approximate Point-Based Value Iteration Algorithm, i.e. the PERSEUS algorithm
#   applied to PU Occupancy Behavior Estimation in Cognitive Radio Networks.
# Author: Bharath Keshavamurthy
# Organization: School of Electrical and Computer Engineering, Purdue University, West Lafayette, IN.
# Copyright (c) 2019. All Rights Reserved.

# WIKI
#   This entity describes the PERSEUS Algorithm for solving the Optimal Spectrum Sensing and Access Problem in Cognitive
#   Radio Networks wherein a SecondaryUser (SU) needs to sense a subset of channels and access channels in the radio
#   environment that it deems to be idle, i.e. un-occupied by the PrimaryUsers (PUs - Incumbents).

# PREMISE OF THE FRAGMENTED PERSEUS ALGORITHM
#   The entire premise behind the fragmented PERSEUS algorithm in a radio environment where there is more than one PU
#   is that the PUs generally are restricted to certain segments of the radio spectrum by design and by bureaucracy.
#   Therefore, in order to speed up the PERSEUS algorithm over a significantly large number of channels, we set out to
#   partition the radio spectrum into fragments, each corresponding to the spectral constraints of incumbents, as set
#   by governments or regulatory agencies. So, we run instances of the PERSEUS agent (SU with PERSEUS as its
#   Channel Sensing and Access strategy/heuristic) over these fragments and then, aggregate the utilities obtained
#   by these individual PERSEUS instances in order to get the overall utility obtained by the SU over numerous
#   episodes of interaction with the radio environment.

# PERSEUS ALGORITHM WITHOUT MODEL FORESIGHT AND STANDARD BELIEF UPDATE PRACTICES

# Visualization: Utility v Episodes | Regret v Iterations | #policy_changes v Iterations

# The imports
import math
import numpy
import plotly
import random
import itertools
import scipy.stats
from enum import Enum
import plotly.graph_objs as go
from collections import namedtuple

# Plotly user account credentials for visualization
plotly.tools.set_credentials_file(username='bkeshava',
                                  api_key='W2WL5OOxLcgCzf8NNlgl')


# Markovian Correlation Class Enumeration
class MarkovianCorrelationClass(Enum):
    # Markovian Correlation across channel indices
    SPATIAL = 0
    # Markovian Correlation across time indices
    TEMPORAL = 1
    # Invalid
    INVALID = 2


# Occupancy State Enumeration
# Based on Energy Detection, \mathbb{E}[|X_k(i)|^2] = 1, if Occupied; else, \mathbb{E}[|X_k(i)|^2] = 0
class OccupancyState(Enum):
    # Occupancy state IDLE
    IDLE = 0
    # Occupancy state OCCUPIED
    OCCUPIED = 1


# The Markov Chain object that can be used via extension or replication in order to imply Markovian correlation
#   across either the channel indices or the time indices
class MarkovChain(object):

    # The initialization sequence
    def __init__(self):
        print('[INFO] MarkovChain Initialization: Initializing the Markov Chain...')
        # The steady-state probabilities (a.k.a start probabilities for each channel / each episode
        #   independently in the Markov Chain)
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
            self.markovian_correlation_class = MarkovianCorrelationClass.INVALID
        print('[INFO] MarkovChain set_markovian_correlation_class: Markovian Correlation Class - ',
              self.markovian_correlation_class.name)

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
            # \mathbb{P}(Occupied|Idle) = p
            self.transition_probabilities[0][1] = p
            # \mathbb{P}(Idle|Idle) = 1 - p
            self.transition_probabilities[0][0] = 1 - p
            # \mathbb{P}(Idle|Occupied) = q = p(1 - pi) / pi
            self.transition_probabilities[1][0] = (p * self.start_probabilities[0]) / self.start_probabilities[1]
            # \mathbb{P}(Occupied|Occupied) = 1 - q
            self.transition_probabilities[1][1] = 1 - self.transition_probabilities[1][0]
        else:
            print(
                '[ERROR] MarkovChain set_transition_probability_parameter: Error while populating the state transition '
                'probabilities matrix! Proceeding with default values...')
            # Default Values...
            self.transition_probabilities = {0: {0: 0.7, 1: 0.3}, 1: {0: 0.2, 1: 0.8}}
        print('[INFO] MarkovChain set_transition_probability_parameter: State Transition Probabilities Matrix - ',
              str(self.transition_probabilities))

    # The termination sequence
    def __exit__(self, exc_type, exc_val, exc_tb):
        print('[INFO] MarkovChain Termination: Tearing things down...')
        # Nothing to do...


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

    # Construct the complete transition probability matrix from \mathbb{P}(Occupied|Idle), i.e. 'p' and
    #   \mathbb{P}(Occupied), i.e. 'pi'
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


# This entity encapsulates the Channel object - simulates the Channel
class Channel(object):

    # The initialization sequence
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
        # Number of episodes in which the PERSEUS-I agent interacts with the radio environment
        self.number_of_episodes = _number_of_episodes
        # The Channel Impulse Response used in the Observation Model
        self.impulse_response = self.get_impulse_response()
        # The AWGN used in the Observation Model
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


# This class encapsulates the Licensed User dynamically occupying the discretized spectrum under analysis
class PrimaryUser(object):

    # The initialization sequence
    def __init__(self, _number_of_channels, _number_of_episodes, _spatial_markov_chain, _temporal_markov_chain):
        print('[INFO] PrimaryUser Initialization: Bringing things up...')
        # The number of channels in the discretized spectrum of interest
        self.number_of_channels = _number_of_channels
        # The number of episodes of PU Occupancy Behavior that's under analysis
        self.number_of_episodes = _number_of_episodes
        # The Markov Chain across channels
        self.spatial_markov_chain = _spatial_markov_chain
        # The Markov Chain across episodes
        self.temporal_markov_chain = _temporal_markov_chain
        # Occupancy Behavior Collection
        self.occupancy_behavior_collection = []
        # The Utility class
        self.util = Util()

    # Generate the initial states for k = 0 (band-0) across time
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
            sample_value = numpy.random.random_sample()
            if previous == 1 and sample_value < q_val:
                previous = 0
            elif previous == 1 and sample_value > q_val:
                previous = 1
            elif previous == 0 and sample_value < p_val:
                previous = 1
            else:
                previous = 0
            initial_state_vector.append(previous)
        return initial_state_vector

    # Get the spatial and temporal occupancy behavior of the Primary User based on the statistics shared during the...
    # ...creation of the Spatial Markov Chain and the Temporal Markov Chain
    def simulate_occupancy_behavior(self):
        # Extracting the statistics from the objects for easy use in this method
        spatial_transition_probabilities_matrix = self.spatial_markov_chain.transition_probabilities
        spatial_start_probabilities = self.spatial_markov_chain.start_probabilities
        temporal_transition_probabilities_matrix = self.temporal_markov_chain.transition_probabilities
        temporal_start_probabilities = self.temporal_markov_chain.start_probabilities
        # Global System Steady-State Analysis - What if it's wrong?
        # Note that both chains are essentially dealing with the same cell (channel and time) and hence, the steady
        #   state probabilities of the cells (over space and time) need to be the same.
        if spatial_start_probabilities != temporal_start_probabilities:
            print(
                '[ERROR] PrimaryUser simulate_occupancy_behavior: Looks like the start probabilities are different '
                'across the Spatial and the Temporal Markov Chains. This is inaccurate! Proceeding with defaults...')
            # Default Values
            spatial_start_probabilities = {0: 0.4, 1: 0.6}
            temporal_start_probabilities = {0: 0.4, 1: 0.6}
            print('[WARN] PrimaryUser simulate_occupancy_behavior: Modified System Steady State Probabilities - ',
                  str(temporal_start_probabilities))
        # Everything's alright with the system steady-state statistics - Start simulating the PU Occupancy Behavior
        # This is global and system-specific. So, it doesn't matter which chain's steady-state probabilities are used...
        pi_val = spatial_start_probabilities[1]
        # SINGLE CHAIN INFLUENCE along all the columns of the first row
        # Get the initial state vector to get things going - row 0
        self.occupancy_behavior_collection.append(
            self.get_initial_states_temporal_variation(temporal_transition_probabilities_matrix[0][1],
                                                       temporal_transition_probabilities_matrix[1][0],
                                                       pi_val
                                                       )
        )
        previous_state = self.occupancy_behavior_collection[0][0]
        # SINGLE CHAIN INFLUENCE along the first column of all the rows
        # Start filling things based on spatial correlation (i.e. across rows for column-0)
        for channel_index in range(1, self.number_of_channels):
            random_sample = numpy.random.random_sample()
            if previous_state == 1 and random_sample < spatial_transition_probabilities_matrix[1][0]:
                previous_state = 0
            elif previous_state == 1 and random_sample > spatial_transition_probabilities_matrix[1][0]:
                previous_state = 1
            elif previous_state == 0 and random_sample < spatial_transition_probabilities_matrix[0][1]:
                previous_state = 1
            else:
                previous_state = 0
            self.occupancy_behavior_collection.append([previous_state])

        # Interpretation #1
        # DOUBLE CHAIN INFLUENCE along all the remaining cells
        # Go on and fill in the remaining cells in the Occupancy Behavior Matrix
        # Use the definitions of Conditional Probabilities to realize the math - \mathbb{P}(A|B,C)
        # \mathbb{P}(A=a|B=b) = \sum_{c\in\{0,1\}}\ \mathbb{P}(A=a|B=b,C=c)P(C=c)
        # Using the definition of Marginal Probability in discrete distributions
        # for channel_index in range(1, self.number_of_channels):
        #     for episode_index in range(1, self.number_of_episodes):
        #         occupied_spatial_transition = spatial_transition_probabilities_matrix[
        #             self.occupancy_behavior_collection[channel_index - 1][episode_index]][1]
        #         random_sample = numpy.random.random_sample()
        #         if random_sample < occupied_spatial_transition:
        #             self.occupancy_behavior_collection[channel_index].append(1)
        #         else:
        #             self.occupancy_behavior_collection[channel_index].append(0)

        # Interpretation #2
        # DOUBLE CHAIN INFLUENCE along all the remaining cells
        # The two Markov Chains are independent and hence, we can treat the transitions independently.
        # \mathbb{P}(X_{k,t} = a | X_{k-1,t} = b, X_{k,t-1} = c) = \mathbb{P}(X_{k,t} = a | X_{k-1,t} = b) * \\
        #                                                          \mathbb{P}(X_{k,t} = a | X_{k,t-1} = c)
        for channel_idx in range(1, self.number_of_channels):
            for episode_idx in range(1, self.number_of_episodes):
                previous_spatial_state = self.occupancy_behavior_collection[channel_idx - 1][episode_idx]
                previous_temporal_state = self.occupancy_behavior_collection[channel_idx][episode_idx - 1]
                occupied_probability = spatial_transition_probabilities_matrix[previous_spatial_state][1] * \
                    temporal_transition_probabilities_matrix[previous_temporal_state][1]
                random_sample = numpy.random.random_sample()
                if random_sample < occupied_probability:
                    self.occupancy_behavior_collection[channel_idx].append(1)
                else:
                    self.occupancy_behavior_collection[channel_idx].append(0)

    # The termination sequence
    def __exit__(self, exc_type, exc_val, exc_tb):
        print('[INFO] PrimaryUser Termination: Tearing things down...')
        # Nothing to do...


# This entity emulates a Secondary User (SU) making observations of all the channels in the discretized spectrum
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
        # Occupancy Status of the cells based on simulated PU behavior - needed to simulate SU observations
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
    # THe "state" member is an enum instance of OccupancyState.
    def get_emission_probabilities(self, state, observation_sample):
        # If the channel is not observed, i.e. if the observation is [phi] or [0], report m_r(y_i) as 1.
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


# This entity encapsulates the routine which provides rewards based on the state of the system and the action taken by
#   the PERSEUS-I agent
class Sweepstakes(object):

    # The initialization sequence
    def __init__(self, _primary_user, _mu):
        print('[INFO] Sweepstakes Initialization: Bringing things up...')
        # The Primary User
        self.primary_user = _primary_user
        # The penalty for missed detections
        self.mu = _mu

    # Get Reward based on the system state and the action taken by the PERSEUS-I agent
    def roll(self, estimated_state, system_state):
        if len(estimated_state) != len(system_state):
            print('[ERROR] Sweepstakes roll: The estimated_state arg and the system_state arg need to be of the same'
                  'dimension!')
            return 0
        # Let B_k(i) denote the actual true occupancy status of the channel in this 'episode'.
        # Let \hat{B}_k(i) denote the estimated occupancy status of the channel in this 'episode'.
        # Utility = R = \sum_{k=1}^{K}\ (1 - B_k(i)) (1 - \hat{B}_k(i)) + \mu B_k(i) (1 - \hat{B}_k(i))
        reward = 0
        for k in range(0, len(system_state)):
            reward += ((1 - estimated_state[k]) * (1 - system_state[k])) + \
                      (self.mu * (system_state[k] * (1 - estimated_state[k])))
        return reward

    # The termination sequence
    def __exit__(self, exc_type, exc_val, exc_tb):
        print('[INFO] Sweepstakes Termination: Tearing things down...')
        # Nothing to do...


# The Markov Chain Parameter Estimator Algorithm (modified EM - Baum-Welch)
# This entity is employed by the PERSEUS-I agent in order to learn the model concurrently
class ParameterEstimator(object):

    # The initialization sequence
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

    # Get the Emission Probabilities -> \mathbb{P}(y|x)
    # The "state" arg is an instance of the OccupancyState enumeration.
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
            OccupancyState.OCCUPIED,
            observation_vector[0]) * self.start_probabilities[1]
        # Idle
        _forward_boundary_condition_idle = self.get_emission_probabilities(
            OccupancyState.IDLE,
            observation_vector[0]) * self.start_probabilities[0]
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
                        current_occupancy_state, observation_vector[channel_index])
                self.forward_probabilities[channel_index][current_occupancy_state.value] = occupancy_state_sum

    # In-situ update of Backward Probabilities
    def fill_up_backward_probabilities(self, iteration, observation_vector):
        # Setting the Boundary conditions for Backward Probabilities - again for coherence
        # Occupied
        _state_sum = 0
        # Outside summation - refer to the definition of backward probability
        for _state in OccupancyState:
            _state_sum += self.transition_probabilities[1][_state.value][iteration - 1] * \
                          self.get_emission_probabilities(_state,
                                                          observation_vector[self.number_of_chain_links - 1])
        _backward_boundary_condition_occupied = _state_sum
        # Idle
        _state_sum = 0
        # outside summation - refer to the definition of backward probability
        for _state in OccupancyState:
            _state_sum += self.transition_probabilities[0][_state.value][iteration - 1] * \
                          self.get_emission_probabilities(_state,
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
                                           self.get_emission_probabilities(next_occupancy_state,
                                                                           observation_vector[channel_index])
                self.backward_probabilities[channel_index][previous_occupancy_state.value] = occupancy_state_sum

    # Get the numerator of the fraction in the Algorithm (refer to the document for more information)
    # The "previous_state" and "next_state" args are instances of the OccupancyState enumeration.
    def get_numerator(self, previous_state, next_state, iteration, observation_vector):
        numerator_sum = self.get_emission_probabilities(next_state, observation_vector[0]) * \
                        self.transition_probabilities[previous_state.value][next_state.value][iteration - 1] * \
                        self.backward_probabilities[1][next_state.value]
        for spatial_index in range(1, self.number_of_chain_links - 1):
            numerator_sum += self.forward_probabilities[spatial_index - 1][previous_state.value] * \
                             self.get_emission_probabilities(next_state, observation_vector[spatial_index]) * \
                             self.transition_probabilities[previous_state.value][next_state.value][iteration - 1] * \
                             self.backward_probabilities[spatial_index + 1][next_state.value]
        numerator_sum += self.forward_probabilities[self.number_of_chain_links - 2][previous_state.value] * \
            self.get_emission_probabilities(next_state, observation_vector[self.number_of_chain_links - 1]) * \
            self.transition_probabilities[previous_state.value][next_state.value][iteration - 1]
        return numerator_sum

    # Get the denominator of the fraction in the Algorithm (refer to the document for more information)
    # The "previous_state" arg is an instance of the OccupancyState enumeration.
    def get_denominator(self, previous_state, iteration, observation_vector):
        denominator_sum = 0
        for nxt_state in OccupancyState:
            denominator_sum_internal = self.get_emission_probabilities(nxt_state, observation_vector[0]) * \
                                       self.transition_probabilities[previous_state.value][nxt_state.value][
                                           iteration - 1] * \
                                       self.backward_probabilities[1][nxt_state.value]
            for _spatial_index in range(1, self.number_of_chain_links - 1):
                denominator_sum_internal += self.forward_probabilities[_spatial_index - 1][previous_state.value] * \
                                            self.get_emission_probabilities(nxt_state,
                                                                            observation_vector[_spatial_index]) * \
                                            self.transition_probabilities[previous_state.value][nxt_state.value][
                                                iteration - 1] * \
                                            self.backward_probabilities[_spatial_index + 1][nxt_state.value]
            denominator_sum_internal += \
                self.forward_probabilities[self.number_of_chain_links - 2][previous_state.value] \
                * self.get_emission_probabilities(nxt_state, observation_vector[self.number_of_chain_links - 1]) \
                * self.transition_probabilities[previous_state.value][nxt_state.value][iteration - 1]
            denominator_sum += denominator_sum_internal
        return denominator_sum

    # Core method
    # Estimate the Markov Chain State Transition Probabilities Matrix
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
                            previous_state, next_state, iteration, observation_vector))
                        denominators_collection[previous_state.value][next_state.value].append(
                            self.get_denominator(previous_state, iteration, observation_vector))
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

    # The termination sequence
    def __exit__(self, exc_type, exc_val, exc_tb):
        print('[INFO] ParameterEstimator Termination: Tearing things down...')
        # Nothing to do...


# The Markov Chain State Estimator algorithm - Viterbi Algorithm
# This entity is employed during the operation of the PERSEUS-I agent
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
        if collection is not None and len(collection) is not 0 and collection[index] is not None:
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
                    # Again finishing off the [0] index first
                    max_pointer = self.get_transition_probabilities(OccupancyState.IDLE, state) * \
                                  value_function_collection[observation_index - 1][
                                      OccupancyState.IDLE.name].current_value
                    confirmed_previous_state = OccupancyState.IDLE.name
                    for candidate_previous_state in OccupancyState:
                        if candidate_previous_state == OccupancyState.IDLE:
                            # Already done
                            continue
                        else:
                            pointer = self.get_transition_probabilities(candidate_previous_state,
                                                                        state) * \
                                      value_function_collection[observation_index - 1][
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
                    value_function_collection[i + 1][previous_state].previous_state))
                previous_state = value_function_collection[i + 1][previous_state].previous_state
            estimated_states_array.append(estimated_states)
        # FIXME: Come up with some averaging/thresholding logic here instead of picking the final estimated vector...
        return estimated_states_array[self.number_of_sampling_rounds - 1]

    # The termination sequence
    def __exit__(self, exc_type, exc_val, exc_tb):
        print('[INFO] StateEstimator Termination: Tearing things down...')
        # Nothing to do...


# This entity encapsulates an Oracle which knows the best possible channels to use in each episode.
# Hence, the policy followed by this Oracle is the most optimal policy.
# The action policy achieved by the PERSEUS-I agent will be evaluated/benchmarked against the Oracle's policy thereby
#   giving us a regret metric.
class Oracle(object):

    # The initialization sequence
    def __init__(self, _number_of_channels, _true_pu_occupancy_states, _mu):
        print('[INFO] Oracle Initialization: Bringing things up...')
        self.number_of_channels = _number_of_channels
        self.true_pu_occupancy_states = _true_pu_occupancy_states
        self.mu = _mu

    def get_return(self, episode):
        estimated_state_vector = [self.true_pu_occupancy_states[k][episode] for k in range(0, self.number_of_channels)]
        utility = 0
        # Let B_k(i) denote the actual true occupancy status of the channel in this 'episode'.
        # Let \hat{B}_k(i) denote the estimated occupancy status of the channel in this 'episode'.
        # Utility = R = \sum_{k=1}^{K}\ (1 - B_k(i)) (1 - \hat{B}_k(i)) + \mu B_k(i) (1 - \hat{B}_k(i))
        for channel in range(0, self.number_of_channels):
            utility += ((1 - self.true_pu_occupancy_states[channel][episode]) * (1 - estimated_state_vector[channel])) \
                       + (self.mu *
                          (1 - estimated_state_vector[channel]) * self.true_pu_occupancy_states[channel][episode])
        return utility

    # The termination sequence
    def __exit__(self, exc_type, exc_val, exc_tb):
        print('[INFO] Oracle Termination: Tearing things down...')
        # Nothing to do...


# This entity encapsulates a POMDP Approximate Point-based Value Iteration Algorithm known as the PERSEUS Algorithm.
# Training to fine-tune the belief space -> Perform backup until convergence -> Re-sample using the most recent policy
# References to the Parameter Estimation Algorithm and the State Estimation Algorithm in the belief analysis phase
# Concurrent Model Learning with no simplification whatsoever!
class PERSEUS(object):

    # Setup the Markov Chain
    @staticmethod
    def setup_markov_chain(_pi, _p, _correlation_class):
        print('[INFO] PERSEUS setup_markov_chain: Setting up the Markov Chain...')
        transient_markov_chain_object = MarkovChain()
        transient_markov_chain_object.set_markovian_correlation_class(_correlation_class)
        transient_markov_chain_object.set_start_probability_parameter(_pi)
        transient_markov_chain_object.set_transition_probability_parameter(_p)
        return transient_markov_chain_object

    # The initialization sequence
    def __init__(self, _number_of_channels, _number_of_sampling_rounds, _number_of_episodes, _exploration_period,
                 _noise_mean, _noise_variance, _impulse_response_mean, _impulse_response_variance, _penalty,
                 _limitation, _confidence_bound, _gamma, _epsilon, _utility_multiplication_factor):
        print('[INFO] PERSEUS Initialization: Bringing things up...')
        # The Utility object
        self.util = Util()
        # The number of channels in the discretized spectrum of interest (fragment size)
        self.number_of_channels = _number_of_channels
        # The number of sampling rounds in each episode
        self.number_of_sampling_rounds = _number_of_sampling_rounds
        # The number of time slots of interaction of the PERSEUS-I agent with the radio environment
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
        # The penalty for missed detections, i.e. PU interference
        self.penalty = _penalty
        # The SU sensing limitation
        self.limitation = _limitation
        # The confidence bound for convergence analysis
        self.confidence_bound = _confidence_bound
        # The discount factor employed in the Bellman equation
        self.gamma = _gamma
        # \mathbb{P}(Occupied) - The steady state probability of being occupied
        self.pi = 0.6
        # \mathbb{P}(Occupied|Idle) - The transition probability parameter
        self.p = 0.3
        # The initial estimate of \mathbb{P}(Occupied|Idle) a.k.a 'p'
        self.initial_p = 0.00001
        # The initial transition probabilities matrix
        self.initial_transition_probabilities_matrix = self.util.construct_transition_probability_matrix(self.initial_p,
                                                                                                         self.pi)
        # The convergence threshold for the parameter estimator
        self.epsilon = _epsilon
        # The start probabilities dict
        self.start_probabilities_dict = self.util.construct_start_probabilities_dict(self.pi)
        # Setup the Spatial Markov Chain
        self.spatial_markov_chain = self.setup_markov_chain(self.pi, self.p, MarkovianCorrelationClass.SPATIAL)
        # Setup the Temporal Markov Chain
        self.temporal_markov_chain = self.setup_markov_chain(self.pi, self.p, MarkovianCorrelationClass.TEMPORAL)
        # The Channel
        self.channel = Channel(self.number_of_channels, self.number_of_sampling_rounds, self.number_of_episodes,
                               self.noise_mean, self.noise_variance, self.impulse_response_mean,
                               self.impulse_response_variance)
        # The Emission Evaluator
        self.emission_evaluator = EmissionEvaluator(self.noise_variance, self.impulse_response_variance)
        # The Primary User
        self.primary_user = PrimaryUser(self.number_of_channels, self.number_of_episodes,
                                        self.spatial_markov_chain, self.temporal_markov_chain)
        self.primary_user.simulate_occupancy_behavior()
        # The Secondary User
        self.secondary_user = SecondaryUser(self.number_of_channels, self.number_of_sampling_rounds,
                                            self.number_of_episodes, self.channel,
                                            self.primary_user.occupancy_behavior_collection)
        # Sweepstakes - The Reward Analyzer
        self.sweepstakes = Sweepstakes(self.primary_user, self.penalty)
        # The Parameter Estimator - Modified Baum-Welch / Modified EM Algorithm
        self.parameter_estimator = ParameterEstimator(self.number_of_channels, self.number_of_sampling_rounds,
                                                      self.initial_transition_probabilities_matrix,
                                                      self.start_probabilities_dict, None,
                                                      self.emission_evaluator, self.epsilon,
                                                      self.confidence_bound, self.util)
        # The State Estimator - Single Markov Chain Viterbi Algorithm
        self.state_estimator = StateEstimator(self.number_of_channels, self.number_of_sampling_rounds, None,
                                              self.start_probabilities_dict,
                                              self.initial_transition_probabilities_matrix,
                                              self.emission_evaluator)
        # The Oracle
        self.oracle = Oracle(self.number_of_channels, self.primary_user.occupancy_behavior_collection, self.penalty)
        # The episodes log collection for regret analysis using an Oracle
        self.episodes_for_regret_analysis = []
        # The utilities as the algorithm progresses towards optimality
        self.utilities = []
        # The utilities obtained by the Oracle during the episodes of interaction of this PERSEUS-I agent with the
        #   radio environment
        self.perfect_utilities = []
        # The number of policy changes as the algorithm progresses towards optimality
        self.policy_changes = []
        # The belief choice for value function tracking
        self.belief_choice = None
        # The progression of the value function as the algorithm progresses towards optimality
        self.value_function_changes_array = []
        # All possible states
        self.all_possible_states = list(map(list, itertools.product([0, 1], repeat=self.number_of_channels)))
        # All possible actions based on the given SU sensing limitation
        self.all_possible_actions = []
        for state in self.all_possible_states:
            if sum(state) == self.limitation:
                self.all_possible_actions.append(state)
        # The utility multiplication factor
        self.utility_multiplication_factor = _utility_multiplication_factor

    # Get the enumeration instance based on the value passed as an argument in order to ensure compliance with the
    #   'state' communication APIs.
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
    def get_transition_probability(self, prev_state, next_state, transition_probabilities_matrix):
        # Temporal/Episodic change for the first channel
        transition_probability = transition_probabilities_matrix[prev_state[0]][next_state[0]]
        for index in range(1, self.number_of_channels):
            transition_probability *= transition_probabilities_matrix[next_state[index - 1]][next_state[index]]
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
            # Set the observations in the ParameterEstimator
            self.parameter_estimator.observation_samples = observations
            # Estimate the transition probabilities matrix
            transition_probabilities_matrix = self.util.construct_transition_probability_matrix(
                self.parameter_estimator.estimate_parameters(),
                self.pi)
            # Perform the Belief Update
            updated_belief_vector = dict()
            # Belief sum for this updated belief vector
            belief_sum = 0
            # Calculate the denominator which is nothing but the normalization constant
            denominator = self.get_normalization_constant(previous_belief_vector,
                                                          observations,
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
        for key, value in policy_collection.items():
            system_state = []
            for channel in range(0, self.number_of_channels):
                # int(key) refers to the episode number
                system_state.append(self.primary_user.occupancy_behavior_collection[channel][int(key)])
            observation_samples = self.secondary_user.make_observations(int(key), value[1])
            self.parameter_estimator.observation_samples = observation_samples
            transition_probabilities_matrix = self.util.construct_transition_probability_matrix(
                self.parameter_estimator.estimate_parameters(),
                self.pi
            )
            self.state_estimator.transition_probabilities = transition_probabilities_matrix
            self.state_estimator.observation_samples = observation_samples
            estimated_state = self.state_estimator.estimate_pu_occupancy_states()
            print('[DEBUG] PERSEUS calculate_utility: Estimated PU Occupancy states - {}'.format(str(estimated_state)))
            utility += self.sweepstakes.roll(estimated_state, system_state)
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
                self.parameter_estimator.observation_samples = observation_samples
                transition_probabilities_matrix = self.util.construct_transition_probability_matrix(
                    self.parameter_estimator.estimate_parameters(),
                    self.pi
                )
                # Estimate the System State
                self.state_estimator.transition_probabilities = transition_probabilities_matrix
                self.state_estimator.observation_samples = observation_samples
                estimated_system_state = self.state_estimator.estimate_pu_occupancy_states()
                reward_sum = 0
                normalization_constant = 0
                for state in self.all_possible_states:
                    emission_probability = self.get_emission_probability(observation_samples, state)
                    multiplier = 0
                    for prev_state in self.all_possible_states:
                        multiplier += self.get_transition_probability(prev_state,
                                                                      state,
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
                self.parameter_estimator.observation_samples = observation_samples
                transition_probabilities_matrix = self.util.construct_transition_probability_matrix(
                    self.parameter_estimator.estimate_parameters(),
                    self.pi
                )
                # Estimate the System State
                self.state_estimator.transition_probabilities = transition_probabilities_matrix
                self.state_estimator.observation_samples = observation_samples
                estimated_system_state = self.state_estimator.estimate_pu_occupancy_states()
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
            utility = self.calculate_utility(next_stage_value_function_collection)
            print('[INFO] PERSEUS backup: '
                  'Logging all the relevant metrics within this Backup stage - [Utility: {}, #policy_changes: {}, '
                  'sampled_value_function: {}]'.format(utility, number_of_belief_changes,
                                                       next_stage_value_function_collection[self.belief_choice][0]))
            self.episodes_for_regret_analysis.append(stage_number)
            self.utilities.append(utility)
        return [next_stage_value_function_collection, number_of_belief_changes]

    # The PERSEUS algorithm
    # Calls to Random Exploration, Initialization, and Backup stages
    def run_perseus(self):
        # Random Exploration - Get the set of reachable beliefs by randomly interacting with the radio environment
        reachable_beliefs = self.random_exploration()
        # Belief choice for value function tracking (visualization component)
        self.belief_choice = (lambda: self.belief_choice,
                              lambda: random.choice([
                                  k for k in reachable_beliefs.keys()]))[self.belief_choice is None]()
        # Initialization - Initializing to -10 for all beliefs in the reachable beliefs set
        # FIXME: Is -10 the right initial value for the beliefs in the reachable beliefs set
        initial_value_function_collection = self.initialize(reachable_beliefs)
        # Relevant collections
        previous_value_function_collection = initial_value_function_collection
        stage_number = self.exploration_period - 1
        # Utility addition for the initial value function
        utility = self.calculate_utility(previous_value_function_collection)
        self.episodes_for_regret_analysis.append(stage_number)
        self.utilities.append(utility)
        # Local confidence check for modeling policy convergence
        confidence = 0
        # Check for termination condition here...
        while confidence < self.confidence_bound:
            self.value_function_changes_array.append(previous_value_function_collection[self.belief_choice][0])
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
            self.policy_changes.append(belief_changes)
            if len(next_value_function_collection) is not 0:
                previous_value_function_collection = next_value_function_collection
            if belief_changes is 0:
                print('[DEBUG] PERSEUS run_perseus: Confidence Update - {}'.format(confidence))
                confidence += 1
            else:
                confidence = 0
                print('[DEBUG] PERSEUS run_perseus: Confidence Stagnation/Fallback - {}'.format(confidence))
        optimal_utilities = []
        for episode_number, results_tuple in previous_value_function_collection.items():
            system_state = []
            for channel in range(0, self.number_of_channels):
                system_state.append(self.primary_user.occupancy_behavior_collection[channel][int(episode_number)])
            optimal_action = results_tuple[1]
            observation_samples = self.secondary_user.make_observations(int(episode_number), optimal_action)
            self.parameter_estimator.observation_samples = observation_samples
            transition_probabilities_matrix = self.util.construct_transition_probability_matrix(
                self.parameter_estimator.estimate_parameters(),
                self.pi
            )
            self.state_estimator.transition_probabilities = transition_probabilities_matrix
            self.state_estimator.observation_samples = observation_samples
            estimated_states = self.state_estimator.estimate_pu_occupancy_states()
            optimal_utilities.append(self.sweepstakes.roll(estimated_states, system_state))
        return optimal_utilities

    # Visualize the progression of regret of this PERSEUS-I agent over numerous backup and wrapper stages
    def visualize_iterative_regret(self):
        self.perfect_utilities = [self.oracle.get_return(i) for i in self.episodes_for_regret_analysis]
        iterative_regret = numpy.array(self.perfect_utilities) - numpy.array(self.utilities)
        # The visualization data trace
        visualization_trace = go.Scatter(x=[k + 1 for k in range(0, len(iterative_regret))],
                                         y=iterative_regret,
                                         mode='lines+markers')
        # The visualization layout
        visualization_layout = dict(title='Iterative Regret of the Model Free PERSEUS Algorithm over numerous '
                                          'backup and wrapper stages',
                                    xaxis=dict(title='Iterations/Stages'),
                                    yaxis=dict(title='Regret of the PERSEUS-I agent'))
        # The visualization figure
        visualization_figure = dict(data=[visualization_trace],
                                    layout=visualization_layout)
        # The figure URL
        figure_url = plotly.plotly.plot(visualization_figure,
                                        filename='Iterative_Regret_of_Model_Free_PERSEUS_Standard_Belief_Update')
        # Print the URL in case you're on an environment where a GUI is not available
        print('PERSEUS visualize_iterative_regret: The visualization figure is available at - {}'.format(figure_url))

    # Visualize the progression of #policy_changes of this PERSEUS-I agent over numerous backup and wrapper stages
    def visualize_iterative_policy_changes_count(self):
        # The visualization data trace
        visualization_trace = go.Scatter(x=[k + 1 for k in range(0, len(self.policy_changes))],
                                         y=self.policy_changes,
                                         mode='lines+markers')
        # The visualization layout
        visualization_layout = dict(
            title='The number of policy changes of the Model Free PERSEUS Algorithm over numerous '
                  'backup and wrapper stages',
            xaxis=dict(title='Iterations/Stages'),
            yaxis=dict(title='#policy_changes'))
        # The visualization figure
        visualization_figure = dict(data=[visualization_trace],
                                    layout=visualization_layout)
        # The figure URL
        figure_url = \
            plotly.plotly.plot(visualization_figure,
                               filename='Iterative_Policy_Changes_Count_of_Model_Free_PERSEUS_Standard_Belief_Update')
        # Print the URL in case you're on an environment where a GUI is not available
        print(
            'PERSEUS visualize_iterative_policy_changes_count: '
            'The visualization figure is available at - {}'.format(figure_url))

    # Visualize the episodic utilities of this PERSEUS-I agent over numerous episodes of interaction with the
    #   radio environment
    def visualize_episodic_utilities(self, optimal_utilities):
        # The visualization data trace
        visualization_trace = go.Scatter(x=[k + 1 for k in range(0, len(optimal_utilities))],
                                         y=list(numpy.array(optimal_utilities) * self.utility_multiplication_factor),
                                         mode='lines+markers')
        # The visualization layout
        visualization_layout = dict(title='Episodic Utilities of the Model Free PERSEUS Algorithm',
                                    xaxis=dict(title=r'$Episodes\ n$'),
                                    yaxis=dict(title=r'$Utility\ \sum_{k=1}^{K}\ (1 - B_k(i)) (1 - \hat{B}_k(i)) - '
                                                     r'\lambda B_k(i) (1 - \hat{B}_k(i))$'))
        # The visualization figure
        visualization_figure = dict(data=[visualization_trace],
                                    layout=visualization_layout)
        # The figure URL
        figure_url = plotly.plotly.plot(visualization_figure,
                                        filename='Episodic_Utilities_of_Model_Free_PERSEUS_Standard_Belief_Update')
        # Print the URL in case you're on an environment where a GUI is not available
        print(
            'PERSEUS visualize_episodic_utilities: The visualization figure is available at - {}'.format(figure_url))

    # The termination sequence
    def __exit__(self, exc_type, exc_val, exc_tb):
        print('[INFO] PERSEUS Termination: Tearing things down...')
        # Nothing to do...


# This class encapsulates the complete evaluation framework for the PERSEUS-I agent detailed in the rest of this script.
class PerseusIEvaluation(object):
    # The number of channels in the discretized spectrum of interest
    NUMBER_OF_CHANNELS = 18

    # The number of sampling rounds per episode
    NUMBER_OF_SAMPLING_ROUNDS = 300

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

    # The size of each agent-assigned individual fragment of the spectrum which is independent from the other fragments
    # I assume the same Markovian correlation within each fragment
    FRAGMENT_SIZE = 6

    # The penalty for missed detections
    PENALTY = -1

    # The convergence threshold for the parameter estimation algorithm
    CONVERGENCE_THRESHOLD = 0.00001

    # The initialization sequence
    def __init__(self):
        print('[INFO] PerseusIEvaluation Initialization: Bringing things up...')
        self.model_free_perseus = PERSEUS(self.FRAGMENT_SIZE, self.NUMBER_OF_SAMPLING_ROUNDS,
                                          self.NUMBER_OF_EPISODES, self.EXPLORATION_PERIOD,
                                          self.NOISE_MEAN, self.NOISE_VARIANCE,
                                          self.IMPULSE_RESPONSE_MEAN,
                                          self.IMPULSE_RESPONSE_VARIANCE, self.PENALTY,
                                          self.FRAGMENTED_SPATIAL_SENSING_LIMITATION,
                                          self.CONFIDENCE_BOUND, self.DISCOUNT_FACTOR,
                                          self.CONVERGENCE_THRESHOLD,
                                          math.ceil(self.NUMBER_OF_CHANNELS / self.FRAGMENT_SIZE))

    # The evaluation routine
    def evaluate(self):
        obtained_optimal_utilities = self.model_free_perseus.run_perseus()
        self.model_free_perseus.visualize_iterative_regret()
        self.model_free_perseus.visualize_iterative_policy_changes_count()
        self.model_free_perseus.visualize_episodic_utilities(obtained_optimal_utilities)

    # The termination sequence
    def __exit__(self, exc_type, exc_val, exc_tb):
        print('[INFO] PerseusIEvaluation Termination: Tearing things down...')
        # Nothing to do...


# Run Trigger
if __name__ == '__main__':
    print('[INFO] PerseusIEvaluation main: Triggering the evaluation of the PERSEUS-I agent, i.e. the Model Free '
          'PERSEUS Algorithm with a Standard Belief Update procedure...')
    perseusI = PerseusIEvaluation()
    perseusI.evaluate()
