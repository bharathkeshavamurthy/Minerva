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
#   The entire premise behind the fragmented PERSEUS algorithm in a radio environment where there are more than one PU
#   is that the PUs generally are restricted to certain segments of the radio spectrum by design and by bureaucracy.
#   Therefore, in order to speed up the PERSEUS algorithm over a significantly large number of channels, we set out to
#   partition the radio spectrum into fragments, each corresponding to the spectral constraints of incumbents, as set
#   by governments or regulatory agencies. So, we run instances of the PERSEUS agent (SU with PERSEUS as its
#   Channel Sensing and Access strategy/heuristic) over these fragments and then, aggregate the utilities obtained
#   by these individual PERSEUS instances in order to get the overall utility obtained by the SU over numerous
#   episodes of interaction with the radio environment.

# PERSEUS ALGORITHM WITHOUT MODEL FORESIGHT AND STANDARD BELIEF UPDATE PRACTICES

# Visualization: Utility v Episodes

# The imports
import numpy
import plotly
import scipy.stats
from enum import Enum
# import plotly.graph_objs as go
# from collections import namedtuple

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
