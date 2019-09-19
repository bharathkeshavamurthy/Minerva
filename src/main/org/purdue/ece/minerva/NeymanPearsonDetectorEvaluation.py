# This entity involves the evaluation of the Neyman Pearson Detector for the Cognitive Radio System developed as
#   a part of Minerva.
# Author: Bharath Keshavamurthy
# Organization: School of Electrical and Computer Engineering, Purdue University, West Lafayette, IN.
# Copyright (c) 2019. All Rights Reserved.

# WIKI
# The entity described here visualizes the performance of the "Neyman Pearson Detector" in a
#   scenario wherein the Secondary User (Cognitive Radio Node) is determining the Spectrum Occupancy Behavior of the
#   Primary User (Incumbent) over time and utilizing the spectrum holes for the flows assigned to it by the C2API.

# Visualization: Utility v Episodes

# NEYMAN PEARSON DETECTOR ASSUMING INDEPENDENCE IN A MARKOV-CORRELATED RADIO ENVIRONMENT

# The imports
import math
import numpy
import plotly
import scipy.stats
from enum import Enum
import plotly.graph_objs as go

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


# OccupancyState Enumeration
# Based on Energy Detection, \mathbb{E}[|X_k(i)|^2] = 1, if Occupied ; else, \mathbb{E}[|X_k(i)|^2] = 0
class OccupancyState(Enum):
    # Occupancy state IDLE
    IDLE = 0
    # Occupancy state OCCUPIED:
    OCCUPIED = 1


# The Markov Chain object that can be used via extension or replication in order to imply Markovian correlation
#   across either the channel indices or the time indices
class MarkovChain(object):

    # The initialization sequence
    def __init__(self):
        print('[INFO] MarkovChain Initialization: Initializing the Markov Chain...')
        # The steady-state probabilities (a.k.a start probabilities for each channel / each sampling round
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
            # P(Idle|Occupied) = p + (1 - pi)
            self.transition_probabilities[1][0] = p + self.start_probabilities[0]
            # P(Occupied|Occupied) = 1 - P(Idle|Occupied)
            self.transition_probabilities[1][1] = 1 - self.transition_probabilities[1][0]
        else:
            print(
                '[ERROR] MarkovChain set_transition_probability_parameter: Error while populating the state transition '
                'probabilities matrix! Proceeding with default values...')
            # Default Values...
            self.transition_probabilities = {0: {0: 0.8, 1: 0.2}, 1: {0: 0.6, 1: 0.4}}
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
    @staticmethod
    def get_state_probability(state, pi):
        return (lambda: 1 - pi, lambda: pi)[state == OccupancyState.OCCUPIED]()

    # Construct the complete transition probability matrix from \mathbb{P}(Occupied|Idle), i.e. 'p' and
    #   \mathbb{P}(Occupied), i.e. 'pi'
    @staticmethod
    def construct_transition_probability_matrix(p, pi):
        # \mathbb{P}(Idle|Occupied)
        q = p + (1 - pi)
        return {0: {0: 1 - p, 1: p}, 1: {0: q, 1: 1 - q}}

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
        # Number of episodes in which the Viterbi-I agent interacts with the radio environment
        self.number_of_episodes = _number_of_episodes
        # Channel Impulse Response used in the Observation Model
        self.impulse_response = self.get_impulse_response()
        # The AWGN used in the Observation Model
        self.noise = self.get_noise()

    # Generate the Channel Impulse Response samples
    def get_impulse_response(self):
        # The metrics to be passed to numpy.random.normal(mu, std, n)
        n_channel_impulse_response = self.number_of_sampling_rounds
        mu_channel_impulse_response = self.impulse_response_mean
        std_channel_impulse_response = numpy.sqrt(self.impulse_response_variance)
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
        if spatial_start_probabilities[1] != temporal_start_probabilities[1]:
            print(
                '[ERROR] PrimaryUser get_occupancy_behavior: Looks like the start probabilities are different across...'
                'the Spatial and the Temporal Markov Chains. This is inaccurate! Proceeding with defaults...')
            # Default Values
            spatial_start_probabilities = {0: 0.4, 1: 0.6}
            temporal_start_probabilities = {0: 0.4, 1: 0.6}
            print('[WARN] PrimaryUser get_occupancy_behavior: Modified System Steady State Probabilities - ',
                  str(temporal_start_probabilities))
        # Everything's alright with the system steady-state statistics - Start simulating the PU Occupancy Behavior
        # This is global and system-specific. So, it doesn't matter which chain's steady-state probabilities is used...
        pi_val = spatial_start_probabilities[1]
        # SINGLE CHAIN INFLUENCE along all the columns of the first row
        # Get the Initial state vector to get things going - row 0
        self.occupancy_behavior_collection.append(
            self.get_initial_states_temporal_variation(temporal_transition_probabilities_matrix[0][1],
                                                       temporal_transition_probabilities_matrix[1][0], pi_val))
        previous_state = self.occupancy_behavior_collection[0][0]
        # SINGLE CHAIN INFLUENCE along the first columns of all the rows
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
        # Use the definitions of Conditional Probabilities to realize the math - P(A|B,C)
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
        # \mathbb{P}(X_{k,t} = a | X_{k-1,t} = b, X_{k, t-1} = c) = \mathbb{P}(X_{k,t} = a | X_{k-1,t} = b) * \\
        #                                                           \mathbb{P}(X_{k,t} = a | X_{k, t-1} = c)
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

    # The observation procedure required for the Neyman-Pearson detector
    def make_sampled_observations_across_all_episodes(self):
        observation_samples = []
        for band in range(0, self.number_of_channels):
            obs_per_band = []
            for episode in range(0, self.number_of_episodes):
                obs_per_band.append(list((numpy.array(self.channel.impulse_response[band][episode]) * self.
                                          true_pu_occupancy_states[band][episode]) +
                                         numpy.array(self.channel.noise[band][episode])))
            observation_samples.append(obs_per_band)
        return observation_samples

    # The termination sequence
    def __exit__(self, exc_type, exc_val, exc_tb):
        print('[INFO] SecondaryUser Termination: Tearing things down...')
        # Nothing to do...


# A Neyman-Pearson Detector assuming independence among channels
# The channels are actually Markov correlated
# But, this class helps me evaluate the performance if I assume independence
# The signal model is s[n] = A (occupancy)
# The observation model is x[n] = w[n], null hypothesis, and
#                          x[n] = A + w[n], alternative hypothesis
class NeymanPearsonDetector(object):

    # The initialization sequence
    def __init__(self, _number_of_channels, _number_of_sampling_rounds, _number_of_episodes, _false_alarm_probability,
                 _noise_mean, _noise_variance, _observations, _true_pu_occupancy_states, _penalty):
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

    # Detect occupancy across episodes by averaging out noisy observations over numerous sampling rounds
    # Likelihood Ratio Test (LRT) based on a test statistic with the threshold determined from the P_{FA} constraint
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
        return utilities


# The evaluation of the Neyman Pearson Detection based non-POMDP agent
class NeymanPearsonDetectorEvaluation(object):
    # The number of channels in the discretized spectrum of interest
    NUMBER_OF_CHANNELS = 18

    # The number of sampling rounds during the observation phase of the SecondaryUser
    # FIXME: Right now, I have the PrimaryUser's behavior logged for every sampling round per episode.
    #   This translates to the SU logging observations for every sampling period per episode.
    #   Although this logic is unused as of today (18-Sept-2019), I'm keeping it in case I want to average out
    #       the inconsistencies I get during the evaluation of this Neyman Pearson Detection based non-POMDP algorithm.
    NUMBER_OF_SAMPLING_ROUNDS = 1000

    # The number of episodes of interaction of this agent with the radio environment
    NUMBER_OF_EPISODES = 1000

    # The mean of the Additive, White, Gaussian Noise samples
    NOISE_MEAN = 0

    # The mean of the Gaussian Impulse Response samples
    IMPULSE_RESPONSE_MEAN = 0

    # The variance of the Additive, White, Gaussian Noise samples
    NOISE_VARIANCE = 1

    # The variance of the Gaussian Impulse Response samples
    IMPULSE_RESPONSE_VARIANCE = 80

    # The penalty term for missed detections during utility evaluation
    PENALTY = -1

    # The constraint on false alarm probability for this agent
    FALSE_ALARM_PROBABILITY_CONSTRAINT = 0.7

    # Plotly Scatter mode
    PLOTLY_SCATTER_MODE = 'lines+markers'

    # Setup the Markov Chain
    @staticmethod
    def setup_markov_chain(_correlation_class, _pi, _p):
        print('[INFO] ViterbiIIEvaluation setup_markov_chain: Setting up the Markov Chain...')
        transient_markov_chain_object = MarkovChain()
        transient_markov_chain_object.set_markovian_correlation_class(_correlation_class)
        transient_markov_chain_object.set_start_probability_parameter(_pi)
        transient_markov_chain_object.set_transition_probability_parameter(_p)
        return transient_markov_chain_object

    # The initialization sequence
    def __init__(self):
        print('[INFO] NeymanPearsonDetectorEvaluation: Bringing things up...')
        # The start probabilities for the spatial Markov chain
        self.spatial_start_probabilities = {0: 0.4, 1: 0.6}
        # The start probabilities for the temporal Markov chain
        self.temporal_start_probabilities = {0: 0.4, 1: 0.6}
        # The transition model of the spatial Markov chain
        self.spatial_transition_probability_matrix = {0: {0: 0.8, 1: 0.2}, 1: {0: 0.6, 1: 0.4}}
        # The transition model of the temporal Markov chain
        self.temporal_transition_probability_matrix = {0: {0: 0.8, 1: 0.2}, 1: {0: 0.6, 1: 0.4}}
        # The spatial Markov chain
        self.spatial_markov_chain = self.setup_markov_chain(MarkovianCorrelationClass.SPATIAL,
                                                            self.spatial_start_probabilities[1],
                                                            self.spatial_transition_probability_matrix[0][1])
        # The temporal Markov chain
        self.temporal_markov_chain = self.setup_markov_chain(MarkovianCorrelationClass.TEMPORAL,
                                                             self.temporal_start_probabilities[1],
                                                             self.temporal_transition_probability_matrix[0][1])
        # The channel instance
        self.channel = Channel(self.NUMBER_OF_CHANNELS, self.NUMBER_OF_SAMPLING_ROUNDS, self.NUMBER_OF_EPISODES,
                               self.NOISE_MEAN, self.NOISE_VARIANCE, self.IMPULSE_RESPONSE_MEAN,
                               self.IMPULSE_RESPONSE_VARIANCE)
        # The Primary User
        self.primary_user = PrimaryUser(self.NUMBER_OF_CHANNELS, self.NUMBER_OF_EPISODES, self.spatial_markov_chain,
                                        self.temporal_markov_chain)
        # Simulate the Primary User behavior
        self.primary_user.simulate_occupancy_behavior()
        # The Secondary User
        self.secondary_user = SecondaryUser(self.NUMBER_OF_CHANNELS, self.NUMBER_OF_SAMPLING_ROUNDS,
                                            self.NUMBER_OF_EPISODES, self.channel,
                                            self.primary_user.occupancy_behavior_collection)
        # Make sampled observations across all episodes
        self.sampled_observations_across_all_episodes = \
            self.secondary_user.make_sampled_observations_across_all_episodes()
        # The Neyman Pearson Detection based non-POMDP agent
        self.neyman_pearson_detection_based_non_pomdp_agent = NeymanPearsonDetector(
            self.NUMBER_OF_CHANNELS, self.NUMBER_OF_SAMPLING_ROUNDS, self.NUMBER_OF_EPISODES,
            self.FALSE_ALARM_PROBABILITY_CONSTRAINT, self.NOISE_MEAN, self.NOISE_VARIANCE,
            self.sampled_observations_across_all_episodes, self.primary_user.occupancy_behavior_collection,
            self.PENALTY)
        # The x-axis corresponds to the episodes of interaction of this agent with the radio environment
        self.x_axis = [k + 1 for k in range(0, self.NUMBER_OF_EPISODES)]
        # The y-axis corresponds to the episodic utilities obtained by this agent
        self.y_axis = self.neyman_pearson_detection_based_non_pomdp_agent.get_utilities()

    # Visualize the episodic utilities of Neyman Pearson Detection based non-POMDP agent using the Plotly API
    def evaluate(self):
        # Data Trace
        visualization_trace = go.Scatter(x=self.x_axis,
                                         y=self.y_axis,
                                         mode=self.PLOTLY_SCATTER_MODE)
        # Figure Layout
        visualization_layout = dict(title='Episodic Utilities of the Neyman-Pearson Detector in a Spatio-Temporal '
                                          'Markovian PU Occupancy Behavior Model',
                                    xaxis=dict(title=r'$Episodes\ n$'),
                                    yaxis=dict(title=r'$Utility\ (1 - P_{FA}) + \mu P_{MD}$'))
        # Figure
        visualization_figure = dict(data=[visualization_trace],
                                    layout=visualization_layout)
        # URL
        figure_url = plotly.plotly.plot(visualization_figure,
                                        filename='Episodic_Utility_of_Neyman_Pearson_Detector')
        # Print the URL in case you're on an environment where a GUI is not available
        print('[INFO] ViterbiIEvaluation evaluate: Data Visualization Figure is available at {}'.format(figure_url))

    # The termination sequence
    def __exit__(self, exc_type, exc_val, exc_tb):
        print('[INFO] NeymanPearsonDetectorEvaluation Termination: Tearing things down...')
        # Nothing to do...


# Run Trigger
if __name__ == '__main__':
    print('[INFO] NeymanPearsonDetectorEvaluation main: Triggering the evaluation of the Neyman Pearson Detection based'
          'non-POMDP agent')
    agent = NeymanPearsonDetectorEvaluation()
    agent.evaluate()
