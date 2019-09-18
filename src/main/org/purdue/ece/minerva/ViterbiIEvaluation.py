# This entity involves the evaluation of the Viterbi Algorithm with Complete Observations for the Cognitive Radio
#   System developed as a part of Minerva.
# Author: Bharath Keshavamurthy
# Organization: School of Electrical and Computer Engineering, Purdue University, West Lafayette, IN.
# Copyright (c) 2019. All Rights Reserved.

# WIKI
# The entity described here visualizes the performance of the "Viterbi Algorithm with Complete Observations" in a
#   scenario wherein the Secondary User (Cognitive Radio Node) is determining the Spectrum Occupancy Behavior of the
#   Primary User (Incumbent) over time and utilizing the spectrum holes for the flows assigned to it by the C2API.
#   The transition, the steady-state, and the emission metrics of the underlying Markov Model are assumed to be either
#   known or learnt beforehand by the Parameter Estimator (EM algorithm).

#   Visualization: Utility v Episodes

# VITERBI ALGORITHM WITH COMPLETE OBSERVATIONS

# The imports
import numpy
import plotly
import warnings
import functools
import scipy.stats
from enum import Enum
import plotly.graph_objs as go
from collections import namedtuple

# Plotly user account credentials for visualization
plotly.tools.set_credentials_file(username='bkeshava',
                                  api_key='W2WL5OOxLcgCzf8NNlgl')


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
    SPATIAL = 0
    # Markovian Correlation across time indices
    TEMPORAL = 1
    # Invalid
    INVALID = 2


# OccupancyState Enumeration
# Based on Energy Detection, E[|X_k(i)|^2] = 1, if Occupied ; else, E[|X_k(i)|^2] = 0
class OccupancyState(Enum):
    # Occupancy state IDLE
    IDLE = 0
    # Occupancy state OCCUPIED:
    OCCUPIED = 1


# The Markov Chain object that can be used via extension or replication in order to imply Markovian correlation...
# ...across either the channel indices or the time indices
class MarkovChain(object):

    # The initialization sequence
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

    # Construct the complete transition probability matrix from P(Occupied|Idle), i.e. 'p' and P(Occupied), i.e. 'pi'
    @staticmethod
    def construct_transition_probability_matrix(p, pi):
        # P(Idle|Occupied)
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

    # Observe everything from a global perspective for the unconstrained non-POMDP agent, i.e. Viterbi-I
    def observe_everything_unconstrained(self):
        observation_samples = []
        for band in range(0, self.number_of_channels):
            obs_per_band = []
            for episode in range(0, self.number_of_episodes):
                obs_per_band.append((self.channel.impulse_response[band][episode][0] *
                                     self.true_pu_occupancy_states[band][episode]
                                     ) + self.channel.noise[band][episode][0])
            observation_samples.append(obs_per_band)
        # The observation_samples member is a kxt (channel x episode) matrix
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


# The unconstrained non-POMDP agent, i.e. Viterbi-I
# The double Markov Chain Viterbi Algorithm with complete observations
class DoubleMarkovChainViterbiAlgorithm(object):
    # Start probabilities of PU occupancy per frequency band
    BAND_START_PROBABILITIES = namedtuple('BandStartProbabilities', ['idle', 'occupied'])

    # Value function named tuple
    VALUE_FUNCTION_NAMED_TUPLE = namedtuple('ValueFunction',
                                            ['current_value', 'previous_temporal_state', 'previous_spatial_state'])

    # The initialization Sequence
    def __init__(self, _number_of_channels, _number_of_episodes, _emission_evaluator, _true_pu_occupancy_states,
                 _observation_samples, _spatial_start_probabilities, _temporal_start_probabilities,
                 _spatial_transition_probabilities_matrix, _temporal_transition_probabilities_matrix, _mu):
        print('[INFO] DoubleMarkovChainViterbiAlgorithm Initialization: Bringing things up ...')
        # The number of channels in the discretized spectrum of interest
        self.number_of_channels = _number_of_channels
        # The number of episodes of interaction of this unconstrained non-POMDP agent with the environment
        self.number_of_episodes = _number_of_episodes
        # The emission evaluator
        self.emission_evaluator = _emission_evaluator
        # True PU Occupancy states
        self.true_pu_occupancy_states = _true_pu_occupancy_states
        # The observed samples at the SU receiver
        self.observation_samples = _observation_samples
        # The start probabilities
        self.start_probabilities = self.BAND_START_PROBABILITIES(idle=_spatial_start_probabilities[0],
                                                                 occupied=_spatial_start_probabilities[1])
        # The spatial transition probabilities matrix
        self.spatial_transition_probabilities_matrix = _spatial_transition_probabilities_matrix
        # The temporal transition probabilities matrix
        self.temporal_transition_probabilities_matrix = _temporal_transition_probabilities_matrix
        # The unified transition probabilities matrix
        # FIXME: For now, the same transition model across both chains
        self.transition_probabilities_matrix = _spatial_transition_probabilities_matrix
        # The missed detections penalty term
        self.mu = _mu

    # Get the start probabilities from the named tuple - a simple getter utility method exclusive to this class
    def get_start_probabilities(self, state):
        # The "state" arg is an enum instance of OccupancyState.
        if state == OccupancyState.OCCUPIED:
            return self.start_probabilities.occupied
        else:
            return self.start_probabilities.idle

    # Interpretation 1
    # Return the transition probabilities from the transition probabilities matrix - two dimensions
    # Using the concept of marginal probability here as I did for simulating the PrimaryUser behavior
    # def get_transition_probabilities(self, temporal_prev_state, spatial_prev_state, current_state):
    #     # temporal_prev_state is unused here...
    #     print('[DEBUG] DoubleMarkovChainViterbiAlgorithm get_transition_probabilities: current_state - {}, '
    #           'spatial_previous_state - {}, temporal_previous_state - {}'.format(current_state, spatial_prev_state,
    #                                                                              temporal_prev_state))
    #     return self.transition_probabilities_matrix[spatial_prev_state][current_state]

    # Interpretation 2
    # Return the transition probabilities from the transition probabilities matrix - two dimensions
    # The two Markov Chains are independent and hence, we can treat the transitions independently.
    # \mathbb{P}(X_{k,t} = a | X_{k-1,t} = b, X_{k, t-1} = c) = \mathbb{P}(X_{k,t} = a | X_{k-1,t} = b) * \\
    #                                                           \mathbb{P}(X_{k,t} = a | X_{k, t-1} = c)
    # The "temporal_prev_state" arg, the "spatial_prev_state" arg, and the "current_state" arg are all enum instances
    #   of OccupancyState.
    def get_transition_probabilities(self, temporal_prev_state, spatial_prev_state, current_state):
        return self.spatial_transition_probabilities_matrix[spatial_prev_state.value][current_state.value] * \
            self.temporal_transition_probabilities_matrix[temporal_prev_state.value][current_state.value]

    # FIXME: For now, the same transition model across both chains {An exclusive method}
    # Return the transition probabilities from the transition probabilities matrix - single dimension
    # THe "previous_state" arg and the "current_state" arg are enum instances of "OccupancyState".
    def get_transition_probabilities_single(self, previous_state, current_state):
        return self.transition_probabilities_matrix[previous_state.value][current_state.value]

    # Evaluate the Probability of False Alarm P_{FA}
    # \mathbb{P}(\hat{X}_k = 1 | X_k = 0) \forall k \in \{0, 1, 2, ...., K-1\}
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

    # Evaluate the Probability of Missed Detection P_{MD}
    # \mathbb{P}(\hat{X}_k = 0 | X_k = 1) \forall k \in \{0, 1, 2, ...., K-1\}
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

    # Get the utility obtained by this unconstrained non-POMDP agent
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

    # Output the episodic utilities of this Unconstrained Double Markov Chain Viterbi Algorithm
    def estimate_episodic_utilities(self):
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
            for x in range(0, self.number_of_episodes):
                row[x] = dict()
            value_function_collection[k] = row
        # t = 0 and k = 0 - No previous state to base the Markovian Correlation on in either dimension
        for state in OccupancyState:
            current_value = self.emission_evaluator.get_emission_probabilities(state,
                                                                               self.observation_samples[0][0]) * \
                            self.get_start_probabilities(state)
            value_function_collection[0][0][state.name] = self.VALUE_FUNCTION_NAMED_TUPLE(current_value=current_value,
                                                                                          previous_temporal_state=None,
                                                                                          previous_spatial_state=None)
        # First row - Only temporal correlation
        i = 0
        for j in range(1, self.number_of_episodes):
            # Trying to find the max pointer here ...
            for state in OccupancyState:
                # Again finishing off the [0] index first
                max_pointer = self.get_transition_probabilities_single(OccupancyState.idle,
                                                                       state) * \
                              value_function_collection[i][j - 1][OccupancyState.idle.name].current_value
                # Using IDLE as the confirmed previous state
                confirmed_previous_state = OccupancyState.idle.name
                for candidate_previous_state in OccupancyState:
                    if candidate_previous_state.name == OccupancyState.idle.name:
                        # Already done
                        continue
                    else:
                        pointer = self.get_transition_probabilities_single(candidate_previous_state,
                                                                           state) * \
                                  value_function_collection[i][j - 1][candidate_previous_state.name].current_value
                        if pointer > max_pointer:
                            max_pointer = pointer
                            confirmed_previous_state = candidate_previous_state.name
                current_value = max_pointer * self.emission_evaluator.get_emission_probabilities(
                    state, self.observation_samples[i][j])
                value_function_collection[i][j][state.name] = self.VALUE_FUNCTION_NAMED_TUPLE(
                    current_value=current_value, previous_temporal_state=confirmed_previous_state,
                    previous_spatial_state=None)
        # First column - Only spatial correlation
        j = 0
        for i in range(1, self.number_of_channels):
            # Trying to find the max pointer here ...
            for state in OccupancyState:
                # Again finishing off the [0] index first
                max_pointer = self.get_transition_probabilities_single(OccupancyState.idle,
                                                                       state) * \
                              value_function_collection[i - 1][j][OccupancyState.idle.name].current_value
                confirmed_previous_state = OccupancyState.idle.name
                for candidate_previous_state in OccupancyState:
                    if candidate_previous_state.name == OccupancyState.idle.name:
                        # Already done
                        continue
                    else:
                        pointer = self.get_transition_probabilities_single(candidate_previous_state,
                                                                           state) * \
                                  value_function_collection[i - 1][j][candidate_previous_state.name].current_value
                        if pointer > max_pointer:
                            max_pointer = pointer
                            confirmed_previous_state = candidate_previous_state.name
                current_value = max_pointer * self.emission_evaluator.get_emission_probabilities(
                    state, self.observation_samples[i][j])
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
                    max_pointer = self.get_transition_probabilities(OccupancyState.idle,
                                                                    OccupancyState.idle, state) * \
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
                                pointer = self.get_transition_probabilities(candidate_previous_state_temporal,
                                                                            candidate_previous_state_spatial,
                                                                            state) * \
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
                        state,
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
                # FIXME: Using the 'name' member to reference and deference the value function collection is not safe.
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
        return utilities

    # Get enumeration field value from name
    @staticmethod
    def value_from_name(name):
        if name == 'occupied':
            return OccupancyState.occupied.value
        else:
            return OccupancyState.idle.value

    # The termination sequence
    def __exit__(self, exc_type, exc_val, exc_tb):
        print('[INFO] DoubleMarkovChainViterbiAlgorithm Clean-up: Cleaning things up ...')
        # Nothing to do...


# A class detailing the evaluation of the Viterbi-I agent (An Unconstrained Double Markov Chain Viterbi Algorithm)
class ViterbiIEvaluation(object):

    # The number of channels in the discretized spectrum of interest
    NUMBER_OF_CHANNELS = 18

    # The number of sampling rounds during the observation phase of the SecondaryUser
    # FIXME: Right now, I have the PrimaryUser's behavior logged for every sampling round per episode.
    #   This translates to the SU logging observations for every sampling period per episode.
    #   Although this logic is unused as of today (18-Sept-2019), I'm keeping it in case I want to average out
    #       the inconsistencies I get during the evaluation of this Viterbi-I algorithm.
    NUMBER_OF_SAMPLING_ROUNDS = 1000

    # The number of episodes of interaction of the unconstrained non-POMDP agent with the radio environment
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

    # Plotly Scatter mode
    PLOTLY_SCATTER_MODE = 'lines+markers'

    # Setup the Markov Chain
    @staticmethod
    def setup_markov_chain(_correlation_class, _pi, _p):
        print('[INFO] ViterbiIEvaluation setup_markov_chain: Setting up the Markov Chain...')
        transient_markov_chain_object = MarkovChain()
        transient_markov_chain_object.set_markovian_correlation_class(_correlation_class)
        transient_markov_chain_object.set_start_probability_parameter(_pi)
        transient_markov_chain_object.set_transition_probability_parameter(_p)
        return transient_markov_chain_object

    # The initialization sequence
    def __init__(self):
        print('[INFO] ViterbiIEvaluation: Bringing things up...')
        # The start probabilities for the spatial Markov chain
        self.spatial_start_probabilities = {0: 0.4, 1: 0.6}
        # The start probabilities for the temporal Markov chain
        self.temporal_start_probabilities = {0: 0.4, 1: 0.6}
        # The transition model of the spatial Markov chain
        self.spatial_transition_probability_matrix = {0: {0: 0.8, 1: 0.2}, 1: {0: 0.6, 1: 0.4}}
        # The transition model of the temporal Markov chain
        self.temporal_transition_probability_matrix = {0: {0: 0.8, 1: 0.2}, 1: {0: 0.6, 1: 0.4}}
        # The spatial Markov chain
        self.spatial_markov_chain = self.setup_markov_chain(MarkovianCorrelationClass.spatial,
                                                            self.spatial_start_probabilities[1],
                                                            self.spatial_transition_probability_matrix[0][1])
        # The temporal Markov chain
        self.temporal_markov_chain = self.setup_markov_chain(MarkovianCorrelationClass.temporal,
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
        # The emission evaluator
        self.emission_evaluator = EmissionEvaluator(self.NOISE_VARIANCE, self.IMPULSE_RESPONSE_VARIANCE)
        # The unconstrained global observation samples
        self.unconstrained_global_observations = self.secondary_user.observe_everything_unconstrained()
        # The unconstrained non-POMDP agent, i.e. Viterbi-I
        self.unconstrained_non_pomdp_agent = DoubleMarkovChainViterbiAlgorithm(
            self.NUMBER_OF_CHANNELS, self.NUMBER_OF_EPISODES, self.emission_evaluator,
            self.primary_user.occupancy_behavior_collection, self.unconstrained_global_observations,
            self.spatial_start_probabilities, self.temporal_start_probabilities,
            self.spatial_transition_probability_matrix,
            self.temporal_transition_probability_matrix, self.PENALTY)
        # The x-axis corresponds to the episodes of interaction
        self.x_axis = [k + 1 for k in range(0, self.NUMBER_OF_EPISODES)]
        # The y-axis corresponds to the episodic utilities of this Viterbi-I agent
        self.y_axis = self.unconstrained_non_pomdp_agent.estimate_episodic_utilities()

    # Visualize the episodic utilities of Viterbi-I using the Plotly API
    def evaluate(self):
        # Data Trace
        visualization_trace = go.Scatter(x=self.x_axis,
                                         y=self.y_axis,
                                         mode=self.PLOTLY_SCATTER_MODE)
        visualization_layout = dict(title='Episodic Utilities of the Viterbi Algorithm with Complete Observations',
                                    xaxis=dict(title=r'$Episodes\ n$'),
                                    yaxis=dict(title=r'$Utility\ (1 - P_{FA}) + \mu P_{MD}$'))
        visualization_figure = dict(data=[visualization_trace],
                                    layout=visualization_layout)
        figure_url = plotly.plotly.plot(visualization_figure,
                                        filename='Episodic_Utility_of_Unconstrained_Viterbi_Algorithm')
        # Print the URL in case you're on an environment where a GUI is not available
        print('[INFO] ViterbiIEvaluation evaluate: Data Visualization Figure is available at {}'.format(figure_url))

    # The termination sequence
    def __exit__(self, exc_type, exc_val, exc_tb):
        print('[INFO] ViterbiIEvaluation: Tearing things down...')
        # Nothing to do...


# Run Trigger
if __name__ == '__main__':
    print('[INFO] ViterbiIEvaluation main: Triggering the evaluation of the Unconstrained Non-POMDP Agent, i.e. the '
          'Viterbi Algorithm with Complete Observations!')
    agent = ViterbiIEvaluation()
    agent.evaluate()
