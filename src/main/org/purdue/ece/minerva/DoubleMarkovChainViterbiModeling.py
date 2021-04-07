# This entity involves the evaluation of the Viterbi Algorithm with Incomplete Observations for the Cognitive Radio
#   System developed as a part of Minerva.
# This is the improved version modified to handle the recent changes to the correlation model.
# Author: Bharath Keshavamurthy
# Organization: School of Electrical and Computer Engineering, Purdue University, West Lafayette, IN.
# Copyright (c) 2020. All Rights Reserved.

# WIKI
# The entity described here analyzes the performance of the "Viterbi Algorithm with Incomplete Observations" in a
#   scenario wherein the Secondary User (Cognitive Radio Node) is determining the Spectrum Occupancy Behavior of the
#   Primary Users (Incumbents) over time and utilizing the spectrum holes for the flows assigned to it by the C2API.
#   The transition, the steady-state, and the emission metrics of the underlying Markov Model are assumed to be either
#   known or learnt beforehand by the Parameter Estimator (EM algorithm).

# VITERBI ALGORITHM WITH INCOMPLETE OBSERVATIONS

# %%% IMPROVED %%%

# The imports
import numpy
import scipy.stats
from enum import Enum
from collections import namedtuple


# Occupancy State Enumeration
# Based on Energy Detection, \mathbb{E}[|X_k(i)|^2] = 1, if Occupied; else, \mathbb{E}[|X_k(i)|^2] = 0
class OccupancyState(Enum):
    # Occupancy state IDLE
    IDLE = 0
    # Occupancy state OCCUPIED
    OCCUPIED = 1


# Delegate
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
        # Number of episodes in which the Viterbi-II agent interacts with the radio environment
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


# Delegate
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

    # Make observations from a global perspective for the constrained non-POMDP agent, i.e. the Viterbi-II agent
    # The same subset of channels are observed in each episode. No temporal patterns.
    def observe_everything_with_spatial_constraints(self, channel_selection_heuristic):
        observation_samples = []
        for band in range(0, self.number_of_channels):
            obs_per_band = [k - k for k in range(0, self.number_of_episodes)]
            if band in channel_selection_heuristic:
                obs_per_band = []
                for episode in range(0, self.number_of_episodes):
                    obs_per_band.append((self.channel.impulse_response[band][episode][0] *
                                         self.true_pu_occupancy_states[band][episode]) +
                                        self.channel.noise[band][episode][0])
            observation_samples.append(obs_per_band)
        # The observation_samples member is a kxt matrix
        return observation_samples

    # The termination sequence
    def __exit__(self, exc_type, exc_val, exc_tb):
        print('[INFO] SecondaryUser Termination: Tearing things down...')
        # Nothing to do...


# Delegate
# Channel Selection Strategy Generator
# Emulates an RL agent or a Multi-Armed Bandit
class ChannelSelectionStrategyGenerator(object):

    # The initialization sequence
    def __init__(self, _number_of_channels):
        print('[INFO] ChannelSelectionStrategyGenerator Initialization: Bringing things up...')
        # The number of channels in the discretized spectrum of interest
        self.number_of_channels = _number_of_channels
        # The discretized spectrum of interest
        self.discretized_spectrum = [k for k in range(0, self.number_of_channels)]

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

    # The termination sequence
    def __exit__(self, exc_type, exc_val, exc_tb):
        print('[INFO] ChannelSelectionStrategyGenerator Termination: Cleaning things up...')
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
        # If the channel is not observed, i.e. if the observation is [$\phi$] or [$0$], report $m_r(y_i)$ as $1$.
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


# The constrained non-POMDP agent, i.e. Viterbi-II
# The double Markov Chain Viterbi Algorithm with incomplete observations
class DoubleMarkovChainViterbiAlgorithm(object):
    # Start probabilities of PU occupancy per frequency band
    BAND_START_PROBABILITIES = namedtuple('BandStartProbabilities', ['idle', 'occupied'])

    # Value function named tuple
    VALUE_FUNCTION_NAMED_TUPLE = namedtuple('ValueFunction',
                                            ['current_value', 'previous_temporal_state', 'previous_spatial_state'])

    # The initialization sequence
    def __init__(self, _number_of_channels, _number_of_episodes, _emission_evaluator, _true_pu_occupancy_states,
                 _observation_samples, _start_probability, _correlation_model, _mu):
        print('[INFO] DoubleMarkovChainViterbiAlgorithm Initialization: Bringing things up ...')
        # The number of channels in the discretized spectrum of interest
        self.number_of_channels = _number_of_channels
        # The number of episodes of interaction of this constrained non-POMDP agent with the environment
        self.number_of_episodes = _number_of_episodes
        # The emission evaluator
        self.emission_evaluator = _emission_evaluator
        # True PU Occupancy states
        self.true_pu_occupancy_states = _true_pu_occupancy_states
        # The observed samples at the SU receiver
        self.observation_samples = _observation_samples
        # The start probabilities defining both chains in this double Markovian correlation structure
        self.start_probabilities = self.BAND_START_PROBABILITIES(idle=1 - _start_probability,
                                                                 occupied=_start_probability)
        # The transition probabilities for both exclusive single chain and combined double chain operations are
        #   modeled by this set of true parameters defining the correlation model
        self.correlation_model = _correlation_model
        # The missed detections penalty term
        self.mu = _mu
        # The analytics returned by this agent
        self.analytics = namedtuple('ANALYTICS',
                                    ['su_throughput', 'pu_interference'])

    # Get the start probabilities from the named tuple - a simple getter utility method exclusive to this class
    def get_start_probabilities(self, state):
        # The "state" arg is an enum instance of OccupancyState.
        if state == OccupancyState.OCCUPIED:
            return self.start_probabilities.occupied
        else:
            return self.start_probabilities.idle

    # Get the transition probability considering the combined effect of both the spatial and the temporal Markov chains
    # The "temporal_prev_state" arg, the "spatial_prev_state" arg, and the "current_state" arg are all enum instances
    #   of OccupancyState.
    def get_transition_probabilities(self, temporal_prev_state, spatial_prev_state, current_state):
        return (lambda: 1 - self.correlation_model[''.join([str(spatial_prev_state), str(temporal_prev_state)])],
                lambda: self.correlation_model[''.join([str(spatial_prev_state), str(temporal_prev_state)])])[
            current_state == OccupancyState.OCCUPIED]()

    # Return the transition probabilities from the transition probabilities matrix - single dimension
    # The "previous_state" arg and the "current_state" arg are enum instances of "OccupancyState".
    def get_transition_probabilities_single(self, previous_state, current_state):
        return (lambda: 1 - self.correlation_model[str(previous_state)],
                lambda: self.correlation_model[str(previous_state)])[current_state == OccupancyState.OCCUPIED]()

    # Output the analytics of this Constrained Double Markov Chain Viterbi Algorithm
    def estimate_agent_analytics(self):
        previous_state_spatial = None
        previous_state_temporal = None
        # Estimated states - kxt matrix
        estimated_states = []
        for x in range(0, self.number_of_channels):
            estimated_states.append([])
        value_function_collection = []
        # A value function collection to store and index the calculated value functions across k and t
        for k in range(0, self.number_of_channels):
            row = []
            for t in range(0, self.number_of_episodes):
                row.append(dict())
            value_function_collection.append(row)
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
                max_pointer = self.get_transition_probabilities_single(OccupancyState.IDLE,
                                                                       state) * \
                              value_function_collection[i][j - 1][OccupancyState.IDLE.name].current_value
                # Using IDLE as the confirmed previous state
                confirmed_previous_state = OccupancyState.IDLE.name
                for candidate_previous_state in OccupancyState:
                    if candidate_previous_state == OccupancyState.IDLE:
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
                    state,
                    self.observation_samples[i][j])
                value_function_collection[i][j][state.name] = self.VALUE_FUNCTION_NAMED_TUPLE(
                    current_value=current_value,
                    previous_temporal_state=confirmed_previous_state,
                    previous_spatial_state=None)
        # First column - Only spatial correlation
        j = 0
        for i in range(1, self.number_of_channels):
            # Trying to find the max pointer here ...
            for state in OccupancyState:
                # Again finishing off the [0] index first
                max_pointer = self.get_transition_probabilities_single(OccupancyState.IDLE,
                                                                       state) * \
                              value_function_collection[i - 1][j][OccupancyState.IDLE.name].current_value
                confirmed_previous_state = OccupancyState.IDLE.name
                for candidate_previous_state in OccupancyState:
                    if candidate_previous_state == OccupancyState.IDLE:
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
                    state,
                    self.observation_samples[i][j])
                value_function_collection[i][j][state.name] = self.VALUE_FUNCTION_NAMED_TUPLE(
                    current_value=current_value,
                    previous_temporal_state=None,
                    previous_spatial_state=confirmed_previous_state)
        # I'm done with the first row and first column
        # Moving on to the other rows and columns
        for i in range(1, self.number_of_channels):
            # For every row, I'm going across laterally (across columns) and populating the value_function_collection
            for j in range(1, self.number_of_episodes):
                for state in OccupancyState:
                    # Again finishing off the [0] index first
                    max_pointer = self.get_transition_probabilities(OccupancyState.IDLE,
                                                                    OccupancyState.IDLE, state) * \
                                  value_function_collection[i][j - 1][OccupancyState.IDLE.name].current_value * \
                                  value_function_collection[i - 1][j][OccupancyState.IDLE.name].current_value
                    confirmed_previous_state_temporal = OccupancyState.IDLE.name
                    confirmed_previous_state_spatial = OccupancyState.IDLE.name
                    for candidate_previous_state_temporal in OccupancyState:
                        for candidate_previous_state_spatial in OccupancyState:
                            if candidate_previous_state_temporal == OccupancyState.IDLE and \
                                    candidate_previous_state_spatial == OccupancyState.IDLE:
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
                    # I found the max of Double Markov Chain value functions from the past and now I'm multiplying it
                    #   with the emission probability of this particular observation
                    current_value = max_pointer * self.emission_evaluator.get_emission_probabilities(
                        state,
                        self.observation_samples[i][j])
                    value_function_collection[i][j][state.name] = self.VALUE_FUNCTION_NAMED_TUPLE(
                        current_value=current_value,
                        previous_temporal_state=confirmed_previous_state_temporal,
                        previous_spatial_state=confirmed_previous_state_spatial)
        max_value = 0
        # Finding the max value among the named tuples
        for _value in value_function_collection[-1][-1].values():
            if _value.current_value > max_value:
                max_value = _value.current_value
        # Finding the state corresponding to this max_value and using this as the final confirmed state to
        #   backtrack and find the previous states
        for k, v in value_function_collection[-1][-1].items():
            if v.current_value == max_value:
                # FIXME: Using the 'name' member to reference and deference the value function collection is not safe.
                estimated_states[self.number_of_channels - 1].append(self.value_from_name(k))
                previous_state_temporal = k
                previous_state_spatial = k
                break
        # Backtracking
        for i in range(self.number_of_channels - 1, -1, -1):
            if len(estimated_states[i]) == 0:
                estimated_states[i].append(
                                           self.value_from_name(value_function_collection[i + 1][
                                                                    self.number_of_episodes - 1][
                                                                    previous_state_spatial].previous_spatial_state
                                                                )
                                           )
                previous_state_spatial = value_function_collection[i + 1][self.number_of_episodes - 1][
                    previous_state_spatial].previous_spatial_state
                previous_state_temporal = previous_state_spatial
            for j in range(self.number_of_episodes - 1, 0, -1):
                estimated_states[i].insert(0, self.value_from_name(
                    value_function_collection[i][j][previous_state_temporal].previous_temporal_state))
                previous_state_temporal = value_function_collection[i][j][
                    previous_state_temporal].previous_temporal_state
        # Post-estimation analytics
        su_throughputs = []
        pu_interferences = []
        for i in range(self.number_of_episodes):
            su_throughput = 0
            pu_interference = 0
            for k in range(self.number_of_channels):
                su_throughput += (1 - estimated_states[k][i]) * (1 - self.true_pu_occupancy_states[k][i])
                pu_interference += self.true_pu_occupancy_states[k][i] * (1 - estimated_states[k][i])
            su_throughputs.append(su_throughput)
            pu_interferences.append(pu_interference)
        return self.analytics(su_throughput=sum(su_throughputs) / self.number_of_episodes,
                              pu_interference=sum(pu_interferences) / self.number_of_episodes)

    # Get enumeration field value from name
    @staticmethod
    def value_from_name(name):
        if name == OccupancyState.OCCUPIED.name:
            return OccupancyState.OCCUPIED.value
        else:
            return OccupancyState.IDLE.value

    # The termination sequence
    def __exit__(self, exc_type, exc_val, exc_tb):
        print('[INFO] DoubleMarkovChainViterbiAlgorithm Termination: Cleaning things up ...')
        # Nothing to do...


# A class detailing the evaluation of the Viterbi-II agent (A Constrained Double Markov Chain Viterbi Algorithm)
# This is the improved version of the original class, modified to handle the changes to the transition model defining
#   the MDP underlying the occupancy behavior of incumbents in the network
class DoubleMarkovChainViterbiModeling(object):

    # The number of channels in the discretized spectrum of interest
    NUMBER_OF_CHANNELS = 18

    # The number of sampling rounds during the observation phase of the SecondaryUser
    # FIXME: If I want, I could have the PrimaryUser's behavior logged for every sampling round per episode.
    #   This translates to the SU logging observations for every sampling round per episode.
    #   Although this logic is unused as of today (18-Sept-2019), I'm keeping it in case I want to average out
    #       the inconsistencies I get during the evaluation of this Viterbi-II algorithm.
    NUMBER_OF_SAMPLING_ROUNDS = 1000

    # The number of episodes of interaction of the constrained non-POMDP agent with the radio environment
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

    # Rendered delegate behavior
    # Simulate the incumbent occupancy behavior in the spectrum of interest according to the true correlation model
    def simulate_pu_occupancy(self):
        # Set Element (0,0)
        self.true_occupancy_states[0].append(
            (lambda: 1, lambda: 0)[numpy.random.random_sample() > self.start_probability]()
        )
        # Temporal chain: Complete row 0 (Use statistics q0 and q1)
        for i in range(1, self.NUMBER_OF_EPISODES):
            if self.true_occupancy_states[0][i - 1] == 1:
                self.true_occupancy_states[0].append(
                    (lambda: 1, lambda: 0)[numpy.random.random_sample() > self.correlation_model['q1']]()
                )
            else:
                self.true_occupancy_states[0].append(
                    (lambda: 1, lambda: 0)[numpy.random.random_sample() > self.correlation_model['q0']]()
                )
        # Spatial chain: Complete column 0 (Use statistics q0 and q1)
        for k in range(1, self.NUMBER_OF_CHANNELS):
            if self.true_occupancy_states[k - 1][0] == 1:
                self.true_occupancy_states[k].append(
                    (lambda: 1, lambda: 0)[numpy.random.random_sample() > self.correlation_model['q1']]()
                )
            else:
                self.true_occupancy_states[k].append(
                    (lambda: 1, lambda: 0)[numpy.random.random_sample() > self.correlation_model['q0']]()
                )
        # Complete the rest of the kxt matrix (Use statistics p00, p01, p10, and p11)
        for k in range(1, self.NUMBER_OF_CHANNELS):
            for i in range(1, self.NUMBER_OF_EPISODES):
                if self.true_occupancy_states[k - 1][i] == 0 and self.true_occupancy_states[k][i - 1] == 0:
                    self.true_occupancy_states[k].append(
                        (lambda: 1, lambda: 0)[numpy.random.random_sample() > self.correlation_model['p00']]()
                    )
                elif self.true_occupancy_states[k - 1][i] == 0 and self.true_occupancy_states[k][i - 1] == 1:
                    self.true_occupancy_states[k].append(
                        (lambda: 1, lambda: 0)[numpy.random.random_sample() > self.correlation_model['p01']]()
                    )
                elif self.true_occupancy_states[k - 1][i] == 1 and self.true_occupancy_states[k][i - 1] == 0:
                    self.true_occupancy_states[k].append(
                        (lambda: 1, lambda: 0)[numpy.random.random_sample() > self.correlation_model['p10']]()
                    )
                else:
                    self.true_occupancy_states[k].append(
                        (lambda: 1, lambda: 0)[numpy.random.random_sample() > self.correlation_model['p11']]()
                    )
        # Return the collection in case an external method needs it...
        return self.true_occupancy_states

    # The initialization sequence
    def __init__(self):
        print('[INFO] DoubleMarkovChainViterbiModeling Initialization: Bringing things up...')
        # The true occupancy states of the incumbents in the network
        self.true_occupancy_states = {k: [] for k in range(self.NUMBER_OF_CHANNELS)}
        # The start probability of the elements in this double Markov structure
        self.start_probability = 0.6
        # The correlation model parameters, i.e., $\vec{\theta}$
        self.correlation_model = {
            '0': 0.3,   # q0
            '1': 0.8,   # q1
            '00': 0.1,  # p00
            '01': 0.3,  # p01
            '10': 0.3,  # p10
            '11': 0.7   # p11
        }
        # The channel instance
        self.channel = Channel(self.NUMBER_OF_CHANNELS, self.NUMBER_OF_SAMPLING_ROUNDS, self.NUMBER_OF_EPISODES,
                               self.NOISE_MEAN, self.NOISE_VARIANCE, self.IMPULSE_RESPONSE_MEAN,
                               self.IMPULSE_RESPONSE_VARIANCE)
        # The Secondary User
        self.secondary_user = SecondaryUser(self.NUMBER_OF_CHANNELS, self.NUMBER_OF_SAMPLING_ROUNDS,
                                            self.NUMBER_OF_EPISODES, self.channel,
                                            self.true_occupancy_states)
        # The emission evaluator
        self.emission_evaluator = EmissionEvaluator(self.NOISE_VARIANCE, self.IMPULSE_RESPONSE_VARIANCE)
        # The channel selection strategy generator
        self.channel_selection_heuristic_generator = ChannelSelectionStrategyGenerator(self.NUMBER_OF_CHANNELS)
        # All possible combinatorial heuristics of the SU's observations
        self.observation_heuristics = self.channel_selection_heuristic_generator.uniform_sensing()

    # Visualize the episodic utilities of Viterbi-II using the Plotly API
    def evaluate(self):
        # The strategic choices (constraints) laid down in the WIKI
        for strategic_choice in range(1, 5):
            sensing_strategy = self.observation_heuristics[strategic_choice]
            # The constrained observations
            constrained_observations = self.secondary_user.observe_everything_with_spatial_constraints(sensing_strategy)
            # The constrained non-POMDP agent, i.e. the Viterbi-II algorithm
            constrained_non_pomdp_agent = DoubleMarkovChainViterbiAlgorithm(
                self.NUMBER_OF_CHANNELS, self.NUMBER_OF_EPISODES, self.emission_evaluator, self.true_occupancy_states,
                constrained_observations, self.start_probability, self.correlation_model, self.PENALTY)
            # The SU throughput and the PU interference analytics of the constrained non-POMDP agent
            analytics = constrained_non_pomdp_agent.estimate_agent_analytics()
            print('[INFO] DoubleMarkovChainViterbiModeling evaluate: Viterbi with channel sensing strategy {} - '
                  'Average Episodic SU Throughput = {} | '
                  'Average Episodic PU Interference = {}\n'.format(str(sensing_strategy),
                                                                   analytics.su_throughput,
                                                                   analytics.pu_interference))

    # The termination sequence
    def __exit__(self, exc_type, exc_val, exc_tb):
        print('[INFO] DoubleMarkovChainViterbiModeling Termination: Tearing things down...')
        # Nothing to do...


# Run Trigger
if __name__ == '__main__':
    print('[INFO] DoubleMarkovChainViterbiModeling main: Triggering the evaluation of the Constrained Non-POMDP Agent, '
          'i.e., the Viterbi Algorithm with Incomplete Observations!')
    agent = DoubleMarkovChainViterbiModeling()
    agent.simulate_pu_occupancy()
    agent.evaluate()
