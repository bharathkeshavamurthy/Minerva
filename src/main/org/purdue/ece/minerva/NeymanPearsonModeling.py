# This entity involves the evaluation of the Neyman Pearson Detector for the Cognitive Radio System developed as
#   a part of Minerva.
# This is the improved version of the Detector modified to handle the new corrected correlation model.
# Author: Bharath Keshavamurthy
# Organization: School of Electrical and Computer Engineering, Purdue University, West Lafayette, IN.
# Copyright (c) 2020. All Rights Reserved.

# WIKI
# The entity described here visualizes the performance of the "Neyman Pearson Detector" in a
#   scenario wherein the Secondary User (Cognitive Radio Node) is determining the Spectrum Occupancy Behavior of the
#   Primary User (Incumbent) over time and utilizing the spectrum holes for the flows assigned to it by the C2API.

# Reason for analytics: In a system wherein there exists a Markovian Correlation Model in the PU Occupancy Behavior,
#   do algorithms that assume independence [this] perform better than algorithms that leverage the correlation
#   information encapsulated in the given Markovian Model [Viterbi-X, PERSEUS-X, State_of_the_Art-X]?

# NEYMAN PEARSON DETECTOR ASSUMING INDEPENDENCE IN A MARKOV-CORRELATED RADIO ENVIRONMENT

# %%% IMPROVED %%%

# The imports
import math
import numpy
import warnings
import scipy.stats
from enum import Enum
from collections import namedtuple

warnings.filterwarnings("ignore",
                        category=DeprecationWarning)


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
        # Number of episodes in which the "Neyman-Pearson Detection" based agent interacts with the radio environment
        self.number_of_episodes = _number_of_episodes
        # Channel Impulse Response used in the Observation Model
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

    # The observation procedure required for the "Neyman-Pearson Detection" based agent
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


# A Neyman-Pearson Detector assuming independence among channels...
# The channels are actually Markov correlated
# But, this class helps me evaluate the performance if I assume independence
# The signal model is s[n] = A (occupancy)
# The observation model is x[n] = w[n], null hypothesis, and
#                          x[n] = A + w[n], alternative hypothesis
# The non-POMDP agent
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
            print('[WARN] NeymanPearsonDetector Initialization: The observation model assumes Zero-Mean Additive White '
                  'Gaussian Noise samples...')
            self.noise_mean = 0
        # The variance of the AWGN samples
        self.noise_variance = _noise_variance
        # The observations made by the Secondary User
        self.observations = _observations
        # The true PU occupancy states - Markov correlation is left unexploited here...
        self.true_pu_occupancy_states = _true_pu_occupancy_states
        # The threshold for the Likelihood Ratio Test (LRT)
        self.threshold = math.sqrt(self.noise_variance / self.number_of_sampling_rounds) * scipy.stats.norm.ppf(
            1 - self.false_alarm_probability
        )
        # The penalty for missed detections
        self.penalty = _penalty
        # The analytics returned by this agent
        self.analytics = namedtuple('ANALYTICS',
                                    ['su_throughput', 'pu_interference'])

    # Get the average number of truly idle channels exploited by the SU per episode along with the average number of
    #   channels in which the SU interferes with the incumbents
    # The actual behavior of the agent is encapsulated in this method - test statistic creation and comparison
    # The threshold is generated during the initialization of this agent
    def get_analytics(self):
        su_throughputs = []
        pu_interferences = []
        for i in range(self.number_of_episodes):
            su_throughput = 0
            pu_interference = 0
            for k in range(self.number_of_channels):
                estimated_state = (lambda: 0, lambda: 1)[
                    (sum(self.observations[k][i]) / self.number_of_sampling_rounds) >= self.threshold]()
                su_throughput += (1 - estimated_state) * (1 - self.true_pu_occupancy_states[k][i])
                pu_interference += self.true_pu_occupancy_states[k][i] * (1 - estimated_state)
            su_throughputs.append(su_throughput)
            pu_interferences.append(pu_interference)
        return self.analytics(su_throughput=sum(su_throughputs) / self.number_of_episodes,
                              pu_interference=sum(pu_interferences) / self.number_of_episodes)


# The evaluation of the "Neyman Pearson Detection" based non-POMDP agent
# This is the improved version modified to handle the corrected correlation model defining the MDP underlying incumbent
#   occupancy behavior
class NeymanPearsonModeling(object):
    # The number of channels in the discretized spectrum of interest
    NUMBER_OF_CHANNELS = 18

    # The number of sampling rounds during the observation phase of the SecondaryUser
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
    FALSE_ALARM_PROBABILITY_CONSTRAINT = 0.3

    # Plotly Scatter mode
    PLOTLY_SCATTER_MODE = 'lines+markers'

    # Rendered delegate behavior
    # Simulate the incumbent occupancy behavior in the spectrum of interest according to the true correlation model
    def simulate_pu_occupancy(self):
        # Set Element (0,0)
        self.true_occupancy_states[0].append(
            (lambda: 1, lambda: 0)[numpy.random.random_sample() > self.start_probabilities]()
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
        print('[INFO] NeymanPearsonModeling Initialization: Bringing things up...')
        # The true occupancy states of the incumbents in the network
        self.true_occupancy_states = {k: [] for k in range(self.NUMBER_OF_CHANNELS)}
        # The start probability of the elements in this double Markov structure
        self.start_probabilities = 0.6
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
        # Make sampled observations across all episodes
        self.sampled_observations_across_all_episodes = \
            self.secondary_user.make_sampled_observations_across_all_episodes()
        # The "Neyman Pearson Detection" based non-POMDP agent
        self.neyman_pearson_detection_based_non_pomdp_agent = NeymanPearsonDetector(
            self.NUMBER_OF_CHANNELS, self.NUMBER_OF_SAMPLING_ROUNDS, self.NUMBER_OF_EPISODES,
            self.FALSE_ALARM_PROBABILITY_CONSTRAINT, self.NOISE_MEAN, self.NOISE_VARIANCE,
            self.sampled_observations_across_all_episodes, self.true_occupancy_states,
            self.PENALTY)

    # Core behavior
    # The evaluation routine that outputs the SU throughput and PU interference analytics for this non-POMDP agent
    def evaluate(self):
        analytics = self.neyman_pearson_detection_based_non_pomdp_agent.get_analytics()
        print('[INFO] NeymanPearsonModeling evaluate: Neyman-Pearson Detection - '
              'Average Episodic SU Throughput = {} | '
              'Average Episodic PU Interference = {}'.format(analytics.su_throughput,
                                                             analytics.pu_interference)
              )

    # The termination sequence
    def __exit__(self, exc_type, exc_val, exc_tb):
        print('[INFO] NeymanPearsonModeling Termination: Tearing things down...')
        # Nothing to do...


# Run Trigger
if __name__ == '__main__':
    print('[INFO] NeymanPearsonModeling main: Triggering the evaluation of the Neyman Pearson Detection based'
          'non-POMDP agent')
    agent = NeymanPearsonModeling()
    agent.simulate_pu_occupancy()
    agent.evaluate()
