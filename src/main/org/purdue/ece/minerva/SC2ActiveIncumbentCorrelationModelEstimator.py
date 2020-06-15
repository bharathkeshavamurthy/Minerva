# This entity describes the Expectation-Maximization (EM) algorithm in order to estimate the transition model underlying
#   the MDP governing the occupancy behavior of incumbents and competitors in the DARPA SC2 Active Incumbent scenario.
# Author: Bharath Keshavamurthy
# Organization: School of Electrical and Computer Engineering, Purdue University, West Lafayette, IN.
# Copyright (c) 2020. All Rights Reserved.


# The imports
import numpy
import plotly
import scipy.stats
from enum import Enum
import DARPASC2ActiveIncumbentAnalysis as Analyser

# Plotly user account credentials for visualization
plotly.tools.set_credentials_file(username='bkeshava',
                                  api_key='W2WL5OOxLcgCzf8NNlgl')


# The OccupancyState enumeration
class OccupancyState(Enum):
    # The channel is idle
    IDLE = 0
    # The channel is occupied
    OCCUPIED = 1


# The main parameter estimator class that encapsulates the EM algorithm to estimate $\vec{\theta}$ offline.
# A convergence analysis of this parameter estimator is done for all 6 parameters in $\vec{\theta}$ in plotly.
class SC2ActiveIncumbentCorrelationModelEstimator(object):

    # The number of sampling rounds corresponding to a complete kxt matrix observation
    NUMBER_OF_SAMPLING_ROUNDS = 360

    # The variance of the Additive, White, Gaussian Noise which is modeled to be a part of our linear observation model
    VARIANCE_OF_AWGN = 1

    # The variance of the Channel Impulse Response which is modeled to be a part of our linear observation model
    VARIANCE_OF_CHANNEL_IMPULSE_RESPONSE = 80

    # The threshold for algorithm termination
    EPSILON = 1e-8

    # The confidence bound for confident convergence analysis
    CONFIDENCE_BOUND = 10

    # The initialization sequence
    def __init__(self, db):
        print('[INFO] SC2ActiveIncumbentCorrelationModelEstimator Initialization: Bringing things up...')
        # The DARPASC2ActiveIncumbentAnalysis object
        sc2_active_incumbent_analyser = Analyser.DARPASC2ActiveIncumbentAnalysis(db)
        # The re-negotiated, condensed, extracted occupancy behavior of the incumbents and competitors
        self.true_occupancy_states = sc2_active_incumbent_analyser.get_occupancy_behavior()
        # The number of condensed and discretized channels of interest
        self.number_of_channels = len(self.true_occupancy_states.keys())
        # The number of episodes in which the collected DARPA SC2 data would be analysed for estimating the time-freq.
        #   correlation structure: note here that "data homogeneity" is assumed!
        self.number_of_episodes = len(self.true_occupancy_states[0])
        # The true start probability of the elements in this double Markov structure
        self.true_start_probability = sc2_active_incumbent_analyser.get_steady_state_occupancy_probability(
            self.true_occupancy_states)
        # The estimates member declaration: initial random values
        self.estimates = {
            '0': 0.5,    # q0
            '1': 0.5,    # q1
            '00': 0.5,   # p00
            '01': 0.5,   # p01
            '10': 0.5,   # p10
            '11': 0.5    # p11
        }

    # Rendered delegate behavior
    # Simulate an observation given the channel and the episode
    def simulate_observations(self, channel, episode):
        # Assuming zero-mean additive white gaussian noise in the observation model
        noise_sample = numpy.random.normal(0, numpy.sqrt(self.VARIANCE_OF_AWGN), 1)
        # Assuming a zero-mean gaussian channel impulse response in the observation model
        impulse_response_sample = numpy.random.normal(0, numpy.sqrt(self.VARIANCE_OF_CHANNEL_IMPULSE_RESPONSE), 1)
        # Let's use these noise and channel impulse response samples to make an observation
        return (impulse_response_sample * self.true_occupancy_states[channel][episode]) + noise_sample

    # Determine the emission probability of the observation sample, given the state, i.e., \mathbb{P}(Y_{k}(i)|B_{k}(i))
    def get_emission_probability(self, state, observation_sample):
        return scipy.stats.norm(0,
                                numpy.sqrt(
                                    (self.VARIANCE_OF_CHANNEL_IMPULSE_RESPONSE * state.value) +
                                    self.VARIANCE_OF_AWGN
                                )).pdf(observation_sample)

    # Convergence analysis across two consecutive iterations
    # |\hat{\theta}_{j}(t) - \hat{\theta}_{j}(t-1)| < \epsilon, \forall \hat{\theta}_{j} \in \hat{\vec{\theta}}
    def convergence_check(self, prev_estimates, current_estimates):
        for j in prev_estimates.keys():
            if abs(current_estimates[j] - prev_estimates[j]) > self.EPSILON:
                return False
        return True

    # Relevant core behavior
    # The core method: estimate the parameters defining the correlation model underlying incumbent occupancy behavior
    def estimate(self):
        confidence = 0
        prev_estimates = {j: 0.0 for j in self.estimates.keys()}
        current_estimates = {j: 1e-8 for j in self.estimates.keys()}
        # The temporal forward probabilities definition for the E-step (The Forward-Backward Algorithm)
        temporal_forward_probabilities = [{i-i: 0.0,
                                           i-i+1: 0.0} for i in range(self.number_of_episodes)]
        # The temporal backward probabilities definition for the E-step (The Forward-Backward Algorithm)
        temporal_backward_probabilities = [{i-i: 0.0,
                                            i-i+1: 0.0} for i in range(self.number_of_episodes)]
        # The spatio-temporal forward probabilities definition for the E-step (The Forward-Backward Algorithm)
        forward_probabilities = {k: [{i-i: 0.0,
                                      i-i+1: 0.0} for i in range(self.number_of_episodes)
                                     ] for k in range(self.number_of_channels)}
        # The spatio-temporal backward probabilities for the E-step (The Forward-Backward Algorithm)
        backward_probabilities = {k: [{i - i: 0.0,
                                       i - i + 1: 0.0} for i in range(self.number_of_episodes)
                                      ] for k in range(self.number_of_channels)}
        for j in self.estimates.keys():
            while self.convergence_check(prev_estimates, current_estimates) is False or \
                    confidence < self.CONFIDENCE_BOUND:
                sampling_round = 0
                confidence = (lambda: confidence,
                              lambda: confidence + 1)[self.convergence_check(prev_estimates, current_estimates)]()
                prev_estimates = {j: current_estimates[j] for j in self.estimates.keys()}
                while sampling_round < self.NUMBER_OF_SAMPLING_ROUNDS:
                    exclusive = False
                    numerator = 0
                    denominator = 0
                    observations = {k: [self.simulate_observations(k, i) for i in range(self.number_of_episodes)]
                                    for k in range(self.number_of_channels)}
                    temporal_transition_matrix = {
                        0: {0: 1 - prev_estimates['0'], 1: prev_estimates['0']},
                        1: {0: 1 - prev_estimates['1'], 1: prev_estimates['1']}
                    }
                    transition_matrix = {
                        '00': {0: 1 - prev_estimates['00'], 1: prev_estimates['00']},
                        '01': {0: 1 - prev_estimates['01'], 1: prev_estimates['01']},
                        '10': {0: 1 - prev_estimates['10'], 1: prev_estimates['10']},
                        '11': {0: 1 - prev_estimates['11'], 1: prev_estimates['11']}
                    }
                    if j == '0' or j == '1':
                        # E-step: Temporal correlation only
                        # The Forward step
                        for i in range(self.number_of_episodes):
                            add = (lambda: False, lambda: True)[i == 0]()
                            for current_state in OccupancyState:
                                for prev_state in OccupancyState:
                                    temporal_forward_probabilities[i][current_state.value] += (
                                        lambda: ((lambda: 0, lambda: 1)[add]()) *
                                                self.get_emission_probability(current_state, observations[0][i]) * (
                                                    (lambda: prev_estimates['0'] /
                                                             (1 + prev_estimates['0'] - prev_estimates['1']),
                                                     lambda: (1 - prev_estimates['1']) /
                                                             (1 + prev_estimates['0'] - prev_estimates['1'])
                                                     )[current_state]()),
                                        lambda: self.get_emission_probability(current_state, observations[0][i]) *
                                                temporal_transition_matrix[prev_state.value][current_state.value] *
                                                temporal_forward_probabilities[i-1][prev_state.value]
                                    )[i > 0]()
                                    if add:
                                        add = False
                        # The Backward step
                        for i in range(self.number_of_episodes - 1, -1, -1):
                            add = (lambda: False, lambda: True)[i == (self.number_of_episodes - 1)]
                            for current_state in OccupancyState:
                                for next_state in OccupancyState:
                                    temporal_backward_probabilities[i][current_state.value] += (
                                        lambda: ((lambda: 0, lambda: 1)[add]()) * 1,
                                        lambda: self.get_emission_probability(next_state, observations[i+1]) *
                                                temporal_transition_matrix[current_state.value][next_state.value] *
                                                temporal_backward_probabilities[i+1][next_state.value]
                                    )[i < (self.number_of_episodes - 1)]()
                                    if add:
                                        add = False
                        # M-step: Temporal correlation only
                        for state in OccupancyState:
                            for i in range(self.number_of_episodes):
                                numerator += (lambda: 0,
                                              lambda: temporal_forward_probabilities[i - 1][int(j)] *
                                                      self.get_emission_probability(OccupancyState(1),
                                                                                    observations[0][i]) *
                                                      prev_estimates[j] *
                                                      temporal_backward_probabilities[i + 1][1]
                                              )[exclusive is False]()
                                denominator += temporal_forward_probabilities[i - 1][int(j)] * \
                                               self.get_emission_probability(state, observations[0][i]) * \
                                               (lambda: prev_estimates[j],
                                                lambda: 1 - prev_estimates[j])[state == OccupancyState.IDLE]() * \
                                               temporal_backward_probabilities[i + 1][state.value]
                            exclusive = True
                    else:
                        spatial_state = int(list(j)[0])
                        temporal_state = int(list(j)[1])
                        # E-step: Spatio-Temporal correlation
                        # The Forward step
                        forward_probabilities[0] = temporal_forward_probabilities
                        for k in range(1, self.number_of_channels):
                            for i in range(self.number_of_episodes):
                                add = (lambda: False, lambda: True)[i == 0]()
                                for current_state in OccupancyState:
                                    for prev_spatial_state in OccupancyState:
                                        for prev_temporal_state in OccupancyState:
                                            forward_probabilities[k][i][current_state.value] += (
                                                lambda: (lambda: 0, lambda: 1)[add]() *
                                                        self.get_emission_probability(current_state,
                                                                                      observations[k][i]) *
                                                        temporal_transition_matrix[prev_spatial_state.value][
                                                            current_state.value] *
                                                        forward_probabilities[k][i][prev_spatial_state],
                                                lambda: self.get_emission_probability(current_state,
                                                                                      observations[k][i]) *
                                                        transition_matrix[''.join([str(prev_spatial_state.value),
                                                                                   str(prev_temporal_state.value)])][
                                                            current_state.value] *
                                                        forward_probabilities[k-1][i][prev_spatial_state.value] *
                                                        forward_probabilities[k][i-1][prev_temporal_state.value]
                                            )[i > 0]()
                                            if add:
                                                add = False
                        # The Backward step
                        backward_probabilities[self.number_of_channels-1] = temporal_backward_probabilities
                        for k in range(self.number_of_channels - 2, -1, -1):
                            for i in range(self.number_of_episodes - 1, -1, -1):
                                add = (lambda: False, lambda: True)[i == (self.number_of_episodes - 1)]()
                                for current_state in OccupancyState:
                                    for next_spatial_state in OccupancyState:
                                        for next_temporal_state in OccupancyState:
                                            backward_probabilities[k][i][current_state.value] += (
                                                lambda: (lambda: 0, lambda: 1)[add]() *
                                                        self.get_emission_probability(next_spatial_state,
                                                                                      observations[k+1][i]) *
                                                        backward_probabilities[k+1][i][next_spatial_state.value] *
                                                        temporal_transition_matrix[current_state.value][
                                                            next_spatial_state.value],
                                                lambda: self.get_emission_probability(next_spatial_state,
                                                                                      observations[k+1][i]) *
                                                        self.get_emission_probability(next_temporal_state,
                                                                                      observations[k][i+1]) *
                                                        temporal_transition_matrix[current_state.value][
                                                            next_spatial_state.value] *
                                                        temporal_transition_matrix[current_state.value][
                                                            next_temporal_state.value] *
                                                        backward_probabilities[k+1][i][next_spatial_state] *
                                                        backward_probabilities[k][i+1][next_temporal_state]
                                            )[i < (self.number_of_episodes - 1)]()
                        # M-step: Spatio-Temporal correlation
                        for state in OccupancyState:
                            for k in range(self.number_of_channels):
                                for i in range(self.number_of_episodes):
                                    numerator += (lambda: 0,
                                                  lambda: forward_probabilities[k-1][i][spatial_state] *
                                                          forward_probabilities[k][i-1][temporal_state] *
                                                          self.get_emission_probability(OccupancyState(1),
                                                                                        observations[k][i]) *
                                                          prev_estimates[j] *
                                                          backward_probabilities[i][k][1]
                                                  )[exclusive is False]()
                                    denominator += forward_probabilities[k-1][i][spatial_state] * \
                                                   forward_probabilities[k][i-1][temporal_state] * \
                                                   self.get_emission_probability(state, observations[k][i]) * \
                                                   prev_estimates[j] * \
                                                   backward_probabilities[k][i][state.value]
                            exclusive = True
                    current_estimates[j] += (numerator / denominator)
                    sampling_round += 1
                current_estimates[j] /= self.NUMBER_OF_SAMPLING_ROUNDS
        # Post-convergence analysis
        self.estimates = {j: current_estimates[j] for j in self.estimates.keys()}

    # The termination sequence
    def __exit__(self, exc_type, exc_val, exc_tb):
        print('[INFO] SC2ActiveIncumbentCorrelationModelEstimator Termination: Tearing things down...')
        # Nothing to do


# Run Trigger
if __name__ == '__main__':
    # The default DB file (Active Incumbent Scenario-8342)
    _db = 'data/active_incumbent_scenario8342.db'
    print('[INFO] SC2ActiveIncumbentCorrelationModelEstimator main: Creating an instance of the estimator with '
          'rendered delegate behavior and core estimation capabilities...')
    _estimator = SC2ActiveIncumbentCorrelationModelEstimator(_db)
    _estimator.estimate()
    print('[INFO] SC2ActiveIncumbentCorrelationModelEstimator main: The estimated parameters defining the correlation '
          'model underlying the incumbent occupancy behavior are: \n')
    print('q0 = {}\n'.format(_estimator.estimates['0']))
    print('q1 = {}\n'.format(_estimator.estimates['1']))
    print('p00 = {}\n'.format(_estimator.estimates['00']))
    print('p01 = {}\n'.format(_estimator.estimates['01']))
    print('p10 = {}\n'.format(_estimator.estimates['10']))
    print('p11 = {}.'.format(_estimator.estimates['11']))
    # Fin
