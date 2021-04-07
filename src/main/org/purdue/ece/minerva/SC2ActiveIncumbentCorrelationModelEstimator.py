# This entity describes the Expectation-Maximization (EM) algorithm in order to estimate the transition model underlying
#   the MDP governing the occupancy behavior of incumbents and competitors in the DARPA SC2 Active Incumbent scenario.
# Author: Bharath Keshavamurthy
# Organization: School of Electrical and Computer Engineering, Purdue University, West Lafayette, IN.
# Copyright (c) 2020. All Rights Reserved.


# The imports
import numpy
import scipy.stats
from enum import Enum
import DARPASC2ActiveIncumbentAnalysis as Analyser


# The OccupancyState enumeration
class OccupancyState(Enum):
    # The channel is idle
    IDLE = 0
    # The channel is occupied
    OCCUPIED = 1


# The main parameter estimator class that encapsulates the EM algorithm to estimate $\vec{\theta}$ offline.
# A convergence analysis of this parameter estimator is done for all 6 parameters in $\vec{\theta}$.
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
    CONFIDENCE_BOUND = 20

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
        # The transient spatial Markov chain transition estimates are logged here because they're needed in the
        #   Viterbi algorithm based state estimation in the PERSEUS-III agent
        self.transient_spatial_estimates = {'0': 0.5, '1': 0.5}

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
    # An initialization of 0.5 for every probabilistic value makes more sense heuristically
    def estimate(self):
        confidence = 0
        # The actual parameter vector for both the temporal chain AND the spatio-temporal chain
        prev_estimates = {j: 0.5 for j in self.estimates.keys()}
        current_estimates = {j: 0.5 for j in self.estimates.keys()}
        # The transient parameter vector exclusively for the spatial chain
        transient_prev_estimates = {'0': 0.5, '1': 0.5}
        transient_current_estimates = {'0': 0.5, '1': 0.5}
        # The temporal forward probabilities definition for the E-step (The Forward-Backward Algorithm) [channel-0]
        temporal_forward_probabilities = [{i-i: 0.0,
                                           i-i+1: 0.0} for i in range(self.number_of_episodes)]
        # The temporal backward probabilities definition for the E-step (The Forward-Backward Algorithm) [channel-0]
        temporal_backward_probabilities = [{i-i: 0.0,
                                            i-i+1: 0.0} for i in range(self.number_of_episodes)]
        # The spatial forward probabilities definition for the E-step (The Forward-Backward Algorithm) [time-slot: 0]
        spatial_forward_probabilities = [{i-i: 0.0,
                                          i-i+1: 0.0} for i in range(self.number_of_channels)]
        # The spatial backward probabilities definition for the E-step (The Forward-Backward Algorithm) [time-slot: 0]
        spatial_backward_probabilities = [{i-i: 0.0,
                                           i-i+1: 0.0} for i in range(self.number_of_channels)]
        # The spatio-temporal forward probabilities definition for the E-step (The Forward-Backward Algorithm)
        forward_probabilities = {k: [{i-i: 0.0,
                                      i-i+1: 0.0} for i in range(self.number_of_episodes)
                                     ] for k in range(self.number_of_channels)}
        # The spatio-temporal backward probabilities for the E-step (The Forward-Backward Algorithm)
        backward_probabilities = {k: [{i-i: 0.0,
                                       i-i+1: 0.0} for i in range(self.number_of_episodes)
                                      ] for k in range(self.number_of_channels)}
        for j in self.estimates.keys():
            while self.convergence_check(transient_prev_estimates, transient_current_estimates) is False or \
                    self.convergence_check(prev_estimates, current_estimates) is False or \
                    confidence < self.CONFIDENCE_BOUND:
                sampling_round = 0
                confidence = (lambda: confidence,
                              lambda: confidence + 1)[self.convergence_check(prev_estimates, current_estimates)]()
                prev_estimates = {j: current_estimates[j] for j in self.estimates.keys()}
                transient_prev_estimates = {'0': transient_current_estimates['0'],
                                            '1': transient_current_estimates['1']}
                temporal_transition_matrix = {
                    0: {0: 1 - prev_estimates['0'], 1: prev_estimates['0']},
                    1: {0: 1 - prev_estimates['1'], 1: prev_estimates['1']}
                }
                spatial_transition_matrix = {
                    0: {0: 1 - transient_prev_estimates['0'], 1: transient_prev_estimates['0']},
                    1: {0: 1 - transient_prev_estimates['1'], 1: transient_prev_estimates['1']}
                }
                transition_matrix = {
                    '00': {0: 1 - prev_estimates['00'], 1: prev_estimates['00']},
                    '01': {0: 1 - prev_estimates['01'], 1: prev_estimates['01']},
                    '10': {0: 1 - prev_estimates['10'], 1: prev_estimates['10']},
                    '11': {0: 1 - prev_estimates['11'], 1: prev_estimates['11']}
                }
                while sampling_round < self.NUMBER_OF_SAMPLING_ROUNDS:
                    exclusive = False
                    numerator, denominator = 0, 0
                    spatial_numerator, spatial_denominator = 0, 0
                    observations = {k: [self.simulate_observations(k, i)[0] for i in range(self.number_of_episodes)]
                                    for k in range(self.number_of_channels)}
                    if j == '0' or j == '1':
                        # E-step: Temporal correlation only
                        # The Forward step
                        for i in range(self.number_of_episodes):
                            for current_state in OccupancyState:
                                add = (lambda: False, lambda: True)[i == 0]()
                                for prev_state in OccupancyState:
                                    temporal_forward_probabilities[i][current_state.value] += (
                                        lambda: ((lambda: 0, lambda: 1)[add]()) *
                                                self.get_emission_probability(current_state, observations[0][i]) * (
                                                    (lambda: prev_estimates['0'] /
                                                             (1 + prev_estimates['0'] - prev_estimates['1']),
                                                     lambda: (1 - prev_estimates['1']) /
                                                             (1 + prev_estimates['0'] - prev_estimates['1'])
                                                     )[current_state.value == 0]()),
                                        lambda: self.get_emission_probability(current_state, observations[0][i]) *
                                                temporal_transition_matrix[prev_state.value][current_state.value] *
                                                temporal_forward_probabilities[i-1][prev_state.value]
                                    )[i > 0]()
                                    if add:
                                        add = False
                        # The Backward step
                        for i in range(self.number_of_episodes - 1, -1, -1):
                            for prev_state in OccupancyState:
                                for current_state in OccupancyState:
                                    temporal_backward_probabilities[i][prev_state.value] += (
                                        lambda: self.get_emission_probability(current_state, observations[0][i]) *
                                                temporal_transition_matrix[prev_state.value][current_state.value],
                                        lambda: self.get_emission_probability(current_state, observations[0][i]) *
                                                temporal_transition_matrix[prev_state.value][current_state.value] *
                                                temporal_backward_probabilities[i+1][current_state.value]
                                    )[i < (self.number_of_episodes - 1)]()
                        # M-step: Temporal correlation only
                        # The first and last summation inputs to the numerator
                        numerator += self.get_emission_probability(OccupancyState.OCCUPIED,
                                                                   observations[0][0]) * prev_estimates[j] * \
                                     temporal_backward_probabilities[1][OccupancyState.OCCUPIED.value]
                        numerator += temporal_forward_probabilities[self.number_of_episodes-2][int(j)] * \
                                     self.get_emission_probability(OccupancyState.OCCUPIED,
                                                                   observations[0][
                                                                       self.number_of_episodes-1]) * \
                                     prev_estimates[j]
                        for state in OccupancyState:
                            for i in range(1, self.number_of_episodes - 1):
                                numerator += (lambda: 0,
                                              lambda: temporal_forward_probabilities[i-1][int(j)] *
                                                      self.get_emission_probability(OccupancyState.OCCUPIED,
                                                                                    observations[0][i]) *
                                                      prev_estimates[j] *
                                                      temporal_backward_probabilities[i+1][
                                                          OccupancyState.OCCUPIED.value]
                                              )[exclusive is False]()
                                if i == 0 or i == self.number_of_episodes:
                                    continue
                                denominator += temporal_forward_probabilities[i-1][int(j)] * \
                                               self.get_emission_probability(state, observations[0][i]) * \
                                               (lambda: prev_estimates[j],
                                                lambda: 1 - prev_estimates[j])[state == OccupancyState.IDLE]() * \
                                               temporal_backward_probabilities[i+1][state.value]
                            exclusive = True
                            # The first and last summation inputs to the denominator
                            denominator += self.get_emission_probability(state,
                                                                         observations[0][0]) * \
                                           (lambda: prev_estimates[j],
                                            lambda: 1 - prev_estimates[j])[state == OccupancyState.IDLE]() * \
                                           temporal_backward_probabilities[1][state.value]
                            denominator += temporal_forward_probabilities[self.number_of_episodes-2][int(j)] * \
                                           self.get_emission_probability(state,
                                                                         observations[0][self.number_of_episodes-1]) * \
                                           (lambda: prev_estimates[j],
                                            lambda: 1 - prev_estimates[j])[state == OccupancyState.IDLE]()
                        exclusive = False
                        # E-step: Spatial correlation only
                        # The Forward step
                        for k in range(self.number_of_channels):
                            for current_state in OccupancyState:
                                add = (lambda: False, lambda: True)[k == 0]()
                                for prev_state in OccupancyState:
                                    spatial_forward_probabilities[k][current_state.value] += (
                                        lambda: ((lambda: 0, lambda: 1)[add]()) *
                                                self.get_emission_probability(current_state, observations[k][0]) * (
                                                    (lambda: prev_estimates['0'] /
                                                             (1 + prev_estimates['0'] - prev_estimates['1']),
                                                     lambda: (1 - prev_estimates['1']) /
                                                             (1 + prev_estimates['0'] - prev_estimates['1'])
                                                     )[current_state.value == 0]()),
                                        lambda: self.get_emission_probability(current_state, observations[k][0]) *
                                                spatial_transition_matrix[prev_state.value][current_state.value] *
                                                spatial_forward_probabilities[k-1][prev_state.value]
                                    )[k > 0]()
                                    if add:
                                        add = False
                        # The Backward step
                        for k in range(self.number_of_channels - 1, -1, -1):
                            for prev_state in OccupancyState:
                                for current_state in OccupancyState:
                                    spatial_backward_probabilities[k][prev_state.value] += (
                                        lambda: self.get_emission_probability(current_state, observations[k][0]) *
                                                spatial_transition_matrix[prev_state.value][current_state.value],
                                        lambda: self.get_emission_probability(current_state, observations[k][0]) *
                                                spatial_transition_matrix[prev_state.value][current_state.value] *
                                                spatial_backward_probabilities[k+1][current_state.value]
                                    )[k < (self.number_of_channels - 1)]()
                        # M-step: Spatial correlation only
                        # The first and last summation inputs to the numerator
                        spatial_numerator += self.get_emission_probability(OccupancyState.OCCUPIED,
                                                                           observations[0][0]) * \
                                     transient_prev_estimates[j] * \
                                     spatial_backward_probabilities[1][OccupancyState.OCCUPIED.value]
                        spatial_numerator += spatial_forward_probabilities[self.number_of_channels - 2][int(j)] * \
                                             self.get_emission_probability(OccupancyState.OCCUPIED,
                                                                           observations[
                                                                               self.number_of_channels - 1][0]) * \
                                             transient_prev_estimates[j]
                        for state in OccupancyState:
                            for k in range(1, self.number_of_channels - 1):
                                spatial_numerator += (lambda: 0,
                                                      lambda: spatial_forward_probabilities[k-1][int(j)] *
                                                              self.get_emission_probability(OccupancyState.OCCUPIED,
                                                                                            observations[k][0]) *
                                                              transient_prev_estimates[j] *
                                                              spatial_backward_probabilities[k+1][
                                                                  OccupancyState.OCCUPIED.value]
                                                      )[exclusive is False]()
                                if k == 0 or k == self.number_of_episodes:
                                    continue
                                spatial_denominator += spatial_forward_probabilities[k-1][int(j)] * \
                                                       self.get_emission_probability(state, observations[k][0]) * \
                                                       (lambda: transient_prev_estimates[j],
                                                        lambda: 1 - transient_prev_estimates[j])[
                                                           state == OccupancyState.IDLE]() * \
                                                       spatial_backward_probabilities[k+1][state.value]
                            exclusive = True
                            # The first and last summation inputs to the denominator
                            spatial_denominator += self.get_emission_probability(state,
                                                                                 observations[0][0]) * \
                                                   (lambda: transient_prev_estimates[j],
                                                    lambda: 1 - transient_prev_estimates[j])[
                                                       state == OccupancyState.IDLE]() * \
                                                   spatial_backward_probabilities[1][state.value]
                            spatial_denominator += spatial_forward_probabilities[self.number_of_channels-2][int(j)] * \
                                                   self.get_emission_probability(state,
                                                                                 observations[
                                                                                     self.number_of_channels-1][0]) * \
                                                   (lambda: transient_prev_estimates[j],
                                                    lambda: 1 - transient_prev_estimates[j])[
                                                       state == OccupancyState.IDLE]()
                    else:
                        spatial_state = int(list(j)[0])
                        temporal_state = int(list(j)[1])
                        # E-step: Spatio-Temporal correlation
                        # The Forward step
                        forward_probabilities[0] = temporal_forward_probabilities
                        for k in range(1, self.number_of_channels):
                            for i in range(self.number_of_episodes):
                                for current_state in OccupancyState:
                                    for prev_spatial_state in OccupancyState:
                                        add = (lambda: False, lambda: True)[i == 0]()
                                        for prev_temporal_state in OccupancyState:
                                            forward_probabilities[k][i][current_state.value] += (
                                                lambda: (lambda: 0, lambda: 1)[add]() *
                                                        self.get_emission_probability(current_state,
                                                                                      observations[k][i]) *
                                                        spatial_transition_matrix[prev_spatial_state.value][
                                                            current_state.value] *
                                                        forward_probabilities[k-1][i][prev_spatial_state.value],
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
                                for current_state in OccupancyState:
                                    for next_spatial_state in OccupancyState:
                                        add = (lambda: False, lambda: True)[i == (self.number_of_episodes - 1)]()
                                        for next_temporal_state in OccupancyState:
                                            backward_probabilities[k][i][current_state.value] += (
                                                lambda: (lambda: 0, lambda: 1)[add]() *
                                                        self.get_emission_probability(next_spatial_state,
                                                                                      observations[k+1][i]) *
                                                        backward_probabilities[k+1][i][next_spatial_state.value] *
                                                        spatial_transition_matrix[current_state.value][
                                                            next_spatial_state.value],
                                                lambda: self.get_emission_probability(next_spatial_state,
                                                                                      observations[k+1][i]) *
                                                        self.get_emission_probability(next_temporal_state,
                                                                                      observations[k][i+1]) *
                                                        spatial_transition_matrix[current_state.value][
                                                            next_spatial_state.value] *
                                                        temporal_transition_matrix[current_state.value][
                                                            next_temporal_state.value] *
                                                        backward_probabilities[k+1][i][next_spatial_state] *
                                                        backward_probabilities[k][i+1][next_temporal_state]
                                            )[i < (self.number_of_episodes - 1)]()
                                            if add:
                                                add = False
                        # M-step: Spatio-Temporal correlation
                        for state in OccupancyState:
                            for k in range(1, self.number_of_channels):
                                for i in range(1, self.number_of_episodes):
                                    numerator += (lambda: 0,
                                                  lambda: forward_probabilities[k-1][i][spatial_state] *
                                                          forward_probabilities[k][i-1][temporal_state] *
                                                          self.get_emission_probability(OccupancyState.OCCUPIED,
                                                                                        observations[k][i]) *
                                                          prev_estimates[j] *
                                                          backward_probabilities[k][i][OccupancyState.OCCUPIED.value]
                                                  )[exclusive is False]()
                                    denominator += forward_probabilities[k-1][i][spatial_state] * \
                                                   forward_probabilities[k][i-1][temporal_state] * \
                                                   self.get_emission_probability(state, observations[k][i]) * \
                                                   prev_estimates[j] * \
                                                   backward_probabilities[k][i][state.value]
                            exclusive = True
                    current_estimates[j] += (numerator / denominator)
                    if j == '0' or j == '1':
                        transient_current_estimates[j] += (spatial_numerator / spatial_denominator)
                    sampling_round += 1
                current_estimates[j] /= self.NUMBER_OF_SAMPLING_ROUNDS
                if j == '0' or j == '1':
                    transient_current_estimates[j] /= self.NUMBER_OF_SAMPLING_ROUNDS
        # Post-convergence analysis
        self.estimates = {j: current_estimates[j] for j in self.estimates.keys()}
        self.transient_spatial_estimates = {'0': transient_current_estimates['0'],
                                            '1': transient_current_estimates['1']}

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
    print('[INFO] SC2ActiveIncumbentCorrelationModelEstimator main: The transient spatial Markov chain transition'
          'estimates are: \n')
    print('P(1|0) = {}\n'.format(_estimator.transient_spatial_estimates['0']))
    print('P(1|1) = {}\n'.format(_estimator.transient_spatial_estimates['1']))
    # Fin
