# This entity describes the Expectation-Maximization (EM) algorithm in order to estimate the transition model underlying
#   the MDP governing the occupancy behavior of incumbents in the radio environment.
# Author: Bharath Keshavamurthy
# Organization: School of Electrical and Computer Engineering, Purdue University, West Lafayette, IN.
# Copyright (c) 2020. All Rights Reserved.


# The imports
import numpy
import plotly
import scipy.stats
from enum import Enum
import plotly.graph_objs as go

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
class MarkovCorrelationModelEstimator(object):
    # The number of channels in the discretized spectrum of interest
    NUMBER_OF_CHANNELS = 18

    # The number of episodes or time-steps for our parameter estimation algorithm
    NUMBER_OF_EPISODES = 1000

    # The number of sampling rounds corresponding to a complete kxt matrix observation
    NUMBER_OF_SAMPLING_ROUNDS = 500

    # The variance of the Additive, White, Gaussian Noise which is modeled to be a part of our linear observation model
    VARIANCE_OF_AWGN = 1

    # The variance of the Channel Impulse Response which is modeled to be a part of our linear observation model
    VARIANCE_OF_CHANNEL_IMPULSE_RESPONSE = 80

    # The threshold for algorithm termination
    EPSILON = 1e-8

    # The confidence bound for confident convergence analysis
    CONFIDENCE_BOUND = 10

    # The initialization sequence
    def __init__(self):
        print('[INFO] MarkovCorrelationModelEstimator Initialization: Bringing things up...')
        # The true occupancy states of the incumbents in the network
        self.true_occupancy_states = {k: [] for k in range(self.NUMBER_OF_CHANNELS)}
        # The true start probability of the elements in this double Markov structure
        self.true_start_probability = 0.6
        # The true correlation model parameters, i.e., $\vec{\theta}$
        self.true_parameters = {
            '0': 0.3,    # q0
            '1': 0.8,    # q1
            '00': 0.1,   # p00
            '01': 0.3,   # p01
            '10': 0.3,   # p10
            '11': 0.7    # p11
        }
        # The estimates member declaration
        self.estimates = None

    # Rendered delegate behavior
    # Simulate the incumbent occupancy behavior in the spectrum of interest according to the true correlation model
    def simulate_pu_occupancy(self):
        # Set Element (0,0)
        self.true_occupancy_states[0].append(
            (lambda: 1, lambda: 0)[numpy.random.random_sample() > self.true_start_probability]()
        )
        # Temporal chain: Complete row 0 (Use statistics q0 and q1)
        for i in range(1, self.NUMBER_OF_EPISODES):
            if self.true_occupancy_states[0][i - 1] == 1:
                self.true_occupancy_states[0].append(
                    (lambda: 1, lambda: 0)[numpy.random.random_sample() > self.true_parameters['q1']]()
                )
            else:
                self.true_occupancy_states[0].append(
                    (lambda: 1, lambda: 0)[numpy.random.random_sample() > self.true_parameters['q0']]()
                )
        # Spatial chain: Complete column 0 (Use statistics q0 and q1)
        for k in range(1, self.NUMBER_OF_CHANNELS):
            if self.true_occupancy_states[k - 1][0] == 1:
                self.true_occupancy_states[k].append(
                    (lambda: 1, lambda: 0)[numpy.random.random_sample() > self.true_parameters['q1']]()
                )
            else:
                self.true_occupancy_states[k].append(
                    (lambda: 1, lambda: 0)[numpy.random.random_sample() > self.true_parameters['q0']]()
                )
        # Complete the rest of the kxt matrix (Use statistics p00, p01, p10, and p11)
        for k in range(1, self.NUMBER_OF_CHANNELS):
            for i in range(1, self.NUMBER_OF_EPISODES):
                if self.true_occupancy_states[k - 1][i] == 0 and self.true_occupancy_states[k][i - 1] == 0:
                    self.true_occupancy_states[k].append(
                        (lambda: 1, lambda: 0)[numpy.random.random_sample() > self.true_parameters['p00']]()
                    )
                elif self.true_occupancy_states[k - 1][i] == 0 and self.true_occupancy_states[k][i - 1] == 1:
                    self.true_occupancy_states[k].append(
                        (lambda: 1, lambda: 0)[numpy.random.random_sample() > self.true_parameters['p01']]()
                    )
                elif self.true_occupancy_states[k - 1][i] == 1 and self.true_occupancy_states[k][i - 1] == 0:
                    self.true_occupancy_states[k].append(
                        (lambda: 1, lambda: 0)[numpy.random.random_sample() > self.true_parameters['p10']]()
                    )
                else:
                    self.true_occupancy_states[k].append(
                        (lambda: 1, lambda: 0)[numpy.random.random_sample() > self.true_parameters['p11']]()
                    )
        # Return the collection in case an external method needs it...
        return self.true_occupancy_states

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
        prev_estimates = {j: 0.0 for j in self.true_parameters.keys()}
        current_estimates = {j: 1e-8 for j in self.true_parameters.keys()}
        # The temporal forward probabilities definition for the E-step (The Forward-Backward Algorithm)
        temporal_forward_probabilities = [{i-i: 0.0,
                                           i-i+1: 0.0} for i in range(self.NUMBER_OF_EPISODES)]
        # The temporal backward probabilities definition for the E-step (The Forward-Backward Algorithm)
        temporal_backward_probabilities = [{i-i: 0.0,
                                            i-i+1: 0.0} for i in range(self.NUMBER_OF_EPISODES)]
        # The spatio-temporal forward probabilities definition for the E-step (The Forward-Backward Algorithm)
        forward_probabilities = {k: [{i-i: 0.0,
                                      i-i+1: 0.0} for i in range(self.NUMBER_OF_EPISODES)
                                     ] for k in range(self.NUMBER_OF_CHANNELS)}
        # The spatio-temporal backward probabilities for the E-step (The Forward-Backward Algorithm)
        backward_probabilities = {k: [{i - i: 0.0,
                                       i - i + 1: 0.0} for i in range(self.NUMBER_OF_EPISODES)
                                      ] for k in range(self.NUMBER_OF_CHANNELS)}
        # The squared errors across iterations for terminal convergence analysis
        squared_errors = {j: [] for j in self.true_parameters.keys()}
        for j in self.true_parameters.keys():
            while self.convergence_check(prev_estimates, current_estimates) is False or \
                    confidence < self.CONFIDENCE_BOUND:
                sampling_round = 0
                confidence = (lambda: confidence,
                              lambda: confidence + 1)[self.convergence_check(prev_estimates, current_estimates)]()
                prev_estimates = {j: current_estimates[j] for j in self.true_parameters.keys()}
                squared_errors[j].append(numpy.square(prev_estimates[j] - self.true_parameters[j]))
                while sampling_round < self.NUMBER_OF_SAMPLING_ROUNDS:
                    exclusive = False
                    numerator = 0
                    denominator = 0
                    observations = {k: [self.simulate_observations(k, i) for i in range(self.NUMBER_OF_EPISODES)]
                                    for k in range(self.NUMBER_OF_CHANNELS)}
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
                        for i in range(self.NUMBER_OF_EPISODES):
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
                        for i in range(self.NUMBER_OF_EPISODES - 1, -1, -1):
                            add = (lambda: False, lambda: True)[i == (self.NUMBER_OF_EPISODES - 1)]
                            for current_state in OccupancyState:
                                for next_state in OccupancyState:
                                    temporal_backward_probabilities[i][current_state.value] += (
                                        lambda: ((lambda: 0, lambda: 1)[add]()) * 1,
                                        lambda: self.get_emission_probability(next_state, observations[i+1]) *
                                                temporal_transition_matrix[current_state.value][next_state.value] *
                                                temporal_backward_probabilities[i+1][next_state.value]
                                    )[i < (self.NUMBER_OF_EPISODES - 1)]()
                                    if add:
                                        add = False
                        # M-step: Temporal correlation only
                        for state in OccupancyState:
                            for i in range(self.NUMBER_OF_EPISODES):
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
                        for k in range(1, self.NUMBER_OF_CHANNELS):
                            for i in range(self.NUMBER_OF_EPISODES):
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
                        backward_probabilities[self.NUMBER_OF_CHANNELS-1] = temporal_backward_probabilities
                        for k in range(self.NUMBER_OF_CHANNELS - 2, -1, -1):
                            for i in range(self.NUMBER_OF_EPISODES - 1, -1, -1):
                                add = (lambda: False, lambda: True)[i == (self.NUMBER_OF_EPISODES - 1)]()
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
                                            )[i < (self.NUMBER_OF_EPISODES - 1)]()
                        # M-step: Spatio-Temporal correlation
                        for state in OccupancyState:
                            for k in range(self.NUMBER_OF_CHANNELS):
                                for i in range(self.NUMBER_OF_EPISODES):
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
        self.estimates = {j: current_estimates[j] for j in self.true_parameters.keys()}
        mean_squared_errors = []
        slowest_parameter = max(squared_errors, key=lambda x: len(squared_errors[x]))
        slowest_time = len(squared_errors[slowest_parameter])
        for key, value in squared_errors.items():
            if key != slowest_parameter:
                for diff in range(slowest_time - len(value)):
                    value.append(value[len(value)-1])
        # \mathbb{E}_{t}[\sum_{j=1}^{6}\ (\theta_{j}(t) - \hat{\theta}_{j}(t))^{2}]
        for index in range(slowest_time):
            mean_squared_errors.append(
                sum([squared_errors[j][index] for j in self.true_parameters.keys()])
            )
        # Visualization
        data_trace = go.Scatter(x=[t for t in range(slowest_time)], y=mean_squared_errors, mode='lines+markers')
        data_layout = dict(title='Mean Square Error Convergence of the Parameters defining the Incumbent Occupancy '
                                 'Correlation Model',
                           xaxis=dict(title='Number of iterations (x300 observations)'),
                           yaxis=dict(type='log', autorange=True,
                                      title=r'Mean Square Error - '
                                            r'$\mathbb{E}_{t}[\sum_{j=1}^{6}(\theta_{j}(t) - \hat{\theta}_{j}(t))^{2}]$'
                                      )
                           )
        figure = dict(data=[data_trace],
                      layout=data_layout)
        figure_url = plotly.plotly.plot(figure,
                                        filename='Mean_Square_Error_Convergence_Plot_of_the_Parameter_Estimator')
        print('[INFO] MarkovCorrelationModelEstimator estimate: The figure is available at [{}]'.format(figure_url))

    # The termination sequence
    def __exit__(self, exc_type, exc_val, exc_tb):
        print('[INFO] MarkovCorrelationModelEstimator Termination: Tearing things down...')
        # Nothing to do


# Run Trigger
if __name__ == '__main__':
    print('[INFO] MarkovCorrelationModelEstimator main: Creating an instance of the estimator with rendered delegate '
          'behavior and core estimation capabilities...')
    estimator = MarkovCorrelationModelEstimator()
    estimator.simulate_pu_occupancy()
    estimator.estimate()
    print('[INFO] MarkovCorrelationModelEstimator main: The estimated parameters defining the correlation model '
          'underlying the incumbent occupancy behavior are: \n')
    print('q0 = {}\n'.format(estimator.estimates['0']))
    print('q1 = {}\n'.format(estimator.estimates['1']))
    print('p00 = {}\n'.format(estimator.estimates['00']))
    print('p01 = {}\n'.format(estimator.estimates['01']))
    print('p10 = {}\n'.format(estimator.estimates['10']))
    print('p11 = {}.'.format(estimator.estimates['11']))
    # Fin
