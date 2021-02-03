# This Python script encapsulates an algorithm to estimate the transition parameters of Markov chains used in our
#   Cognitive Radio Research.
# Static PU with Markovian Correlation across the channel indices
# Author: Bharath Keshavamurthy
# Organization: School of Electrical and Computer Engineering, Purdue University
# Copyright (c) 2019. All Rights Reserved
# For the complete math behind this algorithm, refer to the following link.
# https://github.rcac.purdue.edu/bkeshava/Minerva/blob/master/latex/pdf/PU_Occupancy_Behavior_Estimator_v1_0_0.pdf
# This link is subject to change. Please contact the author at bkeshava@purdue.edu for more details.

# The imports
import numpy
import plotly
import scipy.stats
from enum import Enum
import plotly.graph_objs as go
from collections import namedtuple

# Plotly user account credentials for visualization
plotly.tools.set_credentials_file(username='bkeshava',
                                  api_key='W2WL5OOxLcgCzf8NNlgl')


# Occupancy State Enumeration
# Based on Energy Detection, E[|X_k(i)|^2] = 1, if Occupied; else, E[|X_k(i)|^2] = 0
class OccupancyState(Enum):
    # Occupancy state IDLE
    IDLE = 0
    # Occupancy state OCCUPIED
    OCCUPIED = 1


# Markov Chain Parameter Estimation Algorithm
class MarkovChainParameterEstimator(object):
    # Number of channels in the discretized spectrum of interest
    NUMBER_OF_FREQUENCY_BANDS = 18

    # Number of observations made by the SU during the simulation period
    NUMBER_OF_SAMPLES = 300

    # Number of cycles in order to smoothen the plot
    NUMBER_OF_CYCLES = 1

    # Variance of the Additive White Gaussian Noise Samples
    VARIANCE_OF_AWGN = 1

    # Variance of the Channel Impulse Response which is a zero mean Gaussian
    # This channel and noise model gives an SNR ~ 19 dB
    VARIANCE_OF_CHANNEL_IMPULSE_RESPONSE = 80

    # Start probabilities of PU occupancy per frequency band
    BAND_START_PROBABILITIES = namedtuple('BandStartProbabilities', ['idle', 'occupied'])

    # Confidence bound for termination
    CONFIDENCE_BOUND = 5

    # The initialization sequence
    def __init__(self):
        print('[INFO] MarkovChainParameterEstimator Initialization: Bringing things up...')
        # AWGN samples
        self.noise_samples = {}
        # Channel Impulse Response samples
        self.channel_impulse_response_samples = {}
        # True PU Occupancy states
        self.true_pu_occupancy_states = []
        # The observed samples at the SU receiver
        self.observation_samples = []
        # The start probabilities
        self.start_probabilities = self.BAND_START_PROBABILITIES(idle=0.4, occupied=0.6)
        # The transition probabilities
        self.transition_probabilities = {0: {0: list(), 1: list()}, 1: {0: list(), 1: list()}}
        # The forward probabilities
        self.forward_probabilities = []
        for k in range(0, self.NUMBER_OF_FREQUENCY_BANDS):
            self.forward_probabilities.append(dict())
        # The backward probabilities
        self.backward_probabilities = []
        for k in range(0, self.NUMBER_OF_FREQUENCY_BANDS):
            self.backward_probabilities.append(dict())
        # Convergence check epsilon - a very small value > 0
        # Distance Metric
        self.epsilon = 1e-8
        # Initial p = \mathbb{P}(Occupied|Idle)
        self.p_initial = 0.0
        # Initial Forward Probabilities dict
        self.initial_forward_probabilities = None
        # Initial Backward Probabilities dict
        self.initial_backward_probabilities = None
        # True \mathbb{P}(Occupied|Idle) value
        self.true_p_value = 0.0

    # Generate the true states using the Markovian model
    # Arguments: p -> \mathbb{P}(1|0); q -> \mathbb{P}(0|1); and pi -> \mathbb{P}(1)
    def allocate_true_pu_occupancy_states(self, p_val, q_val, pi_val):
        previous = 1
        # Initial state generation -> band-0 using pi_val
        if numpy.random.random_sample() > pi_val:
            previous = 0
        self.true_pu_occupancy_states.append(previous)
        # Based on the state of band-0 and the (p_val,q_val) values, generate the states of the remaining bands
        for loop_counter in range(1, self.NUMBER_OF_FREQUENCY_BANDS):
            sample = numpy.random.random_sample()
            if previous == 1 and sample < q_val:
                previous = 0
            elif previous == 1 and sample > q_val:
                previous = 1
            elif previous == 0 and sample < p_val:
                previous = 1
            else:
                previous = 0
            self.true_pu_occupancy_states.append(previous)

    # Get the observations vector
    # Generate the observations of all the bands for a number of observation rounds or cycles
    def allocate_observations(self):
        for frequency_band in range(0, self.NUMBER_OF_FREQUENCY_BANDS):
            # The noise statistics
            mu_noise = 0
            std_noise = numpy.sqrt(self.VARIANCE_OF_AWGN)
            n_noise = self.NUMBER_OF_SAMPLES
            # Generation of the AWGN samples
            self.noise_samples[frequency_band] = numpy.random.normal(mu_noise,
                                                                     std_noise,
                                                                     n_noise)
            # The channel impulse response statistics
            mu_channel_impulse_response = 0
            std_channel_impulse_response = numpy.sqrt(self.VARIANCE_OF_CHANNEL_IMPULSE_RESPONSE)
            n_channel_impulse_response = self.NUMBER_OF_SAMPLES
            # Generation of the impulse response samples
            self.channel_impulse_response_samples[frequency_band] = numpy.random.normal(mu_channel_impulse_response,
                                                                                        std_channel_impulse_response,
                                                                                        n_channel_impulse_response)
        # Making the observations using the generated AWGN and Channel Impulse Response samples
        for band in range(0, self.NUMBER_OF_FREQUENCY_BANDS):
            obs_per_band = list()
            for count in range(0, self.NUMBER_OF_SAMPLES):
                obs_per_band.append((self.channel_impulse_response_samples[band][
                                        count] * self.true_pu_occupancy_states[band]) + self.noise_samples[
                                        band][count])
            self.observation_samples.append(obs_per_band)
        return self.observation_samples

    # Get the Emission Probabilities -> \mathbb{P}(y|x)
    # The "_state" arg is an instance of the OccupancyState enumeration.
    def get_emission_probabilities(self, _state, observation_sample):
        emission_probability = scipy.stats.norm(0, numpy.sqrt(
            (self.VARIANCE_OF_CHANNEL_IMPULSE_RESPONSE * _state.value) + self.VARIANCE_OF_AWGN)).pdf(
            observation_sample)
        return emission_probability

    # Convergence check
    def has_it_converged(self, iteration):
        # Obviously, not enough data to check for convergence
        if iteration == 1:
            return False
        for key, value in self.transition_probabilities.items():
            # I get 0: dict() and 1: dict()
            for k, v in value.items():
                # I get 0: list() and 1: list()
                if abs(v[iteration - 1] - v[iteration - 2]) > self.epsilon:
                    return False
        # All have converged
        return True

    # Construct the transition probabilities matrix
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
            observation_vector[0]) * self.start_probabilities.occupied
        # Idle
        _forward_boundary_condition_idle = self.get_emission_probabilities(
            OccupancyState.IDLE,
            observation_vector[0]) * self.start_probabilities.idle
        _temp_forward_probabilities_dict = {0: _forward_boundary_condition_idle,
                                            1: _forward_boundary_condition_occupied}
        self.forward_probabilities[0] = _temp_forward_probabilities_dict
        # j
        for channel_index in range(1, self.NUMBER_OF_FREQUENCY_BANDS):
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
                                                          observation_vector[self.NUMBER_OF_FREQUENCY_BANDS - 1])
        _backward_boundary_condition_occupied = _state_sum
        # Idle
        _state_sum = 0
        # Outside summation - refer to the definition of backward probability
        for _state in OccupancyState:
            _state_sum += self.transition_probabilities[0][_state.value][iteration - 1] * \
                          self.get_emission_probabilities(_state,
                                                          observation_vector[self.NUMBER_OF_FREQUENCY_BANDS - 1])
        _backward_boundary_condition_idle = _state_sum
        _temp_backward_probabilities_dict = {0: _backward_boundary_condition_idle,
                                             1: _backward_boundary_condition_occupied}
        self.backward_probabilities[self.NUMBER_OF_FREQUENCY_BANDS - 1] = _temp_backward_probabilities_dict
        # j
        for channel_index in range(self.NUMBER_OF_FREQUENCY_BANDS - 2, -1, -1):
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

    # Get the numerator of the fraction in the algorithm (refer to the document for more information)
    # The "previous_state" and "next_state" args are instances of the OccupancyState enumeration.
    def get_numerator(self, previous_state, next_state, iteration, observation_vector):
        numerator_sum = self.get_emission_probabilities(next_state, observation_vector[0]) * \
            self.transition_probabilities[previous_state.value][next_state.value][iteration - 1] * \
            self.backward_probabilities[1][next_state.value]
        for spatial_index in range(1, self.NUMBER_OF_FREQUENCY_BANDS - 1):
            numerator_sum += self.forward_probabilities[spatial_index - 1][previous_state.value] * \
                self.get_emission_probabilities(next_state, observation_vector[spatial_index]) * \
                self.transition_probabilities[previous_state.value][next_state.value][iteration - 1] * \
                self.backward_probabilities[spatial_index + 1][next_state.value]
        numerator_sum += self.forward_probabilities[self.NUMBER_OF_FREQUENCY_BANDS - 2][previous_state.value] * \
            self.get_emission_probabilities(next_state, observation_vector[self.NUMBER_OF_FREQUENCY_BANDS - 1]) * \
            self.transition_probabilities[previous_state.value][next_state.value][iteration - 1]
        return numerator_sum

    # Get the denominator of the fraction in the algorithm (refer to the document for more information)
    # The "previous_state" arg is an instance of the OccupancyState enumeration
    def get_denominator(self, previous_state, iteration, observation_vector):
        denominator_sum = 0
        for nxt_state in OccupancyState:
            denominator_sum_internal = self.get_emission_probabilities(nxt_state, observation_vector[0]) * \
                self.transition_probabilities[previous_state.value][nxt_state.value][iteration - 1] * \
                self.backward_probabilities[1][nxt_state.value]
            for _spatial_index in range(1, self.NUMBER_OF_FREQUENCY_BANDS - 1):
                denominator_sum_internal += self.forward_probabilities[_spatial_index - 1][previous_state.value] * \
                    self.get_emission_probabilities(nxt_state, observation_vector[_spatial_index]) * \
                    self.transition_probabilities[previous_state.value][nxt_state.value][iteration - 1] * \
                    self.backward_probabilities[_spatial_index + 1][nxt_state.value]
            denominator_sum_internal += \
                self.forward_probabilities[self.NUMBER_OF_FREQUENCY_BANDS - 2][previous_state.value] \
                * self.get_emission_probabilities(nxt_state, observation_vector[self.NUMBER_OF_FREQUENCY_BANDS - 1]) \
                * self.transition_probabilities[previous_state.value][nxt_state.value][iteration - 1]
            denominator_sum += denominator_sum_internal
        return denominator_sum

    # A controlled reset of all the collections for the next set of observations
    def controlled_reset(self):
        # Controlled reset of forward probabilities
        self.forward_probabilities = []
        for k in range(0, self.NUMBER_OF_FREQUENCY_BANDS):
            self.forward_probabilities.append(dict())
        # Controlled reset of backward probabilities
        self.backward_probabilities = []
        for k in range(0, self.NUMBER_OF_FREQUENCY_BANDS):
            self.backward_probabilities.append(dict())
        # Controlled reset of the transition probabilities matrix
        for key, value in self.transition_probabilities.items():
            for k, v in value.items():
                _new_list = list()
                _new_list.append(v[0])
                self.transition_probabilities[key][k] = _new_list

    # Get the converged transition probabilities matrix
    def get_converged_transition_matrix(self):
        _converged_transition_matrix = {0: dict(), 1: dict()}
        for key, value in self.transition_probabilities.items():
            for k, v in value.items():
                _converged_transition_matrix[key][k] = v[len(v) - 1]
        return _converged_transition_matrix

    # Core method
    # Estimate the Markov Chain State Transition Probabilities Matrix
    def estimate_parameters(self):
        final_estimated_parameters = {0: dict(), 1: dict()}
        collection_of_estimates = []
        mean_square_error_across_cycles = []
        max_number_of_iterations = 0
        for cycle in range(0, self.NUMBER_OF_CYCLES):
            # Confidence variable to determine when the algorithm has converged
            confidence = 0
            # Iteration counter
            iteration = 1
            # Mean-square error
            mean_square_error_across_iterations = list()
            # Until convergence
            while self.has_it_converged(iteration) is False or confidence < self.CONFIDENCE_BOUND:
                mean_square_error_across_iterations.append(
                    numpy.square((self.true_p_value - self.transition_probabilities[0][1][iteration - 1])))
                # A confidence check
                convergence = self.has_it_converged(iteration)
                if convergence is True:
                    # It has converged. Doing this to ensure that the convergence is permanent...
                    confidence += 1
                else:
                    confidence = 0
                # Numerators collection
                numerators_collection = {0: {0: list(), 1: list()}, 1: {0: list(), 1: list()}}
                # Denominators collection
                denominators_collection = {0: {0: list(), 1: list()}, 1: {0: list(), 1: list()}}
                # Let us first construct a reduced observation vector like we did for the Viterbi algorithm
                for _sampling_round in range(0, self.NUMBER_OF_SAMPLES):
                    observation_vector = []
                    for _channel in range(0, self.NUMBER_OF_FREQUENCY_BANDS):
                        observation_vector.append(self.observation_samples[_channel][_sampling_round])
                    # Fill up the remaining elements of the forward probabilities array
                    self.fill_up_forward_probabilities(iteration, observation_vector)
                    # Fill up the remaining elements of the backward probabilities array
                    self.fill_up_backward_probabilities(iteration, observation_vector)
                    # State r in {0, 1}
                    # Total 4 combinations arise from this double loop
                    for previous_state in OccupancyState:
                        # State l in {0, 1}
                        for next_state in OccupancyState:
                            numerators_collection[previous_state.value][next_state.value].append(self.get_numerator(
                                previous_state, next_state, iteration, observation_vector))
                            denominators_collection[previous_state.value][next_state.value].append(self.get_denominator(
                                previous_state, iteration, observation_vector))
                for previous_state in OccupancyState:
                    for next_state in OccupancyState:
                        numerator_sum = 0
                        denominator_sum = 0
                        for numerator in numerators_collection[previous_state.value][next_state.value]:
                            numerator_sum += numerator
                        for denominator in denominators_collection[previous_state.value][next_state.value]:
                            denominator_sum += denominator
                        self.transition_probabilities[previous_state.value][next_state.value].append(
                            numerator_sum/denominator_sum)
                iteration += 1
            if iteration > max_number_of_iterations:
                max_number_of_iterations = iteration
            collection_of_estimates.append(self.get_converged_transition_matrix())
            mean_square_error_across_cycles.append(mean_square_error_across_iterations)
            self.controlled_reset()
        # Padding is done here to make sure the output arrays across different cycles are of the same dimension...
        for entry in mean_square_error_across_cycles:
            converged_value = entry[len(entry) - 1]
            for pad_index in range(len(entry), max_number_of_iterations):
                entry.append(converged_value)
        x_axis = [k for k in range(0, max_number_of_iterations)]
        y_axis = []
        for _index in range(0, max_number_of_iterations):
            averaging_sum = 0
            for entry in mean_square_error_across_cycles:
                averaging_sum += entry[_index]
            y_axis.append(averaging_sum/self.NUMBER_OF_CYCLES)
        # The data trace
        visualization_trace = go.Scatter(x=x_axis,
                                         y=y_axis,
                                         mode='lines+markers')
        # The figure layout
        visualization_layout = dict(title='Mean Square Error Convergence of the Markov Chain Parameter Estimation '
                                          'Algorithm for a Static PU with Complete Information',
                                    xaxis=dict(title='Number of Iterations (x300 observation vectors)'),
                                    yaxis=dict(
                                        type='log',
                                        autorange=True,
                                        title=r'Mean Square Error - $\mathbb{E}\ [(p\ -\ \hat{p})^2]$')
                                    )
        # The figure
        visualization_figure = dict(data=[visualization_trace],
                                    layout=visualization_layout)
        # The figure URL
        figure_url = plotly.plotly.plot(visualization_figure,
                                        filename='Mean_Square_Error_Convergence_Parameter_Estimation_Algorithm')
        # Print the URL in case you're on an environment where a GUI is not available
        print('[INFO] MarkovChainParameterEstimator estimate_parameters: The visualization figure is available at: '
              '{}'.format(figure_url))
        for previous_state in OccupancyState:
            for next_state in OccupancyState:
                sum_for_averaging = 0
                for estimate in collection_of_estimates:
                    sum_for_averaging += estimate[previous_state.value][next_state.value]
                final_estimated_parameters[previous_state.value][next_state.value] = sum_for_averaging / \
                    self.NUMBER_OF_CYCLES
        return final_estimated_parameters

    # The termination sequence
    def __exit__(self, exc_type, exc_val, exc_tb):
        print('[INFO] MarkovChainParameterEstimator Termination: Tearing things down...')
        # Nothing to do...


# Run Trigger
if __name__ == '__main__':
    print('[INFO] MarkovChainParameterEstimator main: Creating instance and beginning the process of parameter '
          'estimation')
    # Create the instance
    markovChainParameterEstimator = MarkovChainParameterEstimator()
    # Steady State Probability \mathbb{P}(Occupied) = \mathbb{P}(X_k = 1)
    pi = markovChainParameterEstimator.start_probabilities.occupied
    # \mathbb{P}(Occupied | Idle)
    p = 0.30
    markovChainParameterEstimator.true_p_value = p
    # \mathbb{P}(Idle|Occupied)
    q = (p * (1 - pi)) / pi
    # Actual State Transition Probabilities Matrix
    actual_transition_probabilities_matrix = {
        0: {0: (1 - p), 1: p},
        1: {0: q, 1: (1 - q)}
    }
    # Generate the true PU Occupancy states based on the Markov chain parameters described above
    markovChainParameterEstimator.allocate_true_pu_occupancy_states(p, q, pi)
    # Allocate observations based on the given System Model and Observation Model
    markovChainParameterEstimator.allocate_observations()
    # Before, we go ahead and estimate the state transition probabilities matrix, let's set them to some initial values
    # \mathbb{P}(Occupied|Idle) initial assumption
    p_initial = 1e-8
    markovChainParameterEstimator.p_initial = p_initial
    # \mathbb{P}(Idle|Occupied) initial assumption
    q_initial = (p_initial * (1 - pi)) / pi
    # Set the initial values to the transition probabilities matrix
    markovChainParameterEstimator.transition_probabilities[0][0].append(1 - p_initial)
    markovChainParameterEstimator.transition_probabilities[0][1].append(p_initial)
    markovChainParameterEstimator.transition_probabilities[1][0].append(q_initial)
    markovChainParameterEstimator.transition_probabilities[1][1].append(1 - q_initial)
    # Estimate the parameters of the Markov chain
    try:
        estimated_markov_chain_parameters = markovChainParameterEstimator.estimate_parameters()
        print('[INFO] MarkovChainParameterEstimator main: Estimated Transition Probabilities Matrix is - ',
              str(estimated_markov_chain_parameters))
    except Exception as e:
        print('[ERROR] MarkovChainParameterEstimation main: Exception caught in the core method - [', e, ']')
