# PU Occupancy Behavior Estimation
# First iteration: Complete information and Markovian across frequency
# Author: Bharath Keshavamurthy
# Organization: School of Electrical and Computer Engineering, Purdue University
# Copyright (c) 2019. All Rights Reserved.

# For the math behind this algorithm, refer to:
# This url may change - Please contact the author at <bkeshava@purdue.edu> for more details.
# https://github.rcac.purdue.edu/bkeshava/Minerva/blob/master/latex/pdf/PU_Occupancy_Behavior_Estimator_v1_0_0.pdf

import numpy
import plotly
import scipy.stats
from enum import Enum
import plotly.graph_objs as go
from collections import namedtuple

# Plotly user account credentials for visualization
plotly.tools.set_credentials_file(username='bkeshava',
                                  api_key='W2WL5OOxLcgCzf8NNlgl')


# Occupancy state enumeration
class OccupancyState(Enum):
    # Channel Idle
    IDLE = 0
    # Channel Occupied
    OCCUPIED = 1


# Main class: PU Occupancy Behavior Estimation
class PUOccupancyBehaviorEstimator(object):
    # Number of samples for this simulation
    NUMBER_OF_SAMPLES = 1000

    # Variance of the Additive White Gaussian Noise Samples (Zero Mean Gaussian)
    VARIANCE_OF_AWGN = 1

    # Variance of the Channel Impulse Response Samples (Zero Mean Gaussian)
    # SNR = 10log_10(80/1) = 19.03 dB
    VARIANCE_OF_CHANNEL_IMPULSE_RESPONSE = 80

    # Number of frequency bands/channels in the wideband spectrum of interest
    NUMBER_OF_FREQUENCY_BANDS = 18

    # Start probabilities of PU occupancy per frequency band
    BAND_START_PROBABILITIES = namedtuple('BandStartProbabilities', ['idle', 'occupied'])

    # Value function named tuple
    VALUE_FUNCTION_NAMED_TUPLE = namedtuple('ValueFunction', ['current_value', 'previous_state'])

    # Number of trials to smoothen the Estimation Accuracy v/s p = \mathbb{P}(1|0) curve
    NUMBER_OF_CYCLES = 50

    # The initialization sequence
    def __init__(self):
        print('[INFO] PUOccupancyBehaviorEstimator Initialization: Bringing things up ...')
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
        # The state transition probabilities
        self.transition_probabilities_matrix = {}

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
        # Each frequency band is observed ${NUMBER_OF_SAMPLES} times.
        # The sampling of each frequency band involves a noise sample corresponding to that SU receiver
        #   and an impulse response sample for the channel in between that SU receiver and the radio environment
        for frequency_band in range(0, self.NUMBER_OF_FREQUENCY_BANDS):
            mu_noise, std_noise = 0, numpy.sqrt(self.VARIANCE_OF_AWGN)
            self.noise_samples[frequency_band] = numpy.random.normal(mu_noise,
                                                                     std_noise,
                                                                     self.NUMBER_OF_SAMPLES)
            mu_channel_impulse_response, std_channel_impulse_response = 0, numpy.sqrt(
                self.VARIANCE_OF_CHANNEL_IMPULSE_RESPONSE)
            self.channel_impulse_response_samples[frequency_band] = numpy.random.normal(mu_channel_impulse_response,
                                                                                        std_channel_impulse_response,
                                                                                        self.NUMBER_OF_SAMPLES)
        # Making observations according to the defined Observation Model
        for band in range(0, self.NUMBER_OF_FREQUENCY_BANDS):
            obs_per_band = []
            for count in range(0, self.NUMBER_OF_SAMPLES):
                obs_per_band.append((self.channel_impulse_response_samples[band][
                                        count] * self.true_pu_occupancy_states[band]) + self.noise_samples[
                                        band][count])
            self.observation_samples.append(obs_per_band)
        return self.observation_samples

    # Get the start probabilities from the named tuple - a simple getter utility method exclusive to this class
    # The "state" arg is an instance of the OccupancyState enumeration.
    def get_start_probabilities(self, state):
        if state == OccupancyState.OCCUPIED:
            return self.start_probabilities.occupied
        else:
            return self.start_probabilities.idle

    # Return the transition probabilities from the transition probabilities matrix
    # The "row" arg and the "column" arg are instances of the OccupancyState enumeration.
    def get_transition_probabilities(self, row, column):
        return self.transition_probabilities_matrix[row.value][column.value]

    # Get the Emission Probabilities -> \mathbb{P}(y|x)
    # The "state" arg is an instance of the OccupancyState enumeration.
    def get_emission_probabilities(self, state, observation_sample):
        return scipy.stats.norm(0,
                                numpy.sqrt(
                                    (self.VARIANCE_OF_CHANNEL_IMPULSE_RESPONSE * state.value) +
                                    self.VARIANCE_OF_AWGN)
                                ).pdf(observation_sample)

    # Evaluate the Estimation Accuracy
    # \mathbb{P}(\hat{X}_k = x_k | X_k = x_k) \forall k \in \{0, 1, 2, ...., K-1\} and x_k \in \{0, 1\}
    # Relative Frequency approach to estimate this parameter
    def get_estimation_accuracy(self, estimated_states):
        accuracies = 0
        for _counter in range(0, self.NUMBER_OF_FREQUENCY_BANDS):
            if self.true_pu_occupancy_states[_counter] == estimated_states[_counter]:
                accuracies += 1
        return accuracies / self.NUMBER_OF_FREQUENCY_BANDS

    # Output the average estimation accuracy of this Viterbi Algorithm with Complete Observations spread over
    #   thousands of sampling rounds
    def estimate_pu_occupancy_states(self):
        # A near-output member which is to be analyzed for estimator performance
        estimated_states_array = []
        # An internal temporary state member
        previous_state = None
        for sampling_round in range(0, self.NUMBER_OF_SAMPLES):
            estimated_states = []
            reduced_observation_vector = []
            for entry in self.observation_samples:
                reduced_observation_vector.append(entry[sampling_round])
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
                value_function_collection[0][state.name] = self.VALUE_FUNCTION_NAMED_TUPLE(current_value=current_value,
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
                            pointer = self.get_transition_probabilities(candidate_previous_state, state) * \
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
        # Return the average estimation accuracy for this algorithm
        return sum([self.get_estimation_accuracy(estimated_states_per_iteration)
                    for estimated_states_per_iteration in estimated_states_array]) / self.NUMBER_OF_SAMPLES

    # Get enumeration field value from name
    @staticmethod
    def value_from_name(name):
        if name == OccupancyState.OCCUPIED.name:
            return OccupancyState.OCCUPIED.value
        else:
            return OccupancyState.IDLE.value

    # Reset every collection for the next run
    def reset(self):
        self.true_pu_occupancy_states.clear()
        self.transition_probabilities_matrix.clear()
        self.noise_samples.clear()
        self.channel_impulse_response_samples.clear()
        self.observation_samples.clear()

    # The termination sequence
    def __exit__(self, exc_type, exc_val, exc_tb):
        print('[INFO] PUOccupancyBehaviorEstimator Termination: Cleaning things up ...')


# Cyclic average evaluation
def cyclic_average(collection, number_of_internal_collections, number_of_cycles):
    collection_for_plotting = []
    for pass_counter in range(0, number_of_internal_collections):
        _sum = 0
        for _entry in collection:
            _sum += _entry[pass_counter]
        collection_for_plotting.append(_sum / number_of_cycles)
    return collection_for_plotting


# Visualize a plot of Estimation Accuracy v/s \mathbb{P}(Occupied|Idle)
def visualize_estimation_accuracy_plot(x_axis_estimation, y_axis_estimation_accuracies):
    # The visualization trace
    data_trace = go.Scatter(x=x_axis_estimation,
                            y=y_axis_estimation_accuracies,
                            mode='lines+markers')
    # The visualization layout
    figure_layout = dict(title=r'Estimation Accuracies of the Viterbi Algorithm with Complete Observations '
                               r'for varying values of $\mathbb{P}(1|0)$',
                         xaxis=dict(title=r'$\mathbb{P}(1|0)$'),
                         yaxis=dict(title='Estimation Accuracy'))
    # The visualization figure
    figure = dict(data=[data_trace],
                  layout=figure_layout)
    # The figure URL
    figure_url = plotly.plotly.plot(figure,
                                    filename='Estimation_Accuracies_of_Unconstrained_Viterbi_Algorithm')
    # Print the URL in case you're on an environment where a GUI is not available
    print('[INFO] PUOccupancyBehaviorEstimator visualize_estimation_accuracy_plot: '
          'Data Visualization Figure is available at {}'.format(figure_url))


# Run Trigger
if __name__ == '__main__':
    # Increment p by this value
    increment_value_for_p = 0.03
    # X-Axis for the Estimation Accuracy vs \mathbb{P}(1|0) plot
    p_values_overall = []
    # Y-Axis for the Estimation Accuracy vs \mathbb{P}(1|0) plot
    estimation_accuracies_across_p_values_overall = []
    puOccupancyBehaviorEstimator = PUOccupancyBehaviorEstimator()
    # \mathbb{P}(1)
    pi = puOccupancyBehaviorEstimator.start_probabilities.occupied
    final_frontier = int(pi / increment_value_for_p)
    for iteration_cycle in range(0, puOccupancyBehaviorEstimator.NUMBER_OF_CYCLES):
        # \mathbb{P}(1|0) - Let's start small and move towards Independence
        p = increment_value_for_p
        p_values = []
        estimation_accuracies_across_p_values = []
        for increment_counter in range(0, final_frontier):
            p_values.append(p)
            # \mathbb{P}(0|1)
            q = (p * puOccupancyBehaviorEstimator.start_probabilities.idle) \
                / puOccupancyBehaviorEstimator.start_probabilities.occupied
            puOccupancyBehaviorEstimator.transition_probabilities_matrix = {
                0: {0: (1 - p), 1: p},
                1: {0: q, 1: (1 - q)}
            }
            # True PU Occupancy States
            puOccupancyBehaviorEstimator.allocate_true_pu_occupancy_states(p, q, pi)
            puOccupancyBehaviorEstimator.allocate_observations()
            estimation_accuracies_across_p_values.append(puOccupancyBehaviorEstimator.estimate_pu_occupancy_states())
            p += increment_value_for_p
            puOccupancyBehaviorEstimator.reset()
        p_values_overall = p_values
        estimation_accuracies_across_p_values_overall.append(estimation_accuracies_across_p_values)
    # Start Plotting
    visualize_estimation_accuracy_plot(p_values_overall,
                                       cyclic_average(estimation_accuracies_across_p_values_overall,
                                                      final_frontier,
                                                      puOccupancyBehaviorEstimator.NUMBER_OF_CYCLES)
                                       )
