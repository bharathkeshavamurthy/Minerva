# PU Occupancy Behavior Estimation
# Second Iteration: Incomplete information (Missing observations) and Markovian across frequency
# Author: Bharath Keshavamurthy
# Organization: School of Electrical and Computer Engineering, Purdue University
# Copyright (c) 2019. All Rights Reserved.

# For the math behind this algorithm, please refer to:
# This url may change - Please contact the author at <bkeshava@purdue.edu> for more details.
# https://github.rcac.purdue.edu/bkeshava/Minerva/blob/master/latex/pdf/PU_Occupancy_Behavior_Estimator_v1_0_0.pdf

import numpy
import plotly
import warnings
import scipy.stats
from enum import Enum
import plotly.graph_objs as go
from collections import namedtuple

warnings.filterwarnings("ignore",
                        category=DeprecationWarning)

# Plotly user account credentials for visualization
plotly.tools.set_credentials_file(username='bkeshava',
                                  api_key='BEp2EMeaooErdwcIF8Ss')


# Occupancy state enumeration
class OccupancyState(Enum):
    # Channel Idle
    IDLE = 0
    # Channel Occupied
    OCCUPIED = 1


# Main class: PU Occupancy Behavior Estimator II
class PUOccupancyBehaviorEstimatorII(object):
    # Number of samples for this simulation
    NUMBER_OF_SAMPLES = 1000

    # Variance of the Additive White Gaussian Noise samples (Zero Mean Gaussian)
    VARIANCE_OF_AWGN = 1

    # Variance of the Channel Impulse Response samples (Zero Mean Gaussian)
    # SNR = 10log_10(80/1) = 19.03 dB
    VARIANCE_OF_CHANNEL_IMPULSE_RESPONSE = 80

    # Number of frequency bands/channels in the wideband spectrum of interest
    NUMBER_OF_FREQUENCY_BANDS = 18

    # Channels in which the PU Occupancy is measured - the remaining channels are not sensed (Energy Efficiency metric)
    BANDS_OBSERVED = ()

    # Empty observation place holder value
    EMPTY_OBSERVATION_PLACEHOLDER_VALUE = 0

    # Start probabilities of PU occupancy per frequency band
    BAND_START_PROBABILITIES = namedtuple('BandStartProbabilities', ['idle', 'occupied'])

    # Value function named tuple
    VALUE_FUNCTION_NAMED_TUPLE = namedtuple('ValueFunction', ['current_value', 'previous_state'])

    # Number of trials to smoothen the Estimation Accuracy v/s \mathbb{P}(1|0) curve
    NUMBER_OF_CYCLES = 50

    # The initialization sequence
    def __init__(self):
        print('[INFO] PUOccupancyBehaviorEstimatorII Initialization: Bringing things up ...')
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
        # Return the true PU occupancy states in case an external entity needs it...
        return self.true_pu_occupancy_states

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
        # Making observations according to the Observation Model
        for band in range(0, self.NUMBER_OF_FREQUENCY_BANDS):
            # Create an empty collection object
            obs_per_band = []
            # If the channel is sensed by the SU, extract the observations
            if band in self.BANDS_OBSERVED:
                for count in range(0, self.NUMBER_OF_SAMPLES):
                    obs_per_band.append((self.channel_impulse_response_samples[band][
                                            count] * self.true_pu_occupancy_states[band]) + self.noise_samples[
                                            band][count])
            self.observation_samples.append(obs_per_band)
        # Return the observation samples in case an external entity needs it...
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
        return (lambda: scipy.stats.norm(0,
                                         numpy.sqrt(
                                             (self.VARIANCE_OF_CHANNEL_IMPULSE_RESPONSE * state.value) +
                                             self.VARIANCE_OF_AWGN)).pdf(observation_sample),
                lambda: 1)[observation_sample == self.EMPTY_OBSERVATION_PLACEHOLDER_VALUE]()

    # Evaluate the Estimation Accuracy
    # \mathbb{P}(\hat{X}_k = x_k | X_k = x_k) \forall k \in \{0, 1, 2, ...., K - 1\} and x_k \in \{0, 1\}
    # Relative Frequency approach to estimate this parameter
    def get_estimation_accuracy(self, _input, estimated_states):
        accuracies = 0
        for _channel_index in _input:
            if estimated_states[_channel_index] == self.true_pu_occupancy_states[_channel_index]:
                accuracies += 1
        return accuracies / len(_input)

    # Safe entry access using indices from a collection object exclusive to this constrained Viterbi algorithm
    def get_entry(self, collection, index):
        if collection is not None and len(collection) != 0 and collection[index] is not None:
            return collection[index]
        else:
            return self.EMPTY_OBSERVATION_PLACEHOLDER_VALUE

    # Output the average estimation accuracy for this Viterbi Algorithm with Incomplete Observations
    def estimate_pu_occupancy_states(self, _parameter_evaluation_input):
        # A near-output member that holds the estimation accuracies over different sampling rounds
        estimated_states_array = []
        # An internal temporary state housing member
        previous_state = None
        for sampling_round in range(0, self.NUMBER_OF_SAMPLES):
            estimated_states = []
            reduced_observation_vector = []
            for entry in self.observation_samples:
                reduced_observation_vector.append(self.get_entry(entry, sampling_round))
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
                value_function_collection[0][state.name] = self.VALUE_FUNCTION_NAMED_TUPLE(
                    current_value=current_value,
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
                            pointer = self.get_transition_probabilities(candidate_previous_state,
                                                                        state) * \
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
        # Return the average estimation accuracy of this algorithm
        return sum([self.get_estimation_accuracy(_parameter_evaluation_input, estimated_states_per_iteration)
                    for estimated_states_per_iteration in estimated_states_array]) / self.NUMBER_OF_SAMPLES

    # Get enumeration field value from name
    @staticmethod
    def value_from_name(name):
        if name == OccupancyState.OCCUPIED.name:
            return OccupancyState.OCCUPIED.value
        else:
            return OccupancyState.IDLE.value

    # Get un-sensed channels from the sensed channels input
    # In other words, find the complement...
    def get_complement(self, sensed_channels):
        un_sensed_channels = []
        for _channel_index in range(0, self.NUMBER_OF_FREQUENCY_BANDS):
            if _channel_index not in sensed_channels:
                un_sensed_channels.append(_channel_index)
        return un_sensed_channels

    # Reset every collection for the next run
    def reset(self):
        self.true_pu_occupancy_states.clear()
        self.transition_probabilities_matrix.clear()
        self.noise_samples.clear()
        self.channel_impulse_response_samples.clear()
        self.observation_samples.clear()

    # The termination sequence
    def __exit__(self, exc_type, exc_val, exc_tb):
        print('[INFO] PUOccupancyBehaviorEstimatorII Termination: Cleaning things up ...')
        # Nothing to do...


# Emulates a Multi-Armed Bandit or a Reinforcement Learning Agent
# Uniform Sensing Generic - Take in the number of channels as an argument
def generic_uniform_sensing(number_of_channels):
    # Array of tuples with varying k
    channel_selection_strategies_based_on_uniform_sensing = []
    k = 0
    while k < number_of_channels - 1:
        i = 0
        temp_array = []
        while i < number_of_channels:
            temp_array.append(i)
            i = i + k + 1
        channel_selection_strategies_based_on_uniform_sensing.append(temp_array)
        k += 1
    return channel_selection_strategies_based_on_uniform_sensing


# Cyclic average evaluation
def cyclic_average(collection, number_of_internal_collections, number_of_cycles):
    collection_for_plotting = []
    for _pass_counter in range(0, number_of_internal_collections):
        _average_sum = 0
        for _entry in collection:
            _average_sum += _entry[_pass_counter]
        collection_for_plotting.append(_average_sum / number_of_cycles)
    return collection_for_plotting


# Visualize a plot of Estimation Accuracies v/s \mathbb{P}(Occupied|Idle) for different
#   sensing strategies and sub-strategies
def visualize_estimation_accuracies(data_traces):
    # The visualization layout
    figure_layout = dict(title=r'Estimation Accuracies of the Viterbi Algorithm with Incomplete Observations '
                               r'over varying values of $\mathbb{P}(1|0)$ for different sensing strategies',
                         xaxis=dict(title=r'$\mathbb{P}(1|0)$'),
                         yaxis=dict(title='Estimation Accuracy')
                         )
    # The visualization figure
    figure = dict(data=data_traces,
                  layout=figure_layout
                  )
    # The figure URL
    figure_url = plotly.plotly.plot(figure,
                                    filename='Estimation_Accuracies_of_Constrained_Viterbi_Algorithm'
                                    )
    # Print the URL in case you're on an environment where a GUI is not available
    print('[INFO] PUOccupancyBehaviorEstimatorII visualize_estimation_accuracies: '
          'Data Visualization Figure is available at {}'.format(figure_url))


# Run Trigger
if __name__ == '__main__':
    # The traces logged over different strategies and sub-strategies
    traces = []
    # The internal legend naming member
    legend_name = None
    # Create the constrained Viterbi agent
    puOccupancyBehaviorEstimator = PUOccupancyBehaviorEstimatorII()
    # Get the channel selection strategies
    # Three strategies (limiting to three for aesthetic purposes)
    # 0: Sense {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17}
    # 1: Sense {0, 2, 4, 6, 8, 10, 12, 14, 16}
    # 2: Sense {0, 3, 6, 9, 12, 15}
    channel_selection_strategies = generic_uniform_sensing(puOccupancyBehaviorEstimator.NUMBER_OF_FREQUENCY_BANDS)[0:3]
    strategy_counter = 0
    # Iterate over multiple channel selection strategies provided by the emulator or the Bandit / RL-agent
    for channel_selection_strategy in channel_selection_strategies:
        # Increment p by this value
        increment_value_for_p = 0.03
        # \mathbb{P}(1)
        pi = puOccupancyBehaviorEstimator.start_probabilities.occupied
        final_frontier = int(pi / increment_value_for_p)
        # Setting the chosen channel selection strategy
        puOccupancyBehaviorEstimator.BANDS_OBSERVED = channel_selection_strategy
        # 2 takes - sensed channels and un-sensed channels
        for _simple_counter in range(0, 2):
            # There's no need to do un-sensed channel analysis for strategy #0 (there are no un-sensed channels)
            if len(channel_selection_strategy) == 18 and _simple_counter == 1:
                continue
            # X-Axis for the Estimation Accuracy vs \mathbb{P}(1|0) plot
            p_values_overall = []
            # Y-Axis for the Estimation Accuracy vs \mathbb{P}(1|0) plot
            estimation_accuracies_across_p_values_overall = []
            # Figure out the input for parameter evaluation here
            if _simple_counter == 0:
                parameter_evaluation_input = channel_selection_strategy
                legend_name = 'Sensed Channels: {}'.format(str(parameter_evaluation_input))
            else:
                # Get the complement here
                parameter_evaluation_input = puOccupancyBehaviorEstimator.get_complement(channel_selection_strategy)
                legend_name = 'Un-sensed Channels: {}'.format(str(parameter_evaluation_input))
            # Multiple iteration cycles to average out inconsistencies
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
                        0: {0: (1 - p),
                            1: p
                            },
                        1: {0: q,
                            1: (1 - q)
                            }
                    }
                    # True PU Occupancy States
                    puOccupancyBehaviorEstimator.allocate_true_pu_occupancy_states(p, q, pi)
                    puOccupancyBehaviorEstimator.allocate_observations()
                    estimation_accuracies_across_p_values.append(
                        puOccupancyBehaviorEstimator.estimate_pu_occupancy_states(parameter_evaluation_input)
                    )
                    puOccupancyBehaviorEstimator.reset()
                    p += increment_value_for_p
                p_values_overall = p_values
                estimation_accuracies_across_p_values_overall.append(estimation_accuracies_across_p_values)
            traces.append(go.Scatter(x=p_values_overall,
                                     y=cyclic_average(estimation_accuracies_across_p_values_overall,
                                                      final_frontier,
                                                      puOccupancyBehaviorEstimator.NUMBER_OF_CYCLES),
                                     mode='lines+markers',
                                     name=legend_name,
                                     marker=dict(symbol=strategy_counter)))
            # Reset everything
            puOccupancyBehaviorEstimator.reset()
        strategy_counter += 1
        # Reset everything
        puOccupancyBehaviorEstimator.reset()
    # Visualize the estimation accuracy of the Viterbi Algorithm over \mathbb{P}(1|0) for numerous strategies and
    #   sub-strategies
    visualize_estimation_accuracies(traces)
