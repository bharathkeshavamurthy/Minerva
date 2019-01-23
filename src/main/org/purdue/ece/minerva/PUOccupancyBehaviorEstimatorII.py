# PU Occupancy Behavior Estimation
# Second Iteration: Incomplete information (Missing observations) and Markovian across frequency
# Second run of Prof. Michelusi's suggestion of obtaining detection accuracies of channels which haven't been sensed
# Author: Bharath Keshavamurthy
# Organization: School of Electrical and Computer Engineering, Purdue University
# Copyright (c) 2019. All Rights Reserved.

# For the math behind this algorithm, please refer to:
# This url may change - Please contact the author at <bkeshava@purdue.edu> for more details.
# https://github.rcac.purdue.edu/bkeshava/Minerva/tree/master/latex

from enum import Enum
import numpy
import scipy.stats
from matplotlib import pyplot as plt
from collections import namedtuple
import ChannelSelectionStrategyGenerator
import os


# Occupancy state enumeration
class OccupancyState(Enum):
    # Channel Idle
    idle = 0
    # Channel Occupied
    occupied = 1


# Main class: PU Occupancy Behavior Estimation
class PUOccupancyBehaviorEstimatorII(object):
    # Number of samples for this simulation
    NUMBER_OF_SAMPLES = 500

    # Variance of the Additive White Gaussian Noise Samples
    VARIANCE_OF_AWGN = 1

    # Variance of the Channel Impulse Response which is a zero mean Gaussian
    # SNR = 20 dB
    # SNR = 10log_10(100/1) = 10 *2 = 20 dB
    VARIANCE_OF_CHANNEL_IMPULSE_RESPONSE = 100

    # Number of frequency bands/channels in the wideband spectrum of interest
    NUMBER_OF_FREQUENCY_BANDS = 18

    # Channels in which the PU Occupancy is measured - the remaining channels are not sensed (Energy Efficiency metric)
    BANDS_OBSERVED = ()

    # Empty observation place holder value
    EMPTY_OBSERVATION_PLACEHOLDER_VALUE = 0

    # Start probabilities of PU occupancy per frequency band
    BAND_START_PROBABILITIES = namedtuple('BandStartProbabilities', ['idle', 'occupied'])

    # Occupancy States (IDLE, OCCUPIED)
    OCCUPANCY_STATES = (OccupancyState.idle, OccupancyState.occupied)

    # Value function named tuple
    VALUE_FUNCTION_NAMED_TUPLE = namedtuple('ValueFunction', ['current_value', 'previous_state'])

    # Number of trials to smoothen the Detection Accuracy v/s P(1|0) curve
    NUMBER_OF_CYCLES = 100

    # Initialization Sequence
    def __init__(self):
        print('[INFO] PUOccupancyBehaviorEstimatorII Initialization: Bringing things up ...')
        # AWGN samples
        self.noise_samples = {}
        # Channel Impulse Response samples
        self.channel_impulse_response_samples = {}
        # True PU Occupancy state
        self.true_pu_occupancy_states = []
        # The parameters of observations of all the bands are stored in this dict here
        self.observations_parameters = {}
        # The observed samples at the SU receiver
        self.observation_samples = []
        # The start probabilities
        self.start_probabilities = self.BAND_START_PROBABILITIES(occupied=0.6, idle=0.4)
        # The complete state transition matrix is defined as a Python dictionary taking in named-tuples
        self.transition_probabilities_matrix = {}

    # Generate the true states using the Markovian model
    # Arguments: p -> P(1|0); q -> P(0|1); and pi -> P(1)
    def allocate_true_pu_occupancy_states(self, p_val, q_val, pi_val):
        previous = 1
        # Initial state generation -> band-0 using pi_val
        if numpy.random.random_sample() > pi_val:
            previous = 0
        self.true_pu_occupancy_states.append(previous)
        # Based on the state of band-0 and the (p_val,q_val) values, generate the states of the remaining bands
        for loop_counter in range(1, self.NUMBER_OF_FREQUENCY_BANDS):
            seed = numpy.random.random_sample()
            if previous == 1 and seed < q_val:
                previous = 0
            elif previous == 1 and seed > q_val:
                previous = 1
            elif previous == 0 and seed < p_val:
                previous = 1
            else:
                previous = 0
            self.true_pu_occupancy_states.append(previous)

    # Get the observations vector
    # Generate the observations of all the bands for a number of observation rounds or cycles
    def allocate_observations(self):
        # Each frequency band is observed ${NUMBER_OF_SAMPLES} times.
        # The sampling of each frequency band involves a noise sample corresponding to that SU receiver
        # ... and a channel impulse response sample between that SU receiver and the radio environment
        for frequency_band in range(0, self.NUMBER_OF_FREQUENCY_BANDS):
            mu_noise, std_noise = 0, numpy.sqrt(self.VARIANCE_OF_AWGN)
            # The Re and Im parts of the noise samples are IID and distributed as N(0,\sigma_V^2/2)
            real_noise_samples = numpy.random.normal(mu_noise, (std_noise / numpy.sqrt(2)), self.NUMBER_OF_SAMPLES)
            img_noise_samples = numpy.random.normal(mu_noise, (std_noise / numpy.sqrt(2)), self.NUMBER_OF_SAMPLES)
            self.noise_samples[frequency_band] = real_noise_samples + (1j * img_noise_samples)
            mu_channel_impulse_response, std_channel_impulse_response = 0, numpy.sqrt(
                self.VARIANCE_OF_CHANNEL_IMPULSE_RESPONSE)
            # The Re and Im parts of the channel impulse response samples are IID and distributed as N(0,\sigma_H^2/2)
            real_channel_impulse_response_samples = numpy.random.normal(mu_channel_impulse_response,
                                                                        (std_channel_impulse_response / numpy.sqrt(2)),
                                                                        self.NUMBER_OF_SAMPLES)
            img_channel_impulse_response_samples = numpy.random.normal(mu_channel_impulse_response,
                                                                       (std_channel_impulse_response / numpy.sqrt(2)),
                                                                       self.NUMBER_OF_SAMPLES)
            self.channel_impulse_response_samples[frequency_band] = real_channel_impulse_response_samples + (
                    1j * img_channel_impulse_response_samples)
        # Re-arranging the vectors
        for band in range(0, self.NUMBER_OF_FREQUENCY_BANDS):
            # Create an empty collection object
            obs_per_band = list()
            # If the channel is sensed by the SU, extract the observations
            if band in self.BANDS_OBSERVED:
                for count in range(0, self.NUMBER_OF_SAMPLES):
                    obs_per_band.append(self.channel_impulse_response_samples[band][
                                            count] * self.true_pu_occupancy_states[band] + self.noise_samples[
                                            band][count])
            self.observation_samples.append(obs_per_band)
        return self.observation_samples

    # Get the start probabilities from the named tuple - a simple getter utility method exclusive to this class
    def get_start_probabilities(self, state_name):
        if state_name == 'occupied':
            return self.start_probabilities.occupied
        else:
            return self.start_probabilities.idle

    # Return the transition probabilities from the transition probabilities matrix
    def get_transition_probabilities(self, row, column):
        return self.transition_probabilities_matrix[row][column]

    # Get the Emission Probabilities -> P(y|x)
    def get_emission_probabilities(self, state, observation_sample):
        # If the channel is not observed, i.e. if the observation is [phi] or [0], report m_r(y_i) as 1
        if observation_sample == self.EMPTY_OBSERVATION_PLACEHOLDER_VALUE:
            return 1
        # Normal Viterbi code
        else:
            # Idea: P(A, B|C) = P(A|B, C)P(B|C) = P(A|C)P(B|C) because the real and imaginary components are independent
            # P(\Re(y_k(i)), \Im(y_k(i))|x_k) = P(\Re(y_k(i))|x_k)P(\Im(y_k(i)|x_k)
            # The emission probability from the real component
            emission_real_gaussian_component = scipy.stats.norm(0, numpy.sqrt(
                ((self.VARIANCE_OF_CHANNEL_IMPULSE_RESPONSE / 2) * state) + (self.VARIANCE_OF_AWGN / 2))).pdf(
                observation_sample.real)
            # The emission probability from the imaginary component
            emission_imaginary_gaussian_component = scipy.stats.norm(0, numpy.sqrt(
                ((self.VARIANCE_OF_CHANNEL_IMPULSE_RESPONSE / 2) * state) + (self.VARIANCE_OF_AWGN / 2))).pdf(
                observation_sample.imag)
            return emission_real_gaussian_component * emission_imaginary_gaussian_component

    # Evaluate the Estimation Accuracy
    # P(\hat{X_k} = x_k | X_k = x_k) \forall k \in \{0, 1, 2, ...., K\} and x_k \in \{0, 1\}
    # Relative Frequency approach to estimate this parameter
    def get_estimation_accuracy(self, _input, estimated_states):
        accuracies = 0
        total = 0
        for _channel_index in _input:
            total += 1
            if estimated_states[_channel_index] == self.true_pu_occupancy_states[_channel_index]:
                accuracies += 1
        return accuracies / total

    # Evaluate the Probability of Detection P_D
    # P(\hat{X_k} = 1 | X_k = 1) \forall k \in \{0, 1, 2, ...., K\}
    # Relative Frequency approach to estimate this parameter
    def get_detection_probability(self, _input, estimated_states):
        occupancies = 0
        correct_detections = 0
        for _channel_index in _input:
            if self.true_pu_occupancy_states[_channel_index] == 1:
                occupancies += 1
                if estimated_states[_channel_index] == self.true_pu_occupancy_states[_channel_index]:
                    correct_detections += 1
        if occupancies == 0:
            return 1
        return correct_detections / occupancies

    # Evaluate the Probability of False Alarm P_FA
    # P(\hat{X_k} = 1 | X_k = 0) \forall k \in \{0, 1, 2, ...., K\}
    # Relative Frequency approach to estimate this parameter
    def get_false_alarm_probability(self, _input, estimated_states):
        idle_count = 0
        false_alarms = 0
        for _channel_index in _input:
            if self.true_pu_occupancy_states[_channel_index] == 0:
                idle_count += 1
                if estimated_states[_channel_index] == 1:
                    false_alarms += 1
        if idle_count == 0:
            return 0
        return false_alarms / idle_count

    # Evaluate the Probability of Missed Detection P_MD
    # P(\hat{X_k} = 1 | X_k = 0) \forall k \in \{0, 1, 2, ...., K\}
    # Relative Frequency approach to estimate this parameter
    def get_missed_detection_probability(self, _input, estimated_states):
        occupancies = 0
        missed_detections = 0
        for _channel_index in _input:
            if self.true_pu_occupancy_states[_channel_index] == 1:
                occupancies += 1
                if estimated_states[_channel_index] == 0:
                    missed_detections += 1
        if occupancies == 0:
            return 0
        return missed_detections / occupancies

    # Output the estimated state of the frequency bands in the wideband spectrum of interest
    # Output a collection consisting of the required parameters of interest
    def estimate_pu_occupancy_states(self, _parameter_evaluation_input):
        estimated_states_array = []
        for sampling_round in range(0, self.NUMBER_OF_SAMPLES):
            estimated_states = []
            reduced_observation_vector = []
            for entry in self.observation_samples:
                reduced_observation_vector.append(self.get_entry(entry, sampling_round))
            # Now, I have to estimate the state of the ${NUMBER_OF_FREQUENCY_BANDS} based on
            # ...this reduced observation vector
            # INITIALIZATION : The array of initial probabilities is known
            # FORWARD RECURSION
            value_function_collection = [dict() for x in range(len(reduced_observation_vector))]
            for state in OccupancyState:
                current_value = self.get_emission_probabilities(state.value, reduced_observation_vector[0]) * \
                                self.get_start_probabilities(state.name)
                value_function_collection[0][state.name] = self.VALUE_FUNCTION_NAMED_TUPLE(
                    current_value=current_value,
                    previous_state=None)
            # For each observation after the first one (I can't apply Markovian to [0])
            for observation_index in range(1, len(reduced_observation_vector)):
                # Trying to find the max pointer here ...
                for state in OccupancyState:
                    # Again finishing off the [0] index first
                    max_pointer = self.get_transition_probabilities(OccupancyState.idle.value, state.value) * \
                                  value_function_collection[observation_index - 1][
                                      OccupancyState.idle.name].current_value
                    confirmed_previous_state = OccupancyState.idle.name
                    for candidate_previous_state in OccupancyState:
                        if candidate_previous_state.name == OccupancyState.idle.name:
                            # Already done
                            continue
                        else:
                            pointer = self.get_transition_probabilities(candidate_previous_state.value,
                                                                        state.value) * \
                                      value_function_collection[observation_index - 1][
                                          candidate_previous_state.name].current_value
                            if pointer > max_pointer:
                                max_pointer = pointer
                                confirmed_previous_state = candidate_previous_state.name
                    current_value = max_pointer * self.get_emission_probabilities(state.value,
                                                                                  reduced_observation_vector[
                                                                                      observation_index])
                    value_function_collection[observation_index][state.name] = self.VALUE_FUNCTION_NAMED_TUPLE(
                        current_value=current_value, previous_state=confirmed_previous_state)
            max_value = 0
            # Finding the max value among the named tuples
            for value in value_function_collection[-1].values():
                if value.current_value > max_value:
                    max_value = value.current_value
            # Finding the state corresponding to this max_value and using this as the final confirmed state to ...
            # ...backtrack and find the previous states
            for k, v in value_function_collection[-1].items():
                if v.current_value == max_value:
                    estimated_states.append(self.value_from_name(k))
                    previous_state = k
                    break
            # Backtracking
            for i in range(len(value_function_collection) - 2, -1, -1):
                estimated_states.insert(0, self.value_from_name(
                    value_function_collection[i + 1][previous_state].previous_state))
                previous_state = value_function_collection[i + 1][previous_state].previous_state
            estimated_states_array.append(estimated_states)
        estimation_accuracies = [
            self.get_estimation_accuracy(_parameter_evaluation_input, estimated_states_per_iteration) for
            estimated_states_per_iteration in estimated_states_array]
        detection_probabilities = [
            self.get_detection_probability(_parameter_evaluation_input, estimated_states_per_iteration) for
            estimated_states_per_iteration in estimated_states_array]
        false_alarm_probabilities = [
            self.get_false_alarm_probability(_parameter_evaluation_input, estimated_states_per_iteration) for
            estimated_states_per_iteration in estimated_states_array]
        missed_detection_probabilities = [
            self.get_missed_detection_probability(_parameter_evaluation_input, estimated_states_per_iteration) for
            estimated_states_per_iteration in estimated_states_array]
        # Output that would be returned by this core method
        output_dict = dict()
        output_dict['Estimation_Accuracy'] = self._average(estimation_accuracies, self.NUMBER_OF_SAMPLES)
        output_dict['Detection_Probability'] = self._average(detection_probabilities, self.NUMBER_OF_SAMPLES)
        output_dict['False_Alarm_Probability'] = self._average(false_alarm_probabilities, self.NUMBER_OF_SAMPLES)
        output_dict['Missed_Detection_Probability'] = self._average(missed_detection_probabilities,
                                                                    self.NUMBER_OF_SAMPLES)
        return output_dict

    @staticmethod
    # A static utility method for averaging
    def _average(collection, number_of_samples):
        sum_for_average = 0
        for entry_in_collection in collection:
            sum_for_average += entry_in_collection
        return sum_for_average / number_of_samples

    # Safe entry access using indices from a collection object
    def get_entry(self, collection, index):
        if len(collection) is not 0 and collection[index] is not None:
            return collection[index]
        else:
            return self.EMPTY_OBSERVATION_PLACEHOLDER_VALUE

    # Get enumeration field value from name
    @staticmethod
    def value_from_name(name):
        if name == 'occupied':
            return OccupancyState.occupied.value
        else:
            return OccupancyState.idle.value

    # Get un-sensed channels from the sensed channels input
    # In other words, find the complement
    def get_complement(self, sensed_channels):
        un_sensed_channels = list()
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

    # Exit strategy
    def __exit__(self, exc_type, exc_val, exc_tb):
        print('[INFO] PUOccupancyBehaviorEstimatorII Clean-up: Cleaning things up ...')


# Cyclic average evaluation
def cyclic_average(collection, number_of_internal_collections, number_of_cycles):
    collection_for_plotting = []
    for _pass_counter in range(0, number_of_internal_collections):
        _average_sum = 0
        for _collection_in_entry in collection:
            _average_sum += _collection_in_entry[_pass_counter]
        collection_for_plotting.append(_average_sum / number_of_cycles)
    return collection_for_plotting


# Visualize a plot of Estimation Accuracies v/s P(Occupied|Idle)
def visualize_estimation_accuracy_plot(estimation_axes, p_estimation_x_axis,
                                       estimation_accuracies_for_plotting, _color, _label):
    estimation_axes.plot(p_estimation_x_axis, estimation_accuracies_for_plotting, linestyle='--', linewidth=1.0,
                         marker='o', color=_color, label=_label)


# Visualize a plot of False Alarm Probabilities v/s P(Occupied|Idle)
def visualize_false_alarm_probability_plot(false_alarm_axes, p_false_alarm_x_axis,
                                           false_alarm_probabilities_y_axis, _color, _label):
    false_alarm_axes.plot(p_false_alarm_x_axis, false_alarm_probabilities_y_axis, linestyle='--', linewidth=1.0,
                          marker='o', color=_color, label=_label)


# Visualize a plot of Missed Detection Probabilities v/s P(Occupied|Idle)
def visualize_missed_detection_probability(missed_detection_axes, p_missed_detection_x_axis, missed_detection_y_axis,
                                           _color, _label):
    missed_detection_axes.plot(p_missed_detection_x_axis, missed_detection_y_axis, linestyle='--',
                               linewidth=1.0, marker='o', color=_color, label=_label)


# Add metadata to the plots and save them to files in the output folder
def add_metadata_to_plots(_collection_of_plots, _metadata, output_folder):
    for collection_index in range(len(_collection_of_plots) - 1, -1, -1):
        _collection_of_plots[collection_index][0].suptitle(_metadata[collection_index][0], fontsize=6.5)
        _collection_of_plots[collection_index][1].set_xlabel(_metadata[collection_index][1], fontsize=8)
        _collection_of_plots[collection_index][1].set_ylabel(_metadata[collection_index][2], fontsize=8)
        plt.legend(loc='upper right', prop={'size': 7})
        _collection_of_plots[collection_index][0].savefig(output_folder + _metadata[collection_index][3] + '.png')
        plt.close(_collection_of_plots[collection_index][0])


# Run Trigger
if __name__ == '__main__':
    # Colors tuple for differentiation in the visualized results
    colors = ('b', 'r')
    # Clear the output folder
    path_to_output_folder = '../../../../../../test/Static_PU_Channel_Sensing_Strategy_Plots/Uniform_Sensing/'
    file_list = [f for f in os.listdir(path_to_output_folder) if f.endswith('.png')]
    for f in file_list:
        os.remove(os.path.join(path_to_output_folder, f))
    # Variety in channel selection for sensing - This'll be given by the Bandit
    # Emulating this for now using the ChannelSelectionStrategyGenerator class
    channel_selection_strategy_generator = ChannelSelectionStrategyGenerator.ChannelSelectionStrategyGenerator()
    puOccupancyBehaviorEstimator = PUOccupancyBehaviorEstimatorII()
    # Get the channel selection strategies
    channel_selection_strategies = channel_selection_strategy_generator.generic_uniform_sensing(
        puOccupancyBehaviorEstimator.NUMBER_OF_FREQUENCY_BANDS)
    strategy_counter = 0
    # Iterate over multiple channel selection strategies provided by the emulator or the bandit
    for channel_selection_strategy in channel_selection_strategies:
        # Protection - Don't go ahead with empty or complete strategies
        if len(channel_selection_strategy) == puOccupancyBehaviorEstimator.NUMBER_OF_FREQUENCY_BANDS or len(
                channel_selection_strategy) == 0:
            continue
        # Figures for plotting
        estimation_fig_obj, estimation_ax_obj = plt.subplots()
        false_alarm_fig_obj, false_alarm_ax_obj = plt.subplots()
        missed_detection_fig_obj, missed_detection_ax_obj = plt.subplots()
        # Strategy Counter for title
        strategy_counter += 1
        # Color index to index into the Colors tuple
        color_index = 0
        # Increment p by this value
        increment_value_for_p = 0.03
        # P(1)
        pi = puOccupancyBehaviorEstimator.start_probabilities.occupied
        final_frontier = int(pi / increment_value_for_p)
        # Setting the chosen channel selection strategy
        puOccupancyBehaviorEstimator.BANDS_OBSERVED = channel_selection_strategy
        # 2 takes - sensed channels and un-sensed channels
        for _simple_counter in range(0, 2):
            # X-Axis for the P(1|0) versus Estimation Accuracy plot
            p_values_overall = []
            # Y-Axis for the P(1|0) versus Estimation Accuracy plot
            estimation_accuracies_across_p_values_overall = []
            # Axes for False Alarm Probabilities and Detection Probabilities plots
            detection_probabilities_overall = []
            false_alarm_probabilities_overall = []
            # Axes for Missed Detection Probabilities plot
            missed_detection_probabilities_overall = []
            # Figure out the input for parameter evaluation here
            if _simple_counter == 0:
                parameter_evaluation_input = channel_selection_strategy
            else:
                # Get the complement here
                parameter_evaluation_input = puOccupancyBehaviorEstimator.get_complement(channel_selection_strategy)
            # Multiple iteration cycles to average out inconsistencies
            for iteration_cycle in range(0, puOccupancyBehaviorEstimator.NUMBER_OF_CYCLES):
                # P(1|0) - Let's start small and move towards independence
                p = increment_value_for_p
                p_values = []
                estimation_accuracies_across_p_values = []
                detection_probabilities_internal = []
                false_alarm_probabilities_internal = []
                missed_detection_probabilities_internal = []
                for increment_counter in range(0, final_frontier):
                    p_values.append(p)
                    # P(0|1)
                    q = (p * puOccupancyBehaviorEstimator.start_probabilities.idle) \
                        / puOccupancyBehaviorEstimator.start_probabilities.occupied
                    puOccupancyBehaviorEstimator.transition_probabilities_matrix = {
                        1: {1: (1 - q), 0: q},
                        0: {1: p, 0: (1 - p)}
                    }
                    # True PU Occupancy State
                    puOccupancyBehaviorEstimator.allocate_true_pu_occupancy_states(p, q, pi)
                    puOccupancyBehaviorEstimator.allocate_observations()
                    obtained_collection = puOccupancyBehaviorEstimator.estimate_pu_occupancy_states(
                        parameter_evaluation_input)
                    estimation_accuracies_across_p_values.append(obtained_collection['Estimation_Accuracy'])
                    detection_probabilities_internal.append(obtained_collection['Detection_Probability'])
                    false_alarm_probabilities_internal.append(obtained_collection['False_Alarm_Probability'])
                    missed_detection_probabilities_internal.append(obtained_collection['Missed_Detection_Probability'])
                    p += increment_value_for_p
                    puOccupancyBehaviorEstimator.reset()
                p_values_overall = p_values
                estimation_accuracies_across_p_values_overall.append(estimation_accuracies_across_p_values)
                detection_probabilities_overall.append(detection_probabilities_internal)
                false_alarm_probabilities_overall.append(false_alarm_probabilities_internal)
                missed_detection_probabilities_overall.append(missed_detection_probabilities_internal)
            # Start Plotting
            visualize_estimation_accuracy_plot(estimation_ax_obj, p_values_overall,
                                               cyclic_average(estimation_accuracies_across_p_values_overall,
                                                              final_frontier,
                                                              puOccupancyBehaviorEstimator.NUMBER_OF_CYCLES),
                                               colors[color_index],
                                               'Estimation Accuracy for strategy: ' + str(parameter_evaluation_input))
            visualize_false_alarm_probability_plot(false_alarm_ax_obj, p_values_overall,
                                                   cyclic_average(false_alarm_probabilities_overall,
                                                                  final_frontier,
                                                                  puOccupancyBehaviorEstimator.NUMBER_OF_CYCLES),
                                                   colors[color_index],
                                                   'False Alarm Probability for strategy: ' +
                                                   str(parameter_evaluation_input))
            visualize_missed_detection_probability(missed_detection_ax_obj, p_values_overall,
                                                   cyclic_average(missed_detection_probabilities_overall,
                                                                  final_frontier,
                                                                  puOccupancyBehaviorEstimator.NUMBER_OF_CYCLES),
                                                   colors[color_index],
                                                   'Missed Detection Probability for strategy: ' + str(
                                                       parameter_evaluation_input))
            color_index += 1
            # Reset everything
            puOccupancyBehaviorEstimator.reset()
        collection_of_plots = [(estimation_fig_obj, estimation_ax_obj), (false_alarm_fig_obj, false_alarm_ax_obj),
                               (missed_detection_fig_obj, missed_detection_ax_obj)]
        estimation_accuracy_metadata = [
            'Estimation Accuracy v P(Occupied|Idle) at P(Occupied)=0.6 with Markovian Correlation across '
            'channels and Missing Observations', 'P(Occupied | Idle)', 'Estimation Accuracy',
            'Uniform_Sensing_Estimation_Accuracy_Plot_' + str(strategy_counter)]
        false_alarm_probability_metadata = [
            'False Alarm Probability v P(Occupied|Idle) at P(Occupied)=0.6 with Markovian Correlation across channels '
            'and Missing Observations', 'P(Occupied | Idle)', 'Probability of False Alarm',
            'Uniform_Sensing_False_Alarm_Plot_' + str(strategy_counter)]
        missed_detection_probability_metadata = [
            'Missed Detection Probability v P(Occupied|Idle) at P(Occupied)=0.6 with Markovian Correlation across '
            'channels and Missing Observations', 'P(Occupied | Idle)', 'Probability of Missed Detection',
            'Uniform_Sensing_Missed_Detection_Plot_' + str(strategy_counter)]
        metadata = [estimation_accuracy_metadata, false_alarm_probability_metadata,
                    missed_detection_probability_metadata]
        add_metadata_to_plots(collection_of_plots, metadata, path_to_output_folder)
        # Reset everything
        puOccupancyBehaviorEstimator.reset()
