# PU Occupancy Behavior Estimation
# Third iteration: Complete Information and Markovian across both time and frequency
# Author: Bharath Keshavamurthy
# Organization: School of Electrical and Computer Engineering, Purdue University
# Copyright (c) 2019. All Rights Reserved.

# For the math behind this algorithm, refer to:
# This url may change - Please contact the author at <bkeshava@purdue.edu> for more details.
# https://github.rcac.purdue.edu/bkeshava/Minerva/tree/master/latex

from enum import Enum
import numpy
import scipy.stats
from collections import namedtuple
from matplotlib import pyplot as plt


# Occupancy state enumeration
class OccupancyState(Enum):
    # Channel Idle
    idle = 0
    # Channel Occupied
    occupied = 1


# Main class: PU Occupancy Behavior Estimation
class PUOccupancyBehaviorEstimatorIII(object):
    # Number of samples for this simulation
    # Also referred to as the Number of Sampling Rounds
    NUMBER_OF_SAMPLES = 1000

    # Variance of the Additive White Gaussian Noise Samples
    VARIANCE_OF_AWGN = 1

    # Variance of the Channel Impulse Response which is a zero mean Gaussian
    # SNR = 10log_10(80/1) = 19.03 dB
    VARIANCE_OF_CHANNEL_IMPULSE_RESPONSE = 80

    # Number of frequency bands/channels in the wideband spectrum of interest
    NUMBER_OF_FREQUENCY_BANDS = 18

    # Start probabilities of PU occupancy per frequency band
    BAND_START_PROBABILITIES = namedtuple('BandStartProbabilities', ['idle', 'occupied'])

    # Occupancy States (IDLE, OCCUPIED)
    OCCUPANCY_STATES = (OccupancyState.idle, OccupancyState.occupied)

    # Value function named tuple
    VALUE_FUNCTION_NAMED_TUPLE = namedtuple('ValueFunction',
                                            ['current_value', 'previous_temporal_state', 'previous_spatial_state'])

    # Number of trials to smoothen the Detection Accuracy v/s P(1|0) curve
    # Iterating the estimation over numerous trials to average out the inconsistencies
    NUMBER_OF_CYCLES = 50

    # Initialization Sequence
    def __init__(self):
        print('[INFO] PUOccupancyBehaviorEstimatorIII Initialization: Bringing things up ...')
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
        # This is gonna be same in either dimension because P(X_i = 1) is same in either dimension
        self.start_probabilities = self.BAND_START_PROBABILITIES(occupied=0.6, idle=0.4)
        # The complete state transition matrix is defined as a Python dictionary taking in named-tuples
        # Both spatially and temporally, I'm using the same transition probability matrix
        # TODO: Maybe use different transition probability matrices and see what happens...
        # This will be defined in the Run Trigger call because of the 'p' variations
        self.transition_probabilities_matrix = {}

    # Generate the initial states for k = 0 across time
    def get_initial_states_temporal_variation(self, p_val, q_val, pi_val):
        initial_state_vector = []
        previous = 1
        # Initial state generation -> band-0 at time-0 using pi_val
        if numpy.random.random_sample() > pi_val:
            previous = 0
        initial_state_vector.append(previous)
        # Based on the state of band-0 at time-0 and the (p_val,q_val) values, generate the states of the remaining...
        # ...bands
        for _loop_iterator in range(1, self.NUMBER_OF_SAMPLES):
            seed = numpy.random.random_sample()
            if previous == 1 and seed < q_val:
                previous = 0
            elif previous == 1 and seed > q_val:
                previous = 1
            elif previous == 0 and seed < p_val:
                previous = 1
            else:
                previous = 0
            initial_state_vector.append(previous)
        return initial_state_vector

    # Generate the true states with Markovian across channels and Markovian across time
    # Arguments: p -> P(1|0); q -> P(0|1); and pi -> P(1)
    def generate_true_pu_occupancy_states(self, p_val, q_val, pi_val):
        # True PU Occupancy states collection will be a list of self.NUMBER_OF_CHANNELS rows each row ...
        # ... with self.NUMBER_OF_SAMPLING_ROUNDS columns
        self.true_pu_occupancy_states.append(self.get_initial_states_temporal_variation(p_val, q_val, pi_val))
        # t = 0 and k = 0
        previous_state = self.true_pu_occupancy_states[0][0]
        for channel_index in range(1, self.NUMBER_OF_FREQUENCY_BANDS):
            seed = numpy.random.random_sample()
            if previous_state == 1 and seed < q_val:
                previous_state = 0
            elif previous_state == 1 and seed > q_val:
                previous_state = 1
            elif previous_state == 0 and seed < p_val:
                previous_state = 1
            else:
                previous_state = 0
            self.true_pu_occupancy_states.append([previous_state])
        # Let's fill in the other states
        for channel_index in range(1, self.NUMBER_OF_FREQUENCY_BANDS):
            for round_index in range(1, self.NUMBER_OF_SAMPLES):
                previous_temporal_state = self.true_pu_occupancy_states[channel_index][round_index - 1]
                previous_spatial_state = self.true_pu_occupancy_states[channel_index - 1][round_index]
                probability_occupied_temporal = \
                    self.transition_probabilities_matrix[previous_temporal_state][1]
                probability_occupied_spatial = self.transition_probabilities_matrix[previous_spatial_state][1]
                probability_occupied = (probability_occupied_spatial * probability_occupied_temporal) / pi_val
                seed = numpy.random.random_sample()
                if seed < probability_occupied:
                    previous_state = 1
                else:
                    previous_state = 0
                self.true_pu_occupancy_states[channel_index].append(previous_state)

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
            obs_per_band = list()
            for count in range(0, self.NUMBER_OF_SAMPLES):
                obs_per_band.append(self.channel_impulse_response_samples[band][
                                        count] * self.true_pu_occupancy_states[band][count] + self.noise_samples[
                                        band][count])
            self.observation_samples.append(obs_per_band)
        # The observation_samples member is a kxt matrix
        return self.observation_samples

    # Get the start probabilities from the named tuple - a simple getter utility method exclusive to this class
    def get_start_probabilities(self, state):
        # name or value
        if state == 'occupied' or state == '1':
            return self.start_probabilities.occupied
        else:
            return self.start_probabilities.idle

    # Return the transition probabilities from the transition probabilities matrix - two dimensions
    def get_transition_probabilities(self, temporal_prev_state, spatial_prev_state, current_state):
        return (self.transition_probabilities_matrix[temporal_prev_state][current_state] *
                self.transition_probabilities_matrix[spatial_prev_state][current_state]) / self.get_start_probabilities(
            str(temporal_prev_state))

    # Return the transition probabilities from the transition probabilities matrix - single dimension
    def get_transition_probabilities_single(self, row, column):
        return self.transition_probabilities_matrix[row][column]

    # Get the Emission Probabilities -> P(y|x)
    def get_emission_probabilities(self, state, observation_sample):
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
    # P(\hat{X_k} = x_k | X_k = x_k) \forall k \in \{0, 1, 2, ...., K-1\} and x_k \in \{0, 1\}
    # Relative Frequency approach to estimate this parameter
    def get_estimation_accuracy(self, estimated_states):
        total_count = 0
        accuracies = 0
        for _counter in range(0, self.NUMBER_OF_FREQUENCY_BANDS):
            for _round in range(0, self.NUMBER_OF_SAMPLES):
                total_count += 1
                if self.true_pu_occupancy_states[_counter][_round] == estimated_states[_counter][_round]:
                    accuracies += 1
        return accuracies / total_count

    # Evaluate the Probability of Detection P_D
    # P(\hat{X_k} = 1 | X_k = 1) \forall k \in \{0, 1, 2, ...., K-1\}
    # Relative Frequency approach to estimate this parameter
    def get_probability_of_detection(self, estimated_states):
        occupancies = 0
        number_of_detections = 0
        for channel_index in range(0, self.NUMBER_OF_FREQUENCY_BANDS):
            for time_index in range(0, self.NUMBER_OF_SAMPLES):
                pu_state = self.true_pu_occupancy_states[channel_index][time_index]
                if pu_state == 1:
                    occupancies += 1
                    if estimated_states[channel_index][time_index] == pu_state:
                        number_of_detections += 1
        if occupancies == 0:
            return 1
        return number_of_detections / occupancies

    # Evaluate the Probability of False Alarm P_FA
    # P(\hat{X_k} = 1 | X_k = 0) \forall k \in \{0, 1, 2, ...., K-1\}
    # Relative Frequency approach to estimate this parameter
    def get_probability_of_false_alarm(self, estimated_states):
        idle_count = 0
        number_of_false_alarms = 0
        for channel_index in range(0, self.NUMBER_OF_FREQUENCY_BANDS):
            for time_index in range(0, self.NUMBER_OF_SAMPLES):
                pu_state = self.true_pu_occupancy_states[channel_index][time_index]
                if pu_state == 0:
                    idle_count += 1
                    if estimated_states[channel_index][time_index] == 1:
                        number_of_false_alarms += 1
        if idle_count == 0:
            return 0
        return number_of_false_alarms / idle_count

    # Evaluate the Probability of Missed Detection P_MD
    # P(\hat{X_k} = 0 | X_k = 1) \forall k \in \{0, 1, 2, ...., K-1\}
    # Relative Frequency approach to estimate this parameter
    def get_probability_of_missed_detection(self, estimated_states):
        occupancies = 0
        number_of_missed_detections = 0
        for channel_index in range(0, self.NUMBER_OF_FREQUENCY_BANDS):
            for time_index in range(0, self.NUMBER_OF_SAMPLES):
                pu_state = self.true_pu_occupancy_states[channel_index][time_index]
                if pu_state == 1:
                    occupancies += 1
                    if estimated_states[channel_index][time_index] == 0:
                        number_of_missed_detections += 1
        if occupancies == 0:
            return 0
        return number_of_missed_detections / occupancies

    # Output the estimated state of the frequency bands in the wideband spectrum of interest
    # Output the collection consisting of the parameters of interest
    def estimate_pu_occupancy_states(self):
        # Estimated states - kxt matrix
        estimated_states = [[] for x in range(0, self.NUMBER_OF_FREQUENCY_BANDS)]
        # A value function collection to store and index the calculated value functions across t and k
        value_function_collection = [[dict() for x in range(self.NUMBER_OF_SAMPLES)] for k in
                                     range(0, self.NUMBER_OF_FREQUENCY_BANDS)]
        # t = 0 and k = 0 - No previous state to base the Markovian Correlation on in either dimension
        for state in OccupancyState:
            current_value = self.get_emission_probabilities(state.value, self.observation_samples[0][0]) * \
                            self.get_start_probabilities(state.name)
            value_function_collection[0][0][state.name] = self.VALUE_FUNCTION_NAMED_TUPLE(current_value=current_value,
                                                                                          previous_temporal_state=None,
                                                                                          previous_spatial_state=None)
        # First row - Only temporal correlation
        i = 0
        for j in range(1, self.NUMBER_OF_SAMPLES):
            # Trying to find the max pointer here ...
            for state in OccupancyState:
                # Again finishing off the [0] index first
                max_pointer = self.get_transition_probabilities_single(OccupancyState.idle.value,
                                                                       state.value) * \
                              value_function_collection[i][j - 1][OccupancyState.idle.name].current_value
                # Using IDLE as the confirmed previous state
                confirmed_previous_state = OccupancyState.idle.name
                for candidate_previous_state in OccupancyState:
                    if candidate_previous_state.name == OccupancyState.idle.name:
                        # Already done
                        continue
                    else:
                        pointer = self.get_transition_probabilities_single(candidate_previous_state.value,
                                                                           state.value) * \
                                  value_function_collection[i][j - 1][
                                      candidate_previous_state.name].current_value
                        if pointer > max_pointer:
                            max_pointer = pointer
                            confirmed_previous_state = candidate_previous_state.name
                current_value = max_pointer * self.get_emission_probabilities(state.value,
                                                                              self.observation_samples[
                                                                                  i][j])
                value_function_collection[i][j][state.name] = self.VALUE_FUNCTION_NAMED_TUPLE(
                    current_value=current_value, previous_temporal_state=confirmed_previous_state,
                    previous_spatial_state=None)
        # First column - Only spatial correlation
        j = 0
        for i in range(1, self.NUMBER_OF_FREQUENCY_BANDS):
            # Trying to find the max pointer here ...
            for state in OccupancyState:
                # Again finishing off the [0] index first
                max_pointer = self.get_transition_probabilities_single(OccupancyState.idle.value,
                                                                       state.value) * \
                              value_function_collection[i - 1][j][
                                  OccupancyState.idle.name].current_value
                confirmed_previous_state = OccupancyState.idle.name
                for candidate_previous_state in OccupancyState:
                    if candidate_previous_state.name == OccupancyState.idle.name:
                        # Already done
                        continue
                    else:
                        pointer = self.get_transition_probabilities_single(candidate_previous_state.value,
                                                                           state.value) * \
                                  value_function_collection[i - 1][j][
                                      candidate_previous_state.name].current_value
                        if pointer > max_pointer:
                            max_pointer = pointer
                            confirmed_previous_state = candidate_previous_state.name
                current_value = max_pointer * self.get_emission_probabilities(state.value,
                                                                              self.observation_samples[
                                                                                  i][j])
                value_function_collection[i][j][state.name] = self.VALUE_FUNCTION_NAMED_TUPLE(
                    current_value=current_value, previous_temporal_state=None,
                    previous_spatial_state=confirmed_previous_state)
        # I'm done with the first row and first column
        # Moving on to the other rows and columns
        for i in range(1, self.NUMBER_OF_FREQUENCY_BANDS):
            # For every row, I'm going across laterally (across columns) and populating the value_function_collection
            for j in range(1, self.NUMBER_OF_SAMPLES):
                for state in OccupancyState:
                    # Again finishing off the [0] index first
                    max_pointer = self.get_transition_probabilities(OccupancyState.idle.value,
                                                                    OccupancyState.idle.value, state.value) * \
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
                                pointer = self.get_transition_probabilities(candidate_previous_state_temporal.value,
                                                                            candidate_previous_state_spatial.value,
                                                                            state.value) * \
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
                    current_value = max_pointer * self.get_emission_probabilities(state.value,
                                                                                  self.observation_samples[
                                                                                      i][j])
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
                estimated_states[self.NUMBER_OF_FREQUENCY_BANDS - 1].append(
                    self.value_from_name(k))
                previous_state_temporal = k
                previous_state_spatial = k
                break
        # Backtracking
        for i in range(self.NUMBER_OF_FREQUENCY_BANDS - 1, -1, -1):
            for j in range(self.NUMBER_OF_SAMPLES - 1, -1, -1):
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
            previous_state_spatial = value_function_collection[i][self.NUMBER_OF_SAMPLES - 1][
                previous_state_spatial].previous_spatial_state
        return {'Estimation_Accuracy': self.get_estimation_accuracy(estimated_states),
                'Detection_Probability': self.get_probability_of_detection(estimated_states),
                'False_Alarm_Probability': self.get_probability_of_false_alarm(estimated_states),
                'Missed_Detection_Probability': self.get_probability_of_missed_detection(estimated_states)}

    # Get enumeration field value from name
    @staticmethod
    def value_from_name(name):
        if name == 'occupied':
            return OccupancyState.occupied.value
        else:
            return OccupancyState.idle.value

    # Reset every collection for the next run
    def reset(self):
        self.true_pu_occupancy_states.clear()
        self.transition_probabilities_matrix.clear()
        self.noise_samples.clear()
        self.channel_impulse_response_samples.clear()
        self.observation_samples.clear()

    # Exit strategy
    def __exit__(self, exc_type, exc_val, exc_tb):
        print('[INFO] PUOccupancyBehaviorEstimatorIII Clean-up: Cleaning things up ...')


# Visualize a plot of Estimation Accuracies v/s P(Occupied|Idle)
def visualize_estimation_accuracy_plot(p_estimation_x_axis, estimation_accuracies_for_plotting):
    estimation_fig, estimation_axes = plt.subplots()
    estimation_axes.plot(p_estimation_x_axis, estimation_accuracies_for_plotting, linestyle='--', linewidth=1.0,
                         marker='o', color='r')
    estimation_fig.suptitle(
        'Estimation Accuracy v P(Occupied|Idle) at P(Occupied)=0.6 with Markovian Correlation across channels and '
        'across observation rounds', fontsize=7.5)
    estimation_axes.set_xlabel('P(Occupied | Idle)', fontsize=10)
    estimation_axes.set_ylabel('Estimation Accuracy', fontsize=10)
    plt.show()


# Visualize a plot of False Alarm Probabilities v/s P(Occupied|Idle)
def visualize_false_alarm_probability_plot(p_false_alarm_x_axis, false_alarm_probabilities_for_plotting):
    false_alarm_fig, false_alarm_axes = plt.subplots()
    false_alarm_axes.plot(p_false_alarm_x_axis, false_alarm_probabilities_for_plotting, linestyle='--', linewidth=1.0,
                          marker='o', color='b')
    false_alarm_fig.suptitle('False Alarm Probability v P(Occupied|Idle) at P(Occupied)=0.6 with Markovian Correlation '
                             'across channels and across observation rounds', fontsize=7.5)
    false_alarm_axes.set_xlabel('P(Occupied | Idle)', fontsize=10)
    false_alarm_axes.set_ylabel('Probability of False Alarm', fontsize=10)
    plt.show()


# Visualize a plot of Missed Detection Probabilities v/s P(Occupied|Idle)
def visualize_missed_detection_probability_plot(p_missed_detection_x_axis, missed_detection_probabilities_for_plotting):
    missed_detection_fig, missed_detection_axes = plt.subplots()
    missed_detection_axes.plot(p_missed_detection_x_axis, missed_detection_probabilities_for_plotting, linestyle='--',
                               linewidth=1.0, marker='o', color='m')
    missed_detection_fig.suptitle(
        'Missed Detection Probability v P(Occupied|Idle) at P(Occupied)=0.6 with Markovian Correlation '
        'across channels and across observation rounds', fontsize=7.5)
    missed_detection_axes.set_xlabel('P(Occupied | Idle)', fontsize=10)
    missed_detection_axes.set_ylabel('Probability of Missed Detection', fontsize=10)
    plt.legend(loc='upper right', prop={'size': 8})
    plt.show()


# Cyclic average evaluation
def cyclic_average(collection, number_of_internal_collections, number_of_cycles):
    collection_for_plotting = []
    for pass_counter in range(0, number_of_internal_collections):
        _sum = 0
        for _entry in collection:
            _sum += _entry[pass_counter]
        collection_for_plotting.append(_sum / number_of_cycles)
    return collection_for_plotting


# Run Trigger
if __name__ == '__main__':
    # External estimation accuracies array
    external_estimation_accuracies = []
    # External detection probabilities array
    external_detection_probabilities = []
    # External false_alarm_probabilities array
    external_false_alarm_probabilities = []
    # External missed detection probabilities array
    external_missed_detection_probabilities = []
    puOccupancyBehaviorEstimator = PUOccupancyBehaviorEstimatorIII()
    # P(1)
    pi = puOccupancyBehaviorEstimator.start_probabilities.occupied
    p_initial = 0.03
    for cycle in range(0, puOccupancyBehaviorEstimator.NUMBER_OF_CYCLES):
        # Internal estimation accuracies array
        internal_estimation_accuracies = []
        # Internal detection probabilities array
        internal_detection_probabilities = []
        # Internal false_alarm_probabilities array
        internal_false_alarm_probabilities = []
        # Internal missed detection probabilities array
        internal_missed_detection_probabilities = []
        # P(Occupied|Idle)
        p = p_initial
        # Varying p all the way up to independence
        for iteration in range(0, int(pi / p)):
            q = (p * (1 - pi)) / pi
            puOccupancyBehaviorEstimator.transition_probabilities_matrix = {
                1: {1: (1 - q), 0: q},
                0: {1: p, 0: (1 - p)}
            }
            # True PU Occupancy State
            puOccupancyBehaviorEstimator.generate_true_pu_occupancy_states(p, q, pi)
            puOccupancyBehaviorEstimator.allocate_observations()
            obtained_collection = puOccupancyBehaviorEstimator.estimate_pu_occupancy_states()
            internal_estimation_accuracies.append(obtained_collection['Estimation_Accuracy'])
            internal_detection_probabilities.append(obtained_collection['Detection_Probability'])
            internal_false_alarm_probabilities.append(obtained_collection['False_Alarm_Probability'])
            internal_missed_detection_probabilities.append(obtained_collection['Missed_Detection_Probability'])
            p += p_initial
            puOccupancyBehaviorEstimator.reset()
        external_estimation_accuracies.append(internal_estimation_accuracies)
        external_detection_probabilities.append(internal_detection_probabilities)
        external_false_alarm_probabilities.append(internal_false_alarm_probabilities)
        external_missed_detection_probabilities.append(internal_missed_detection_probabilities)
    x_axis = []
    for value in range(1, int(pi / p_initial) + 1):
        x_axis.append(value * p_initial)
    final_frontier = int(pi / p_initial)
    # Start Plotting
    visualize_estimation_accuracy_plot(x_axis,
                                       cyclic_average(external_estimation_accuracies, final_frontier,
                                                      puOccupancyBehaviorEstimator.NUMBER_OF_CYCLES))
    visualize_false_alarm_probability_plot(x_axis, cyclic_average(external_false_alarm_probabilities, final_frontier,
                                                                  puOccupancyBehaviorEstimator.NUMBER_OF_CYCLES))
    visualize_missed_detection_probability_plot(x_axis,
                                                cyclic_average(external_missed_detection_probabilities, final_frontier,
                                                               puOccupancyBehaviorEstimator.NUMBER_OF_CYCLES))
