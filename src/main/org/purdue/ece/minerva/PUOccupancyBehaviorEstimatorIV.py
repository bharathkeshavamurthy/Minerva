# This Python script deals with the Occupancy Behavior Estimation of a PU in a wideband spectrum of interest
# There exists a Markovian Correlation across both the channel indices and the time indices
# Only a few channels are observed by the SU estimating the occupancy status of channels in the spectrum of interest
# Dynamic PU with Double Markov Chain and Missing observations
# Author: Bharath Keshavamurthy
# Organization: School of Electrical and Computer Engineering, Purdue University
# Copyright (c) 2018. All Rights Reserved.

# For the math behind this algorithm, refer to:
# This url may change - Please contact the author at <bkeshava@purdue.edu> for more details.
# https://github.rcac.purdue.edu/bkeshava/Minerva/blob/master/SystemModelAndEstimator_v3_5_0.pdf

from enum import Enum
import numpy
import scipy.stats
from collections import namedtuple
from matplotlib import pyplot as plt
import ChannelSelectionStrategyGenerator
import os


# Occupancy state enumeration
class OccupancyState(Enum):
    # Channel Idle
    idle = 0
    # Channel Occupied
    occupied = 1


# Main class: PU Occupancy Behavior Estimation
class PUOccupancyBehaviorEstimatorIV(object):
    # Number of samples for this simulation
    # Also referred to as the Number of Sampling Rounds
    NUMBER_OF_SAMPLES = 100

    # Variance of the Additive White Gaussian Noise Samples
    VARIANCE_OF_AWGN = 1

    # Variance of the Channel Impulse Response which is a zero mean Gaussian
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
    NUMBER_OF_CYCLES = 500

    # The set of channels that are sensed based on recommendations from the RL agent / bandit / emulator
    BANDS_OBSERVED = []

    # Empty observation place holder value
    EMPTY_OBSERVATION_PLACEHOLDER_VALUE = 0

    # Initialization Sequence
    def __init__(self):
        print('[INFO] PUOccupancyBehaviorEstimatorIV Initialization: Bringing things up ...')
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
                # TODO: Maybe return the output instead of doing a instance-level in-situ update

    # Get the observations vector
    # Generate the observations of all the bands for a number of observation rounds or cycles
    def allocate_observations(self):
        # Each frequency band is observed ${NUMBER_OF_SAMPLES} times.
        # The sampling of each frequency band involves a noise sample corresponding to that SU receiver
        # ... and a channel impulse response sample between that SU receiver and the radio environment
        for frequency_band in range(0, self.NUMBER_OF_FREQUENCY_BANDS):
            mu_noise, std_noise = 0, numpy.sqrt(self.VARIANCE_OF_AWGN)
            self.noise_samples[frequency_band] = numpy.random.normal(mu_noise, std_noise, self.NUMBER_OF_SAMPLES)
            mu_channel_impulse_response, std_channel_impulse_response = 0, numpy.sqrt(
                self.VARIANCE_OF_CHANNEL_IMPULSE_RESPONSE)
            self.channel_impulse_response_samples[frequency_band] = numpy.random.normal(mu_channel_impulse_response,
                                                                                        std_channel_impulse_response,
                                                                                        self.NUMBER_OF_SAMPLES)
        # Re-arranging the vectors
        for band in range(0, self.NUMBER_OF_FREQUENCY_BANDS):
            obs_per_band = [self.EMPTY_OBSERVATION_PLACEHOLDER_VALUE for k in range(0, self.NUMBER_OF_SAMPLES)]
            # I'm sensing a smaller subset of channels
            # However, I'm sensing across all time steps / rounds / time indices for the bands that are being sensed!
            if band in self.BANDS_OBSERVED:
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
        # If the channel was not sensed, there's no point in calculating the emission probability of that observation
        # Report it as 1, i.e. m_r(y_{tk}) = 1
        if observation_sample == self.EMPTY_OBSERVATION_PLACEHOLDER_VALUE:
            return 1
        # Else, return the actual emission probability from the distribution
        else:
            return scipy.stats.norm(0, numpy.sqrt(
                (self.VARIANCE_OF_CHANNEL_IMPULSE_RESPONSE * state) + self.VARIANCE_OF_AWGN)).pdf(
                observation_sample)

    # Calculate the detection accuracy
    def get_detection_accuracy(self, _input, estimated_states):
        accuracies = 0
        try:
            for _counter in range(0, len(_input)):
                for _round in range(0, self.NUMBER_OF_SAMPLES):
                    if self.true_pu_occupancy_states[_input[_counter]][_round] == \
                            estimated_states[_input[_counter]][_round]:
                        accuracies += 1
        except Exception as e:
            print(
                '[ERROR] PUOccupancyBehaviorEstimatorIV get_detection_accuracy: Exception caught while calculating '
                'detection accuracy of the estimator- [', e, ']')
        return accuracies / (len(_input) * self.NUMBER_OF_SAMPLES)

    # Output the estimated state of the frequency bands in the wideband spectrum of interest
    # TODO: Remove unnecessary comments and Verified tags which were added in because the code is too complex to keep...
    # ...track of without them...Remove them once you get the hang of it!
    # TODO: Refactor this method - it's way too huge!
    def estimate_pu_occupancy_states(self, detection_input):
        # Estimated states - kxt matrix
        # Verified
        estimated_states = [[] for x in range(0, self.NUMBER_OF_FREQUENCY_BANDS)]
        # A value function collection to store and index the calculated value functions across t and k
        # Verified
        value_function_collection = [[dict() for x in range(self.NUMBER_OF_SAMPLES)] for k in
                                     range(0, self.NUMBER_OF_FREQUENCY_BANDS)]
        # t = 0 and k = 0 - No previous state to base the Markovian Correlation on in either dimension
        # Verified
        for state in OccupancyState:
            # Verified - [0][0] is being finished here because there's no Markovian correlation that exists here
            current_value = self.get_emission_probabilities(state.value, self.observation_samples[0][0]) * \
                            self.get_start_probabilities(state.name)
            # Verified
            value_function_collection[0][0][state.name] = self.VALUE_FUNCTION_NAMED_TUPLE(current_value=current_value,
                                                                                          previous_temporal_state=None,
                                                                                          previous_spatial_state=None)
        # First row - Only temporal correlation
        # Verified
        i = 0
        # Verified
        for j in range(1, self.NUMBER_OF_SAMPLES):
            # Trying to find the max pointer here ...
            # Verified
            for state in OccupancyState:
                # Again finishing off the [0] index first
                # Verified - a_{lr} * V_{j-1}^{(l)}
                # TODO: max_pointer may be a kind of misnomer here and you may confuse it with the backtrack pointer
                max_pointer = self.get_transition_probabilities_single(OccupancyState.idle.value,
                                                                       state.value) * \
                              value_function_collection[i][j - 1][
                                  OccupancyState.idle.name].current_value
                # Using IDLE as the confirmed previous state
                # Verified
                confirmed_previous_state = OccupancyState.idle.name
                # Verified
                for candidate_previous_state in OccupancyState:
                    # Verified
                    if candidate_previous_state.name == OccupancyState.idle.name:
                        # Already done
                        # Verified
                        continue
                    else:
                        # Verified
                        pointer = self.get_transition_probabilities_single(candidate_previous_state.value,
                                                                           state.value) * \
                                  value_function_collection[i][j - 1][
                                      candidate_previous_state.name].current_value
                        # Verified
                        if pointer > max_pointer:
                            # Verified
                            max_pointer = pointer
                            # Verified
                            confirmed_previous_state = candidate_previous_state.name
                # Verified
                current_value = max_pointer * self.get_emission_probabilities(state.value,
                                                                              self.observation_samples[
                                                                                  i][j])
                # Verified
                value_function_collection[i][j][state.name] = self.VALUE_FUNCTION_NAMED_TUPLE(
                    current_value=current_value, previous_temporal_state=confirmed_previous_state,
                    previous_spatial_state=None)
        # First column - Only spatial correlation
        # Verified
        j = 0
        # Verified
        for i in range(1, self.NUMBER_OF_FREQUENCY_BANDS):
            # Trying to find the max pointer here ...
            # Verified
            for state in OccupancyState:
                # Again finishing off the [0] index first
                # Verified
                max_pointer = self.get_transition_probabilities_single(OccupancyState.idle.value,
                                                                       state.value) * \
                              value_function_collection[i - 1][j][
                                  OccupancyState.idle.name].current_value
                # Verified
                confirmed_previous_state = OccupancyState.idle.name
                # Verified
                for candidate_previous_state in OccupancyState:
                    # Verified
                    if candidate_previous_state.name == OccupancyState.idle.name:
                        # Already done
                        # Verified
                        continue
                    else:
                        # Verified
                        pointer = self.get_transition_probabilities_single(candidate_previous_state.value,
                                                                           state.value) * \
                                  value_function_collection[i - 1][j][
                                      candidate_previous_state.name].current_value
                        # Verified
                        if pointer > max_pointer:
                            # Verified
                            max_pointer = pointer
                            # Verified
                            confirmed_previous_state = candidate_previous_state.name
                # Verified
                current_value = max_pointer * self.get_emission_probabilities(state.value,
                                                                              self.observation_samples[
                                                                                  i][j])
                # Verified
                value_function_collection[i][j][state.name] = self.VALUE_FUNCTION_NAMED_TUPLE(
                    current_value=current_value, previous_temporal_state=None,
                    previous_spatial_state=confirmed_previous_state)
        # I'm done with the first row and first column
        # Moving on to the other rows and columns
        # Verified
        for i in range(1, self.NUMBER_OF_FREQUENCY_BANDS):
            # Verified
            # For every row, I'm going across laterally (across columns) and populating the value_function_collection
            for j in range(1, self.NUMBER_OF_SAMPLES):
                # Verified
                for state in OccupancyState:
                    # Again finishing off the [0] index first
                    # Verified - Double checked
                    max_pointer = self.get_transition_probabilities(OccupancyState.idle.value,
                                                                    OccupancyState.idle.value, state.value) * \
                                  value_function_collection[i][j - 1][OccupancyState.idle.name].current_value * \
                                  value_function_collection[i - 1][j][OccupancyState.idle.name].current_value
                    # Verified
                    confirmed_previous_state_temporal = OccupancyState.idle.name
                    # Verified
                    confirmed_previous_state_spatial = OccupancyState.idle.name
                    # Verified
                    for candidate_previous_state_temporal in OccupancyState:
                        # Verified
                        for candidate_previous_state_spatial in OccupancyState:
                            # Verified - if both are IDLE, I've already covered them
                            if candidate_previous_state_temporal.name == OccupancyState.idle.name and \
                                    candidate_previous_state_spatial.name == OccupancyState.idle.name:
                                # Already done
                                # Verified
                                continue
                            else:
                                # Verified
                                pointer = self.get_transition_probabilities(candidate_previous_state_temporal.value,
                                                                            candidate_previous_state_spatial.value,
                                                                            state.value) * \
                                          value_function_collection[i][j - 1][
                                              candidate_previous_state_temporal.name].current_value * \
                                          value_function_collection[i - 1][j][
                                              candidate_previous_state_spatial.name].current_value
                                # Verified
                                if pointer > max_pointer:
                                    # Verified
                                    max_pointer = pointer
                                    # Verified
                                    confirmed_previous_state_temporal = candidate_previous_state_temporal.name
                                    # Verified
                                    confirmed_previous_state_spatial = candidate_previous_state_spatial.name
                    # Now, I have the value function for this i and this j
                    # Populate the value function collection with this value
                    # Verified
                    # I found maximum of Double Markov Chain value functions from the past and now I'm multiplying it...
                    # ...with the emission probability of this particular observation
                    current_value = max_pointer * self.get_emission_probabilities(state.value,
                                                                                  self.observation_samples[
                                                                                      i][j])
                    # Verified
                    value_function_collection[i][j][state.name] = self.VALUE_FUNCTION_NAMED_TUPLE(
                        current_value=current_value, previous_temporal_state=confirmed_previous_state_temporal,
                        previous_spatial_state=confirmed_previous_state_spatial)
        # I think the forward path is perfect
        # I have doubts in the backtrack path
        max_value = 0
        # Finding the max value among the named tuples
        # Verified
        for _value in value_function_collection[-1][-1].values():
            # Verified
            if _value.current_value > max_value:
                # Verified
                max_value = _value.current_value
        # Finding the state corresponding to this max_value and using this as the final confirmed state to ...
        # ...backtrack and find the previous states
        # Verified
        for k, v in value_function_collection[-1][-1].items():
            # Verified
            if v.current_value == max_value:
                # Verified
                estimated_states[self.NUMBER_OF_FREQUENCY_BANDS - 1].append(
                    self.value_from_name(k))
                # Verified
                previous_state_temporal = k
                # Verified
                previous_state_spatial = k
                # Verified
                break
        # Backtracking
        # Verified
        for i in range(self.NUMBER_OF_FREQUENCY_BANDS - 1, -1, -1):
            # Verified
            for j in range(self.NUMBER_OF_SAMPLES - 1, -1, -1):
                # Verified
                if len(estimated_states[i]) == 0:
                    estimated_states[i].insert(0, self.value_from_name(
                        value_function_collection[i + 1][j][previous_state_spatial].previous_spatial_state))
                    previous_state_temporal = value_function_collection[i][j][
                        previous_state_spatial].previous_temporal_state
                    continue
                estimated_states[i].insert(0, self.value_from_name(
                    value_function_collection[i][j][previous_state_temporal].previous_temporal_state))
                # Verified
                previous_state_temporal = value_function_collection[i][j][
                    previous_state_temporal].previous_temporal_state
            previous_state_spatial = value_function_collection[i][self.NUMBER_OF_SAMPLES - 1][
                previous_state_spatial].previous_spatial_state
        return self.get_detection_accuracy(detection_input, estimated_states)

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
        self.BANDS_OBSERVED = []

    # Exit strategy
    def __exit__(self, exc_type, exc_val, exc_tb):
        print('[INFO] PUOccupancyBehaviorEstimatorIV Clean-up: Cleaning things up ...')


# Run Trigger
if __name__ == '__main__':
    # Colors tuple for differentiation in the visualized results
    colors = ('b', 'r')
    # Clear the output folder
    path_to_output_folder = '../../../../../../test/Dynamic_PU_Channel_Sensing_Strategy_Plots/Uniform_Sensing'
    file_list = [f for f in os.listdir(path_to_output_folder) if f.endswith('.png')]
    for f in file_list:
        os.remove(os.path.join(path_to_output_folder, f))
    print(
        '[INFO] PUOccupancyBehaviorEstimatorIV main: Creating an instance and starting the initialization process ...')
    # Emulate the recommendations of an RL agent / a bandit
    channelSelectionStrategyGenerator = ChannelSelectionStrategyGenerator.ChannelSelectionStrategyGenerator()
    # Estimator instance
    puOccupancyBehaviorEstimator = PUOccupancyBehaviorEstimatorIV()
    # Strategy Counter
    strategy_counter = 0
    for channel_selection_strategy in channelSelectionStrategyGenerator.generic_uniform_sensing(
            puOccupancyBehaviorEstimator.NUMBER_OF_FREQUENCY_BANDS):
        # Singular identification boolean
        singular = False
        # Increment the strategy counter for file name and other aesthetic considerations
        strategy_counter += 1
        color_index = 0
        fig, ax = plt.subplots()
        for complement_counter in range(0, 2):
            # If I'm sensing all the channels, there's no point in doing complement analysis
            if len(channel_selection_strategy) == 0 or len(
                    puOccupancyBehaviorEstimator.get_complement(channel_selection_strategy)) == 0:
                # Reverse the increment
                strategy_counter = strategy_counter - 1
                # Singular is true
                singular = True
                # Break this complement_counter loop
                break
            # Sensed channels
            if complement_counter == 0:
                detection_input = channel_selection_strategy
            # Un-sensed channels
            else:
                detection_input = puOccupancyBehaviorEstimator.get_complement(channel_selection_strategy)
            # Global detection accuracies array
            global_detection_accuracies = []
            # Reset everything for the next strategy
            puOccupancyBehaviorEstimator.reset()
            puOccupancyBehaviorEstimator.BANDS_OBSERVED = channel_selection_strategy
            # P(1)
            pi = puOccupancyBehaviorEstimator.start_probabilities.occupied
            p_initial = 0.03
            for cycle in range(0, puOccupancyBehaviorEstimator.NUMBER_OF_CYCLES):
                # Internal detection accuracies array
                local_detection_accuracies = []
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
                    local_detection_accuracies.append(
                        puOccupancyBehaviorEstimator.estimate_pu_occupancy_states(detection_input))
                    p += p_initial
                global_detection_accuracies.append(local_detection_accuracies)
                # Reset everything for the next pass
                puOccupancyBehaviorEstimator.reset()
            y_axis = []
            for _loop_counter in range(0, int(pi / p_initial)):
                _sum = 0
                for entry in global_detection_accuracies:
                    _sum = _sum + entry[_loop_counter]
                y_axis.append(_sum / puOccupancyBehaviorEstimator.NUMBER_OF_CYCLES)
            x_axis = []
            for value in range(1, int(pi / p_initial) + 1):
                x_axis.append(value * p_initial)
            label = 'Detection Accuracy for ' + str(detection_input)
            ax.plot(x_axis, y_axis, linestyle='--', linewidth=1.0, marker='o',
                    color=colors[color_index], label=label)
            color_index += 1
        if singular is False:
            fig.suptitle('Detection Accuracy v/s P(Occupied | Idle) at P( Xi = 1 ) = 0.6', fontsize=6)
            ax.set_xlabel('P(Occupied | Idle)', fontsize=12)
            ax.set_ylabel('Detection Accuracy', fontsize=12)
            title = 'Uniform_Sensing_' + str(strategy_counter)
            plt.legend(loc='upper right', prop={'size': 6})
            fig.savefig(
                '../../../../../../test/Dynamic_PU_Channel_Sensing_Strategy_Plots/Uniform_Sensing/' + title + '.png')
            plt.close(fig)
