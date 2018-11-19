# PU Occupancy Behavior Estimation
# Author: Bharath Keshavamurthy
# School of Electrical and Computer Engineering
# Purdue University
# Copyright (c) 2018. All Rights Reserved.

# For the math behind this algorithm, refer to:
# https://github.rcac.purdue.edu/bkeshava/Minerva/blob/master/SystemModelAndEstimator_v3_3_0.pdf

from enum import Enum
import threading
import numpy
import scipy.stats
import matplotlib.pyplot
from collections import namedtuple


# Elements of uncertainties in my simulation
class Randomness(Enum):
    # Channel Impulse Response
    channel_impulse_response = 1
    # Additive White Gaussian Noise
    noise = 2
    # Cumulative effect of both channel_impulse_response_samples and AWGN on the PU occupancy observation
    cumulative = 3


# Occupancy state enumeration
class OccupancyState(Enum):
    # Channel Idle
    idle = 0
    # Channel Occupied
    occupied = 1


# Main class: PU Occupancy Behavior Estimation
class PUOccupancyBehaviorEstimation(object):
    # Number of samples for this simulation
    NUMBER_OF_SAMPLES = 1000

    # Variance of the Additive White Gaussian Noise Samples
    VARIANCE_OF_AWGN = 6.25

    # Variance of the Channel Impulse Response which is a zero mean Gaussian
    VARIANCE_OF_CHANNEL_IMPULSE_RESPONSE = 10

    # Precision of allocate_noise_samples generation by numpy
    PRECISION_OF_NOISE_GENERATION = 0.1

    # Number of frequency bands/channels in the wideband spectrum of interest
    NUMBER_OF_FREQUENCY_BANDS = 10

    # Maximum Number of workers
    MAX_WORKERS = 20

    # The parameters of observations per frequency band
    OBSERVATION_PARAMETERS_PER_BAND = namedtuple('OBSERVATION_PARAMETERS_PER_BAND', ['mean', 'standard_deviation'])

    # Start probabilities of PU occupancy per frequency band
    BAND_START_PROBABILITIES = namedtuple('BandStartProbabilities', ['idle', 'occupied'])

    # Occupancy States (IDLE, OCCUPIED)
    OCCUPANCY_STATES = (OccupancyState.idle, OccupancyState.occupied)

    # State Transition Probabilities column entries per row
    TRANSITION_PROBABILITY_TUPLE = namedtuple('TransitionProbabilities', ['idle', 'occupied'])

    # Value function named tuple
    VALUE_FUNCTION_NAMED_TUPLE = namedtuple('ValueFunction', ['current_value', 'previous_state'])

    # Initialization Sequence
    def __init__(self):
        print('PUOccupancyBehaviorEstimation Initialization: Bringing things up ...')
        # AWGN samples
        self.noise_samples = []
        # Channel Impulse Response samples
        self.channel_impulse_response_samples = []
        # True PU Occupancy state
        self.true_pu_occupancy_states = []
        # The parameters of observations of all the bands are stored in this dict here
        self.observations_parameters = {}
        # The observed samples at the SU receiver
        self.observation_samples = []
        # The start probabilities
        self.start_probabilities = namedtuple('BandStartProbabilities', ['idle', 'occupied'])
        # The complete state transition matrix is defined as a Python dictionary taking in named-tuples
        self.transition_probabilities_matrix = {}

    # Generate channel impulse response samples
    # H_k ~ \mathcal{N}(0,\ \sigma_H^2)
    def allocate_channel_impulse_response_samples(self):
        print('PUOccupancyBehaviorEstimation allocate_channel_impulse_response_samples: Executor{}'.format(
            threading.current_thread()))
        mu_channel_impulse_response, std_channel_impulse_response = 0, numpy.sqrt(
            self.VARIANCE_OF_CHANNEL_IMPULSE_RESPONSE)
        channel_impulse_response_samples = numpy.random.normal(mu_channel_impulse_response,
                                                               std_channel_impulse_response, self.NUMBER_OF_SAMPLES)
        if self.validate_gaussian_samples(mu_channel_impulse_response, std_channel_impulse_response,
                                          channel_impulse_response_samples):
            self.plot_gaussian(mu_channel_impulse_response, std_channel_impulse_response,
                               channel_impulse_response_samples, Randomness.channel_impulse_response)
            print('PUOccupancyBehaviorEstimation allocate_channel_impulse_response_samples: ',
                  channel_impulse_response_samples)
            self.channel_impulse_response_samples = channel_impulse_response_samples
        else:
            print('PUOccupancyBehaviorEstimation allocate_channel_impulse_response_samples(): '
                  'Incorrect channel_impulse_response_samples generated ...')
            return None

    # Generate the allocate_noise_samples samples
    # V_k(i) ~ \mathcal{N}(0,\ \sigma_V^2)
    def allocate_noise_samples(self):
        print('PUOccupancyBehaviorEstimation allocate_noise_samples(): Executor {}'.format(threading.current_thread()))
        mu_noise, std_noise = 0, numpy.sqrt(self.VARIANCE_OF_AWGN)
        noise_samples = numpy.random.normal(mu_noise, std_noise, self.NUMBER_OF_SAMPLES)
        if self.validate_gaussian_samples(mu_noise, std_noise, noise_samples):
            self.plot_gaussian(mu_noise, std_noise, noise_samples, Randomness.noise)
            print(
                'PUOccupancyBehaviorEstimation allocate_noise_samples(): ', noise_samples)
            self.noise_samples = noise_samples
        else:
            print('PUOccupancyBehaviorEstimation allocate_noise_samples(): '
                  'Incorrect noise samples generated ...')
            return None

    # Validate the gaussian samples generated by numpy up to a certain precision
    def validate_gaussian_samples(self, mu, std, gaussian_samples):
        # return (abs(mu - numpy.mean(gaussian_samples)) < self.PRECISION_OF_NOISE_GENERATION) and (
        #         abs(std - numpy.std(gaussian_samples, ddof=1)) < self.PRECISION_OF_NOISE_GENERATION)
        return True

    # Plot the gaussian samples against a formulaic Gaussian
    def plot_gaussian(self, mu, std, gaussian_samples, randomness_index):
        count, bins, ignored = matplotlib.pyplot.hist(gaussian_samples, 30, density=True)
        matplotlib.pyplot.plot(bins, self.gaussian(mu, std, bins), linewidth=1, color='g')
        if randomness_index == Randomness.noise:
            matplotlib.pyplot.title('AWGN Samples')
            matplotlib.pyplot.savefig('AWGN_Samples.png')
        elif randomness_index == Randomness.channel_impulse_response:
            matplotlib.pyplot.title('Channel Impulse Response Samples')
            matplotlib.pyplot.savefig('Channel_Impulse_Response_Samples.png')
        else:
            print('PUOccupancyBehaviorEstimation plot_gaussian(): Not plotting for {}'.format(
                Randomness(randomness_index).name))
        matplotlib.pyplot.close()

    # Get the formulaic Gaussian
    @staticmethod
    def gaussian(mu, std, bins):
        return (1 / (numpy.sqrt(2 * numpy.pi) * std)) * numpy.exp(-(bins - mu) ** 2 / (2 * std ** 2))

    # Get the true occupancy behavior of the PU
    # For the sake of experimentation, I randomize PU occupancy behavior using numpy
    def allocate_true_pu_occupancy_states(self):
        print('PUOccupancyBehaviorEstimation allocate_true_pu_occupancy_states(): Executor {}'.format(
            threading.current_thread()))
        self.true_pu_occupancy_states = numpy.random.randint(0, 2, size=self.NUMBER_OF_FREQUENCY_BANDS)
        print('PUOccupancyBehaviorEstimation allocate_true_pu_occupancy_states(): ',
              self.true_pu_occupancy_states)

    # Get the observations vector
    def allocate_observations(self):
        # PU occupancy state added with some White Gaussian Noise
        # The observations are zero mean Gaussians with variance std_obs**2
        observation_parameters = {}
        band_index = 0
        for true_pu_occupancy_state in self.true_pu_occupancy_states:
            std_observation = numpy.sqrt(((numpy.std(self.channel_impulse_response_samples, ddof=1) ** 2)
                                          * true_pu_occupancy_state) + numpy.std(self.noise_samples, ddof=1) ** 2)
            observation_parameters[band_index] = self.OBSERVATION_PARAMETERS_PER_BAND(mean=self.get_mean_observations(),
                                                                                      standard_deviation=std_observation
                                                                                      )

            band_index += 1
        self.observation_samples = [self.make_observations(observation_parameters[k], k) for k in
                                    observation_parameters.keys()]
        print('PUOccupancyBehaviorEstimation observations: ', self.observation_samples)

    # Make observations
    def make_observations(self, observation_parameters, band_index):
        print('PUOccupancyBehaviorEstimation make_observations: Executor {}'.format(threading.current_thread()))
        mu_observations, std_observations = observation_parameters.mean, observation_parameters.standard_deviation
        observation_samples = numpy.random.normal(mu_observations, std_observations, self.NUMBER_OF_SAMPLES)
        if self.validate_gaussian_samples(mu_observations, std_observations, observation_samples) is False:
            print('PUOccupancyBehaviorEstimation make_observations(): '
                  'Imprecise observation samples generated for band [', band_index, ']. Just a warning!')
        return observation_samples

    # Generating the start probabilities
    def allocate_start_probabilities(self):
        success_probability = numpy.random.random_sample()
        self.start_probabilities = self.BAND_START_PROBABILITIES(occupied=1 - success_probability,
                                                                 idle=success_probability)
        print('PUOccupancyBehaviorEstimation start_probabilities: ', self.start_probabilities)

    # Generating the state transition probabilities
    def allocate_transition_probabilities(self):
        for state in self.OCCUPANCY_STATES:
            success_probability = numpy.random.random_sample()
            self.transition_probabilities_matrix[state] = self.TRANSITION_PROBABILITY_TUPLE(
                occupied=1 - success_probability, idle=success_probability)
        print('PUOccupancyBehaviorEstimation transition_probabilities: ', self.transition_probabilities_matrix)

    # The observations are Gaussians with zero mean and variance \sigma_H^2x_k + \sigma_V^2
    @staticmethod
    def get_mean_observations():
        # All are zero for now based on my math which can be found in the doc listed above
        return 0

    # Get the Emission Probabilities
    def get_emission_probabilities(self, state, observation_sample):
        return scipy.stats.norm(self.get_mean_observations(),
                                numpy.sqrt(((numpy.std(self.channel_impulse_response_samples, ddof=1) ** 2)
                                            * state) + numpy.std(self.noise_samples, ddof=1) ** 2)).pdf(
            observation_sample)

    # Return the start probabilities from the named_tuple
    def get_start_probabilities(self, state_name):
        if state_name == 'occupied':
            return self.start_probabilities.occupied
        else:
            return self.start_probabilities.idle

    # Return the transition probabilities from the named_tuple
    def get_transition_probabilities(self, dict_key, named_tuple_key):
        if named_tuple_key == 'occupied':
            return self.transition_probabilities_matrix[dict_key].occupied
        else:
            return self.transition_probabilities_matrix[dict_key].idle

    # Output the estimated state of the frequency bands in the wideband spectrum of interest
    def estimate_pu_occupancy_states(self):
        # The output of this state estimator
        estimated_states = []
        # Let's simplify the observations for now
        # Assuming only a single time instance of sampling each band
        reduced_observation_vector = []
        for entry in self.observation_samples:
            reduced_observation_vector.append(entry[0])
        # Now, I have to estimate the state of the ${NUMBER_OF_FREQUENCY_BANDS} based on this reduced observation vector
        # INITIALIZATION : The array of initial probabilities is known\
        # FORWARD RECURSION
        value_function_collection = [dict() for x in range(len(reduced_observation_vector))]
        for state in OccupancyState:
            current_value = self.get_emission_probabilities(state.value, reduced_observation_vector[0]) * \
                            self.get_start_probabilities(state.name)
            value_function_collection[0][state.name] = self.VALUE_FUNCTION_NAMED_TUPLE(current_value=current_value,
                                                                                       previous_state=None)
        # For each observation after the first one (I can't apply Markovian to [0])
        for observation_index in range(1, len(reduced_observation_vector)):
            # Trying to find the max pointer here ...
            for state in OccupancyState:
                # Again finishing off the [0] index first
                max_pointer = self.get_transition_probabilities(OccupancyState.idle, state.name) * \
                              value_function_collection[observation_index - 1][OccupancyState.idle.name].current_value
                confirmed_previous_state = OccupancyState.idle.name
                for candidate_previous_state in OccupancyState:
                    if candidate_previous_state.name == OccupancyState.idle.name:
                        # Already done
                        continue
                    else:
                        pointer = self.get_transition_probabilities(candidate_previous_state, state.name) * \
                                  value_function_collection[observation_index - 1][
                                      candidate_previous_state.name].current_value
                        if pointer > max_pointer:
                            max_pointer = pointer
                            confirmed_previous_state = candidate_previous_state.name
                current_value = max_pointer * self.get_emission_probabilities(state.value, reduced_observation_vector[
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
        return estimated_states

    # Get enumeration field value from name
    @staticmethod
    def value_from_name(name):
        if name == 'occupied':
            return OccupancyState.occupied.value
        else:
            return OccupancyState.idle.value

    # Get enumeration field name from value
    @staticmethod
    def name_from_value(value):
        if value == 0:
            return OccupancyState.idle.name
        else:
            return OccupancyState.occupied.name

    # Exit strategy
    def __exit__(self, exc_type, exc_val, exc_tb):
        print('PUOccupancyBehaviorEstimation Clean-up: Cleaning things up ...')


# Run Trigger
if __name__ == '__main__':
    print(
        'PUOccupancyBehaviorEstimation main: Creating an instance and starting the initialization process ...')
    puOccupancyBehaviorEstimation = PUOccupancyBehaviorEstimation()
    jobs = list()
    # This is way faster than the other non-concurrent approach I did earlier ...
    # AWGN Noise Generation
    jobs.append(threading.Thread(target=puOccupancyBehaviorEstimation.allocate_noise_samples()))
    # Channel Impulse Response Generation
    jobs.append(threading.Thread(target=puOccupancyBehaviorEstimation.allocate_channel_impulse_response_samples()))
    # True PU Occupancy State
    jobs.append(threading.Thread(target=puOccupancyBehaviorEstimation.allocate_true_pu_occupancy_states()))
    # Generating Start Probabilities
    jobs.append(threading.Thread(target=puOccupancyBehaviorEstimation.allocate_start_probabilities()))
    # Generating the State Transition Probability Matrix
    jobs.append(threading.Thread(target=puOccupancyBehaviorEstimation.allocate_transition_probabilities()))
    # Start the jobs
    for job in jobs:
        job.start()
    # Ensure that all of them have been completed
    for job in jobs:
        job.join()
    print('PUOccupancyBehaviorEstimation main: All initialization tasks have been completed ...')
    puOccupancyBehaviorEstimation.allocate_observations()
    print(
        'PUOccupancyBehaviorEstimation main: Now, let us estimate the PU occupancy states in these frequency bands...')
    print('PUOccupancyBehaviorEstimation main: True states: ',
          puOccupancyBehaviorEstimation.true_pu_occupancy_states)
    print('PUOccupancyBehaviorEstimation main: Estimated states: ',
          puOccupancyBehaviorEstimation.estimate_pu_occupancy_states())
