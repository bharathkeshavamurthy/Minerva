# This Python script encapsulates an algorithm to estimate the parameters of Markov chains used in our research
# Static PU with Markovian Correlation across the channel indices
# Author: Bharath Keshavamurthy
# Organization: School of Electrical and Computer Engineering, Purdue University
# Copyright (c) 2019. All Rights Reserved

from enum import Enum
from collections import namedtuple
import numpy
import scipy.stats


# OccupancyState Enumeration
class OccupancyState(Enum):
    # Occupancy state IDLE
    IDLE = 0
    # Occupancy state OCCUPIED:
    OCCUPIED = 1


# Markov Chain Parameter Estimation Algorithm
class MarkovChainParameterEstimator(object):
    # Number of channels in the discretized spectrum of interest
    NUMBER_OF_FREQUENCY_BANDS = 18

    # Number of observations made by the SU during the simulation period
    NUMBER_OF_SAMPLES = 1000

    # Variance of the Additive White Gaussian Noise Samples
    VARIANCE_OF_AWGN = 1

    # Variance of the Channel Impulse Response which is a zero mean Gaussian
    VARIANCE_OF_CHANNEL_IMPULSE_RESPONSE = 80

    # Start probabilities of PU occupancy per frequency band
    BAND_START_PROBABILITIES = namedtuple('BandStartProbabilities', ['idle', 'occupied'])

    # Initialization sequence
    def __init__(self):
        print('[INFO] MarkovChainParameterEstimator Initialization: Bringing things up...')
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
        # The transition probabilities
        self.transition_probabilities = {'0': {'0': list(), '1': list()}, '1': {'0': list(), '1': list()}}
        # The forward probabilities collection across simulation time
        self.forward_probabilities = [dict() for k in range(0, self.NUMBER_OF_FREQUENCY_BANDS)]
        # The backward probabilities collection across simulation time
        self.backward_probabilities = [dict() for k in range(0, self.NUMBER_OF_FREQUENCY_BANDS)]

    # Evaluate Forward Probability based on the passed args
    def evaluate_forward_probability(self, iteration, current_state, observation_sample):
        evaluated_value = 0
        for iterative_state in OccupancyState:
            evaluated_value += (self.forward_probabilities[iteration - 1][iterative_state] *
                                self.get_emission_probabilities(current_state, observation_sample) *
                                self.transition_probabilities[iterative_state][current_state][iteration])
        return evaluated_value

    # Evaluate Backward Probability based on the passed args
    def evaluate_backward_probability(self, iteration, current_state, observation_sample):
        evaluated_value = 0
        for iterative_state in OccupancyState:
            evaluated_value += (self.backward_probabilities[iteration - 1][iterative_state] *
                                self.get_emission_probabilities(current_state, observation_sample) *
                                self.transition_probabilities[iterative_state][current_state][iteration])
        return evaluated_value

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
            self.noise_samples[frequency_band] = numpy.random.normal(mu_noise, std_noise, self.NUMBER_OF_SAMPLES)
            mu_channel_impulse_response, std_channel_impulse_response = 0, numpy.sqrt(
                self.VARIANCE_OF_CHANNEL_IMPULSE_RESPONSE)
            self.channel_impulse_response_samples[frequency_band] = numpy.random.normal(mu_channel_impulse_response,
                                                                                        std_channel_impulse_response,
                                                                                        self.NUMBER_OF_SAMPLES)
        # Re-arranging the vectors
        for band in range(0, self.NUMBER_OF_FREQUENCY_BANDS):
            obs_per_band = list()
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

    # Get the Emission Probabilities -> P(y|x)
    def get_emission_probabilities(self, state, observation_sample):
        return scipy.stats.norm(0, numpy.sqrt(
            (self.VARIANCE_OF_CHANNEL_IMPULSE_RESPONSE * state) + self.VARIANCE_OF_AWGN)).pdf(
            observation_sample)

    # Termination sequence
    def __exit__(self, exc_type, exc_val, exc_tb):
        print('[INFO] MarkovChainParameterEstimator Termination: Tearing things down...')
