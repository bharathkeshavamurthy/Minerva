# PU Occupancy Behavior Estimation - A Crude Viterbi Algorithm
# For the official, refined, fine-tuned version, please refer to PUOccupancyBehaviorEstimator
# Author: Bharath Keshavamurthy
# Organization: School of Electrical and Computer Engineering, Purdue University
# Copyright (c) 2019. All Rights Reserved.

# For the math behind this algorithm, refer to:
# This url may change - Please contact the author at <bkeshava@purdue.edu> for more details.
# https://github.rcac.purdue.edu/bkeshava/Minerva/tree/master/latex

import numpy
import scipy.stats
from matplotlib import pyplot as plt


# PU Occupancy Behavior Estimator
# Crude Viterbi Algorithm
# Markovian across Frequency - Viterbi algorithm used to predict the most likely sequence of states
class CrudeViterbiAlgorithm:

    # Initialization Sequence
    def __init__(self):
        self.true_pu_occupancy_states = []
        self.observation_samples = []
        self.noise_samples = dict()
        self.channel_impulse_response_samples = dict()
        self.detection_accuracy = []

    # Generate the true states using the Markovian model
    def generate_true_states(self, prob, qrob, start_probabilities):
        initial_seed = numpy.random.random_sample()
        previous = 1
        if initial_seed > start_probabilities['Occupied']:
            previous = 0
        self.true_pu_occupancy_states.append(previous)
        for loop_counter in range(1, 18):
            seed = numpy.random.random_sample()
            if previous == 1 and seed < qrob:
                previous = 0
            elif previous == 1 and seed > qrob:
                previous = 1
            elif previous == 0 and seed < prob:
                previous = 1
            else:
                previous = 0
            self.true_pu_occupancy_states.append(previous)
        print(self.true_pu_occupancy_states)

    # Generate the observations of all the bands for a number of observation rounds or cycles
    def allocate_observations(self):
        for iband in range(0, 18):
            mu_noise, std_noise = 0, numpy.sqrt(1)
            self.noise_samples[iband] = numpy.random.normal(mu_noise, std_noise, 500)
            mu_channel_impulse_response, std_channel_impulse_response = 0, numpy.sqrt(80)
            self.channel_impulse_response_samples[iband] = numpy.random.normal(mu_channel_impulse_response,
                                                                               std_channel_impulse_response, 500)

        # PU occupancy get_state added with some White Gaussian Noise
        # The observations are zero mean Gaussians with variance std_obs**2
        for freq_band in range(0, 18):
            obs_per_band = list()
            for count in range(0, 500):
                obs_per_band.append(self.channel_impulse_response_samples[freq_band][
                                        count] * self.true_pu_occupancy_states[freq_band] + self.noise_samples[
                                        freq_band][count])
            self.observation_samples.append(obs_per_band)
        return self.observation_samples

    # Get the value of the state - a utility function
    @staticmethod
    def get_state(state):
        if state == 'Occupied':
            return 1
        else:
            return 0

    # Get the Emission Probabilities
    def get_emit(self, state, observation_sample):
        return scipy.stats.norm(0, numpy.sqrt((80 * self.get_state(state)) + 1)).pdf(observation_sample)

    # Calculate the detection accuracy
    def get_detection_accuracy(self, estimated_states):
        accuracies = 0
        for _counter in range(0, 18):
            if self.true_pu_occupancy_states[_counter] == self.get_state(estimated_states[_counter]):
                accuracies += 1
        return accuracies / 18

    # Estimate the most likely sequence of states of the |B| frequency bands using the Viterbi algorithm
    def viterbi(self, obs, states, start_p, trans_p):
        v = [{}]
        for st in states:
            v[0][st] = {"prob": start_p[st] * self.get_emit(st, obs[0]), "prev": None}
        # Run Viterbi when t > 0
        for t in range(1, len(obs)):
            v.append({})
            for st in states:
                max_tr_prob = v[t - 1][states[0]]["prob"] * trans_p[states[0]][st]
                prev_st_selected = states[0]
                for prev_st in states[1:]:
                    tr_prob = v[t - 1][prev_st]["prob"] * trans_p[prev_st][st]
                    if tr_prob > max_tr_prob:
                        max_tr_prob = tr_prob
                        prev_st_selected = prev_st
                max_prob = max_tr_prob * self.get_emit(st, obs[t])
                v[t][st] = {"prob": max_prob, "prev": prev_st_selected}
        opt = []
        max_prob = max(value["prob"] for value in v[-1].values())
        previous = None
        for st, data in v[-1].items():
            if data["prob"] == max_prob:
                opt.append(st)
                previous = st
                break
        for t in range(len(v) - 2, -1, -1):
            opt.insert(0, v[t + 1][previous]["prev"])
            previous = v[t + 1][previous]["prev"]
        self.detection_accuracy.append(self.get_detection_accuracy(opt))


# Run Trigger
if __name__ == '__main__':
    state_values = ('Occupied', 'Idle')
    start_prob = {'Occupied': 0.6, 'Idle': 0.4}
    iterations_array = []
    state_array = [1, 0]
    state_start_probs = [0.6, 0.4]
    ps = []
    final_da = []
    viterbi_algorithm = CrudeViterbiAlgorithm()
    for counter in range(0, 50):
        ps = []
        das = []
        p = 0.030
        for i in range(0, 20):
            ps.append(p)
            q = (p * 0.4) / 0.6
            trans_prob = {
                'Occupied': {'Occupied': (1 - q), 'Idle': q},
                'Idle': {'Occupied': p, 'Idle': (1 - p)}
            }
            trans_names = [['10', '11'], ['00', '01']]
            trans_matrix = [[q, (1 - q)], [(1 - p), p]]
            viterbi_algorithm.generate_true_states(p, q, start_prob)
            observations_all = viterbi_algorithm.allocate_observations()
            for round_number in range(0, 500):
                observations_round = []
                for band in range(0, 18):
                    observations_round.append(observations_all[band][round_number])
                viterbi_algorithm.viterbi(observations_round, state_values, start_prob, trans_prob)
            da_sum = 0
            for k in viterbi_algorithm.detection_accuracy:
                da_sum = da_sum + k
            viterbi_algorithm.detection_accuracy.clear()
            viterbi_algorithm.true_pu_occupancy_states.clear()
            viterbi_algorithm.observation_samples.clear()
            viterbi_algorithm.noise_samples.clear()
            viterbi_algorithm.channel_impulse_response_samples.clear()
            print('pi = 0.6 | p = ', p, ' | Detection Accuracy = ', round(da_sum / 500, 2))
            das.append(round(da_sum / 500, 2))
            p = p + 0.030
        final_da.append(das)
    final_davg = []
    for n in range(0, 20):
        _sum = 0
        for entry in final_da:
            _sum += entry[n]
        final_davg.append(_sum / 50)
    fig, ax = plt.subplots()
    ax.plot(ps, final_davg, linestyle='--', linewidth=1.0, marker='o', color='b')
    fig.suptitle('Detection Accuracy v/s P(Occupied | Idle) at P( Xi = 1 ) = 0.6', fontsize=20)
    ax.set_xlabel('p', fontsize=12)
    ax.set_ylabel('Detection Accuracy', fontsize=12)
    plt.show()
