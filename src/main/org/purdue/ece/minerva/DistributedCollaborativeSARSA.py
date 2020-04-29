# This module describes a collaborative channel sensing and access strategy employing an $\epsilon$-greedy strategy
#   dictated by a TD-SARSA with linear function approximation algorithm. Furthermore, the transition probabilities are
#   learned using a stochastic approximation update. Additionally, standard neighbor discovery and channel access rank
#   pre-allocation algorithms are employed during network-initialization and/or network bootstrapping.
# Author: Bharath Keshavamurthy
# Organization: School of Electrical and Computer Engineering, Purdue University, West Lafayette, IN.
# Copyright (c) 2020. All Rights Reserved.

# The imports
import numpy
import plotly
import random
import scipy.stats
import plotly.graph_objs as go
from collections import namedtuple

# Plotly user account credentials for visualization
plotly.tools.set_credentials_file(username='bkeshava',
                                  api_key='W2WL5OOxLcgCzf8NNlgl')


# This entity describes the distributed collaborative multi-agent multi-band TD-SARSA with LFA framework for optimal
#   channel sensing and access--along with additional heuristics for neighbor discovery and channel access rank/order
#   allocation--for a system without apriori model knowledge.
class DistributedCollaborativeSARSA(object):
    # The number of time-steps of framework evaluation
    NUMBER_OF_TIME_STEPS = 1000

    # The number of sensing sampling rounds to account for the AWGN observation model
    NUMBER_OF_SAMPLING_ROUNDS = 300

    # The number of channels in the discretized spectrum of interest $K$
    NUMBER_OF_CHANNELS = 18

    # The number of cognitive radios (Secondary Users-SUs) in the network $N$
    NUMBER_OF_COGNITIVE_RADIOS = 18

    # The number of incumbents (Primary Users-PUs: military/licensed/priority users) in the network $J$
    NUMBER_OF_INCUMBENTS = 3

    # The steady-state channel occupancy probability of a time-frequency cell $\Pi$
    STEADY_STATE_OCCUPANCY = 0.6

    # The time-frequency incumbent occupancy correlation structure $\vec{p} and \vec{q}$
    # Note: The frequency chain and the temporal chain have the same transition probabilities in this analysis--
    #   Of course, that can be changed without significant modifications to this module
    CORRELATION_MODEL = {'p00': 0.1, 'p01': 0.3, 'p10': 0.3, 'p11': 0.7, 'q0': 0.3, 'q1': 0.8}

    # The noise variance $\sigma_{V}^{2}$
    NOISE_VARIANCE = 1

    # The channel impulse response variance $\sigma_{H}^{2}$
    CHANNEL_IMPULSE_RESPONSE_VARIANCE = 80

    # The discount factor $\gamma$
    DISCOUNT_FACTOR = 0.9

    # The number of members needed for a quorum
    QUORUM_REQUIREMENT = 10

    # The transmission power of all cognitive radio nodes for RSSI analysis (dB)
    TRANSMISSION_POWER = 20

    # The mean of the random variable representing the mobility patterns
    MOBILITY_MEAN = 1

    # The variance of the random variable representing the mobility patterns
    MOBILITY_VARIANCE = 1

    # The RSSI threshold for neighbor discovery (dB)
    RSSI_THRESHOLD = 10

    # The acceptable level of false alarm probability per channel given that only one cognitive radio node is sensing it
    RAW_FALSE_ALARM_PROBABILITY = 0.5

    # The fixed $\epsilon$ value for the $\epsilon$-greedy policy in the TD-SARSA with LFA algorithm
    EPSILON = 0.1

    # The constant employed in the belief update heuristic $\lambda$
    LAMBDA = 0.9

    # The penalty for missed detections
    PENALTY = 1

    # Plotly Scatter mode
    PLOTLY_SCATTER_MODE = 'lines+markers'

    # The initialization sequence
    def __init__(self):
        print('DistributedCollaborativeSARA Initialization: Bringing things up...')
        self.temporal_action_pair = namedtuple('ActionPair', ['past', 'present'])
        self.test_statistics_counter = namedtuple('TestStatisticsCounter', ['sum', 'count'])
        self.incumbent_occupancy_states = self.simulate_incumbent_occupancy_behavior()
        self.global_rssi_lists = {n: {m: 0.0 for m in range(self.NUMBER_OF_COGNITIVE_RADIOS)}
                                  for n in range(self.NUMBER_OF_COGNITIVE_RADIOS)}
        self.cognitive_radio_ensemble = {n: 0 for n in self.simulate_neighbor_discovery()}
        self.simulate_channel_access_order_allocation()
        # Initialization sequence complete

    # Simulate the occupancy behavior of the incumbents according to the double Markov chain time-frequency correlation
    #   structure described by $\vec{p} and \vec{q}$
    def simulate_incumbent_occupancy_behavior(self):
        # Output initialization
        _incumbent_occupancy_states = []
        for k in range(self.NUMBER_OF_CHANNELS):
            _incumbent_occupancy_states.append([])
        # Time-0 Frequency-0
        _incumbent_occupancy_states[0].append((lambda: 0,
                                               lambda: 1)[numpy.random.random() < self.STEADY_STATE_OCCUPANCY]())
        # Time-{1,T} Frequency-0
        for i in range(1, self.NUMBER_OF_TIME_STEPS):
            if _incumbent_occupancy_states[0][i - 1]:
                _incumbent_occupancy_states[0].append((lambda: 0,
                                                       lambda: 1)[
                                                          numpy.random.random() < self.CORRELATION_MODEL['q1']]())
            else:
                _incumbent_occupancy_states[0].append((lambda: 0,
                                                       lambda: 1)[
                                                          numpy.random.random() < self.CORRELATION_MODEL['q0']]())
        # Time-0 Frequency-{1,K}
        for k in range(1, self.NUMBER_OF_CHANNELS):
            if _incumbent_occupancy_states[k - 1][0]:
                _incumbent_occupancy_states[k].append((lambda: 0,
                                                       lambda: 1)[
                                                          numpy.random.random() < self.CORRELATION_MODEL['q1']]())
            else:
                _incumbent_occupancy_states[k].append((lambda: 0,
                                                       lambda: 1)[
                                                          numpy.random.random() < self.CORRELATION_MODEL['q0']]())
        # Time-{1,T} Frequency-{1,K}
        for k in range(1, self.NUMBER_OF_CHANNELS):
            for i in range(1, self.NUMBER_OF_TIME_STEPS):
                if _incumbent_occupancy_states[k - 1][i] and _incumbent_occupancy_states[k][i - 1]:
                    _incumbent_occupancy_states[k].append(
                        (lambda: 0,
                         lambda: 1)[numpy.random.random() < self.CORRELATION_MODEL['p11']]()
                    )
                elif _incumbent_occupancy_states[k - 1][i] == 0 and _incumbent_occupancy_states[k][i - 1]:
                    _incumbent_occupancy_states[k].append(
                        (lambda: 0,
                         lambda: 1)[numpy.random.random() < self.CORRELATION_MODEL['p01']]()
                    )
                elif _incumbent_occupancy_states[k - 1][i] and _incumbent_occupancy_states[k][i - 1] == 0:
                    _incumbent_occupancy_states[k].append(
                        (lambda: 0,
                         lambda: 1)[numpy.random.random() < self.CORRELATION_MODEL['p10']]()
                    )
                else:
                    _incumbent_occupancy_states[k].append(
                        (lambda: 0,
                         lambda: 1)[numpy.random.random() < self.CORRELATION_MODEL['p00']]()
                    )
        # Return the output
        return _incumbent_occupancy_states

    # Simulate the quorum-based RSSI-based neighbor discovery algorithm
    def simulate_neighbor_discovery(self):
        # Neighbor initialization--node discovery over the control channel
        node_specific_neighbors = {}
        for n in range(self.NUMBER_OF_COGNITIVE_RADIOS):
            discovered_neighbors = []
            for m in range(self.NUMBER_OF_COGNITIVE_RADIOS):
                if m == n:
                    continue
                else:
                    discovered_neighbors.append(m)
            node_specific_neighbors[n] = discovered_neighbors
        # RSSI-threshold based discovery list truncation
        for n in range(self.NUMBER_OF_COGNITIVE_RADIOS):
            discovered_neighbors = []
            for m in node_specific_neighbors.get(n):
                rssi = (numpy.random.normal(self.MOBILITY_MEAN,
                                            numpy.sqrt(self.MOBILITY_VARIANCE),
                                            1) * self.TRANSMISSION_POWER) / self.NOISE_VARIANCE
                if rssi > self.RSSI_THRESHOLD:
                    discovered_neighbors.append(m)
                    self.global_rssi_lists[n][m] = rssi
            node_specific_neighbors[n] = discovered_neighbors
        # Consensus for final ensemble list
        local_score_tracker = {n: 0 for n in range(self.NUMBER_OF_COGNITIVE_RADIOS)}
        for n, v in node_specific_neighbors.items():
            for m in v:
                local_score_tracker[m] += 1
        # Return the output (subject to RSSI consensus scores and quorum requirements)
        return sorted(local_score_tracker, key=local_score_tracker.get, reverse=True)[:self.QUORUM_REQUIREMENT]

    # Simulate the quorum-based preferential voting channel access rank/order allocation algorithm
    def simulate_channel_access_order_allocation(self):
        preferential_ballots = {n: {} for n in self.cognitive_radio_ensemble}
        # Quorum check is inherent in the ensemble creation routine...
        for n in self.cognitive_radio_ensemble:
            rssi_neighbors = self.global_rssi_lists.get(n)
            rssi_neighbors = sorted(rssi_neighbors, key=rssi_neighbors.get, reverse=True)
            preferential_ballot = [n] + [m for m in rssi_neighbors[0:rssi_neighbors.index(n)]]
            preferential_ballots[n] = preferential_ballot
        # Vote count--preferential ballot
        votes = {n: 0 for n in self.cognitive_radio_ensemble}
        for n, v in preferential_ballots.items():
            for j in range(len(v)):
                if v[j] in self.cognitive_radio_ensemble:
                    votes[v[j]] += len(self.cognitive_radio_ensemble) - j
        channel_access_order = sorted(votes, key=votes.get, reverse=True)
        # Set the channel access ranks globally
        for n in self.cognitive_radio_ensemble:
            self.cognitive_radio_ensemble[n] = channel_access_order.index(n)
        # Return the channel access order in case a third party needs it...
        return channel_access_order

    # Get the allowed false alarm probability based on the number of cognitive radios sensing a particular channel for
    #   action selection
    def get_pfa1(self, _k, k, actions, nodes):
        denominator = (lambda: 0, lambda: 1)[_k == k]() + \
                      numpy.sum([(lambda: 0, lambda: 1)[actions.get(m).present is not None
                                                        and actions.get(m).present == k]()
                                 for m in nodes])
        return (lambda: self.RAW_FALSE_ALARM_PROBABILITY,
                lambda: self.RAW_FALSE_ALARM_PROBABILITY / denominator)[denominator.item() > 0]()

    # Get the allowed false alarm probability based on the number of cognitive radios sensing a particular channel for
    #   belief and SARSA $\vec{\theta}$ (past)
    def get_pfa2(self, actions, k):
        denominator = numpy.sum([(lambda: 0, lambda: 1)[actions.get(m).past == k]()
                                 for m in self.cognitive_radio_ensemble.keys()])
        return (lambda: self.RAW_FALSE_ALARM_PROBABILITY,
                lambda: self.RAW_FALSE_ALARM_PROBABILITY / denominator)[denominator.item() > 0]()

    # Get the allowed false alarm probability based on the number of cognitive radios sensing a particular channel for
    #   belief and SARSA $\vec{\theta}$ (present)
    def get_pfa3(self, actions, k):
        denominator = numpy.sum([(lambda: 0, lambda: 1)[actions.get(m).present == k]()
                                 for m in self.cognitive_radio_ensemble.keys()])
        return (lambda: self.RAW_FALSE_ALARM_PROBABILITY,
                lambda: self.RAW_FALSE_ALARM_PROBABILITY / denominator)[denominator.item() > 0]()

    # The distributed collaborative SARSA with Linear Function Approximation algorithm
    def collaborative_sarsa(self):
        # Initialize the output to be returned
        time_slots = []
        utilities = []
        # Initialize the temporary variables--collaboration through global view (system-wide dicts and lists)
        thetas = {n: [(0.0 * ((k+1)/(k+1))) for k in range(self.NUMBER_OF_CHANNELS)]
                  for n in self.cognitive_radio_ensemble.keys()}
        beliefs = {n: [(0.5 * ((k+1)/(k+1))) for k in range(self.NUMBER_OF_CHANNELS)]
                   for n in self.cognitive_radio_ensemble.keys()}
        actions = {n: self.temporal_action_pair(past=numpy.random.randint(0, self.NUMBER_OF_CHANNELS - 1),
                                                present=None) for n in self.cognitive_radio_ensemble.keys()}
        for i in range(self.NUMBER_OF_TIME_STEPS):
            time_slots.append(i)
            # Observation
            observations = {n: (numpy.sum([((t+1)/(t+1)) * ((self.incumbent_occupancy_states[actions.get(n).past][i] *
                                                             numpy.random.normal(0, numpy.sqrt(
                                                                 self.CHANNEL_IMPULSE_RESPONSE_VARIANCE))
                                                             ) + numpy.random.normal(0,
                                                                                     numpy.sqrt(self.NOISE_VARIANCE))
                                                            ) for t in range(self.NUMBER_OF_SAMPLING_ROUNDS)]
                                          ) / self.NUMBER_OF_SAMPLING_ROUNDS)
                            for n in self.cognitive_radio_ensemble.keys()}
            all_nodes = [m for m in self.cognitive_radio_ensemble.keys()]
            random.shuffle(all_nodes)
            # Action selection
            for n in all_nodes:
                nodes = [m for m in self.cognitive_radio_ensemble.keys()]
                nodes.remove(n)
                feature_vector = {_k: [beliefs.get(n)[k] * (1 - self.get_pfa1(_k, k, actions, nodes))
                                       for k in range(self.NUMBER_OF_CHANNELS)]
                                  for _k in range(self.NUMBER_OF_CHANNELS)}
                q_values = [numpy.sum([thetas.get(n)[k] * feature_vector.get(_k)[k]
                                       for k in range(self.NUMBER_OF_CHANNELS)])
                            for _k in range(self.NUMBER_OF_CHANNELS)]
                max_action = max([k for k in range(len(q_values))], key=lambda x: q_values[x])
                action = (lambda: numpy.random.randint(0, self.NUMBER_OF_CHANNELS - 1), lambda: max_action)[
                    numpy.random.random() > self.EPSILON]()
                actions[n] = self.temporal_action_pair(past=actions.get(n).past,
                                                       present=action)
            test_statistics = {k: self.test_statistics_counter(sum=0.0, count=0)
                               for k in range(self.NUMBER_OF_CHANNELS)}
            # Aggregation
            for n in self.cognitive_radio_ensemble.keys():
                test_statistics[actions.get(n).past] = self.test_statistics_counter(
                    sum=test_statistics[actions.get(n).past].sum + observations.get(n),
                    count=test_statistics[actions.get(n).past].count + 1)
            estimated_state = [((k+1)/(k+1)) for k in range(self.NUMBER_OF_CHANNELS)]
            for k in range(self.NUMBER_OF_CHANNELS):
                if test_statistics.get(k).count > 0:
                    if (test_statistics.get(k).sum / test_statistics.get(k).count) < \
                            (numpy.sqrt(self.NOISE_VARIANCE / (test_statistics.get(k).count *
                                                               self.NUMBER_OF_SAMPLING_ROUNDS)) *
                             scipy.stats.norm.ppf(1 - (self.RAW_FALSE_ALARM_PROBABILITY /
                                                       test_statistics.get(k).count))):
                        estimated_state[k] = 0
            # Reward
            utilities.append(numpy.sum([((lambda: 0, lambda: 1)[self.incumbent_occupancy_states[k][i] == 0 and
                                                                estimated_state[k] == 0]()) -
                                        (self.PENALTY *
                                         (lambda: 0, lambda: 1)[self.incumbent_occupancy_states[k][i] == 1 and
                                                                estimated_state[k] == 0]())
                                        for k in range(self.NUMBER_OF_CHANNELS)]))
            # Belief update and TD-SARSA $\vec{\theta}$ update
            for n in self.cognitive_radio_ensemble.keys():
                past_feature_vector = [beliefs.get(n)[k] * (1 - self.get_pfa2(actions, k))
                                       for k in range(self.NUMBER_OF_CHANNELS)]
                past_q_value = numpy.sum([thetas.get(n)[k] * past_feature_vector[k]
                                          for k in range(self.NUMBER_OF_CHANNELS)])
                for k in range(self.NUMBER_OF_CHANNELS):
                    if actions.get(n).past == k and estimated_state[k] == 0:
                        beliefs[n][k] = 1
                    elif actions.get(n).past == k and estimated_state[k] == 1:
                        beliefs[n][k] = 0
                    elif actions.get(n).past != k and beliefs.get(n)[k] >= 0.5:
                        beliefs[n][k] = max([0.5, (self.LAMBDA * beliefs.get(n)[k])])
                    else:
                        beliefs[n][k] = max([0.5, 1 - (self.LAMBDA * (1 - beliefs.get(n)[k]))])
                present_feature_vector = [beliefs.get(n)[k] * (1 - self.get_pfa3(actions, k))
                                          for k in range(self.NUMBER_OF_CHANNELS)]
                present_q_value = numpy.sum([thetas.get(n)[k] * present_feature_vector[k]
                                             for k in range(self.NUMBER_OF_CHANNELS)])
                step_size = (lambda: 1, lambda: (1 / i))[i > 0]()
                additive_factor = numpy.multiply(past_feature_vector,
                                                 [step_size * (utilities[i] +
                                                               (self.DISCOUNT_FACTOR * present_q_value) -
                                                               past_q_value)])
                thetas[n] = (numpy.array(thetas.get(n)) + additive_factor).tolist()
        print('DistributedCollaborativeSARSA collaborative_sarsa: The system-wide average utility per time-step'
              'in an 18-channel 18-SU 3-PU radio environment with RSSI-based neighbor discovery and '
              'preferential-ballot based channel access rank pre-allocation--in a double Markov chain time-frequency'
              'correlation structure occupancy model--with a sensing/access restriction of 1 is: {}'.format(
                numpy.sum(utilities) / self.NUMBER_OF_TIME_STEPS))
        # Return the output for visualization
        return time_slots, utilities

    # Visualize the utilities per time-slot obtained by the "TD-SARSA with LFA" based RL agent using the Plotly API
    def evaluate(self):
        x_axis, y_axis = self.collaborative_sarsa()
        # Data Trace
        visualization_trace = go.Scatter(x=x_axis,
                                         y=y_axis,
                                         mode=self.PLOTLY_SCATTER_MODE)
        # Figure Layout
        visualization_layout = dict(title='Utilities per time-slot obtained by the "TD-SARSA with LFA" based RL agent '
                                          'in a Spatio-Temporal Markovian PU Occupancy Behavior Model',
                                    xaxis=dict(title=r'Time-slots\ i$'),
                                    yaxis=dict(title=r'$Utility\ \sum_{k=1}^{K}\ (1 - B_k(i)) (1 - \hat{B}_k(i)) - '
                                                     r'\lambda B_k(i) (1 - \hat{B}_k(i))$'))
        # Figure
        visualization_figure = dict(data=[visualization_trace],
                                    layout=visualization_layout)
        # URL
        figure_url = plotly.plotly.plot(visualization_figure,
                                        filename='Utilities_per_time_step_TD_SARSA_LFA')
        # Print the URL in case you're on an environment where a GUI is not available
        print('DistributedCollaborativeSARSA evaluate: '
              'Data Visualization Figure is available at {}'.format(figure_url))

    # The termination sequence
    def __exit__(self, exc_type, exc_val, exc_tb):
        print('DistributedCollaborativeSARSA Termination: Tearing things down...')
        # Termination complete


# Run trigger
if __name__ == '__main__':
    print('DistributedCollaborativeSARSA main: '
          'Starting the distributed multi-agent multi-band collaborative SARSA framework')
    sarsa = DistributedCollaborativeSARSA()
    sarsa.evaluate()
    print('DistributedCollaborativeSARSA main: '
          'Distributed multi-agent multi-band collaborative SARSA framework evaluation completed')
    # Framework evaluation complete
