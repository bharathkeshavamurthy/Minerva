# This module details the design of an opportunistic channel sensing and access scheme for distributed multi-agent
#   multi-band cognitive radio networks. There are two variants to this scheme:
#   1. A pre-allocation of channel access ranks/orders via communication over a common control channel and
#   employs past-horizon channel availability updates for determining channel access; and
#   2. An $\epsilon$-greedy strategy based on g-statistics, past-horizon channel availability metrics, and ACKs for
#   distributed, opportunistic channel sensing and access without any information exchange among the nodes whatsover.
# Author: Bharath Keshavamurthy
# Organization: School of Electrical and Computer Engineering, Purdue University, West Lafayette, IN.
# Copyright (c) 2020. All Rights Reserved.

# The imports
import numpy
import plotly
import scipy.stats
import plotly.graph_objs as go

# Plotly user account credentials for visualization
plotly.tools.set_credentials_file(username='bkeshava',
                                  api_key='W2WL5OOxLcgCzf8NNlgl')


# This entity describes the neighbor discovery, channel access rank/order pre-allocation, and the two variants of this
#   opportunistic channel sensing and access scheme detailed earlier.
class DistributedOpportunisticAccess(object):
    # The number of time-steps of framework evaluation
    NUMBER_OF_TIME_STEPS = 3000

    # The number of sensing sampling rounds per time-step to account for the AWGN observation model
    NUMBER_OF_SAMPLING_ROUNDS = 3000

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

    # The number of members needed for a quorum
    QUORUM_REQUIREMENT = 12

    # The transmission power of all cognitive radio nodes for RSSI analysis (dB)
    TRANSMISSION_POWER = 20

    # The mean of the random variable representing the mobility patterns
    MOBILITY_MEAN = 1

    # The variance of the random variable representing the mobility patterns
    MOBILITY_VARIANCE = 1

    # The RSSI threshold for neighbor discovery (dB)
    RSSI_THRESHOLD = 10

    # The penalty for missed detections
    PENALTY = 1

    # The false alarm probability constraint for threshold determination
    FALSE_ALARM_PROBABILITY = 0.3

    # Plotly Scatter mode
    PLOTLY_SCATTER_MODE = 'lines+markers'

    # A flag for inactive/invalid entries
    INVALIDITY_IDENITIFIER = 7777

    # The initialization sequence
    def __init__(self):
        print('DistributedOpportunisticAccess Initialization: Bringing things up...')
        self.incumbent_occupancy_states, self.average_occupancies = self.simulate_incumbent_occupancy_behavior()
        self.global_rssi_lists = {n: {m: 0.0 for m in range(self.NUMBER_OF_COGNITIVE_RADIOS)}
                                  for n in range(self.NUMBER_OF_COGNITIVE_RADIOS)}
        self.cognitive_radio_ensemble = {n: 0 for n in self.simulate_neighbor_discovery()}
        self.simulate_channel_access_order_allocation()
        # The initialization sequence has been completed

    # Simulate the occupancy behavior of the incumbents according to the double Markov chain time-frequency correlation
    #   structure described by $\vec{p} and \vec{q}$
    def simulate_incumbent_occupancy_behavior(self):
        # Initialization of the outputs to be returned
        _incumbent_occupancy_states = []
        _average_occupancies = 0.0
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
        _average_occupancies = sum([sum([_incumbent_occupancy_states[k][i] for k in range(self.NUMBER_OF_CHANNELS)])
                                    for i in range(self.NUMBER_OF_TIME_STEPS)]) / self.NUMBER_OF_TIME_STEPS
        # Return the output
        return _incumbent_occupancy_states, _average_occupancies

    # Simulate the quorum-controlled RSSI-based neighbor discovery algorithm
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

    # The "greedy distributed learning under pre-allocation" variant of opportunistic spectrum sensing and access
    def greedy_distributed_learning_under_pre_allocation(self):
        # Transient variables
        sensings = {n: [] for n in self.cognitive_radio_ensemble.keys()}
        actions = {n: [] for n in self.cognitive_radio_ensemble.keys()}
        ordered_ensemble = sorted(self.cognitive_radio_ensemble, key=self.cognitive_radio_ensemble.get)
        availabilities = {n: [((k+1)/(k+1)) * numpy.random.random() for k in range(self.NUMBER_OF_CHANNELS)]
                          for n in self.cognitive_radio_ensemble.keys()}
        for i in range(self.NUMBER_OF_TIME_STEPS):
            for n in ordered_ensemble:
                relevant_channels = sorted([k for k in range(self.NUMBER_OF_CHANNELS)],
                                           key=lambda x: availabilities[n][x],
                                           reverse=True)[:len(ordered_ensemble)]
                channel_pairs = [(k, _k) for k in relevant_channels for _k in relevant_channels if k != _k]
                # Smallest Kullback-Liebler distance based on channel availabilities
                d_min = min([abs(availabilities.get(n)[k] - availabilities.get(n)[_k]) for k, _k in channel_pairs])
                # This projection is better--otherwise, the $\frac{4}{d_{min}^{2}}$ term dominates...
                d_min = (lambda: 0.2,
                         lambda: d_min)[d_min >= 0.2]()
                beta = max([20, (4/(d_min**2))]) + 1
                epsilon = (lambda: min([beta/i, 1]), lambda: 1)[i == 0]()
                action = (lambda: numpy.random.randint(0, self.NUMBER_OF_CHANNELS - 1),
                          lambda: relevant_channels[ordered_ensemble.index(n)])[numpy.random.random() > epsilon]()
                test_statistic_n_avg = sum([((t+1)/(t+1)) *
                                            ((numpy.random.normal(0,
                                                                  numpy.sqrt(self.CHANNEL_IMPULSE_RESPONSE_VARIANCE)
                                                                  ) * self.incumbent_occupancy_states[action][i]
                                              ) + numpy.random.normal(0,
                                                                      numpy.sqrt(self.NOISE_VARIANCE)
                                                                      )
                                             )
                                            for t in range(self.NUMBER_OF_SAMPLING_ROUNDS)
                                            ]
                                           ) / self.NUMBER_OF_SAMPLING_ROUNDS
                estimated_occupancy = (lambda: 0,
                                       lambda: 1)[test_statistic_n_avg >
                                                  (numpy.sqrt(self.NOISE_VARIANCE /
                                                              self.NUMBER_OF_SAMPLING_ROUNDS)).item() *
                                                  scipy.stats.norm.ppf(1 - self.FALSE_ALARM_PROBABILITY).item()]()
                sensings[n].append(action)
                if estimated_occupancy == 0:
                    actions[n].append(action)
                else:
                    actions[n].append(self.INVALIDITY_IDENITIFIER)
                availabilities[n][action] += 1 - estimated_occupancy
                # +1 for the initial availability random assignment
                availabilities[n][action] /= (sum([((s+1)/(s+1)) for s in sensings.get(n) if s == action]) + 1)
        su_throughputs = [sum([((actions.get(n)[i]+1)/(actions.get(n)[i]+1))
                               for n in self.cognitive_radio_ensemble.keys()
                               if (actions.get(n)[i] != self.INVALIDITY_IDENITIFIER and
                                   self.incumbent_occupancy_states[actions.get(n)[i]][i] == 0)])
                          for i in range(self.NUMBER_OF_TIME_STEPS)]
        pu_interferences = [sum([((actions.get(n)[i]+1)/(actions.get(n)[i]+1))
                                 for n in self.cognitive_radio_ensemble.keys()
                                 if (actions.get(n)[i] != self.INVALIDITY_IDENITIFIER and
                                     self.incumbent_occupancy_states[actions.get(n)[i]][i] == 1)])
                            for i in range(self.NUMBER_OF_TIME_STEPS)]
        utilities = [su_throughputs[i] - (self.PENALTY * pu_interferences[i]) for i in range(self.NUMBER_OF_TIME_STEPS)]
        print('DistributedOpportunisticAccess greedy_distributed_learning_under_pre_allocation: '
              'Performance Analysis of "the greedy distributed learning under pre-allocation" variant of the '
              'distributed opportunistic spectrum sensing and access policy: '
              'SU Throughput = {} | PU Interference = {}'.format(sum(su_throughputs) / self.NUMBER_OF_TIME_STEPS,
                                                                 sum(pu_interferences) /
                                                                 (self.NUMBER_OF_TIME_STEPS * self.average_occupancies)
                                                                 )
              )
        return [i for i in range(self.NUMBER_OF_TIME_STEPS)], utilities

    # The distributed learning and allocation variant of opportunistic spectrum sensing and access
    def distributed_learning_and_allocation(self):
        # Transient variables
        choice = 1
        availabilities = {n: [k-k for k in range(self.NUMBER_OF_CHANNELS)]
                          for n in self.cognitive_radio_ensemble.keys()}
        sensings = {n: [k for k in range(self.NUMBER_OF_CHANNELS)]
                    for n in self.cognitive_radio_ensemble.keys()}
        actions = {n: [((i+1)/(i+1)) * self.INVALIDITY_IDENITIFIER
                       for i in range(self.NUMBER_OF_TIME_STEPS)]
                   for n in self.cognitive_radio_ensemble.keys()}
        acknowledgements = {n: True for n in self.cognitive_radio_ensemble.keys()}
        # Initialization
        for n in self.cognitive_radio_ensemble.keys():
            for k in range(self.NUMBER_OF_CHANNELS):
                test_statistic_n_avg = sum([((t + 1) / (t + 1)) *
                                            ((numpy.random.normal(0,
                                                                  numpy.sqrt(self.CHANNEL_IMPULSE_RESPONSE_VARIANCE)
                                                                  ) * self.incumbent_occupancy_states[k][0]
                                              ) + numpy.random.normal(0,
                                                                      numpy.sqrt(self.NOISE_VARIANCE)
                                                                      )
                                             )
                                            for t in range(self.NUMBER_OF_SAMPLING_ROUNDS)
                                            ]
                                           ) / self.NUMBER_OF_SAMPLING_ROUNDS
                estimated_occupancy = (lambda: 0,
                                       lambda: 1)[test_statistic_n_avg >
                                                  (numpy.sqrt(self.NOISE_VARIANCE /
                                                              self.NUMBER_OF_SAMPLING_ROUNDS)).item() *
                                                  scipy.stats.norm.ppf(1 - self.FALSE_ALARM_PROBABILITY).item()]()
                availabilities[n][k] += 1 - estimated_occupancy
                # +1 for the initial availability random assignment
                availabilities[n][k] /= (sum([((s+1)/(s+1)) for s in sensings.get(n) if s == k]) + 1)
            actions[n][0] = sorted([k for k in range(self.NUMBER_OF_CHANNELS)],
                                   key=lambda x: availabilities.get(n)[x],
                                   reverse=True)[choice]
            # Check for collisions
            for m in self.cognitive_radio_ensemble.keys():
                if m == n:
                    continue
                else:
                    if actions.get(m)[0] == actions.get(n)[0]:
                        acknowledgements[n] = False
        # Core Loop
        for i in range(1, self.NUMBER_OF_TIME_STEPS):
            for n in self.cognitive_radio_ensemble.keys():
                g_statistics = [availabilities.get(n)[k] + numpy.sqrt((2 * numpy.log(i))/sum([((s+1)/(s+1))
                                                                                              for s in sensings.get(n)
                                                                                              if s == k]))
                                for k in range(self.NUMBER_OF_CHANNELS)]
                if not acknowledgements.get(n):
                    choice = numpy.random.randint(0, self.NUMBER_OF_CHANNELS - 1)
                action = sorted([k for k in range(self.NUMBER_OF_CHANNELS)],
                                key=lambda x: g_statistics[x],
                                reverse=True)[choice]
                test_statistic_n_avg = sum([((t + 1) / (t + 1)) *
                                            ((numpy.random.normal(0,
                                                                  numpy.sqrt(self.CHANNEL_IMPULSE_RESPONSE_VARIANCE)
                                                                  ) * self.incumbent_occupancy_states[action][i]
                                              ) + numpy.random.normal(0,
                                                                      numpy.sqrt(self.NOISE_VARIANCE)
                                                                      )
                                             )
                                            for t in range(self.NUMBER_OF_SAMPLING_ROUNDS)
                                            ]
                                           ) / self.NUMBER_OF_SAMPLING_ROUNDS
                estimated_occupancy = (lambda: 0,
                                       lambda: 1)[test_statistic_n_avg >
                                                  (numpy.sqrt(self.NOISE_VARIANCE /
                                                              self.NUMBER_OF_SAMPLING_ROUNDS)).item() *
                                                  scipy.stats.norm.ppf(1 - self.FALSE_ALARM_PROBABILITY).item()]()
                sensings[n].append(action)
                # Unavailable channel = ACK
                if estimated_occupancy == 1:
                    acknowledgements[n] = True
                else:
                    actions[n][i] = action
                    # Available channel and Collision = Action taken and NACK
                    if sum([((actions.get(m)[i]+1)/(actions.get(m)[i]+1))
                            for m in self.cognitive_radio_ensemble.keys()
                            if (m != n and actions.get(m)[i] == action)]) > 0:
                        acknowledgements[n] = False
                    # Available channel and No collision = Action taken and ACK
                    else:
                        acknowledgements[n] = True
                availabilities[n][action] += 1 - estimated_occupancy
                # +1 for the initial availability random assignment
                availabilities[n][action] /= (sum([((s+1)/(s+1)) for s in sensings[n] if s == action]) + 1)
        su_throughputs = [sum([((actions.get(n)[i]+1)/(actions.get(n)[i]+1))
                               for n in self.cognitive_radio_ensemble.keys()
                               if (actions.get(n)[i] != self.INVALIDITY_IDENITIFIER and
                                   self.incumbent_occupancy_states[actions.get(n)[i]][i] == 0)])
                          for i in range(self.NUMBER_OF_TIME_STEPS)]
        pu_interferences = [sum([((actions.get(n)[i]+1)/(actions.get(n)[i]+1))
                                 for n in self.cognitive_radio_ensemble.keys()
                                 if (actions.get(n)[i] != self.INVALIDITY_IDENITIFIER and
                                     self.incumbent_occupancy_states[actions.get(n)[i]][i] == 1)])
                            for i in range(self.NUMBER_OF_TIME_STEPS)]
        utilities = [su_throughputs[i] - (self.PENALTY * pu_interferences[i])
                     for i in range(self.NUMBER_OF_TIME_STEPS)]
        print('DistributedOpportunisticAccess distributed_learning_and_allocation: '
              'Performance Analysis of "the distributed learning and allocation" variant of the '
              'distributed opportunistic spectrum sensing and access policy: '
              'SU Throughput = {} | PU Interference = {}'.format(sum(su_throughputs) / (self.NUMBER_OF_TIME_STEPS - 1),
                                                                 sum(pu_interferences) /
                                                                 (self.NUMBER_OF_TIME_STEPS * self.average_occupancies)
                                                                 )
              )
        return utilities

    # Visualize the utilities per time-slot obtained by both variants of this distributed opportunistic spectrum sensing
    #   and access policy using the Plotly API
    def evaluate(self):
        x_axis, y_axis1 = self.greedy_distributed_learning_under_pre_allocation()
        y_axis2 = self.distributed_learning_and_allocation()
        # Data Trace
        visualization_trace1 = go.Scatter(x=x_axis,
                                          y=y_axis1,
                                          mode=self.PLOTLY_SCATTER_MODE)
        visualization_trace2 = go.Scatter(x=x_axis,
                                          y=y_axis2,
                                          mode=self.PLOTLY_SCATTER_MODE)
        # Figure Layout
        visualization_layout = dict(
            title='Utilities per time-slot obtained by both variants of the distributed opportunistic spectrum sensing '
                  'and access policy in a Spatio-Temporal Markovian PU Occupancy Behavior Model',
            xaxis=dict(title=r'Time-slots\ i$'),
            yaxis=dict(title=r'$Utility\ \sum_{k=1}^{K}\ (1 - B_k(i)) (1 - \hat{B}_k(i)) - '
                             r'\lambda B_k(i) (1 - \hat{B}_k(i))$'))
        # Figure
        visualization_figure = dict(data=[visualization_trace1, visualization_trace2],
                                    layout=visualization_layout)
        # URL
        figure_url = plotly.plotly.plot(visualization_figure,
                                        filename='Utilities_per_time_step_Distributed_Opportunistic_Access')
        # Print the URL in case you're on an environment where a GUI is not available
        print('DistributedOpportunisticAccess evaluate: '
              'Data Visualization Figure is available at {}'.format(figure_url))

    # The termination sequence
    def __exit__(self, exc_type, exc_val, exc_tb):
        print('DistributedOpportunisticAccess Termination: Tearing things down...')
        # The termination sequence has been completed


# Run Trigger
if __name__ == '__main__':
    print('DistributedOpportunisticAccess main: Starting the evaluation of the two variants of the '
          'distributed opportunistic spectrum sensing and access policy...')
    expediency = DistributedOpportunisticAccess()
    expediency.evaluate()
    print('DistributedOpportunisticAccess main: Completed the evaluation of both variants of the '
          'distributed opportunistic spectrum sensing and access policy...')
    # The framework evaluation has been completed
