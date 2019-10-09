# This entity describes the Greedy Algorithm for Clustering highly correlated channels in the discretized spectrum of
#   interest for our Cognitive Radio Research.
# Author: Bharath Keshavamurthy
# Organization: School of Electrical and Computer Engineering, Purdue University, West Lafayette, IN.
# Copyright (c) 2019. All Rights Reserved.

# Using the same notations as in the reference paper....
# Reference: M Gao, et. al., "Fast Spectrum Sensing: A Combination of Channel Correlation and Markov Model", 2014.

# The imports
import numpy
from collections import namedtuple


# This class details the Greedy Clustering Algorithm outputting Clustered Groups of Highly Correlated Channels and their
#   corresponding Detected Channels (DCs) used to estimate the occupancy states of other channels in its group (ECs).
class GreedyClusteringAlgorithm(object):

    # The number of channels $N$ in the discretized spectrum of interest
    NUMBER_OF_CHANNELS = 18

    # The number of episodes $M$ of SU sensing/interaction with the radio environment
    NUMBER_OF_EPISODES = 1000

    # The \rho_{th} parameter for channel correlation transformation
    CORRELATION_THRESHOLD = 0.7

    # The initialization sequence
    def __init__(self):
        print('[INFO] GreedyClusteringAlgorithm Initialization: Bringing things up...')
        # Both the spatial Markov chain and the temporal Markov chain exhibit the same transition model and
        #   steady-state model as shown below.
        # The start probabilities a.k.a the steady-state probabilities
        self.start_probabilities = {0: 0.4,
                                    1: 0.6
                                    }
        # The transition probabilities
        self.transition_probabilities = {0: {0: 0.7,
                                             1: 0.3
                                             },
                                         1: {
                                             0: 0.2,
                                             1: 0.8}
                                         }
        # The true incumbent(s) occupancy behavior
        self.true_pu_occupancy_states = self.get_true_pu_occupancy_states()
        # Construct the correlation matrix $A$ from the true PU occupancy states
        self.correlation_matrix = self.construct_correlation_matrix()
        # The output of the Greedy Clustering Algorithm
        self.greedy_clustering_output = namedtuple('GREEDY_CLUSTERING_OUTPUT',
                                                   ['G', 'D'])

    # Get the true PU occupancy states by emulating the occupancy behavior of an incumbent(s) exhibiting Markovian
    #   correlation
    def get_true_pu_occupancy_states(self):
        # The output to be returned
        pu_occupancy_states = []
        # The initial state band-0 at time-0 $X_{0}(0)$
        state = 1
        random_sample = numpy.random.random_sample()
        if random_sample > self.start_probabilities[1]:
            state = 0
        pu_occupancy_states.append([state])
        # Go through row-0: Only temporal correlation $X_{0}(m),\ \forall m \in \{1, 2, 3, \dots, M\}$
        for episode in range(1, self.NUMBER_OF_EPISODES):
            random_sample = numpy.random.random_sample()
            if state == 0 and random_sample < self.transition_probabilities[0][1]:
                state = 1
            elif state == 0 and random_sample > self.transition_probabilities[0][1]:
                state = 0
            elif state == 1 and random_sample < self.transition_probabilities[1][0]:
                state = 0
            else:
                state = 1
            pu_occupancy_states[0].append(state)
        state = pu_occupancy_states[0][0]
        # Go through column-0: Only spatial correlation $X_{n}(0),\ \forall n \in \{1, 2, 3, \dots, N\}$
        for channel in range(1, self.NUMBER_OF_CHANNELS):
            random_sample = numpy.random.random_sample()
            if state == 0 and random_sample < self.transition_probabilities[0][1]:
                state = 1
            elif state == 0 and random_sample > self.transition_probabilities[0][1]:
                state = 0
            elif state == 1 and random_sample < self.transition_probabilities[1][0]:
                state = 0
            else:
                state = 1
            pu_occupancy_states.append([state])
        # Go through the rest of the N x M matrix: Both temporal correlation and spatial correlation
        # $X_{n}(m),\ \forall n \in \{1, 2, 3, \dots, N\}\ and\ m \in \{1, 2, 3, \dots, M\}$
        for channel in range(1, self.NUMBER_OF_CHANNELS):
            for episode in range(1, self.NUMBER_OF_EPISODES):
                previous_spatial_state = pu_occupancy_states[channel - 1][episode]
                previous_temporal_state = pu_occupancy_states[channel][episode - 1]
                occupied_probability = \
                    self.transition_probabilities[previous_spatial_state][1] * \
                    self.transition_probabilities[previous_temporal_state][1]
                random_sample = numpy.random.random_sample()
                if random_sample < occupied_probability:
                    pu_occupancy_states[channel].append(1)
                else:
                    pu_occupancy_states[channel].append(0)
        return pu_occupancy_states

    # Construct the correlation matrix $A$ from the true PU occupancy states
    def construct_correlation_matrix(self):
        # Setting up the output collection to be returned
        channel_correlation_matrix = []
        for channel in range(self.NUMBER_OF_CHANNELS):
            channel_correlation_matrix.append([])
        # Channel $i$
        for i in range(self.NUMBER_OF_CHANNELS):
            # Channel $j$
            for j in range(self.NUMBER_OF_CHANNELS):
                correlation_sum = 0
                # The episode loop iterator $m$
                for m in range(self.NUMBER_OF_EPISODES):
                    # The XNOR operation
                    if self.true_pu_occupancy_states[i][m] == self.true_pu_occupancy_states[j][m]:
                        correlation_sum += 1
                channel_correlation_matrix[i].append(correlation_sum / self.NUMBER_OF_EPISODES)
        return channel_correlation_matrix

    # The core greedy clustering algorithm
    # We'll follow the same exact GC algorithm laid down in the reference work...
    def cluster(self):
        # The group of clusters $G$
        g = []
        # The group od detected channels corresponding to these clusters $D$
        d = []
        # The group of correlated channel sets $s_{i},\ \forall i \in \{0, 1, 2, \dots, N\}$
        correlated_channel_sets = {k: [] for k in range(self.NUMBER_OF_CHANNELS)}
        for row in range(self.NUMBER_OF_CHANNELS):
            for column in range(self.NUMBER_OF_CHANNELS):
                if self.correlation_matrix[row][column] >= self.CORRELATION_THRESHOLD:
                    correlated_channel_sets[row].append(column)
        # The set $U$
        u = [k for k in range(self.NUMBER_OF_CHANNELS)]
        while len(u) > 0:
            # Select $s_{i}$ that maximizes $|s_{i} \cap U|$
            max_hits = 0
            max_hit_index = 0
            for i, s_i in correlated_channel_sets.items():
                hits = 0
                for j in s_i:
                    if j in u:
                        hits += 1
                if hits > max_hits:
                    max_hits = hits
                    max_hit_index = i
            # The detected channel $d_{k}$
            d_k = max_hit_index
            # The group corresponding to the detected channel $g_{k}$
            # $g_{k} \leftarrow U \cap s_{i}$
            g_k = []
            for j in correlated_channel_sets[d_k]:
                if j in u:
                    g_k.append(j)
            # $U \leftarrow U - s_{i}$
            for j in correlated_channel_sets[d_k]:
                del u[j]
            # $G \leftarrow G \cup \{g_{k}\}$
            g.append(g_k)
            # $D \leftarrow D \cup \{d_{k}\}$
            d.append(d_k)
        return self.greedy_clustering_output(G=g,
                                             D=d)

    # The termination sequence
    def __exit__(self, exc_type, exc_val, exc_tb):
        print('[INFO] GreedyClusteringAlgorithm Termination: Tearing things down...')
        # Nothing to do...


# Run Trigger
if __name__ == '__main__':
    print('[INFO] GreedyClusteringAlgorithm main: Triggering the Greedy Clustering Algorithm...')
    agent = GreedyClusteringAlgorithm()
    clustered_output = agent.cluster()
    print('[INFO] GreedyClusteringAlgorithm main: The set of grouped channels is - {}'.format(str(clustered_output.G)))
    print('[INFO] GreedyClusteringAlgorithm main: The set of corresponding DCs is - {}'.format(str(clustered_output.D)))
