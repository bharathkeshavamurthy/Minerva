# This entity describes the Minimum Entropy Increment Algorithm (MEI) for Clustering highly correlated channels
#   in the discretized spectrum of interest for our Cognitive Radio Research.
# Author: Bharath Keshavamurthy
# Organization: School of Electrical and Computer Engineering, Purdue University, West Lafayette, IN.
# Copyright (c) 2019. All Rights Reserved.

# Using the same notations as in the reference paper...
# Reference: M Gao, et. al., "Fast Spectrum Sensing: A Combination of Channel Correlation and Markov Model", 2014.

# The imports
import numpy
from collections import namedtuple


# This class describes the Minimum Entropy Increment Clustering Algorithm (MEI) designed to bypass some of the
#   limitations and drawbacks of the Greedy Clustering Algorithm (GC).
class MinimumEntropyIncrementAlgorithm(object):

    # The number of channels $N$ in the discretized spectrum of interest
    NUMBER_OF_CHANNELS = 18

    # The number of episodes $M$ of SU sensing/interaction with the radio environment
    NUMBER_OF_EPISODES = 1000

    # The number of clusters $T$ mandated by our evaluation framework constraints
    NUMBER_OF_CLUSTERS = 6

    # The initialization sequence
    def __init__(self):
        print('[INFO] MinimumEntropyIncrementAlgorithm Initialization: Bringing things up...')
        # The start probabilities a.k.a steady-state probabilities of the spatial and temporal Markov chains
        self.start_probabilities = {0: 0.4,
                                    1: 0.6
                                    }
        # The transition probabilities of both the spatial and the temporal Markov chains
        self.transition_probabilities = {0: {0: 0.7,
                                             1: 0.3
                                             },
                                         1: {0: 0.2,
                                             1: 0.8
                                             }
                                         }
        # The true PU occupancy states over these $M$ episodes
        self.true_pu_occupancy_states = self.get_true_pu_occupancy_states()
        # Construct the channel correlation matrix using the XNOR operation
        self.channel_correlation_matrix = self.get_channel_correlation_matrix()
        # The set $G$ which will contain the set of clusters at the end of this MEI algorithm
        self.g = [k for k in range(self.NUMBER_OF_CHANNELS)]
        # The set $D$ which will contain the set of Detected Channels (DCs) corresponding to the clusters in $G$
        #   at the end of this MEI algorithm
        self.d = [k for k in range(self.NUMBER_OF_CHANNELS)]
        # The output structure/format of the MPEG evaluation routine
        self.mpeg_output_format = namedtuple('MPEG_ROUTINE_OUTPUT',
                                             ['mpeg', 'd'])
        # The output structure/format of the MEI algorithm
        self.mei_output_format = namedtuple('MEI_OUTPUT',
                                            ['G', 'D'])

    # Simulate the occupancy behavior of the incumbents over these $M$ episodes
    def get_true_pu_occupancy_states(self):
        # The output to be returned
        pu_occupancy_states = []
        state = 1
        # The occupancy of band-0 and time-0 $X_{0}(0)$
        if numpy.random.random_sample() > self.start_probabilities[1]:
            state = 0
        pu_occupancy_states.append([state])
        # The occupancy of band-0 across all the episodes $X_{0}(m),\ \forall m \in \{1, 2, 3, \dots, M\}$
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
        # Get the appropriate previous state cell in the output N x M matrix
        state = pu_occupancy_states[0][0]
        # The occupancy of all the bands in episode-0 $X_{n}(0),\ \forall n \in \{1, 2, 3, \dots, N\}$
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
        # Go through the rest of the N x M matrix using both the spatial Markov chain statistics and the temporal
        #   Markov chain statistics
        # $X_{n}(m),\ \forall n \in \{1, 2, 3, \dots, N\}\ and\ m \in \{1, 2, 3, \dots, M\}$
        for channel in range(1, self.NUMBER_OF_CHANNELS):
            for episode in range(1, self.NUMBER_OF_EPISODES):
                random_sample = numpy.random.random_sample()
                previous_spatial_state = pu_occupancy_states[channel - 1][episode]
                previous_temporal_state = pu_occupancy_states[channel][episode - 1]
                occupied_probability = \
                    self.transition_probabilities[previous_spatial_state][1] * \
                    self.transition_probabilities[previous_temporal_state][1]
                if random_sample < occupied_probability:
                    state = 1
                else:
                    state = 0
                pu_occupancy_states[channel].append(state)
        return pu_occupancy_states

    # Construct the channel correlation matrix $A$ using the XNOR operation
    def get_channel_correlation_matrix(self):
        # Setting up the output to the returned
        correlation_matrix = []
        for channel in range(self.NUMBER_OF_CHANNELS):
            correlation_matrix.append([])
        # Row index: Channel $i$
        for i in range(self.NUMBER_OF_CHANNELS):
            # Column index: Channel $j$
            for j in range(self.NUMBER_OF_CHANNELS):
                correlation_sum = 0
                # Episode iterator $m$
                for m in range(self.NUMBER_OF_EPISODES):
                    if self.true_pu_occupancy_states[i][m] == self.true_pu_occupancy_states[j][m]:
                        correlation_sum += 1
                correlation_matrix[i].append(correlation_sum / self.NUMBER_OF_EPISODES)
        return correlation_matrix

    # Calculate the Minimum Prediction Entropy of the given Group (MPEG)
    def calculate_mpeg(self, g_k):
        # The internal evaluation members are initialized here
        peg_gk = 0
        mpeg_gk = 1e100
        d_k = None
        for c_i in g_k:
            for c_j in range(self.NUMBER_OF_CHANNELS):
                # $j \neq i$ in the summation
                if c_j == c_i:
                    continue
                # $H(c_{j}|c_{i}) = -rho_{ij} \log(\rho_{ij}) - (1 - \rho_{ij}) log(1 - \rho_{ij})$
                peg_gk += (-1 *
                           self.channel_correlation_matrix[c_i][c_j] *
                           numpy.log(self.channel_correlation_matrix[c_i][c_j])
                           ) + \
                          (-1 *
                           (1 - self.channel_correlation_matrix[c_i][c_j]) *
                           numpy.log(1 - self.channel_correlation_matrix[c_i][c_j])
                           )
            # $MPEG(g_{k})$ evaluation and subsequent inference of $d_{k}$
            if peg_gk < mpeg_gk:
                mpeg_gk = peg_gk
                d_k = c_i
        return self.mpeg_output_format(mpeg=mpeg_gk,
                                       d=d_k)

    # The core algorithm: Minimum Entropy Increment (MEI)
    # We'll follow the same exact MEI algorithm as laid down in the reference work...
    def cluster(self):
        # The internal members are initialized here
        mei = 1e100
        gi_min = None
        gj_min = None
        gnew_min = None
        di_min = None
        dj_min = None
        dnew_min = None
        # while $|G| > T$
        while len(self.g) > self.NUMBER_OF_CLUSTERS:
            for g_i in self.g:
                # $MPEG(g_{i})$ and $d_{i}$
                mpeg_output = self.calculate_mpeg(g_i)
                mpeg_gi = mpeg_output.mpeg
                d_i = mpeg_output.d
                for g_j in self.g:
                    # $MPEG(g_{j})$ and $d_{j}$
                    mpeg_output = self.calculate_mpeg(g_j)
                    mpeg_gj = mpeg_output.mpeg
                    d_j = mpeg_output.d
                    # Reduce the consumption of sensing by merging these two channel groups
                    g_new = g_i + g_j
                    # $MPEG(g_{new})$ and $d_{new}$
                    mpeg_output = self.calculate_mpeg(g_new)
                    mpeg_gnew = mpeg_output.mpeg
                    d_new = mpeg_output.d
                    # The entropy increment as a result of the merging
                    ei = mpeg_gnew - mpeg_gi - mpeg_gj
                    # MEI evaluation and subsequent inference of the merged group with the smallest entropy increment
                    if ei < mei:
                        mei = ei
                        gi_min = g_i
                        gj_min = g_j
                        gnew_min = g_new
                        di_min = d_i
                        dj_min = d_j
                        dnew_min = d_new
            # $G \leftarrow G - g_{i} - g_{j} + g_{new}$
            self.g.remove(gi_min)
            self.g.remove(gj_min)
            self.g.append(gnew_min)
            # $D \leftarrow D - d_{i} - d_{j} + d_{new}$
            self.d.remove(di_min)
            self.d.remove(dj_min)
            self.d.append(dnew_min)
        # Return the output in the prescribed format in case an external entity needs it...
        return self.mei_output_format(G=self.g,
                                      D=self.d)

    # The termination sequence
    def __exit__(self, exc_type, exc_val, exc_tb):
        print('[INFO] MinimumEntropyIncrementAlgorithm Termination: Tearing things down...')
        # Nothing to do...


# Run Trigger
if __name__ == '__main__':
    print('[INFO] MinimumEntropyIncrementAlgorithm main: Triggering the Minimum Entropy Increment Algorithm...')
    agent = MinimumEntropyIncrementAlgorithm()
    clustered_output = agent.cluster()
    print('[INFO] MinimumEntropyIncrementAlgorithm main: The set of grouped channels is - {}'.format(
        str(clustered_output.G))
    )
    print('[INFO] MinimumEntropyIncrementAlgorithm main: The set of corresponding DCs is - {}'.format(
        str(clustered_output.D))
    )
