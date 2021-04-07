# This entity details the Minimum Entropy Merging (MEM) Algorithm for estimating the occupancy states of channels in the
#   discretized spectrum of interest for our Cognitive Radio Research.
# This entity is the improved version modified to handle the new changes to the MDP transition model.
# Author: Bharath Keshavamurthy
# Organization: School of Electrical and Computer Engineering, Purdue University, West Lafayette, IN.
# Copyright (c) 2020. All Rights Reserved.

# Using the same notations as in the reference paper...
# Reference: M Gao, et. al., "Fast Spectrum Sensing: A Combination of Channel Correlation and Markov Model", 2014.

# %%% IMPROVED %%%

# The imports
import numpy
from enum import Enum
from collections import namedtuple


# The Occupancy State Enumeration
class OccupancyState(Enum):
    # The channel is idle $X_{n}(m) = 0$
    IDLE = 0
    # The channel is occupied $X_{n}(m) = 1$
    OCCUPIED = 1


# The different kinds of clustering for the Channel Correlation based Estimation (CCE) technique
class ClusteringTechnique(Enum):
    # The Greedy Clustering Algorithm (GC)
    GREEDY = 0
    # The Minimum Entropy Increment Clustering Algorithm (MEI)
    MINIMUM_ENTROPY_INCREMENT = 1


# This class describes the evaluation of the Minimum Entropy Merging (MEM) Algorithm which involves a combined strategy
#   incorporating both Channel Correlation based Estimation (CCE) and Markov Process based Estimation (MPE).
# This is the improved version capable of handling the correct double Markov chain structure.
class ClusteringAndMarkovianModeling(object):

    # The number of channels $N$ in the discretized spectrum of interest
    NUMBER_OF_CHANNELS = 18

    # The number of episodes $M$ of SU sensing/interaction with the radio environment
    NUMBER_OF_EPISODES = 1000

    # The correlation threshold $\rho_{th}$ for constructing the Channel Correlation Matrix
    CORRELATION_THRESHOLD = 0.77

    # The number of clusters $T$ mandated by our evaluation framework constraints
    NUMBER_OF_CLUSTERS = 6

    # The penalty $\mu$ for missed detections
    MU = -1

    # The scatter plot visualization mode in the Plotly API
    PLOTLY_VISUALIZATION_MODE = 'lines+markers'

    # The initialization sequence
    def __init__(self):
        print('[INFO] ClusteringAndMarkovianModeling Initialization: Bringing things up...')
        # The start probabilities a.k.a steady-state probabilities of both the spatial and the temporal Markov chains
        self.start_probabilities = {0: 0.4, 1: 0.6}
        # The transition probabilities of both the spatial and the temporal Markov chains defined by q0 and q1
        self.temporal_transition_matrix = {0: {0: 0.7, 1: 0.3}, 1: {0: 0.2, 1: 0.8}}
        # The double Markov chain transition probabilities matrix defined by parameters p00, p01, p10, and p11
        self.transition_matrix = {'00': {0: 0.9, 1: 0.1}, '01': {0: 0.7, 1: 0.3},
                                  '10': {0: 0.7, 1: 0.3}, '11': {0: 0.3, 1: 0.7}}
        # The output format of the Greedy Clustering and the Minimum Entropy Increment Channel Correlation Algorithms
        self.clustering_output_format = namedtuple('CLUSTERING_OUTPUT',
                                                   ['G', 'D'])
        # The output format of the calculate_mpeg routine for the Minimum Entropy Increment Algorithm
        self.mpeg_output_format = namedtuple('MPEG_ROUTINE_OUTPUT',
                                             ['mpeg', 'd'])
        # The output format of the cce_estimate_occupancy_states and the mpe_estimate_occupancy_states routines
        self.estimation_routine_output = namedtuple('ESTIMATION_ROUTINE_OUTPUT',
                                                    ['estimated_occupancy_states', 'likelihoods'])
        # The analytics returned by this agent
        self.analytics = namedtuple('ANALYTICS',
                                    ['su_throughput', 'pu_interference'])
        # The true occupancy states of the incumbents in the network
        self.true_occupancy_states = {k: [] for k in range(self.NUMBER_OF_CHANNELS)}
        # The channel correlation matrix
        self.channel_correlation_matrix = self.construct_correlation_matrix()

    # Simulate the incumbent occupancy behavior in the spectrum of interest according to the true correlation model
    def simulate_pu_occupancy(self):
        # Set Element (0,0)
        self.true_occupancy_states[0].append(
            (lambda: 1, lambda: 0)[numpy.random.random_sample() > self.start_probabilities[1]]()
        )
        # Temporal chain: Complete row 0 (Use statistics q0 and q1)
        for i in range(1, self.NUMBER_OF_EPISODES):
            if self.true_occupancy_states[0][i - 1] == 1:
                self.true_occupancy_states[0].append(
                    (lambda: 1, lambda: 0)[numpy.random.random_sample() > self.temporal_transition_matrix[1][1]]()
                )
            else:
                self.true_occupancy_states[0].append(
                    (lambda: 1, lambda: 0)[numpy.random.random_sample() > self.temporal_transition_matrix[0][1]]()
                )
        # Spatial chain: Complete column 0 (Use statistics q0 and q1)
        for k in range(1, self.NUMBER_OF_CHANNELS):
            if self.true_occupancy_states[k - 1][0] == 1:
                self.true_occupancy_states[k].append(
                    (lambda: 1, lambda: 0)[numpy.random.random_sample() > self.temporal_transition_matrix[1][1]]()
                )
            else:
                self.true_occupancy_states[k].append(
                    (lambda: 1, lambda: 0)[numpy.random.random_sample() > self.temporal_transition_matrix[0][1]]()
                )
        # Complete the rest of the kxt matrix (Use statistics p00, p01, p10, and p11)
        for k in range(1, self.NUMBER_OF_CHANNELS):
            for i in range(1, self.NUMBER_OF_EPISODES):
                if self.true_occupancy_states[k - 1][i] == 0 and self.true_occupancy_states[k][i - 1] == 0:
                    self.true_occupancy_states[k].append(
                        (lambda: 1, lambda: 0)[numpy.random.random_sample() > self.transition_matrix['00'][1]]()
                    )
                elif self.true_occupancy_states[k - 1][i] == 0 and self.true_occupancy_states[k][i - 1] == 1:
                    self.true_occupancy_states[k].append(
                        (lambda: 1, lambda: 0)[numpy.random.random_sample() > self.transition_matrix['01'][1]]()
                    )
                elif self.true_occupancy_states[k - 1][i] == 1 and self.true_occupancy_states[k][i - 1] == 0:
                    self.true_occupancy_states[k].append(
                        (lambda: 1, lambda: 0)[numpy.random.random_sample() > self.transition_matrix['10'][1]]()
                    )
                else:
                    self.true_occupancy_states[k].append(
                        (lambda: 1, lambda: 0)[numpy.random.random_sample() > self.transition_matrix['11'][1]]()
                    )
        # Return the collection in case an external method needs it...
        return self.true_occupancy_states

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
                    if self.true_occupancy_states[i][m] == self.true_occupancy_states[j][m]:
                        correlation_sum += 1
                channel_correlation_matrix[i].append(correlation_sum / self.NUMBER_OF_EPISODES)
        return channel_correlation_matrix

    # The core greedy clustering algorithm
    # We'll follow the same exact GC algorithm laid down in the reference work...
    def greedy_cluster(self):
        # The group of clusters $G$
        g = []
        # The group od detected channels corresponding to these clusters $D$
        d = []
        # The group of correlated channel sets $s_{i},\ \forall i \in \{0, 1, 2, \dots, N\}$
        correlated_channel_sets = {k: [] for k in range(self.NUMBER_OF_CHANNELS)}
        for row in range(self.NUMBER_OF_CHANNELS):
            for column in range(self.NUMBER_OF_CHANNELS):
                if self.channel_correlation_matrix[row][column] >= self.CORRELATION_THRESHOLD:
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
            for j in u:
                if j in correlated_channel_sets[d_k]:
                    g_k.append(j)
            # $U \leftarrow U - s_{i}$
            for j in correlated_channel_sets[d_k]:
                if j in u:
                    u.remove(j)
            # $G \leftarrow G \cup \{g_{k}\}$
            g.append(g_k)
            # $D \leftarrow D \cup \{d_{k}\}$
            d.append(d_k)
        return self.clustering_output_format(G=g,
                                             D=d)

    # Calculate the Minimum Prediction Entropy of the given Group (MPEG)
    def calculate_mpeg(self, g_k):
        # The internal evaluation members are initialized here
        peg_gk = 0
        mpeg_gk = 1e100
        d_k = None
        for _i in range(numpy.array(g_k).size):
            c_i = (lambda: g_k[_i], lambda: g_k)[isinstance(g_k, int)]()
            for _j in range(numpy.array(g_k).size):
                c_j = (lambda: g_k[_j], lambda: g_k)[isinstance(g_k, int)]()
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
    def mei_cluster(self):
        # The output members are initialized here
        g = [[k] for k in range(self.NUMBER_OF_CHANNELS)]
        d = [k for k in range(self.NUMBER_OF_CHANNELS)]
        # The internal members are initialized here
        # while $|G| > T$
        while len(g) > self.NUMBER_OF_CLUSTERS:
            mei = 1e100
            gi_min = None
            gj_min = None
            gnew_min = None
            di_min = None
            dj_min = None
            dnew_min = None
            for g_i in g:
                # $MPEG(g_{i})$ and $d_{i}$
                mpeg_output = self.calculate_mpeg(g_i)
                mpeg_gi = mpeg_output.mpeg
                d_i = mpeg_output.d
                for g_j in g:
                    if g_i == g_j:
                        continue
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
                    if ei <= mei:
                        mei = ei
                        gi_min = g_i
                        gj_min = g_j
                        gnew_min = g_new
                        di_min = d_i
                        dj_min = d_j
                        dnew_min = d_new
            # $G \leftarrow G - g_{i} - g_{j} + g_{new}$
            if gi_min in g:
                g.remove(gi_min)
            if gj_min in g:
                g.remove(gj_min)
            g.append(gnew_min)
            # $D \leftarrow D - d_{i} - d_{j} + d_{new}$
            if di_min in d:
                d.remove(di_min)
            if dj_min in d:
                d.remove(dj_min)
            d.append(dnew_min)
        # Return the output in the prescribed format in case an external entity needs it...
        return self.clustering_output_format(G=g,
                                             D=d)

    # Estimate the Occupancy States of the Incumbents using the Markov Process based Estimation Algorithm
    # \[cs_{n}^{t} = \argmax_{x \in \{0,1\}}\ \{\mathbb{P}(x|\vec{cs}_n^{h} = \vec{y})
    #   \mathbb{P}(\vec{cs}_{n}^{h} = \vec{y})\}
    def mpe_estimate_occupancy_states(self):
        # The outputs to be returned
        estimated_occupancy_states = []
        likelihoods = []
        # The supported occupancy states
        # FIXME: This seems pretentious...
        start_probabilities_supported_states = [self.start_probabilities[member.value] for member in OccupancyState]
        for channel in range(self.NUMBER_OF_CHANNELS):
            # $cs_{n}^{0*}$
            cs_n0 = max(range(0, len(start_probabilities_supported_states)),
                        key=lambda idx: start_probabilities_supported_states[idx])
            estimated_occupancy_states.append([cs_n0])
            likelihoods.append([self.start_probabilities[cs_n0]])
            for episode in range(1, self.NUMBER_OF_EPISODES):
                # The order of this temporal Markov process on an individual channel basis is 1
                # $\vec{cs}_{n}^{h} = cs_{n}^{t-1} = y$ [Remember, this is a first order Markov process]
                previous_state = estimated_occupancy_states[channel][episode - 1]
                posterior_probability_members = [
                    (self.temporal_transition_matrix[previous_state][member.value] *
                     self.start_probabilities[previous_state]
                     ) for member in OccupancyState
                ]
                # $cs_{n}^{t*} = \argmax_{x \in \{0,1\}}\ \{\mathbb{P}(x|cs_{n}^{t-1}) \mathbb{P}(cs_{n}^{t-1} = y)\}$
                cs_nt = max(range(0, len(posterior_probability_members)),
                            key=lambda idx: posterior_probability_members[idx])
                estimated_occupancy_states[channel].append(cs_nt)
                likelihoods[channel].append(
                    self.temporal_transition_matrix[estimated_occupancy_states[channel][episode - 1]][cs_nt]
                )
        return self.estimation_routine_output(estimated_occupancy_states=estimated_occupancy_states,
                                              likelihoods=likelihoods)

    # Estimate the Occupancy States of the Incumbents using the Channel Correlation based Estimation Algorithms
    # Greedy Clustering and Minimum Entropy Increment Algorithms
    # $cs_{j}^{m*} = \argmax_{x \in \{cs_{i}, (1-cs_{i}^{m}\}\ \{\mathbb{P}(cs_{j}^{m} = x|cs_{i}^{m} = y)
    #   \mathbb{P}(cs_{i}^{m} = y)\}$
    def cce_estimate_occupancy_states(self, clustering_technique):
        # Setting up the outputs to be returned
        estimated_occupancy_states = []
        for channel in range(self.NUMBER_OF_CHANNELS):
            estimated_occupancy_states.append([])
        likelihoods = []
        for channel in range(self.NUMBER_OF_CHANNELS):
            likelihoods.append([])
        # Perform clustering
        clustered_output = (lambda: self.greedy_cluster(),
                            lambda: self.mei_cluster())[
            clustering_technique == ClusteringTechnique.MINIMUM_ENTROPY_INCREMENT
        ]()
        # Extract the groups
        g = clustered_output.G
        # Extract the corresponding Detected Channels (DCs)
        d = clustered_output.D
        for m in range(self.NUMBER_OF_EPISODES):
            for i in range(0, len(g)):
                # Sense the DC for this cluster
                cs_i = self.true_occupancy_states[d[i]][m]
                estimated_occupancy_states[d[i]].append(cs_i)
                likelihoods[d[i]].append(1)
                for j in g[i]:
                    posterior_probability_members = [
                        (self.temporal_transition_matrix[cs_i][member.value] *
                         self.start_probabilities[cs_i]
                         ) for member in OccupancyState
                    ]
                    cs_j = max(range(0, len(posterior_probability_members)),
                               key=lambda idx: posterior_probability_members[idx])
                    estimated_occupancy_states[j].append(cs_j)
                    likelihoods[j].append((lambda: 1 - self.channel_correlation_matrix[d[i]][j],
                                           lambda: self.channel_correlation_matrix[d[i]][j])[cs_i == cs_j]())
        return self.estimation_routine_output(estimated_occupancy_states=estimated_occupancy_states,
                                              likelihoods=likelihoods)

    # Get the average number of truly idle channels exploited by the SU per episode along with the average number of
    #   channels in which the SU interferes with the incumbents
    def get_analytics(self, estimated_occupancy_states):
        su_throughputs = []
        pu_interferences = []
        for i in range(self.NUMBER_OF_EPISODES):
            su_throughput = 0
            pu_interference = 0
            for k in range(self.NUMBER_OF_CHANNELS):
                su_throughput += (1 - estimated_occupancy_states[k][i]) * (1 - self.true_occupancy_states[k][i])
                pu_interference += self.true_occupancy_states[k][i] * (1 - estimated_occupancy_states[k][i])
            su_throughputs.append(su_throughput)
            pu_interferences.append(pu_interference)
        return self.analytics(su_throughput=sum(su_throughputs) / self.NUMBER_OF_EPISODES,
                              pu_interference=sum(pu_interferences) / self.NUMBER_OF_EPISODES)

    # Use the Minimum Entropy Merge (MEM) strategy here to determine the final estimated occupancy states of the
    #   channels across all episodes (2 variants: MEM_with_GC_CCE_and_MPE and MEM_with_MEI_CCE_and_MPE)
    def minimum_entropy_merge(self, clustering_technique):
        # Setting up the output to be returned
        estimated_occupancy_states = []
        for channel in range(self.NUMBER_OF_CHANNELS):
            estimated_occupancy_states.append([])
        # Get the estimation output from the CCE technique
        cce_estimation_output = self.cce_estimate_occupancy_states(clustering_technique)
        cce_estimated_occupancy_states = cce_estimation_output.estimated_occupancy_states
        cce_likelihoods = cce_estimation_output.likelihoods
        # Get the estimation output from the MPE technique
        mpe_estimation_output = self.mpe_estimate_occupancy_states()
        mpe_estimated_occupancy_states = mpe_estimation_output.estimated_occupancy_states
        mpe_likelihoods = mpe_estimation_output.likelihoods
        for channel in range(self.NUMBER_OF_CHANNELS):
            for episode in range(self.NUMBER_OF_EPISODES):
                entropies = []
                for k in (cce_likelihoods[channel][episode], mpe_likelihoods[channel][episode]):
                    if k == 0:
                        entropies.append(10e9)
                    elif k == 1:
                        entropies.append(0)
                    else:
                        entropies.append(((-k * numpy.log(k)) - ((1 - k) * numpy.log(1 - k))))
                minimum_entropy_index = min(range(0, len(entropies)),
                                            key=lambda idx: entropies[idx])
                if minimum_entropy_index == 0:
                    estimated_occupancy_states[channel].append(cce_estimated_occupancy_states[channel][episode])
                else:
                    estimated_occupancy_states[channel].append(mpe_estimated_occupancy_states[channel][episode])
        return estimated_occupancy_states

    # Print the average episodic SU throughput and PU interference analytics for both SoA variants
    # Using the same notation as in the reference work...
    def evaluate(self):
        # Minimum Entropy Merging with Greedy Clustering Channel Correlation Estimation and Markov Process Estimation
        mem_gc_cce_mpe_estimated_occupancy_states = self.minimum_entropy_merge(ClusteringTechnique.GREEDY)
        mem_gc_cce_mpe_analytics = self.get_analytics(mem_gc_cce_mpe_estimated_occupancy_states)
        print('[INFO] ClusteringAndMarkovianModeling evaluate: MEM with GC-CCE and MPE Analytics - '
              'Average Episodic SU Throughput = {} | '
              'Average Episodic PU Interference = {}'.format(mem_gc_cce_mpe_analytics.su_throughput,
                                                             mem_gc_cce_mpe_analytics.pu_interference))
        # Minimum Entropy Merging with Minimum Entropy Increment Clustering Channel Correlation Estimation and
        #   Markov Process Estimation
        mem_mei_cce_mpe_estimated_occupancy_states = self.minimum_entropy_merge(
            ClusteringTechnique.MINIMUM_ENTROPY_INCREMENT)
        mem_mei_cce_mpe_analytics = self.get_analytics(mem_mei_cce_mpe_estimated_occupancy_states)
        print('[INFO] ClusteringAndMarkovianModeling evaluate: MEM with MEI-CCE and MPE Analytics - '
              'Average Episodic SU Throughput = {} | '
              'Average Episodic PU Interference = {}'.format(mem_mei_cce_mpe_analytics.su_throughput,
                                                             mem_mei_cce_mpe_analytics.pu_interference))

    # The termination sequence
    def __exit__(self, exc_type, exc_val, exc_tb):
        print('[INFO] ClusteringAndMarkovianModeling Termination: Tearing things down...')
        # Nothing to do...


# Run Trigger
if __name__ == '__main__':
    print('[INFO] ClusteringAndMarkovianModeling main: '
          'Triggering the evaluation of the Minimum Entropy Merging (MEM) Algorithm '
          'for utility maximization in our model of a Cognitive Radio Network with one SU and multiple PUs...')
    agent = ClusteringAndMarkovianModeling()
    agent.evaluate()
