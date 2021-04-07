# This entity describes an adaptive DQN framework for spectrum sensing and access in multi-channel radio environments
#   with multiple licensed users and a single cognitive radio node trying to intelligently access spectral white-spaces.
# Author: Bharath Keshavamurthy
# Organization: School of Electrical and Computer Engineering, Purdue University, West Lafayette, IN.
# Copyright (c) 2020. All Rights Reserved.

# Note here that the adaptive DQN design adopted in this script is exactly similar to the one adopted in the DQN work
#   from Wang, et. al.--because, this class evaluates the performance of this design in a multi-channel radio ecosystem
#   with multiple licensed users and a single cognitive radio node wherein the incumbents occupy the spectrum according
#   to a double Markovian time-frequency correlation structure, and the simulation framework incorporates an AWGN
#   observation model and a Rayleigh channel fading model.

# Energy efficiency and Sensing aggregation time requirements force us to limit the number of channels that can be
#   sensed by the cognitive radio node in any given time-slot.

# Reference: Wang, et. al., "Deep Reinforcement Learning for Dynamic Multichannel Access in Wireless Networks",
#   IEEE TRANSACTIONS ON COGNITIVE COMMUNICATIONS AND NETWORKING, VOL. 4, NO. 2, JUNE 2018

# Nomenclature
# Primary Users (PUs): Incumbents, Licensed Users, Priority Users
# Secondary Users (SUs): Cognitive Radios, Nodes
# Deep Q-Network (DQN)
# Neural Network (NN)
# State space of the MDP $\mathcal{B}$
# Action space of the MDP $\mathcal{A}$

# The imports
import numpy
import itertools
import tensorflow
import scipy.stats
from collections import deque


# This class describes the adaptive DQN components for learning the optimal channel sensing and access policy--and
#   additionally, describes the channel emulation and incumbent occupancy behavioral components needed to simulate
#   real-world radio environments in which multiple agents are vying for the same limited resources.
class AdaptiveDQN(object):

    # The number of channels in the discretized spectrum of interest $K, k$
    NUMBER_OF_CHANNELS = 18

    # The number of time-slots for which this scheme is to be evaluated $N, i$
    NUMBER_OF_TIME_SLOTS = 10000

    # The number of sampling rounds per time-slot for determining observation averages $M, j$
    NUMBER_OF_SAMPLING_ROUNDS = 3000

    # The noise variance $\sigma_{V}^{2}$--the noise mean is zero
    NOISE_VARIANCE = 1

    # The channel impulse response variance $\sigma_{H}^{2}$--the channel impulse response mean is zero
    CHANNEL_IMPULSE_RESPONSE_VARIANCE = 80

    # The steady state occupancy probability of a cell in the time-frequency occupancy grid $\Pi$
    #   $\Pi \triangleq \mathbb{P}(B_{k+1}(i+1){=}1), k{\in}\{0,1,\dots,K{-}1\}, i{\in}\{0,1,\dots,T{-}1\}
    STEADY_STATE_OCCUPANCY_PROBABILITY = 0.6

    # The transition probability parameters defining the transition model of the MDP underlying the occupancy behavior
    #   of the PUs in the network $\vec{\theta}$
    #   $pXY \triangleq \mathbb{P}(B_{k}(i+1){=}1|B_{k-1}(i+1){=}X,B_{k}(i){=}Y), X,Y{\in}\{0,1\},
    #   i{\in}\{0,1,\dots,T{-}1\}, k{\in}\{1,2,\dots,K{-}1\}$
    #   $qZ$ \triangleq \mathbb{P}(B_{1}(i+1){=}1|B_{1}(i){=}Z), Z{\in}\{0,1\}, i{\in}\{0,1,\dots,T{-}1\}
    #   Note here that the double Markov chain is defined by the $pXY$ members, while the single Markov chains (both
    #   time and frequency) are defined by the $qZ$ members
    TRANSITION_MODEL = {'p00': 0.1, 'p01': 0.3, 'p10': 0.3, 'p11': 0.7, 'q0': 0.3, 'q1': 0.8}

    # The size of the experiential replay memory for the DQN $C$
    EXPERIENTIAL_REPLAY_MEMORY_CAPACITY = 1000000

    # The batch size: the number of experiences randomly sampled from the experiential replay memory for training $W$
    BATCH_SIZE = 32

    # The epsilon value for the $\epsilon$-greedy policy designed to evaluate the exploration-exploitation trade-off
    #   Note here that the $\epsilon$ is fixed throughout
    EPSILON = 0.1

    # The learning rate $\alpha$
    LEARNING_RATE = 1e-4

    # The discount factor $\gamma$
    DISCOUNT_FACTOR = 0.9

    # The number of channels that can be simultaneously sensed by the cognitive radio node in a given time-slot
    SENSING_RESTRICTION = 6

    # The penalty for missed detections $\mu$
    PENALTY = -1

    # The allowed false alarm probability for threshold determination w.r.t the Likelihood Ratio Test (LRT)--1%
    ALLOWED_FALSE_ALARM_PROBABILITY = 0.01

    # The initialization sequence
    def __init__(self):
        print('[INFO] AdaptiveDQN Initialization: Bringing things up...')
        # The threshold for the Likelihood Ratio Test (LRT)
        self.threshold = numpy.sqrt(self.NOISE_VARIANCE / self.NUMBER_OF_SAMPLING_ROUNDS) * scipy.stats.norm.ppf(
            1 - self.ALLOWED_FALSE_ALARM_PROBABILITY
        )
        # Based on the number of channels in the discretized spectrum of interest and the sensing restriction imposed
        #   determine the size of the action space
        self.action_space = {index: action for index, action in
                             enumerate([a for a in itertools.product([0, 1], repeat=self.NUMBER_OF_CHANNELS)
                                        if a.count(1) == self.SENSING_RESTRICTION])
                             }
        # Get a dict() {channel (k): [occupancy per time-slot (i)]} of incumbent occupancy over the entire simulation
        #   period--emulated based on the double Markovian time-frequency correlation structure
        # Also, get the average number of occupancies per time-slot observed in the emulated incumbent occupancy dict
        self.incumbent_occupancy, self.average_occupancies = self.get_incumbent_occupancy()
        # Initialize the experiential replay memory: the use of this feature mitigates the effect of the NN's
        #   non-linearities, stabilizes the Q-value approximation operation, and helps the DQN converge
        self.experiential_replay_memory = deque(maxlen=self.EXPERIENTIAL_REPLAY_MEMORY_CAPACITY)
        # Build and Compile the Neural Network (NN)
        self.model = self.build_model()
        # Initialize the time-slot specific su_throughputs and pu_interferences output statistics for evaluation
        self.su_throughputs = []
        self.pu_interferences = []
        # The initialization sequence has been completed

    # Simulate the occupancy behavior of the incumbents in the network according the double-Markovian time-frequency
    #   correlation structure parameterized by $\vec{\theta}$
    def get_incumbent_occupancy(self):
        # The output initialization
        _incumbent_occupancy = dict()
        # Frequency-0 Time-0 PU occupancy determination
        _incumbent_occupancy[0] = [(lambda: 0,
                                    lambda: 1)[numpy.random.random() <= self.STEADY_STATE_OCCUPANCY_PROBABILITY]()]
        # Frequency-0 Time-[1,T-1] PU occupancy determination
        for i in range(1, self.NUMBER_OF_TIME_SLOTS):
            if _incumbent_occupancy[0][i-1]:
                _incumbent_occupancy[0].append((lambda: 0,
                                                lambda: 1)[numpy.random.random() <= self.TRANSITION_MODEL['q1']]())
            else:
                _incumbent_occupancy[0].append((lambda: 0,
                                                lambda: 1)[numpy.random.random() <= self.TRANSITION_MODEL['q0']]())
        # Frequency-[1,K-1] Time-0 PU occupancy determination
        for k in range(1, self.NUMBER_OF_CHANNELS):
            if _incumbent_occupancy[k-1][0]:
                _incumbent_occupancy[k] = [(lambda: 0,
                                            lambda: 1)[numpy.random.random() <= self.TRANSITION_MODEL['q1']]()]
            else:
                _incumbent_occupancy[k] = [(lambda: 0,
                                            lambda: 1)[numpy.random.random() <= self.TRANSITION_MODEL['q0']]()]
        # Frequency-[1,K-1] Time-[1,T-1] PU occupancy determination
        for k in range(1, self.NUMBER_OF_CHANNELS):
            for i in range(1, self.NUMBER_OF_TIME_SLOTS):
                if _incumbent_occupancy[k-1][i] and _incumbent_occupancy[k][i-1]:
                    _incumbent_occupancy[k].append((lambda: 0,
                                                    lambda: 1)[
                                                       numpy.random.random() <= self.TRANSITION_MODEL['p11']]())
                elif _incumbent_occupancy[k-1][i] and not _incumbent_occupancy[k][i-1]:
                    _incumbent_occupancy[k].append((lambda: 0,
                                                    lambda: 1)[
                                                       numpy.random.random() <= self.TRANSITION_MODEL['p10']]())
                elif not _incumbent_occupancy[k-1][i] and _incumbent_occupancy[k][i-1]:
                    _incumbent_occupancy[k].append((lambda: 0,
                                                    lambda: 1)[
                                                       numpy.random.random() <= self.TRANSITION_MODEL['p01']]())
                else:
                    _incumbent_occupancy[k].append((lambda: 0,
                                                    lambda: 1)[
                                                       numpy.random.random() <= self.TRANSITION_MODEL['p00']]())
        _average_occupancies = sum([sum([_incumbent_occupancy[k][i] for k in range(self.NUMBER_OF_CHANNELS)])
                                    for i in range(self.NUMBER_OF_TIME_SLOTS)]) / self.NUMBER_OF_TIME_SLOTS
        # Return the emulated incumbent occupancy dictionary and the average number of occupancies per time-slot
        return _incumbent_occupancy, _average_occupancies

    # Build and Compile the Neural Network (NN) used to determine the Q-values for the state-action pairs
    #   Note here that the high-dimensionality of the problem in terms of the number of possible states and the number
    #   of available actions, makes it computationally infeasible to exhaustively determine the Q-values for all the
    #   state-action pairs--hence, a Neural Network!
    def build_model(self):
        # Construct the Neural Network: Input layer dimensions are dictated by the number of channels (MDP state),
        #   2 hidden layers, and the number of neurons in the output layer is dictated by the size of the action space
        _model = tensorflow.keras.Sequential([
            tensorflow.keras.layers.Dense(units=2048,
                                          input_dim=self.NUMBER_OF_CHANNELS,
                                          activation=tensorflow.nn.relu),
            tensorflow.keras.layers.Dense(units=4096,
                                          activation=tensorflow.nn.relu),
            tensorflow.keras.layers.Dense(units=4096,
                                          activation=tensorflow.nn.relu),
            tensorflow.keras.layers.Dense(units=len(self.action_space),
                                          activation='linear')
        ])
        # Compile the NN with an MSE loss function and an Adam optimizer (stochastic gradient descent with $\alpha$)
        #  Loss = \mathbb{E}[((Reward(i) + \gamma \max_{a'{\in}\mathcal{A}} Q(\vec{B}(i+1),a')) - Q(\vec{B}(i), a))^{2}]
        _model.compile(loss=tensorflow.keras.losses.mean_squared_error,
                       optimizer=tensorflow.keras.optimizers.Adam(learning_rate=self.LEARNING_RATE),
                       metrics=[tensorflow.keras.metrics.mean_absolute_error,
                                tensorflow.keras.metrics.mean_squared_error]
                       )
        # Return the model
        return _model

    # Store the experience in the experiential replay memory
    def store_experience(self, state, action, reward, next_state, done):
        self.experiential_replay_memory.append((state, action, reward, next_state, done))
        # Nothing to be returned

    # Given the state, choose the action that maximizes the Q-value of the state-action pair, i.e.,
    #   $\mathcal{K}(i) = \argmax_{a{\in}\mathcal{A}} Q(\vec{B}(i), a), \vec{B}(i){\in}\mathcal{B}$
    #   $0{\leq}|\mathcal{K}(i)|{\leq}K$
    def choose_action(self, state):
        # Return the chosen action
        return self.action_space[(lambda: numpy.argmax(self.model.predict(state)[0]),
                                  lambda: numpy.random.randint(0, len(self.action_space)))[
            numpy.random.random() <= self.EPSILON]()]

    # Train the NN with the experiences stored in the experiential replay memory
    def learn_from_stored_experiences(self):
        sampled_experiences = numpy.random.choice(self.experiential_replay_memory, self.BATCH_SIZE).tolist()
        for state, action, reward, next_state, done in sampled_experiences:
            target = (lambda: reward + (self.DISCOUNT_FACTOR * numpy.amax(self.model.predict(next_state)[0])),
                      lambda: reward)[done]()
            target_train = self.model.predict(state)
            target_train[0][action] = target
            self.model.fit(state, target_train, epochs=1, verbose=0)
        # Nothing to be returned

    # The noisy observation (AWGN noise + limited observational capability) of the MDP state based on the chosen action
    def observe(self, action, time_slot, prev_state):
        state = [prev_state[k] for k in range(self.NUMBER_OF_CHANNELS)]
        for k in action:
            test_statistic = 0
            for m in range(self.NUMBER_OF_SAMPLING_ROUNDS):
                test_statistic += (numpy.random.normal(0, numpy.sqrt(self.CHANNEL_IMPULSE_RESPONSE_VARIANCE)) *
                                   self.incumbent_occupancy[k][time_slot]) + \
                                  numpy.random.normal(0, numpy.sqrt(self.NOISE_VARIANCE))
            state[k] = (lambda: 0,
                        lambda: 1)[(test_statistic / self.NUMBER_OF_SAMPLING_ROUNDS) >= self.threshold]()
        # Return the observed state
        return state

    # Get the reward of this adaptive DQN agent
    # Let $B_{k}(i)$ denote the actual true occupancy state of channel $k$ in time-slot $i$
    # Let $\hat{B}_{k}(i)$ denote the estimated occupancy state of channel $k$ in time-slot $i$
    # Reward = R = \sum_{k{=}1}^{K} (1{-}B_{k}(i))(1{-}\hat{B}_{k}(i)) + \mu B_{k}(i)(1{-}\hat{B}_{k}(i))
    def get_reward(self, action, time_slot, prev_state):
        # Initialize the transient variables
        su_throughput = 0
        pu_interference = 0
        # The estimated state vector--relevant for reward evaluation
        estimated_state = self.observe(action, time_slot, prev_state)
        for k in range(self.NUMBER_OF_CHANNELS):
            su_throughput += (1 - estimated_state[k][time_slot]) * (1 - self.incumbent_occupancy[k][time_slot])
            pu_interference += (1 - estimated_state[k][time_slot]) * self.incumbent_occupancy[k][time_slot]
        self.su_throughputs.append(su_throughput)
        self.pu_interferences.append(pu_interference)
        # Return the reward
        return su_throughput + (self.PENALTY * pu_interference)

    # Evaluate the performance of this adaptive DQN strategy
    def evaluate(self):
        prev_state = [k-k for k in range(self.NUMBER_OF_CHANNELS)]
        action = self.action_space[numpy.random.randint(0, len(self.action_space))]
        state = self.observe(action, 0, prev_state)
        for i in range(1, self.NUMBER_OF_TIME_SLOTS):
            action = self.choose_action(state)
            next_time_slot = (lambda: i, lambda: i + 1)[i < self.NUMBER_OF_TIME_SLOTS - 1]()
            next_state, reward, done = self.observe(action, next_time_slot, state), \
                                       self.get_reward(action, i, prev_state), \
                                       (lambda: True,
                                        lambda: False)[i < self.NUMBER_OF_TIME_SLOTS - 1]()
            prev_state = state
            self.store_experience(state, action, next_state, reward, done)
            state = next_state
            if done:
                print('[INFO] AdaptiveDQN evaluate: Average SU Throughput = {} | Average PU Interference = {}'.format(
                    self.su_throughputs / (self.NUMBER_OF_TIME_SLOTS - 1),
                    self.pu_interferences / (self.average_occupancies * (self.NUMBER_OF_TIME_SLOTS - 1))))
                # This is redundant--but, I might need this structure (of 'done') in case I want to view the behavior
                #   differently (i.e., as a game: exit after achieving a certain level of performance for a certain
                #   period of time)
                break
            if len(self.experiential_replay_memory) >= self.BATCH_SIZE:
                self.learn_from_stored_experiences()
        # Nothing to be returned

    # The termination sequence
    def __exit__(self, exc_type, exc_val, exc_tb):
        print('[INFO] AdaptiveDQN Termination: Tearing things down...')
        # Nothing to be done here...


# Run trigger
if __name__ == '__main__':
    print('[INFO] AdaptiveDQN Evaluation: Starting the evaluation of the adaptive DQN framework in a radio environment '
          'emulation scenario consisting of multiple licensed users and a single cognitive radio--double Markovian '
          'time-frequency PU occupancy correlation structure, an AWGN observation model, and a Rayleigh channel model.')
    adaptiveDQN = AdaptiveDQN()
    adaptiveDQN.evaluate()
    print('[INFO] AdaptiveDQN Evaluation: Completed the evaluation of the adaptive DQN framework!')
    # The adaptive DQN framework evaluation has been completed
