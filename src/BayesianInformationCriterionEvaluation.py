# This class describes the evaluation of the Bayesian Information Criterion (BIC) metrics in relation to the DARPA SC2
#   Active Incumbent data--in order to study the "goodness of fit" of our double Markovian time-frequency correlation
#   structure ($\hat{\vec{\theta}}$) -- in both forward and backward directions vis-a-vis the correlation across
#   frequency, as opposed to time-frequency independence models ($\mathbf{Q}$), pure temporal correlation models
#   ($\mathbf{R}$), and pure frequency correlation models ($\mathbf{S}).
# Author: Bharath Keshavamurthy
# Organization: School of Electrical and Computer Engineering, West Lafayette, IN.
# Copyright (c) 2020. All Rights Reserved.

# https://en.wikipedia.org/wiki/Bayesian_information_criterion

# MINERVA | BLEEDING EDGE | PHOENIX |

# The imports
import numpy
from enum import Enum
from DARPASC2ActiveIncumbentAnalysis import DARPASC2ActiveIncumbentAnalysis


# This enumeration entity lists the various types of correlation we intend to analyze in this script
class CorrelationModelType(Enum):
    # Double Markovian Time-Frequency Correlation (Our Model - $\hat{\vec{\theta}}$) -- with top-down correlation across
    #   frequencies, i.e., a forward-facing Markov chain across the channels in the discretized spectrum of interest
    MARKOVIAN_TIME_FREQUENCY_CORRELATION_FORWARD = 0

    # Double Markovian Time-Frequency Correlation -- with bottom-up correlation across frequencies, i.e., a
    #   backward-facing Markov chain across the channels in the discretized spectrum of interest
    MARKOVIAN_TIME_FREQUENCY_CORRELATION_BACKWARD = 1

    # Time-Frequency Independence ($\mathbf{Q}$)
    TIME_FREQUENCY_INDEPENDENCE = 2

    # Markovian Time Correlation Only ($\hat{\vec{\theta}}$)
    MARKOVIAN_TIME_CORRELATION_ONLY = 3

    # Markovian Frequency Correlation Only ($\mathbf{S})
    MARKOVIAN_FREQUENCY_CORRELATION_ONLY = 4


# Evaluation of the BIC metrics to analyze the "goodness of fit" of our model against those in the state-of-the-art...
class BayesianInformationCriterionEvaluation(object):
    # The default number of channels to be used in case something goes wrong during instance creation
    DEFAULT_NUMBER_OF_CHANNELS = 20

    # The number of episodes to be used in case something goes wrong during instance creation
    DEFAULT_NUMBER_OF_EPISODES = 10000

    # The initialization sequence
    def __init__(self, _occupancy_behavior, _training_test_split):
        print('[INFO] BayesianInformationCriterionEvaluation Initialization: Bringing things up...')

        # The binary occupancy behavior of the incumbent and the competitors obtained by applying thresholding over the
        #   aggregated PSD observations at the BAM! Wireless Gateway Node
        self.occupancy_behavior = (lambda: {k: [0 for _ in range(0, self.DEFAULT_NUMBER_OF_EPISODES)]
                                            for k in range(0, self.DEFAULT_NUMBER_OF_CHANNELS)},
                                   lambda: _occupancy_behavior)[_occupancy_behavior is not None
                                                                and len(_occupancy_behavior.keys()) > 0
                                                                and len(_occupancy_behavior[0]) > 0]()
        # Training-Test split validation
        if _training_test_split is None or _training_test_split == 0:
            print('[ERROR] BayesianInformationCriterionEvaluation Initialization: Unsupported training_test_split - '
                  'The training/test split refers to the fraction of the dataset that was used for model estimation | '
                  '0.0 < training_test_split < 1.0 | Exiting BayesianInformationCriterionEvaluation instance creation!')
            return

        # The number of discretized channels in the spectrum of interest (10 MHz scenario bandwidth) $K$
        self.number_of_channels = len(self.occupancy_behavior.keys())
        # The total number of episodes under analysis (330s emulation period) $T$
        # NOTE: Index-0 is the pilot for time-slot length determination
        self.number_of_episodes = len(self.occupancy_behavior[0])
        # The sample size under analysis $n$
        self.n = self.number_of_channels
        # The time period (number of episodes) under consideration for averaging the likelihoods across this period
        self.evaluation_period = int(numpy.ceil((1 - _training_test_split) * len(self.occupancy_behavior[0])))
        # The sample start index (based on the provided training/test split) $m$
        self.m = int(numpy.floor(_training_test_split * len(self.occupancy_behavior[0])))

        # The initialization sequence has been completed.

    # Determine the Bayesian Information Criterion for the given model
    def get_bic(self, _correlation_model_type, _model, _pi):
        # The number of parameters in the model under analysis $k$
        k = (lambda: len(_model.keys()),
             lambda: 1)[_correlation_model_type == CorrelationModelType.TIME_FREQUENCY_INDEPENDENCE]()

        # The likelihood collection $\mathbb{P}_{i}(\vec{x}|\hat{\vec{\phi}}, \mathbf{M})$
        likelihoods = []

        # Determine the likelihood probability across the final ${test}% of the data--based on the model under analysis

        # MARKOVIAN_TIME_FREQUENCY_CORRELATION_FORWARD
        # Refer to the Channel Occupancy Model equation in the journal manuscript (\eqref{6})
        if _correlation_model_type == CorrelationModelType.MARKOVIAN_TIME_FREQUENCY_CORRELATION_FORWARD:
            # Episodes-$m$ to $T{-}1$
            for i in range(self.m, self.number_of_episodes):
                # Reset the likelihood for the next episode
                likelihood = 1.0
                # Channel-$0$ | Episode-$i$ | 'q' parameters | Temporal Correlation
                if self.occupancy_behavior[0][i]:
                    likelihood *= _model['1'] if self.occupancy_behavior[0][i - 1] else _model['0']
                else:
                    likelihood *= 1 - (_model['1'] if self.occupancy_behavior[0][i - 1] else _model['0'])
                # Channel-$1$ to $K{-}1$ | Episode-$i$ | 'p' parameters | Time-Frequency Correlation
                for k in range(1, self.number_of_channels):
                    if self.occupancy_behavior[k][i]:
                        likelihood *= (lambda: _model['10'], lambda: _model['11'])[
                            self.occupancy_behavior[k][i - 1]]() if self.occupancy_behavior[k - 1][i] else \
                            (lambda: _model['00'], lambda: _model['01'])[self.occupancy_behavior[k][i - 1]]()
                    else:
                        likelihood *= 1 - ((lambda: _model['10'],
                                            lambda: _model['11'])[self.occupancy_behavior[k][i - 1]]()
                                           if self.occupancy_behavior[k - 1][i]
                                           else (lambda: _model['00'],
                                                 lambda: _model['01'])[self.occupancy_behavior[k][i - 1]]())
                # Append the calculated likelihood for this episode to the likelihoods collection for averaging...
                likelihoods.append(likelihood)
        # MARKOVIAN_TIME_FREQUENCY_CORRELATION_BACKWARD
        # Refer to the Channel Occupancy Model equation in the journal manuscript (\eqref{7})
        elif _correlation_model_type == CorrelationModelType.MARKOVIAN_TIME_FREQUENCY_CORRELATION_BACKWARD:
            # Episodes-$m$ to $T{-}1$
            for i in range(self.m, self.number_of_episodes):
                # Reset the likelihood for the next episode
                likelihood = 1.0
                # Channel-$K{-}1$ | Episode-$i$ | 'q' parameters | Temporal Correlation
                if self.occupancy_behavior[self.number_of_channels - 1][i]:
                    likelihood *= _model['1'] if self.occupancy_behavior[self.number_of_channels - 1][i - 1] \
                        else _model['0']
                else:
                    likelihood *= 1 - (_model['1'] if self.occupancy_behavior[self.number_of_channels - 1][i - 1]
                                       else _model['0'])
                # Channel-$0$ to $K{-}2$ | Episode-$i$ | 'p' parameters | Time-Frequency Correlation
                for k in range(0, self.number_of_channels - 1):
                    if self.occupancy_behavior[k][i]:
                        likelihood *= (lambda: _model['10'], lambda: _model['11'])[
                            self.occupancy_behavior[k][i - 1]]() if self.occupancy_behavior[k + 1][i] else \
                            (lambda: _model['00'], lambda: _model['01'])[self.occupancy_behavior[k][i - 1]]()
                    else:
                        likelihood *= 1 - ((lambda: _model['10'],
                                            lambda: _model['11'])[self.occupancy_behavior[k][i - 1]]()
                                           if self.occupancy_behavior[k + 1][i]
                                           else (lambda: _model['00'],
                                                 lambda: _model['01'])[self.occupancy_behavior[k][i - 1]]())
                # Append the calculated likelihood for this episode to the likelihoods collection for averaging...
                likelihoods.append(likelihood)
        # TIME_FREQUENCY_INDEPENDENCE
        elif _correlation_model_type == CorrelationModelType.TIME_FREQUENCY_INDEPENDENCE:
            # Episodes-$m$ to $T{-}1$
            for i in range(self.m, self.number_of_episodes):
                # Reset the likelihood for the next episode
                likelihood = 1.0
                # Channel-$0$ to $K{-}1$ | Episode-$i$
                for k in range(0, self.number_of_channels):
                    likelihood *= (lambda: 1 - _pi,
                                   lambda: _pi)[self.occupancy_behavior[k][i]]()
                # Append the calculated likelihood for this episode to the likelihoods collection for averaging...
                likelihoods.append(likelihood)
        # MARKOVIAN_TIME_CORRELATION_ONLY
        elif _correlation_model_type == CorrelationModelType.MARKOVIAN_TIME_CORRELATION_ONLY:
            # Episodes-$m$ to $T{-}1$
            for i in range(self.m, self.number_of_episodes):
                # Reset the likelihood for the next episode
                likelihood = 1.0
                # Channel-$0$ to $K{-}1$ | Episode-$i$
                for k in range(0, self.number_of_channels):
                    if self.occupancy_behavior[k][i]:
                        likelihood *= _model['1'] if self.occupancy_behavior[k][i - 1] else _model['0']
                    else:
                        likelihood *= 1 - (_model['1'] if self.occupancy_behavior[k][i - 1] else _model['0'])
                # Append the calculated likelihood for this episode to the likelihoods collection for averaging...
                likelihoods.append(likelihood)
        # MARKOVIAN_FREQUENCY_CORRELATION_ONLY
        else:
            # Episodes-$m$ to $T{-}1$
            for i in range(self.m, self.number_of_episodes):
                # Reset the likelihood for the next episode
                likelihood = 1.0
                # Channel-$0$ | Episode-$i$
                likelihood *= (lambda: 1 - _pi,
                               lambda: _pi)[self.occupancy_behavior[0][i]]()
                for k in range(1, self.number_of_channels):
                    if self.occupancy_behavior[k][i]:
                        likelihood *= _model['1'] if self.occupancy_behavior[k - 1][i] else _model['0']
                    else:
                        likelihood *= 1 - (_model['1'] if self.occupancy_behavior[k - 1][i] else _model['0'])
                # Append the calculated likelihood for this episode to the likelihoods collection for averaging...
                likelihoods.append(likelihood)

        # Return the Bayesian Information Criterion (BIC) metric
        # $BIC_{\mathbf{M}) = kln(n) - 2ln(\mathbb{P}(\vec{x}|\hat{\vec{\phi}}, \mathbf{M}))
        return sum([(k * numpy.log(self.n)) - (2 * numpy.log(hood)) for hood in likelihoods]) / self.evaluation_period

    # The termination sequence
    def __exit__(self, exc_type, exc_val, exc_tb):
        print('[INFO] BayesianInformationCriterionEvaluation Termination: Tearing things down...')
        # Nothing to do here...


# Run Trigger
if __name__ == '__main__':
    print('[INFO] BayesianInformationCriterionEvaluation main: Starting BIC Evaluation...')
    # ${train}-${test} training/test split
    training_test_split = 0.7

    # The analyzer instance in order to extract the occupancy behavior information from the DARPA SC2 data
    analyzer = DARPASC2ActiveIncumbentAnalysis('data/active_incumbent_scenario8342.db')
    # The BIC evaluator instance
    evaluator = BayesianInformationCriterionEvaluation(analyzer.get_occupancy_behavior(), training_test_split)

    # The model estimates obtained by running SC2ActiveIncumbentCorrelationModelEstimator.py on the DARPA SC2 Active
    #   Incumbent data: ${train}% of the emulation data is used for this model estimation...
    # The models under analysis

    # The time-frequency Markovian correlation model (our proposed model: MARKOVIAN_TIME_FREQUENCY_CORRELATION_FORWARD)
    time_freq_correlation_model_forward = {
        '0': 0.67,  # q0
        '1': 0.75,  # q1
        '00': 0.25,  # p00
        '01': 0.75,  # p01
        '10': 0.71,  # p10
        '11': 0.80  # p11
    }
    # The time-frequency Markovian correlation model (our proposed model: MARKOVIAN_TIME_FREQUENCY_CORRELATION_BACKWARD)
    time_freq_correlation_model_backward = {
        '0': 0.5,  # q0
        '1': 0.85,  # q1
        '00': 0.36,  # p00
        '01': 0.50,  # p01
        '10': 0.85,  # p10
        '11': 0.92  # p11
    }
    # The steady state occupancy probability (independence model: TIME_FREQUENCY_INDEPENDENCE)
    steady_state_occupancy = 0.7
    # The purely temporal correlation model (temporal correlation only model: MARKOVIAN_TIME_CORRELATION_ONLY)
    time_correlation_model = {
        '0': 0.67,  # p
        '1': 0.75  # 1 - q
    }
    # The purely frequency correlation model (frequency correlation only model: MARKOVIAN_FREQUENCY_CORRELATION_ONLY)
    frequency_correlation_model = {
        '0': 0.3,  # p
        '1': 0.8  # 1 - q
    }

    # Correlation Model Type AND Model selection
    model_type = CorrelationModelType.MARKOVIAN_TIME_FREQUENCY_CORRELATION_BACKWARD
    if model_type == CorrelationModelType.MARKOVIAN_TIME_FREQUENCY_CORRELATION_FORWARD:
        model = time_freq_correlation_model_forward
    elif model_type == CorrelationModelType.MARKOVIAN_TIME_FREQUENCY_CORRELATION_BACKWARD:
        model = time_freq_correlation_model_backward
    elif model_type == CorrelationModelType.TIME_FREQUENCY_INDEPENDENCE:
        model = None
    elif model_type == CorrelationModelType.MARKOVIAN_TIME_CORRELATION_ONLY:
        model = time_correlation_model
    else:
        model = frequency_correlation_model

    # Get the BIC metric for the model under analysis
    print('[INFO] BayesianInformationCriterionEvaluation main: BIC for {} = '
          '{}'.format(model_type.name,
                      evaluator.get_bic(model_type, model, steady_state_occupancy)))
    # The evaluation ends here...
