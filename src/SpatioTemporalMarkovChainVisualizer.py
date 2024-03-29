# A Python Script that visualizes a Double Markov Chain
# A Markov Chain across the frequency sub-bands
# A Markov Chain across time (sampling rounds or iterations)
# Spatio-Temporal Markov chain
# Author: Bharath Keshavamurthy
# Organization: School of Electrical and Computer Engineering, Purdue University
# Copyright (c) 2019. All Rights Reserved.

import numpy
import plotly
import plotly.graph_objs as graph_objs

plotly.tools.set_credentials_file(username='bkeshava', api_key='6OIjH4XrjB5IyX8bwcaL')


# Channel Object
# Encapsulates the channel index and PU occupancy
class Channel(object):
    # Constructor - Channel Index and PU Occupancy
    def __init__(self, index, occupancy):
        self.index = index
        self.occupancy = occupancy


# Spatio-Temporal Markov Chain Visualization
class SpatioTemporalMarkovChainVisualizer(object):
    # P(Occupied|Idle) = p = 0.3 (some sample value)
    # I'm gonna use this sample for p across both time indices and channel indices
    SAMPLE_VALUE_OF_p = 0.3

    # P(Occupied) = 0.6
    # I'm gonna use this sample for PI across both time indices and channel indices
    SAMPLE_VALUE_OF_PI = 0.6

    # Number of channels (the channels can be discretized from the wideband spectrum of interest using the...
    # ...Wavelet Transform
    # k goes all the way to 18
    NUMBER_OF_CHANNELS = 18

    # Number of sampling rounds
    # t goes all the way to 300
    NUMBER_OF_SAMPLING_ROUNDS = 300

    # (t,k) represents one "pixel"

    # Initialization
    def __init__(self):
        print('[INFO] SpatioTemporalMarkovChainVisualizer Initialization: Bringing things up...')
        self.true_pu_occupancy_states = []
        self.p = self.SAMPLE_VALUE_OF_p
        self.pi = self.SAMPLE_VALUE_OF_PI
        # P(0|1)
        self.q = (self.p * (1 - self.pi)) / self.pi
        # The wideband spectrum of interest: A collection (tuple) of the frequency channels
        self.spectrum = ()
        # Transition Probability Matrix for the Markov Chain across time
        self.temporal_transition_probability_matrix = {
            1: {1: (1 - self.q), 0: self.q},
            0: {1: self.p, 0: (1 - self.p)}
        }
        # Transition Probability Matrix for the Markov Chain across channels
        self.spatial_transition_probability_matrix = {
            1: {1: (1 - self.q), 0: self.q},
            0: {1: self.p, 0: (1 - self.p)}
        }

    # Generate the true states with Markovian across channels and Markovian across time
    # Arguments: p -> P(1|0); q -> P(0|1); and pi -> P(1)
    def generate_true_pu_occupancy_states(self, p_val, q_val, pi_val):
        # True PU Occupancy states collection will be a list of self.NUMBER_OF_CHANNELS rows each row ...
        # ... with self.NUMBER_OF_SAMPLING_ROUNDS columns
        self.true_pu_occupancy_states.append(self.get_initial_states_temporal_variation(p_val, q_val, pi_val))
        # t = 0 and k = 0
        previous_state = self.true_pu_occupancy_states[0][0]
        for channel_index in range(1, self.NUMBER_OF_CHANNELS):
            seed = numpy.random.random_sample()
            if previous_state == 1 and seed < q_val:
                previous_state = 0
            elif previous_state == 1 and seed > q_val:
                previous_state = 1
            elif previous_state == 0 and seed < p_val:
                previous_state = 1
            else:
                previous_state = 0
            self.true_pu_occupancy_states.append([previous_state])
        # Let's fill in the other states
        for channel_index in range(1, self.NUMBER_OF_CHANNELS):
            for round_index in range(1, self.NUMBER_OF_SAMPLING_ROUNDS):
                # b
                previous_temporal_state = self.true_pu_occupancy_states[channel_index][round_index - 1]
                # c
                previous_spatial_state = self.true_pu_occupancy_states[channel_index - 1][round_index]
                # P(A=1|B=b)
                probability_occupied_temporal = self.temporal_transition_probability_matrix[previous_temporal_state][1]
                # P(A=1|C=c)
                probability_occupied_spatial = self.spatial_transition_probability_matrix[previous_spatial_state][1]
                # Calculating P(B=b)
                if previous_temporal_state == 1:
                    pi_temporal = pi_val
                else:
                    pi_temporal = 1 - pi_val
                # P(A=1|B=b,C=c) = [P(A=1|B=b)*P(A=1|C=c)]/P(B=b)
                # This formula is obtained by using Bayes' theorem, independence, and definitions of conditional...
                # ...probability
                probability_occupied = (probability_occupied_spatial * probability_occupied_temporal) / pi_temporal
                seed = numpy.random.random_sample()
                if seed < probability_occupied:
                    previous_state = 1
                else:
                    previous_state = 0
                self.true_pu_occupancy_states[channel_index].append(previous_state)

    # Generate the initial states for k = 0 across time
    def get_initial_states_temporal_variation(self, p_val, q_val, pi_val):
        initial_state_vector = []
        previous = 1
        # Initial state generation -> band-0 at time-0 using pi_val
        if numpy.random.random_sample() > pi_val:
            previous = 0
        initial_state_vector.append(previous)
        # Based on the state of band-0 at time-0 and the (p_val,q_val) values, generate the states of the remaining...
        # ...bands
        for loop_counter in range(1, self.NUMBER_OF_SAMPLING_ROUNDS):
            seed = numpy.random.random_sample()
            if previous == 1 and seed < q_val:
                previous = 0
            elif previous == 1 and seed > q_val:
                previous = 1
            elif previous == 0 and seed < p_val:
                previous = 1
            else:
                previous = 0
            initial_state_vector.append(previous)
        return initial_state_vector

    # Visualization: Core method
    def visualize(self):
        print('[INFO] SpatioTemporalMarkovChainVisualizer Visualization: Inside the core method...')
        # Get the states
        self.generate_true_pu_occupancy_states(self.p, self.q, self.pi)
        # Time
        horizontal_axis = [k + 1 for k in range(0, self.NUMBER_OF_SAMPLING_ROUNDS)]
        # Channels
        vertical_axis = [k + 1 for k in range(0, self.NUMBER_OF_CHANNELS)]
        # Data
        # Plotly API's HeatMap
        data = [
            graph_objs.Heatmap(z=self.true_pu_occupancy_states, x=horizontal_axis,
                               y=vertical_axis, xgap=1, ygap=1, colorscale=[[0, 'rgb(0,255,0)'], [1, 'rgb(255,0,0)']],
                               colorbar=dict(title='PU Occupancy', titleside='right', tickmode='array', tickvals=[0, 1],
                                             ticktext=['Unoccupied', 'Occupied'], ticks='outside'),
                               showscale=True)]
        # Layout
        layout = graph_objs.Layout(
            title='Spectrum Occupancy Map with Dynamic PU and Dual-Dimension Markovian Correlation '
                  'with P(Occupied|Idle) = 0.3 and P(Occupied) = 0.6',
            xaxis=dict(title='Sampling Rounds (Time)', showgrid=True, showticklabels=True),
            yaxis=dict(title='Frequency Channels', showgrid=True, showticklabels=True))
        figure = graph_objs.Figure(data=data, layout=layout)
        try:
            # Interactive plotting online so that the plot can be saved later for analytics
            plotly.plotly.iplot(figure, filename='Spectrum Occupancy Map with Dual-Dimension Markovian Correlation')
        except Exception as e:
            print(
                '[ERROR] SpatioTemporalMarkovChainVisualizer Visualization: Plotly Heatmap- '
                'Exception caught while plotting [', e, ']')
        print('[INFO] SpatioTemporalMarkovChainVisualizer Visualization: Completed visualization!')

    # Termination
    def __exit__(self, exc_type, exc_val, exc_tb):
        print('[INFO] SpatioTemporalMarkovChainVisualizer Termination: Cleaning things up...')


# Run Trigger
if __name__ == '__main__':
    # Instance creation for visualization
    spatio_temporal_markov_chain_visualizer = SpatioTemporalMarkovChainVisualizer()
    spatio_temporal_markov_chain_visualizer.visualize()
