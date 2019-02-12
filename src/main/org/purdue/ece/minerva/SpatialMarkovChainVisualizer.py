# A Python Script that visualizes a single Markov Chain across the channel indices
# Spatial Markov chain
# A Markov Chain across the frequency sub-bands
# Static Primary User Occupancy Behavior
# Author: Bharath Keshavamurthy
# Organization: School of Electrical and Computer Engineering, Purdue University
# Copyright (c) 2019. All Rights Reserved.

import numpy
import plotly
import plotly.graph_objs as graph_objs

plotly.tools.set_credentials_file(username='bkeshava', api_key='RHqYrDdThygiJEPiEW5S')


# Channel Object
# Encapsulates the channel index and PU occupancy
class Channel(object):
    # Constructor - Channel Index and PU Occupancy
    def __init__(self, index, occupancy):
        self.index = index
        self.occupancy = occupancy


# Spatial Markov Chain Visualization
# Markovian Correlation only across channel indices
# Static PU
class SpatialMarkovChainVisualizer(object):
    # P(Occupied|Idle) = p = 0.3 (some sample value)
    SAMPLE_VALUE_OF_p = 0.3

    # P(Occupied) = 0.6
    SAMPLE_VALUE_OF_PI = 0.6

    # Number of channels (the channels can be discretized from the wideband spectrum of interest using the...
    # ...Wavelet Transform
    # k goes all the way to 18
    NUMBER_OF_CHANNELS = 18

    # Number of sampling rounds
    # t goes all the way to 100
    NUMBER_OF_SAMPLING_ROUNDS = 100

    # Initialization
    def __init__(self):
        print('[INFO] SpatialMarkovChainVisualizer Initialization: Bringing things up...')
        self.true_pu_occupancy_states = []
        self.p = self.SAMPLE_VALUE_OF_p
        self.pi = self.SAMPLE_VALUE_OF_PI
        # P(0|1)
        self.q = (self.p * (1 - self.pi)) / self.pi
        # The wideband spectrum of interest: A collection (tuple) of the frequency channels
        self.spectrum = ()
        # Transition Probability Matrix for the Markov Chain across channels
        self.spatial_transition_probability_matrix = {
            1: {1: (1 - self.q), 0: self.q},
            0: {1: self.p, 0: (1 - self.p)}
        }

    # Generate the states using Markovian Correlation across the channel indices
    # Static PU behavior
    def allocate_states_with_markovian_across_channel_indices(self, p_val, q_val, pi_val):
        previous = 1
        if numpy.random.random_sample() > pi_val:
            previous = 0
        self.true_pu_occupancy_states.append(previous)
        # After generating the initial state, fill in the other states, i.e. the states of the other successive channels
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
            self.true_pu_occupancy_states.append(previous)

    # Visualization: Core method
    def visualize(self):
        print('[INFO] SpatialMarkovChainVisualizer Visualization: Inside the core method...')
        # Get the states
        self.allocate_states_with_markovian_across_channel_indices(self.p, self.q, self.pi)
        # Time
        horizontal_axis = [k + 1 for k in range(0, self.NUMBER_OF_SAMPLING_ROUNDS)]
        # Channels
        vertical_axis = [k + 1 for k in range(0, self.NUMBER_OF_CHANNELS)]
        # Occupancy Metric
        occupancy_metric = []
        for channel in range(0, self.NUMBER_OF_CHANNELS):
            occupancy_metric.append(
                [self.true_pu_occupancy_states[channel] for k in range(0, self.NUMBER_OF_SAMPLING_ROUNDS)])
        # Data
        # Plotly API's HeatMap
        data = [
            graph_objs.Heatmap(z=occupancy_metric, x=horizontal_axis,
                               y=vertical_axis, xgap=1, ygap=1, colorscale=[[0, 'rgb(0,255,0)'], [1, 'rgb(255,0,0)']],
                               colorbar=dict(title='PU Occupancy', titleside='right', tickmode='array', tickvals=[0, 1],
                                             ticktext=['Unoccupied', 'Occupied'], ticks='outside'),
                               showscale=True)]
        # Layout
        layout = graph_objs.Layout(
            title='Spectrum Occupancy HeatMap with Static PU and Markovian Correlation across channel indices '
                  'with P(Occupied|Idle) = 0.3 and P(Occupied) = 0.6',
            xaxis=dict(title='Sampling Rounds (Time)', showgrid=True, showticklabels=True),
            yaxis=dict(title='Frequency Channels', showgrid=True, showticklabels=True))
        figure = graph_objs.Figure(data=data, layout=layout)
        try:
            # Interactive plotting online so that the plot can be saved later for analytics
            plotly.plotly.iplot(figure, filename='Spectrum Occupancy Map with Single-Dimension Markovian Correlation')
        except Exception as e:
            print(
                '[ERROR] SpatialMarkovChainVisualizer Visualization: Plotly Heatmap- '
                'Exception caught while plotting [', e, ']')
        print('[INFO] SpatialMarkovChainVisualizer Visualization: Completed visualization!')

    # Termination
    def __exit__(self, exc_type, exc_val, exc_tb):
        print('[INFO] SpatialMarkovChainVisualizer Termination: Cleaning things up...')


# Run Trigger
if __name__ == '__main__':
    # Instance creation for visualization
    spatio_markov_chain_visualizer = SpatialMarkovChainVisualizer()
    spatio_markov_chain_visualizer.visualize()
