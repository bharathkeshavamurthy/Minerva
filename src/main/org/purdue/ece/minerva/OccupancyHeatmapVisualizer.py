# This entity helps visualize a heatmap of the occupancy behavior of incumbents whose characteristics are governed by
#   a time-frequency Markov correlation structure.
# Author: Bharath Keshavamurthy
# Organization: School of Electrical & Computer Engineering, Purdue University, West Lafayette, IN.
# Copyright (c) 2020. All Rights Reserved.

# The imports
import numpy
import plotly
from enum import Enum
import plotly.graph_objs as graph_objs

plotly.tools.set_credentials_file(username='bkeshava', api_key='BEp2EMeaooErdwcIF8Ss')


# This enumeration entity lists the various types of errors (and their associated error codes) that can potentially be
#   caught during the emulation process
class ErrorCode(Enum):

    # No errors, i.e., successful operation
    SUCCESS = 0

    # An error in occupancy behavior emulation due to an invalid ${visualization_type}
    INVALID_VISUALIZATION_TYPE = 1

    # An error in occupancy behavior emulation due to an unsupported ${visualization_type}
    UNSUPPORTED_VISUALIZATION_TYPE = 2

    # An error corresponding to an invalid fragmentation model (bureaucracy, regulations, and traffic patterns dictate
    #   the fragmentation model w.r.t the incumbents in the radio environment)
    INVALID_FRAGMENTATION_MODEL = 3

    # An error in indexing during manipulation on collection objects
    COLLECTION_INDEX_ERROR = 4

    # An unknown error: print the exception details for more information on this unknown error
    UNKNOWN_ERROR = 5


# This enumeration entity lists the various types of visualization we intend to pursue in this script
class VisualizationType(Enum):
    # The occupancy behavior of 3 incumbents (modeled as 1) with occupancy in a cell independent of occupancies of other
    #   cells in the rendered heatmap
    CLUSTERED_INDEPENDENCE = 0

    # The occupancy behavior of 3 incumbents (fragmented into 3 externally-independent internally-correlated subsets)
    #   with occupancy in a cell independent of occupancies of other cells in the rendered heatmap
    FRAGMENTED_INDEPENDENCE = 1

    # The occupancy behavior of 3 incumbents (modeled as 1) with occupancy in a cell correlated with occupancies of
    #   other cells in the rendered heatmap--in a time-frequency double Markovian structure
    CLUSTERED_MARKOVIAN = 2

    # The occupancy behavior of 3 incumbents (fragmented into 3 externally-independent internally-correlated subsets)
    #   with occupancy in a cell correlated with occupancies of other cells in the rendered heatmap--in a
    #   time-frequency double Markovian structure
    FRAGMENTED_MARKOVIAN = 3


# This class encapsulates the emulation of the time-frequency correlated occupancy behavior of the incumbents in the
#   radio environment; visualizes this behavior in terms of a heatmap; and compares it with an occupancy heatmap with
#   independence assumptions.
class OccupancyHeatmapVisualizer(object):

    # The initialization sequence
    def __init__(self):
        print('[INFO] OccupancyHeatmapVisualizer Initialization: Bringing things up...')
        # The error code
        self.error_code = ErrorCode.SUCCESS
        # The number of incumbents in the radio environment $J, j$
        self.number_of_incumbents = 3
        # The number of channels in the discretized spectrum of interest $K, k$
        self.number_of_channels = 20
        # The fragmentation model for the incumbents
        # NOTE: The values should add up to ${number_of_channels}
        self.fragmentation_model = {0: 6, 1: 6, 2: 6}
        if sum(self.fragmentation_model.values()) != self.number_of_channels:
            print('[ERROR] OccupancyHeatmapVisualizer Initialization: Invalid fragmentation model')
            self.error_code = ErrorCode.INVALID_FRAGMENTATION_MODEL
        # The number of time-slots/episodes of occupancy behavior emulation $T, t$
        self.number_of_timeslots = 50
        # The steady-state occupancy probability of a time-frequency cell/pixel $\Pi$
        # Note here that all cells (w.r.t all incumbents) have the same steady-state occupancy (for a different mosaic
        #   in the 'fragmented independence' case, use different $\Pi$s for different incumbent spectrum fragments)
        self.pi = 0.7
        # The time-frequency double Markovian correlation structure $\mathbf{A}$
        # Note here that the $q$ parameters correspond to both the spatial (frequency) and the temporal (time) chains
        # Also, note here that in the 'fragmented Markovian" case, all cells (w.r.t all incumbents) have the same
        #   correlation model.
        self.time_frequency_correlation_structure = {'p00': 0.25, 'p01': 0.75, 'p10': 0.71, 'p11': 0.8,
                                                     'q0': 0.67, 'q1': 0.88}
        # The data object that constitutes the time-frequency occupancy behavior of incumbents in the network
        self.occupancy_behavior = [[k*(t-t) for t in range(self.number_of_timeslots)]
                                   for k in range(self.number_of_channels)]
        # The initialization sequence has been completed

    # Simulate the occupancy behavior of the incumbents based on the provided 'visualization_type'
    def simulate_incumbent_occupancy_behavior(self, visualization_type):
        if not isinstance(visualization_type, VisualizationType):
            print('[ERROR] OccupancyHeatmapVisualizer simulate_incumbent_occupancy_behavior: '
                  'Invalid ${visualization_type}.')
            return ErrorCode.INVALID_VISUALIZATION_TYPE
        # Visualization Routing based on the provided ${visualization_type}
        if visualization_type == VisualizationType.CLUSTERED_INDEPENDENCE:
            self.error_code = self.simulate_clustered_independence()
        elif visualization_type == VisualizationType.FRAGMENTED_INDEPENDENCE:
            self.error_code = self.simulate_fragmented_independence()
        elif visualization_type == VisualizationType.CLUSTERED_MARKOVIAN:
            self.error_code = self.simulate_clustered_markovian()
        elif visualization_type == VisualizationType.FRAGMENTED_MARKOVIAN:
            self.error_code = self.simulate_fragmented_markovian()
        else:
            print('[ERROR] OccupancyHeatmapVisualizer simulate_incumbent_occupancy_behavior: '
                  'Unsupported ${visualization_type}.')
            return ErrorCode.UNSUPPORTED_VISUALIZATION_TYPE
        return (lambda: self.error_code, lambda: self.visualize_heatmap(visualization_type))[
            self.error_code == ErrorCode.SUCCESS]()

    # Simulate the occupancy behavior of incumbents in the 'clustered_independence' case
    def simulate_clustered_independence(self):
        try:
            for k in range(self.number_of_channels):
                for t in range(self.number_of_timeslots):
                    if numpy.random.random_sample() <= self.pi:
                        self.occupancy_behavior[k][t] = 1
        except IndexError as index_error:
            print('[ERROR] OccupancyHeatmapVisualizer simulate_clustered_independence: Exception caught while emulating'
                  ' the clustered independent occupancy behavior of incumbents - {}'.format(index_error))
            return ErrorCode.COLLECTION_INDEX_ERROR
        except Exception as exception:
            print('[ERROR] OccupancyHeatmapVisualizer simulate_clustered_independence: Exception caught while emulating'
                  ' the clustered independent occupancy behavior of incumbents - {}'.format(exception))
            return ErrorCode.UNKNOWN_ERROR
        return ErrorCode.SUCCESS

    # Simulate the occupancy behavior of incumbents in the 'fragmented_independence' case
    def simulate_fragmented_independence(self):
        pointer = 0
        try:
            for j in range(self.number_of_incumbents):
                for k in range(pointer, pointer + self.fragmentation_model[j]):
                    for t in range(self.number_of_timeslots):
                        if numpy.random.random_sample() <= self.pi:
                            self.occupancy_behavior[k][t] = 1
                pointer += self.fragmentation_model[j]
        except IndexError as index_error:
            print('[ERROR] OccupancyHeatmapVisualizer simulate_fragmented_independence: Exception caught while '
                  'emulating the clustered independent occupancy behavior of incumbents - {}'.format(index_error))
            return ErrorCode.COLLECTION_INDEX_ERROR
        except Exception as exception:
            print('[ERROR] OccupancyHeatmapVisualizer simulate_fragmented_independence: Exception caught while '
                  'emulating the clustered independent occupancy behavior of incumbents - {}'.format(exception))
            return ErrorCode.UNKNOWN_ERROR
        return ErrorCode.SUCCESS

    # Simulate the occupancy behavior of incumbents in the 'clustered Markovian' case
    def simulate_clustered_markovian(self):
        try:
            # k = 0, t = 0
            if numpy.random.random_sample() <= self.pi:
                self.occupancy_behavior[0][0] = 1
            # k = 0, t
            previous = self.occupancy_behavior[0][0]
            for t in range(1, self.number_of_timeslots):
                if previous == 0 and numpy.random.random_sample() <= self.time_frequency_correlation_structure['q0']:
                    self.occupancy_behavior[0][t] = 1
                elif previous == 0 and numpy.random.random_sample() > self.time_frequency_correlation_structure['q0']:
                    self.occupancy_behavior[0][t] = 0
                elif previous == 1 and numpy.random.random_sample() <= self.time_frequency_correlation_structure['q1']:
                    self.occupancy_behavior[0][t] = 1
                else:
                    self.occupancy_behavior[0][t] = 0
            # t = 0, k
            previous = self.occupancy_behavior[0][0]
            for k in range(1, self.number_of_channels):
                if previous == 0 and numpy.random.random_sample() <= self.time_frequency_correlation_structure['q0']:
                    self.occupancy_behavior[k][0] = 1
                elif previous == 0 and numpy.random.random_sample() > self.time_frequency_correlation_structure['q0']:
                    self.occupancy_behavior[k][0] = 0
                elif previous == 1 and numpy.random.random_sample() <= self.time_frequency_correlation_structure['q1']:
                    self.occupancy_behavior[k][0] = 1
                else:
                    self.occupancy_behavior[k][0] = 0
            # k, t
            for k in range(1, self.number_of_channels):
                for t in range(1, self.number_of_timeslots):
                    previous_spatial = self.occupancy_behavior[k-1][t]
                    previous_temporal = self.occupancy_behavior[k][t-1]
                    if previous_spatial == 0 and previous_temporal == 0:
                        self.occupancy_behavior[k][t] = (lambda: 0, lambda: 1)[
                            numpy.random.random_sample() <= self.time_frequency_correlation_structure['p00']]()
                    elif previous_spatial == 0 and previous_temporal == 1:
                        self.occupancy_behavior[k][t] = (lambda: 0, lambda: 1)[
                            numpy.random.random_sample() <= self.time_frequency_correlation_structure['p01']]()
                    elif previous_spatial == 1 and previous_temporal == 0:
                        self.occupancy_behavior[k][t] = (lambda: 0, lambda: 1)[
                            numpy.random.random_sample() <= self.time_frequency_correlation_structure['p10']]()
                    else:
                        self.occupancy_behavior[k][t] = (lambda: 0, lambda: 1)[
                            numpy.random.random_sample() <= self.time_frequency_correlation_structure['p11']]()
        except IndexError as index_error:
            print(
                '[ERROR] OccupancyHeatmapVisualizer simulate_clustered_markovian: Exception caught while '
                'emulating the clustered Markovian occupancy behavior of incumbents - {}'.format(index_error))
            return ErrorCode.COLLECTION_INDEX_ERROR
        except Exception as exception:
            print('[ERROR] OccupancyHeatmapVisualizer simulate_clustered_markovian: Exception caught while '
                  'emulating the clustered Markovian occupancy behavior of incumbents - {}'.format(exception))
            return ErrorCode.UNKNOWN_ERROR
        return ErrorCode.SUCCESS

    # Simulate the occupancy behavior of incumbents in the 'fragmented Markovian' case
    def simulate_fragmented_markovian(self):
        pointer = 0
        try:
            for j in range(self.number_of_incumbents):
                if numpy.random.random_sample() <= self.pi:
                    self.occupancy_behavior[pointer][0] = 1
                previous = self.occupancy_behavior[pointer][0]
                for t in range(1, self.number_of_timeslots):
                    if previous == 0 and numpy.random.random_sample() <= \
                            self.time_frequency_correlation_structure['q0']:
                        self.occupancy_behavior[pointer][t] = 1
                    elif previous == 0 and numpy.random.random_sample() > \
                            self.time_frequency_correlation_structure['q0']:
                        self.occupancy_behavior[pointer][t] = 0
                    elif previous == 1 and numpy.random.random_sample() <= \
                            self.time_frequency_correlation_structure['q1']:
                        self.occupancy_behavior[pointer][t] = 1
                    else:
                        self.occupancy_behavior[pointer][t] = 0
                previous = self.occupancy_behavior[pointer][0]
                for k in range(pointer + 1, pointer + self.fragmentation_model[j]):
                    if previous == 0 and numpy.random.random_sample() <= \
                            self.time_frequency_correlation_structure['q0']:
                        self.occupancy_behavior[k][0] = 1
                    elif previous == 0 and numpy.random.random_sample() > \
                            self.time_frequency_correlation_structure['q0']:
                        self.occupancy_behavior[k][0] = 0
                    elif previous == 1 and numpy.random.random_sample() <= \
                            self.time_frequency_correlation_structure['q1']:
                        self.occupancy_behavior[k][0] = 1
                    else:
                        self.occupancy_behavior[k][0] = 0
                for k in range(pointer + 1, pointer + self.fragmentation_model[j]):
                    for t in range(1, self.number_of_timeslots):
                        previous_spatial = self.occupancy_behavior[k - 1][t]
                        previous_temporal = self.occupancy_behavior[k][t - 1]
                        if previous_spatial == 0 and previous_temporal == 0:
                            self.occupancy_behavior[k][t] = (lambda: 0, lambda: 1)[
                                numpy.random.random_sample() <= self.time_frequency_correlation_structure['p00']]()
                        elif previous_spatial == 0 and previous_temporal == 1:
                            self.occupancy_behavior[k][t] = (lambda: 0, lambda: 1)[
                                numpy.random.random_sample() <= self.time_frequency_correlation_structure['p01']]()
                        elif previous_spatial == 1 and previous_temporal == 0:
                            self.occupancy_behavior[k][t] = (lambda: 0, lambda: 1)[
                                numpy.random.random_sample() <= self.time_frequency_correlation_structure['p10']]()
                        else:
                            self.occupancy_behavior[k][t] = (lambda: 0, lambda: 1)[
                                numpy.random.random_sample() <= self.time_frequency_correlation_structure['p11']]()
                pointer += self.fragmentation_model[j]
        except IndexError as index_error:
            print(
                '[ERROR] OccupancyHeatmapVisualizer simulate_clustered_markovian: Exception caught while '
                'emulating the clustered Markovian occupancy behavior of incumbents - {}'.format(index_error))
            return ErrorCode.COLLECTION_INDEX_ERROR
        except Exception as exception:
            print('[ERROR] OccupancyHeatmapVisualizer simulate_clustered_markovian: Exception caught while '
                  'emulating the clustered Markovian occupancy behavior of incumbents - {}'.format(exception))
            return ErrorCode.UNKNOWN_ERROR
        return ErrorCode.SUCCESS

    # Visualize the occupancy behavior heatmap using the Plotly API
    def visualize_heatmap(self, visualization_type):
        # Time-slots/Episodes
        horizontal_axis = [k+1 for k in range(0, self.number_of_timeslots)]
        # Channels
        vertical_axis = [k+1 for k in range(0, self.number_of_channels)]
        # Data
        # Plotly API's HeatMap
        data = [
            graph_objs.Heatmap(z=self.occupancy_behavior, x=horizontal_axis, y=vertical_axis, xgap=1, ygap=1,
                               colorscale=[[0, 'rgb(255,255,255)'], [1, 'rgb(0,0,0)']],
                               colorbar=dict(title='Occupancy Behavior of Incumbents with '
                                                   '{}'.format(visualization_type.name), titleside='right',
                                             tickmode='array', tickvals=[0, 1], ticktext=['Idle (White Space)',
                                                                                          'Occupied by an Incumbent'],
                                             ticks='outside'), showscale=True)]
        # Layout
        layout = graph_objs.Layout(
            xaxis=dict(title='Time-slots', showgrid=True, showticklabels=True),
            yaxis=dict(title='Frequency Channels', showgrid=True, showticklabels=True))
        figure = graph_objs.Figure(data=data, layout=layout)
        try:
            plotly.plotly.plot(figure, filename='Minerva_Channel_Occupancy_Model')
        except Exception as exception:
            print(
                '[ERROR] OccupancyHeatmapVisualizer visualize_heatmap: Plotly Heatmap- '
                'Exception caught while plotting - {}'.format(exception))
            return ErrorCode.UNKNOWN_ERROR
        return ErrorCode.SUCCESS

    # The termination sequence
    def __exit__(self, exc_type, exc_val, exc_tb):
        print('[INFO] OccupancyHeatmapVisualizer Termination: Tearing things down...')
        # Nothing to do


# Run Trigger
if __name__ == '__main__':
    print('[INFO] OccupancyHeatmapVisualizer main: Starting occupancy behavior emulation...')
    # Create the OccupancyHeatmapVisualizer instance
    occupancy_heatmap_visualizer = OccupancyHeatmapVisualizer()

    # Uncomment for 'clustered independence' heatmap
    # error_code = occupancy_heatmap_visualizer.simulate_incumbent_occupancy_behavior(
    #     VisualizationType.CLUSTERED_INDEPENDENCE
    # )

    # Uncomment for 'fragmented independence' heatmap
    # error_code = occupancy_heatmap_visualizer.simulate_incumbent_occupancy_behavior(
    #     VisualizationType.FRAGMENTED_INDEPENDENCE
    # )

    # Uncomment for 'clustered Markovian' heatmap
    # error_code = occupancy_heatmap_visualizer.simulate_incumbent_occupancy_behavior(
    #     VisualizationType.CLUSTERED_MARKOVIAN
    # )

    # Uncomment for 'fragmented Markovian' heatmap
    error_code = occupancy_heatmap_visualizer.simulate_incumbent_occupancy_behavior(
        VisualizationType.CLUSTERED_MARKOVIAN
    )

    # A rudimentary error handling mechanism
    if error_code == ErrorCode.SUCCESS:
        print('[INFO] OccupancyHeatmapVisualizer main: Successfully completed occupancy behavior emulation!')
    else:
        print('[INFO] OccupancyHeatmapVisualizer main: Occupancy behavior emulation FAILED | Error - {} | '
              'Refer to the earlier logs for more details on the error'.format(error_code.name))
    # The occupancy behavior emulation procedure ends here
