# This entity visualizes the Occupancy Behavior of an SC2 Incumbent in the DSRC traffic scenario
# Author: Bharath Keshavamurthy
# Organization: School of Electrical and Computer Engineering, Purdue University
# Copyright (c) 2018. All Rights Reserved.

import os
import json
import configparser
import collections
import plotly
import plotly.graph_objs as graph_objs

plotly.tools.set_credentials_file(username='bkeshava', api_key='RHqYrDdThygiJEPiEW5S')


# DARPA SC2 DSRC Incumbent Behavior Visualization
class SC2IncumbentBehaviorVisualizer(object):
    # COLOSSEUM_CONFIG_INI_FILE
    COLOSSEUM_CONFIG_INI_FILE = 'config/example_incumbent_dsrc_colosseum_config.ini'

    # RADIO_CONF_FILE
    RADIO_CONF_FILE = 'config/example_incumbent_dsrc_radio.conf'

    # SRN_ID
    SRN_ID = 111

    # NETWORK_TYPE
    NETWORK_TYPE = 'incumbent-dsrc'

    # SCHEDULE_KEY
    SCHEDULE_KEY = 'Schedule'

    # TIMESTAMP_KEY
    TIMESTAMP_KEY = 'Timestamp'

    # RF_CENTER_FREQ_OFFSET_KEY
    RF_CENTER_FREQ_OFFSET_KEY = 'RFCenterFrequencyOffset'

    # CENTER_FREQUENCY_KEY
    CENTER_FREQUENCY_KEY = 'center_frequency'

    # RF_BANDWIDTH_KEY
    RF_BANDWIDTH_KEY = 'rf_bandwidth'

    # RF_SECTION_KEY
    RF_SECTION_KEY = 'RF'

    # OFFICIAL_INCUMBENT_RF_BW_KEY
    OFFICIAL_INCUMBENT_RF_BW_KEY = 'RFBandwidth'

    # OCCUPANCY_NAMED_TUPLE
    OCCUPANCY_NAMED_TUPLE = collections.namedtuple('Occupancy', ['lower_cutoff', 'upper_cutoff'])

    # PRECISION = 1MHz
    PRECISION = 1000000

    # DEFAULT LOWEST FREQUENCY
    LOWEST_FREQUENCY_DEFAULT = 100000000000

    # Initialization sequence
    def __init__(self):
        print('[INFO] SC2IncumbentBehaviorVisualizer Initialization: Bringing things up...')
        self.true_pu_occupancy_states = []
        self.radio_conf_data = None
        self.timestamp_parameter_map = {}
        self.official_incumbent_rf_bandwidth = 0.0
        self.official_incumbent_center_frequency = 0.0
        self.lower_end_of_the_spectrum = 0.0
        self.higher_end_of_the_spectrum = 0.0
        self.time_axis = []
        self.channel_axis = []
        self.temporal_termination_point = 0.0

    # Get Properties
    def get_properties(self):
        if os.path.isfile(self.RADIO_CONF_FILE):
            with open(self.RADIO_CONF_FILE) as radio_conf_file:
                self.radio_conf_data = json.load(radio_conf_file)
                self.official_incumbent_rf_bandwidth = self.radio_conf_data[self.OFFICIAL_INCUMBENT_RF_BW_KEY]
        else:
            print('[ERROR] SC2IncumbentBehaviorVisualizer Initialization: Invalid radio.conf path!')

    # Process the radio configurations and store them in suitable collection(s)
    def preprocess_radio_configurations(self):
        if self.radio_conf_data is not None:
            try:
                for schedule_change_count in range(0, len(self.radio_conf_data[self.SCHEDULE_KEY])):
                    self.timestamp_parameter_map[
                        self.radio_conf_data[self.SCHEDULE_KEY][schedule_change_count][self.TIMESTAMP_KEY]] = \
                        self.radio_conf_data[self.SCHEDULE_KEY][schedule_change_count][self.RF_CENTER_FREQ_OFFSET_KEY]
                return True
            except Exception as e:
                print('[ERROR] SC2IncumbentBehaviorVisualizer Preprocessing: ''Exception caught while processing the '
                      'configurations- [', e, ']')
        return False

    # Process the colosseum configurations and update the collections object
    def preprocess_colosseum_configurations(self):
        if os.path.isfile(self.COLOSSEUM_CONFIG_INI_FILE):
            config_parser = configparser.ConfigParser()
            config_parser.read(self.COLOSSEUM_CONFIG_INI_FILE)
            if config_parser[self.RF_SECTION_KEY] is not None and len(config_parser[self.RF_SECTION_KEY]) > 0:
                center_frequency = config_parser[self.RF_SECTION_KEY][self.CENTER_FREQUENCY_KEY]
            else:
                print('[ERROR] SC2IncumbentBehaviorVisualizer visualize: Invalid Colosseum Config .ini file for '
                      'this incumbent')
                return False
        else:
            print('[ERROR] SC2IncumbentBehaviorVisualizer visualize: No file found!')
            return False
        temp_timestamp_parameter_map = {}
        for k, v in self.timestamp_parameter_map.items():
            new_center_frequency = int(center_frequency) + int(v)
            lower_cutoff_frequency = new_center_frequency - (self.official_incumbent_rf_bandwidth / 2)
            upper_cutoff_frequency = new_center_frequency + (self.official_incumbent_rf_bandwidth / 2)
            temp_timestamp_parameter_map[k] = self.OCCUPANCY_NAMED_TUPLE(lower_cutoff=lower_cutoff_frequency,
                                                                         upper_cutoff=upper_cutoff_frequency)
        self.timestamp_parameter_map.clear()
        self.timestamp_parameter_map = temp_timestamp_parameter_map
        return True

    # Process time axis for visualization
    def process_time_axis(self):
        previous_timestamp = 0.0
        temporal_sample_difference = []
        if len(self.timestamp_parameter_map) > 0:
            if len(self.timestamp_parameter_map) > 1:
                for k, v in self.timestamp_parameter_map.items():
                    temporal_sample_difference.append(abs(previous_timestamp - k))
                    if k > self.temporal_termination_point:
                        self.temporal_termination_point = k
                    previous_timestamp = k
                _sum = 0.0
                for i in range(1, len(temporal_sample_difference)):
                    _sum += temporal_sample_difference[i]
                self.temporal_termination_point += _sum / (len(temporal_sample_difference) - 1)
            else:
                self.temporal_termination_point = list(self.timestamp_parameter_map.keys())[0] * 2
        else:
            print(
                '[ERROR] SC2IncumbentBehaviorVisualizer visualize: Error reading the pre-processed properties. '
                'Please refer to the previous error messages for more details.')
            return False
        self.time_axis = [k for k in range(0, len(self.timestamp_parameter_map))]
        return True

    # Process Channel Axis for visualization
    def process_channel_axis(self):
        if len(self.timestamp_parameter_map) > 0:
            lowest_frequency = self.LOWEST_FREQUENCY_DEFAULT
            highest_frequency = 0.0
            for k, v in self.timestamp_parameter_map.items():
                if v.lower_cutoff < lowest_frequency:
                    lowest_frequency = v.lower_cutoff
                if v.upper_cutoff > highest_frequency:
                    highest_frequency = v.upper_cutoff
            number_of_channels = round((highest_frequency - lowest_frequency) / self.PRECISION)
            self.channel_axis = [k for k in range(0, number_of_channels)]
            self.lower_end_of_the_spectrum = lowest_frequency
            self.higher_end_of_the_spectrum = highest_frequency
            return True
        else:
            print('[ERROR] SC2IncumbentBehaviorVisualizer visualize: Invalid timestamp parameter mapping encountered!')
            return False

    # Visualization method
    # Core method
    def visualize(self):
        if self.process_time_axis() and self.process_channel_axis():
            for channel_index in range(0, len(self.channel_axis)):
                self.true_pu_occupancy_states.append(list())
                for k, v in self.timestamp_parameter_map.items():
                    if v.lower_cutoff < (self.lower_end_of_the_spectrum + (
                            self.PRECISION * channel_index)) < v.upper_cutoff or v.lower_cutoff < (
                            self.lower_end_of_the_spectrum + (
                            self.PRECISION * channel_index) + self.PRECISION) < v.upper_cutoff:
                        self.true_pu_occupancy_states[channel_index].append(1)
                    else:
                        self.true_pu_occupancy_states[channel_index].append(0)
            return True
        else:
            print('[ERROR] SC2IncumbentBehaviorVisualizer visualize: Visualization Failed. Please refer to the earlier '
                  'logs for more information on this error...')
            return False

    # Final Heatmap visualization stage
    def visualize_heat_map(self):
        # Plotly API's HeatMap
        data = [
            graph_objs.Heatmap(z=self.true_pu_occupancy_states, x=self.time_axis,
                               y=self.channel_axis, colorscale=[[0, 'rgb(0,255,0)'], [1, 'rgb(255,0,0)']],
                               colorbar=dict(title='PU Occupancy', titleside='right', tickmode='array', tickvals=[0, 1],
                                             ticktext=['Unoccupied', 'Occupied'], ticks='outside'),
                               showscale=True)]
        # Layout
        layout = graph_objs.Layout(title='Spectrum Occupancy Map of an SC2 DSRC WLAN Incumbent with SRN_ID: ' +
                                         str(self.SRN_ID) + ' and spectrum ''ranging from ' +
                                         str(self.lower_end_of_the_spectrum) + ' Hz to ' +
                                         str(self.higher_end_of_the_spectrum) + ' Hz',
                                   xaxis=dict(title='Sampling Rounds (Time) extracted from TimeStamps', showgrid=True,
                                              linecolor='black', showticklabels=True),
                                   yaxis=dict(
                                       title='Frequency Channels extracted from changes in center frequency offsets',
                                       showgrid=True,
                                       linecolor='black',
                                       showticklabels=True))
        figure = graph_objs.Figure(data=data, layout=layout)
        try:
            # Interactive plotting online so that the plot can be saved later for analytics
            plotly.plotly.iplot(figure, filename='Spectrum Occupancy Map of an SC2 DSRC WLAN Incumbent')
        except Exception as e:
            print(
                '[ERROR] SC2IncumbentBehaviorVisualizer visualize: Plotly Heatmap- '
                'Exception caught while plotting [', e, ']')

    # Termination sequence
    def __exit__(self, exc_type, exc_val, exc_tb):
        print('[INFO] SC2IncumbentBehaviorVisualizer Termination: Cleaning things up...')


# Run Trigger
if __name__ == '__main__':
    print('[INFO] SC2IncumbentBehaviorVisualizer Run Trigger: Starting Visualization...')
    sC2IncumbentBehaviorVisualizer = SC2IncumbentBehaviorVisualizer()
    sC2IncumbentBehaviorVisualizer.get_properties()
    if sC2IncumbentBehaviorVisualizer.preprocess_radio_configurations() and \
            sC2IncumbentBehaviorVisualizer.preprocess_colosseum_configurations() and \
            sC2IncumbentBehaviorVisualizer.visualize():
        sC2IncumbentBehaviorVisualizer.visualize_heat_map()
        print('[INFO] SC2IncumbentBehaviorVisualizer Run Trigger: Visualization Complete!')
    else:
        print('[ERROR] SC2IncumbentBehaviorVisualizer Run Trigger: Visualization Incomplete due to previous errors')
