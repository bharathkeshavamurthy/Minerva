# This entity visualizes the PSD observations made at the BAM! Wireless gateway during an Active Incumbent scenario.
# Author: Bharath Keshavamurthy <bkeshava@purdue.edu>
# Organization: School of Electrical & Computer Engineering, Purdue University, West Lafayette, IN.
# Copyright (c) 2020. All Rights Reserved.

# The imports
import numpy
import plotly
import sqlite3
import os.path
import traceback
import contextlib
from enum import Enum
import plotly.graph_objs as graph_objs

# Plotly Access Credentials
plotly.tools.set_credentials_file(username='bkeshava', api_key='T2AiFO1n02nYt0euM7WX')


# An enumeration listing the various potential sources of the PSD observations
class PSDDataSource(Enum):
    # The SRN itself ('local')
    LOCAL = 0
    # The PSD observations aggregated at the gateway and received at individual SRNs over the common control channel
    GATEWAY = 1
    # Unknown (TODO: Initiate Error Handling [UNKNOWN_PSD_DATA_SOURCE_EXCEPTION])
    INVALID = 2


# This class employs the Plotly API to visualize the PSD measurement heatmap aggregated at the BAM! Wireless gateway
#   during an Active Incumbent scenario emulation on the Colosseum.
class ActiveIncumbentPSDVisualizer(object):

    # The initialization sequence
    def __init__(self, db_file, reference_source, reference_srn_id, scenario_bandwidth, scenario_fc, window):
        print('[INFO] ActiveIncumbentPSDVisualizer Initialization: Bringing things up...')
        # TODO: Initiate Error Handling [FILE_NOT_FOUND_EXCEPTION]
        if not os.path.exists(db_file):
            print('[ERROR] ActiveIncumbentPSDVisualizer Initialization: The specified DB file does not exist - '
                  '{}'.format(db_file))
            return
        # The database file (SQLite DB) constituting the runtime logs of the BAM! Wireless network during this emulation
        self.db_file = db_file
        # The source of PSD measurements: Local (at node) OR Gateway (aggregated at the gateway & received over the
        #   common control channel)
        self.reference_source = reference_source
        # The SRN_ID of the radio node whose PSD observations (local/gateway) are to be visualized (for IEEE TCCN)
        self.reference_srn_id = reference_srn_id
        # The SC2 Active Incumbent scenario bandwidth
        self.scenario_bandwidth = scenario_bandwidth
        # The scenario center frequency (in MHz)
        self.scenario_fc = scenario_fc
        # Visualize only a window [time:=-freq] of the PSD data for a more focused visualization
        self.windowed = False
        self.window = None
        if window is not None and len(window) == 2:
            self.windowed = True
            self.window = window
        # The initialization sequence has been completed.

    # Utility method
    # Execute DB call
    #   Using contextlib to:
    #   1. Ensure that all transactions are committed automatically
    #   2. Ensure that the connection and the cursor are closed automatically
    def execute(self, statement, just_one):
        try:
            with contextlib.closing(sqlite3.connect(self.db_file)) as connection:
                with connection:
                    with contextlib.closing(connection.cursor()) as cursor:
                        cursor.execute(statement)
                        return cursor.fetchone() if just_one else cursor.fetchall()
        except IOError as exception:
            print('[ERROR] ActiveIncumbentPSDVisualizer execute: IOError caught while executing statement [{}] on the'
                  'SQLite DB [{}]'.format(statement, self.db_file))
            traceback.print_tb(exception.__traceback__)
            # TODO: Initiate Error Handling [GENERIC_IO_EXCEPTION]
            return []

    # Core method: Start evaluation and/or visualization
    #   1. Extract the PSD data from the appropriate tables and columns in the provided DB
    #   2. Process the extracted data
    #   3. Visualize the HeatMap of PSD observations implying spectrum occupancy
    def start(self):
        srn_ids = [entry[0] for entry in
                   self.execute('SELECT DISTINCT srnID as d_srnid FROM Start ORDER BY d_srnid ASC', False)]
        if srn_ids is not None and len(srn_ids) != 0 and self.reference_srn_id in srn_ids:
            # Transient members for data collection & processing
            extracted_times = []
            extracted_psds = []
            psd_data = self.execute('SELECT time_ns, psd FROM {} WHERE {}={} ORDER BY time_ns ASC'.format(
                # table_name
                (lambda: 'PSDUpdateEvent',
                 lambda: 'PSDRxEvent')[self.reference_source == PSDDataSource.GATEWAY](),
                # column_name
                (lambda: 'srnID',
                 lambda: 'srcNodeID')[self.reference_source == PSDDataSource.GATEWAY](),
                # srn_id
                self.reference_srn_id
            ), False)
            for entry in psd_data:
                extracted_times.append(entry[0])
                extracted_psds.append(10 * numpy.log10(numpy.frombuffer(entry[1], dtype='<f4')))
            # Process the extracted data
            return self.process_data(numpy.array(extracted_times),
                                     numpy.clip(numpy.array(extracted_psds), -55, None))
        else:
            # TODO: Initiate Error Handling [INVALID_SRN_IDS_LIST_EXCEPTION | INVALID_SRN_ID_EXCEPTION]
            print('[ERROR] ActiveIncumbentPSDVisualizer extract_data: An INVALID_SRN_IDS_LIST_EXCEPTION or an '
                  'INVALID_SRN_ID_EXCEPTION was thrown. Please ensure the validity of [srn_ids] and '
                  '[reference_srn_id].')
            return numpy.array([]), numpy.array([]), numpy.array([])

    # Core method: Process the extracted data
    def process_data(self, times, psds):
        # Collections to be returned
        processed_psds = []
        processed_validity_flags = []
        # Empty arguments (TODO: Initiate Error Handling [INVALID_DATA_ARRAYS_EXCEPTION | EMPTY_DATA_ARRAYS_EXCEPTION]
        if len(times) == 0 or len(psds) == 0:
            print(
                '[WARN] ActiveIncumbentPSDVisualizer process_data: No data extracted for SRN_ID=[{}] '
                'with source=[{}]. Please verify that PSD data exists for this SRN according to the source that is '
                'being requested.'.format(self.reference_srn_id, self.reference_source))
            return times, psds, numpy.array([])
        # Convert the times in nanoseconds to milliseconds
        times = times / 1e6
        time_differences = list(numpy.diff(times).astype(int))
        max_time_interval = max(time_differences, key=time_differences.count)
        # Projection with threshold = 6
        max_time_interval = 6 if max_time_interval < 6 else max_time_interval
        processed_times = numpy.arange(self.execute("SELECT MIN(time) FROM C2APIEvent WHERE txt='START'",
                                                    True)[0] / 1e6,
                                       times[-1], max_time_interval)
        for time in processed_times:
            index = numpy.searchsorted(times, time, side='left')
            if index > 0 and (index == len(times) or abs(time - times[index - 1]) < abs(time - times[index])):
                index -= 1
            processed_psds.append(psds[index])
            # TODO: Incorporate validity checks into the HeatMap visualization logic
            processed_validity_flags.append(abs(time - times[index]) <= 1000)
        # Visualize the extracted and processed data
        return self.visualize(processed_times, numpy.transpose(processed_psds))

    # Core method: Render a heatmap of the relevant PSD observations (based on the reference_source & reference_srn_id
    #   constructor arguments)
    def visualize(self, processed_times, processed_psds):
        # The tick values for the X and Y axes
        time_ticks = [i for i in range(len(processed_times))]
        frequency_ticks = [k for k in range(len(processed_psds))]
        # Windowing
        if self.windowed and self.window[0] in frequency_ticks and self.window[1] in frequency_ticks and \
                self.window[0] < self.window[1]:
            print("[WARN] ActiveIncumbentPSDVisualizer visualize: Windowing is enabled--by the user's choice. "
                  "WINDOWING_STATUS = [{}] | WINDOW = [{}]".format(self.windowed, self.window))
            frequency_ticks = numpy.linspace(self.scenario_fc - (self.scenario_bandwidth / 2),
                                             self.scenario_fc + (self.scenario_bandwidth / 2),
                                             self.window[1] - self.window[0])
            relevant_psds = processed_psds[self.window[0]:self.window[1]]
        else:
            print("[WARN] ActiveIncumbentPSDVisualizer visualize: Windowing is disabled--either by the user's choice or"
                  " due to an invalid window argument. WINDOWING_STATUS = [{}] | WINDOW = [{}]".format(self.windowed,
                                                                                                       self.window))
            frequency_ticks = numpy.linspace(self.scenario_fc - (self.scenario_bandwidth / 2),
                                             self.scenario_fc + (self.scenario_bandwidth / 2),
                                             len(processed_psds))
            relevant_psds = processed_psds
        # frequency_ticks = [k for k in range(len(processed_psds))]
        # Plotly API's HeatMap
        data = [graph_objs.Heatmap(z=relevant_psds, x=time_ticks,
                                   y=frequency_ticks,
                                   colorscale='Rainbow',
                                   colorbar=dict(title='PSD Observations (in dBm) aggregated at the BAM! Wireless '
                                                       'Gateway',
                                                 titleside='right', tickmode='auto',
                                                 ticks='outside'),
                                   showscale=True)]
        # Layout
        #   title='PSD Observations at [{}] captured at [{}]'.format(self.reference_source.name, self.reference_srn_id)
        layout = graph_objs.Layout(xaxis=dict(title='Scenario Emulation Time in seconds', showgrid=True,
                                              linecolor='black', showticklabels=True,
                                              tickmode='auto', ticklen=8),
                                   yaxis=dict(title='Frequencies in MHz (Scenario BW = 10 MHz | Scenario fc = 1 GHz)',
                                              showgrid=True, linecolor='black', showticklabels=True,
                                              tickmode='auto', ticklen=8),
                                   font=dict(family='Droid Serif', size=20, color='#000000'),
                                   margin=dict(l=85, r=35, t=50, b=70))
        figure = graph_objs.Figure(data=data, layout=layout)
        try:
            # Interactive plotting online so that the plot can be saved later for analytics
            plotly.plotly.plot(figure, filename='SC2ActiveIncumbent_Scenario_OccupancyHeatMap.html')
        except Exception as e:
            # TODO: Initiate Error Handling [PLOTLY_EXCEPTION | TOO_MANY_ATTEMPTS_EXCEPTION | PAY_WALL_EXCEPTION]
            print(
                '[ERROR] ActiveIncumbentPSDVisualizer visualize: Plotly SC2 Active Incumbent Scenario '
                'Occupancy Heatmap - Exception caught while plotting [', e, ']')
            traceback.print_tb(e.__traceback__)

    # The termination sequence
    def __exit__(self, exc_type, exc_val, exc_tb):
        print('[INFO] ActiveIncumbentPSDVisualizer Termination: Tearing things down...')
        # Nothing to do...


# Run Trigger
if __name__ == '__main__':
    print('[INFO] ActiveIncumbentPSDVisualizer main: Starting the Active Incumbent PSD Visualization...')
    # The PSD observations heatmap visualizer instance creation...
    visualizer = ActiveIncumbentPSDVisualizer('data/full.db',
                                              PSDDataSource.GATEWAY, 99, 10, 1000, (400, 625))
    # Initiate visualization: Extract --> Process --> Visualize
    visualizer.start()
    # Active Incumbent PSD Visualization has been completed.
