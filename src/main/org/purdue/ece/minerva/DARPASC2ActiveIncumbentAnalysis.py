# This entity describes the analysis of the occupancy behavior of the active incumbent and our competitors in the DARPA
#   SC2 "Active Incumbent" scenario. The occupancy behavior that is studied as a part of this entity, will be used in
#   the performance evaluation of our HMM & Approximate POMDP framework--which is a component of Project Minerva.
# Author: Bharath Keshavamurthy
# Organization: School of Electrical and Computer Engineering, Purdue University, West Lafayette, IN.
# Copyright (c) 2020. All Rights Reserved.

# Note that in this analysis, BAM Wireless (without the PERSEUS MAC) will be treated as a competitor (or incumbent).

# The imports
import numpy
import sqlite3
import contextlib
from collections import OrderedDict


# The purpose of this class is analyze the occupancy behavior of the DARPA SC2 Active Incumbent and other competitors
#   and determine two things: Can we extract a double Markov chain time-frequency correlation structure from this
#   occupancy behavior--and if yes, use this occupancy information in the evaluation of our PERSEUS-III framework.
class DARPASC2ActiveIncumbentAnalysis(object):

    # PSD Occupancy Threshold (in dB)
    PSD_OCCUPANCY_THRESHOLD = -25

    # The number of channels in a group--extracted from the 1024 database discretization
    # 10 MHz scenario--20 channels of 500kHz BW each
    GROUP_COUNT = 51

    # Start time data acquisition query
    START_TIME_ACQUISITION_QUERY = "SELECT MIN(time) from C2APIEvent where txt='START'"

    # Stop time data acquisition query
    STOP_TIME_ACQUISITION_QUERY = "SELECT MAX(time) from C2APIEvent where txt='STOP'"

    # SRN data acquisition query
    SRN_DATA_ACQUISITION_QUERY = 'SELECT DISTINCT srnID as d_srnID FROM Start ORDER BY d_srnID ASC'

    # PSD data acquisition query
    PSD_DATA_ACQUISITION_QUERY = 'SELECT time_ns, psd FROM PSDUpdateEvent WHERE srnID={} ORDER BY time_ns ASC'

    # The initialization sequence
    def __init__(self, db_file):
        print('[INFO] DARPASC2ActiveIncumbentAnalysis Initialization: Bringing things up...')
        self.db_file = db_file
        # Scenario start time in nanoseconds
        self.simulation_start_time = self.execute(self.START_TIME_ACQUISITION_QUERY)[0]
        # Scenario end time in nanoseconds
        self.simulation_stop_time = self.execute(self.STOP_TIME_ACQUISITION_QUERY)[0]
        # Identifiers for all the Standard Radio Nodes (SRNs) in the BAM_Wireless network--deployed in this scenario
        self.srns = [x[0] for x in self.execute(self.SRN_DATA_ACQUISITION_QUERY)]
        # The initialization process has been completed

    # Independent, safe, and compact SQLite3 connection instance creation, query execution, and connection closure
    def execute(self, query):
        with contextlib.closing(sqlite3.connect(self.db_file)) as db_connection:
            with db_connection:
                with contextlib.closing(db_connection.cursor()) as db_cursor:
                    db_cursor.execute(query)
                    return db_cursor.fetchall()
        # Nothing to return...

    # Get the local PSD data of the SRNs of the BAM_Wireless network
    def get_psd_data(self):
        # The output to be returned
        psds = OrderedDict()
        # The transient global PSD data variable
        global_psd_data = {n: {} for n in self.srns}
        # Unfiltered PSD data extraction
        for srn in self.srns:
            srn_psd_data = self.execute(self.PSD_DATA_ACQUISITION_QUERY.format(srn))
            for entry in srn_psd_data:
                global_psd_data[srn][entry[0]] = 10*numpy.log10(numpy.frombuffer(entry[1], dtype='<f4'))
        # Filtering, Processing, and Consolidating the extracted PSD data
        index_srn = max(global_psd_data.keys(), key=lambda x: min(global_psd_data[x].keys()))
        index_timestamps = sorted(global_psd_data[index_srn].keys())
        for timestamp in index_timestamps:
            psds[timestamp] = global_psd_data[index_srn][timestamp]
            for srn in filter(lambda x: x != index_srn, self.srns):
                closest_timestamp = min(global_psd_data[srn].keys(), key=lambda x: abs(timestamp - x))
                psds[timestamp] += global_psd_data[srn][closest_timestamp]
            psds[timestamp] /= len(self.srns)
        # Return the processed PSD data
        return psds

    # After processing the PSD data, determine the occupancy behavior of the incumbents and the competitors in the
    #   network--during the emulation of the DARPA SC2 Active Incumbent scenario
    def get_occupancy_behavior(self):
        # The output to be returned
        condensed_occupancy_behavior = OrderedDict()
        # Transient variables
        psds = self.get_psd_data()
        grouping_counter = 0
        occupancy_behavior_collection = OrderedDict()
        # Extraction
        for timestamp, avg_psd_values in psds.items():
            # The discretization remains constant throughout (1024)
            for channel in range(len(avg_psd_values)):
                occupancy_behavior_collection.setdefault(channel, []).append(
                    (lambda: 0, lambda: 1)[int(avg_psd_values[channel]) >= self.PSD_OCCUPANCY_THRESHOLD]())
        # Condensation
        for channel, occupancies in occupancy_behavior_collection.items():
            condensed_channel_index = int(grouping_counter / self.GROUP_COUNT)
            if condensed_channel_index not in condensed_occupancy_behavior.keys():
                condensed_occupancy_behavior[condensed_channel_index] = numpy.array(occupancies)
            else:
                condensed_occupancy_behavior[condensed_channel_index] += numpy.array(occupancies)
            grouping_counter += 1
        # Re-negotiation
        for channel, occupancies in condensed_occupancy_behavior.items():
            for x in range(len(occupancies)):
                condensed_occupancy_behavior[channel][x] = (lambda: 0, lambda: 1)[
                    condensed_occupancy_behavior[channel][x] >= int(self.GROUP_COUNT / 2)]()
        # Return the re-negotiated, condensed, extracted occupancy behavior collection
        return condensed_occupancy_behavior

    # The termination sequence
    def __exit__(self, exc_type, exc_val, exc_tb):
        print('[INFO] DARPASC2ActiveIncumbentAnalysis Termination: Tearing things down...')
        # Nothing to do here...


# Run Trigger
if __name__ == '__main__':
    # The default DB file (Active Incumbent Scenario-8342)
    db = 'data/active_incumbent_scenario8342.db'
    print('[INFO] DARPASC2ActiveIncumbentAnalysis main: Starting the analysis of the occupancy behavior of the Active '
          'Incumbent and our competitors in a DARPA SC2 Active Incumbent scenario...')
    analyser = DARPASC2ActiveIncumbentAnalysis(db)
    _occupancy_behavior_collection = analyser.get_occupancy_behavior()
    print('[INFO] DARPASC2ActiveIncumbentAnalysis main: Completed the analysis of the occupancy behavior of the Active '
          'Incumbent and our competitors in a DARPA SC2 Active Incumbent scenario!')
