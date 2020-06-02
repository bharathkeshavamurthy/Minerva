# This entity describes the analysis of the occupancy behavior of the active incumbent and our competitors in the DARPA
#   SC2 "Active Incumbent" scenario. The occupancy behavior that is studied as a part of this entity, will be used in
#   the performance evaluation of our HMM & Approximate POMDP framework--which is a component of Project Minerva.
# Author: Bharath Keshavamurthy
# Organization: School of Electrical and Computer Engineering, Purdue University, West Lafayette, IN.
# Copyright (c) 2020. All Rights Reserved.

# The imports
import sqlite3
import contextlib


# The purpose of this class is analyze the occupancy behavior of the DARPA SC2 Active Incumbent and other competitors
#   and determine two things: Can we extract a double Markov chain time-frequency correlation structure from this
#   occupancy behavior--and if yes, use this occupancy information in the evaluation of our PERSEUS-III framework.
class DARPASC2ActiveIncumbentAnalysis(object):

    # Start time data acquisition query
    START_TIME_ACQUISITION_QUERY = "SELECT MIN(time) from C2APIEvent where txt='START'"

    # Stop time data acquisition query
    STOP_TIME_ACQUISITION_QUERY = "SELECT MAX(time) from C2APIEvent where txt='STOP'"

    # SRN data acquisition query
    SRN_DATA_ACQUISITION_QUERY = 'SELECT DISTINCT srnID as d_srnID FROM Start ORDER BY d_srnID ASC'

    # PSD data acquisition query
    PSD_DATA_ACQUISITION_QUERY = 'SELECT time_ns, '

    # The initialization sequence
    def __init__(self, db_file):
        print('[INFO] DARPASC2ActiveIncumbentAnalysis Initialization: Bringing things up...')
        self.db_file = db_file
        # Scenario start time in nanoseconds
        self.simulation_start_time = self.execute(self.START_TIME_ACQUISITION_QUERY)[0]
        # Scenario end time in nanoseconds
        self.simulation_stop_time = self.execute(self.STOP_TIME_ACQUISITION_QUERY)[0]
        # Identifiers for all the Standard Radio Nodes (SRNs) in the BAM_Wireless network--deployed in this scenario
        self.srn_ids = [x[0] for x in self.execute(self.SRN_DATA_ACQUISITION_QUERY)]

    # Independent, safe, and compact SQLite3 connection instance creation, query execution, and connection closure
    def execute(self, query):
        with contextlib.closing(sqlite3.connect(self.db_file)) as db_connection:
            with db_connection:
                with contextlib.closing(db_connection.cursor()) as db_cursor:
                    db_cursor.execute(query)
                    return db_cursor.fetchall()

    # Get the IDs of the SRNs in the BAM_Wireless network
    def get_srn_data(self):
        return

    # Get the local PSD data of the SRNs of the BAM_Wireless network
    def get_psd_data(self):
        db_data = self.execute()

    # The termination sequence
    def __exit__(self, exc_type, exc_val, exc_tb):
        print('[INFO] DARPASC2ActiveIncumbentAnalysis Termination: Tearing things down...')


# Run Trigger
if __name__ == '__main__':
    # The default DB file (Active Incumbent Scenario-8342)
    db = 'data/active_incumbent_scenario8342.db'
    print('[INFO] DARPASC2ActiveIncumbentAnalysis main: Starting the analysis of the occupancy behavior of the Active '
          'Incumbent and our competitors in a DARPA SC2 Active Incumbent scenario...')
    analyser = DARPASC2ActiveIncumbentAnalysis(db)
    print('[INFO] DARPASC2ActiveIncumbentAnalysis main: Completed the analysis of the occupancy behavior of the Active '
          'Incumbent and our competitors in a DARPA SC2 Active Incumbent scenario!')
