# This entity parses the CIL collaboration messages received from our peers in a SC2 traffic scenario
# This parsed information will be used to visualize the occupancy behavior of the Incumbent ...
# ... in the given SC2 traffic scenario
# Author: Bharath Keshavamurthy
# School of Electrical and Computer Engineering, Purdue University
# 2018 (c) Copyright. All Rights Reserved.

import sqlite3
import os
import proto_cil_pb2
from google.protobuf.json_format import MessageToJson


class SC2CILMessageParser(object):
    # SRN_ID REPLACEMENT KEY
    SRN_ID_REPLACEMENT_KEY = '${SRN_ID}'

    # CIL Message Query from CollabCILRx
    CIL_MESSAGE_QUERY = 'SELECT time, msg FROM CollabCILRx WHERE srnID = ${SRN_ID} ORDER BY time ASC'

    # CIL Network ID Query - This is the SRN_ID of the Gateway of the Collaborating competitor
    CIL_NETWORK_IDENTIFIER_QUERY = 'SELECT DISTINCT srnID FROM CollabCILRx'

    # The Initialization sequence
    def __init__(self):
        print('[INFO] SC2CILMessageParser Initialization: Bringing things up...')
        self.db_file = input('Please enter the DB File Path: ')

    # The core method
    def parse(self):
        if os.path.isfile(self.db_file):
            cursor = sqlite3.connect(self.db_file)
            peers = cursor.execute(self.CIL_NETWORK_IDENTIFIER_QUERY).fetchall()
            for peer in peers:
                collab_msgs_from_peer = cursor.execute(
                    self.CIL_MESSAGE_QUERY.replace(self.SRN_ID_REPLACEMENT_KEY, str(peer[0]))).fetchall()
                for row in collab_msgs_from_peer:
                    cil_message = proto_cil_pb2.CilMessage()
                    cil_message.ParseFromString(row[1])
                    json_object = MessageToJson(cil_message)
                    print(json_object)
        else:
            print('[ERROR] SC2CILMessageParser Initialization: Incorrect file path - [', self.db_file, ']')

    # Termination sequence
    def __exit__(self, exc_type, exc_val, exc_tb):
        print('[INFO] SC2CILMessageParser Termination: Cleaning things up...')


# Run Trigger
if __name__ == '__main__':
    sc2CILMessageParser = SC2CILMessageParser()
    sc2CILMessageParser.parse()
