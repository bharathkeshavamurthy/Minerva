# This Python script constitutes a playground to understand the workings of the various utilities available in the...
# ...Python community
# Author: Bharath Keshavamurthy
# Organization: School of Electrical and Computer Engineering, Purdue University
# Copyright (c) 2018. All Rights Reserved.

# Direct Run Trigger
if __name__ == '__main__':
    sample_array = [k + 1 for k in range(0, 10)]
    print(sample_array[-1])
    collection_of_dicts = [[dict() for j in range(0, 10)] for i in range(0, 10)]
    for k, v in collection_of_dicts[-1][-1].items():
        print(v)
    print('Range check...')
    for i in range(18 - 1, -1, -1):
        print(i)
    print('Loop check...')
    for _simple_counter in range(0, 2):
        print(_simple_counter)
