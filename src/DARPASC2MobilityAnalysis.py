# DARPA SC2 Mobility Analysis
# Author: Bharath Keshavamurthy <bkeshava@purdue.edu>
# Organization: School of Electrical & Computer Engineering, Purdue University, West Lafayette, IN
# Copyright (c) 2021. All Rights Reserved.

import numpy
import random
import sqlite3
import datetime
import contextlib


def execute(query):
    with contextlib.closing(sqlite3.connect(db_file)) as db_connection:
        with db_connection:
            with contextlib.closing(db_connection.cursor()) as db_cursor:
                db_cursor.execute(query)
                return db_cursor.fetchall()


def converged(r_list):
    for k__, v__ in r_list.items():
        for e__, f__ in r_list.items():
            if k__ != e__ and v__ != f__:
                return False
    return True


snr_threshold, time_resolution, emulation_period = 22.5, 10, 340
snapshots = int(emulation_period / time_resolution)
db_file = 'data/logs/minerva/payline/full.db'
srns = [x[0] for x in execute('SELECT DISTINCT srnID as d_srnID FROM Start ORDER BY d_srnID ASC')]
avg_changes = {srn: [] for srn in srns}
global_neighbors_list = {srn: {} for srn in srns}
for srn in srns:
    neighbor_list = {t: [] for t in range(snapshots)}
    for neighbor_srn in srns:
        if neighbor_srn == srn or \
                len(execute('SELECT time FROM SynchronizationEvent WHERE srnID={} AND srcNodeID={} '
                            'AND snr IS NOT NULL ORDER BY time ASC'.format(srn, neighbor_srn))) == 0:
            continue
        initial_timestamp = int((execute('SELECT time FROM SynchronizationEvent WHERE srnID={} AND srcNodeID={} '
                                         'AND snr IS NOT NULL ORDER BY time ASC'.format(srn,
                                                                                        neighbor_srn))[0][0]) / 1e9)
        for t in range(snapshots):
            terminal_timestamp = int((datetime.datetime.fromtimestamp(initial_timestamp) +
                                      datetime.timedelta(seconds=time_resolution)).timestamp()) * 1e9
            if len(execute('SELECT snr FROM SynchronizationEvent WHERE srnID={} AND '
                           'srcNodeID={} AND snr IS NOT NULL AND time>={} AND '
                           'time<={}'.format(srn, neighbor_srn, initial_timestamp,
                                             terminal_timestamp))) == 0:
                continue
            avg_snr = numpy.average(numpy.array(execute('SELECT snr FROM SynchronizationEvent WHERE srnID={} AND '
                                                        'srcNodeID={} AND snr IS NOT NULL AND time>={} AND '
                                                        'time<={}'.format(srn, neighbor_srn, initial_timestamp,
                                                                          terminal_timestamp)), dtype=float))
            if avg_snr >= snr_threshold:
                neighbor_list[t].append((neighbor_srn, avg_snr))
            initial_timestamp = int(terminal_timestamp / 1e9)
    global_neighbors_list[srn] = neighbor_list
    avg_changes[srn].append(len(neighbor_list[0]))
    for t__ in range(snapshots - 1):
        a, b = numpy.array([k[0] for k in neighbor_list[t__]]), numpy.array([k[0] for k in neighbor_list[t__ + 1]])
        avg_changes[srn].append(len(numpy.setdiff1d(a, b)) + len(numpy.setdiff1d(b, a)))
final_changes = numpy.array([])
for srn, avg_changes_array in avg_changes.items():
    if len(final_changes) == 0:
        final_changes = numpy.array(avg_changes_array)
    else:
        final_changes = final_changes + numpy.array(avg_changes_array)
final_changes = final_changes / len(srns)
print('DARPASC2MobilityAnalysis Alleys-of-Austin: Average Neighbor Discovery List Changes = [{}]'.format(
    [change for change in final_changes]))

confidence_threshold, canary_srn = 5, 41
changes = {srn: [0 for _ in range(snapshots)] for srn in srns}
average_changes = [0.0 for _ in range(snapshots)]
for t in range(snapshots):
    if t == 0:
        prev_ranked_lists = {srn: random.sample(srns, k=len(srns)) for srn in srns}
    else:
        prev_ranked_lists = ranked_lists
    ranked_lists = {srn: [srn] for srn in srns}
    assigned_lists = {srn: {s: 0 for s in srns} for srn in srns}
    for srn in srns:
        max_points = len(srns)
        n_list = global_neighbors_list[srn][t]
        sorted_list = sorted([i for i in range(len(n_list))], key=lambda x: n_list[x][1], reverse=True)
        for k in sorted_list:
            ranked_lists[srn].append(n_list[k][0])
        for n in srns:
            if n not in ranked_lists[srn]:
                ranked_lists[srn].append(n)
        for i in range(len(ranked_lists[srn])):
            s = ranked_lists[srn][i]
            assigned_lists[srn][s] = max_points - i
    iteration, while_loop_changes, confidence = 0, [], 0
    while not converged(ranked_lists) or confidence < confidence_threshold:
        if converged(ranked_lists):
            confidence += 1
        else:
            confidence = 0
        if iteration == 0:
            p_r = random.sample(srns, k=len(srns))
        else:
            p_r = ranked_lists[canary_srn]
        for srn in srns:
            neighbors = [k[0] for k in global_neighbors_list[srn][t]]
            for n_srn in ranked_lists[srn]:
                for s in neighbors:
                    assigned_lists[srn][n_srn] += assigned_lists[s][n_srn]
            ranked_lists[srn] = sorted([n for n in assigned_lists[srn].keys()], key=assigned_lists[srn].get,
                                       reverse=True)
        ch = 0
        for _j in range(len(p_r)):
            if p_r[_j] != ranked_lists[canary_srn][_j]:
                ch += 1
        while_loop_changes.append(ch)
        iteration += 1
    print('DARPASC2MobilityAnalysis Alleys-of-Austin: Channel Access Order Convergence within {}s period for '
          'SRN={} - {}'.format(time_resolution, canary_srn, while_loop_changes))
    for srn in srns:
        a, b = prev_ranked_lists[srn], ranked_lists[srn]
        for _i in range(len(a)):
            if a[_i] != b[_i]:
                changes[srn][t] += 1
        average_changes[t] += changes[srn][t]
    average_changes[t] /= len(srns)
print('DARPASC2MobilityAnalysis Alleys-of-Austin: Average Channel Access Order Changes = [{}]'.format(
    [c for c in average_changes]))
# The mobility analysis ends here...