# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import numpy as np


class KrausOperator:
    def __init__(self, line, qubit, op_int, probability):
        self.line = line
        self.qubit = qubit
        self.op_int = op_int
        self.probability = probability
        
    def __eq__(self, other):
        return (self.line==other.line) and (self.qubit==other.qubit) and (self.op_int==other.op_int)
def XError(kraus_list, ko_line, qubit, probability):
    kraus_list.append(KrausOperator(ko_line, qubit, 1, probability))
    
def YError(kraus_list, ko_line, qubit, probability):
    kraus_list.append(KrausOperator(ko_line, qubit, 2, probability))
    
def ZError(kraus_list, ko_line, qubit, probability):
    kraus_list.append(KrausOperator(ko_line, qubit, 3, probability))

def Depolarization1(kraus_list, ko_line, qubit, probability):
    kraus_list.append(KrausOperator(ko_line, qubit, 1, probability/3))
    kraus_list.append(KrausOperator(ko_line, qubit, 2, probability/3))
    kraus_list.append(KrausOperator(ko_line, qubit, 3, probability/3))

def Depolarization2(kraus_list, ko_line, qubit1, qubit2, probability):
    kraus_list.append(KrausOperator(ko_line, (qubit1,qubit2), 1, probability/15))
    kraus_list.append(KrausOperator(ko_line, (qubit1,qubit2), 2, probability/15))
    kraus_list.append(KrausOperator(ko_line, (qubit1,qubit2), 3, probability/15))
    kraus_list.append(KrausOperator(ko_line, (qubit1,qubit2), 4, probability/15))
    kraus_list.append(KrausOperator(ko_line, (qubit1,qubit2), 5, probability/15))
    kraus_list.append(KrausOperator(ko_line, (qubit1,qubit2), 6, probability/15))
    kraus_list.append(KrausOperator(ko_line, (qubit1,qubit2), 7, probability/15))
    kraus_list.append(KrausOperator(ko_line, (qubit1,qubit2), 8, probability/15))
    kraus_list.append(KrausOperator(ko_line, (qubit1,qubit2), 9, probability/15))
    kraus_list.append(KrausOperator(ko_line, (qubit1,qubit2), 10, probability/15))
    kraus_list.append(KrausOperator(ko_line, (qubit1,qubit2), 11, probability/15))
    kraus_list.append(KrausOperator(ko_line, (qubit1,qubit2), 12, probability/15))
    kraus_list.append(KrausOperator(ko_line, (qubit1,qubit2), 13, probability/15))
    kraus_list.append(KrausOperator(ko_line, (qubit1,qubit2), 14, probability/15))
    kraus_list.append(KrausOperator(ko_line, (qubit1,qubit2), 15, probability/15))

def pts_proportional(kraus_list, samples, shots, keep_all_trajectories=False):
    trajectories, nkraus = [], len(kraus_list)
    for _ in range(samples):
        trajectory = []
        for kraus in kraus_list:
            if kraus.probability >= np.random.uniform():
                distinct = True
                for k in trajectory:
                    if (kraus.line==k.line) and (kraus.qubit==k.qubit):
                        distinct = False
                if distinct:
                    trajectory.append(kraus)
        trajectories.append(trajectory)
    
    if not keep_all_trajectories:
        ### remove redundant trajectories so that no unnecesary work is done
        unique_trajectories = []
        for trajectory in trajectories:
            same = True
            if unique_trajectories == []:
                same = False
            for t in unique_trajectories:
                if (t==[]) and (trajectory!=[]):
                    same = False
                else:
                    for trajectory_k, t_k in zip(trajectory, t):
                        if trajectory_k != t_k:
                            same = False
            if not same:
                unique_trajectories.append(trajectory)
        trajectories = unique_trajectories
        del unique_trajectories

    # [(serial_number, [(line_number,(qubit),op_int), (line_number,(qubit1,qubit2),op_int1)], shots), ....()]
    run_list = []
    for serial_number, trajectory in zip(range(len(trajectories)), trajectories):
        ops = []
        for kraus in trajectory:
            if type(kraus.qubit)==int:
                ops.append((kraus.line,(kraus.qubit,),kraus.op_int))
            elif type(kraus.qubit)==tuple:
                ops.append((kraus.line,kraus.qubit,kraus.op_int))
        #print((serial_number, ops, shots))
        run_list.append((serial_number, ops, shots))
        
    return run_list
