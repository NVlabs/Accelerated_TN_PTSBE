# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
### assumptions:
### gate number of noise operators is ko_line as long as every coherent gate has exactly one noise gate


# WAR: Monkeypatch an internal utility function such that the gate order will remain consistent with user provided circuit
from cuquantum.tensornet._internal import circuit_parser_utils_qiskit
from qiskit.circuit import Measure
    
def remove_measurements(circuit):
    for instruction in circuit.data:
        if isinstance(instruction.operation, Measure): 
            raise ValueError('the input circuit can not contain measurements')
    return circuit

circuit_parser_utils_qiskit.remove_measurements = remove_measurements

import qiskit
import qiskit.qpy
import cupy
import subprocess
import pickle
import sys
import os
import time
current_dir = os.path.dirname(os.path.abspath(__file__))
from utils_circuit import get_rdm_expr_gate_map, build_circuit_and_script, ket_contract, bra_contract, get_error_operand, random_gate_sample, get_noisy_shots_batched, plot_counts
from collections import Counter

dtype = 'complex128'


nqubits = 20
max_free_qubits = 10
ngates = 50
prob_range = [2e-2, 2e-1] ### somewhat realistic error range
proportion_two_qubits = 0.2
local_gates = False
nnoise_samples = 10
nshots = 100
ps = False
circuit_filename = 'random_circuit.qpy'
stim_script_filename = 'stim_script.stim'

### randomly sample the gates
gates_sampled = random_gate_sample(nqubits, ngates, prob_range, proportion_two_qubit=proportion_two_qubits, local_gates=local_gates)
build_circuit_and_script(nqubits, gates_sampled, circuit_filename, stim_script_filename) ### build and save the circuit and stim script
call0 = subprocess.call([sys.executable, 'stim_to_cudaq.py', stim_script_filename, os.path.join(current_dir, 'cudaq_output.py')])
cudaq_start_time = time.time()
call1 = subprocess.call(['python cudaq_output.py ' + str(nqubits) + ' ' + str(nshots)], shell=True)
cudaq_stop_time = time.time()
print('CUDA-Q Time Taken: ' + str(cudaq_stop_time - cudaq_start_time) + ' seconds')
print()
call2 = subprocess.call([sys.executable, 'stim_to_pts.py', stim_script_filename, os.path.join(current_dir, 'pts_output.py')])
call3 = subprocess.call(['python pts_output.py ' + str(nnoise_samples) + ' ' + str(nshots)], shell=True)

with open('error_sets.pickle', 'rb') as file:
    noise_samples = pickle.load(file)
start_time = time.time()
shots = get_noisy_shots_batched(circuit_filename, noise_samples, nqubits, max_free_qubits, nshots, dtype, proportional_sampling=ps)
stop_time = time.time()
recombined_shots, total_shots, per_error_unique_shots = Counter({}), 0, 0
for error in shots:
    per_error_unique_shots += len(error.values())
    recombined_shots = recombined_shots + Counter(error)
    total_shots += sum(error.values())
recombined_shots = dict(recombined_shots)
print('Total number of distinct PTSBE shots collected: ' + str(len(recombined_shots)))
print('Total number of distinct PTSBE pieces of labeled training data collected: ' + str(per_error_unique_shots))
print('Total number of overall PTSBE shots collected: ' + str(total_shots))
print('PTSBE Time Taken: ' + str(stop_time - start_time) + ' seconds')
