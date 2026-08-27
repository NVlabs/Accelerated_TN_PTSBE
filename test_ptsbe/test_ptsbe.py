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

import subprocess
import pickle

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
from utils_circuit import build_circuit_and_script, random_gate_sample, plot_counts
from utils_noisy_shots import get_noisy_shots2


nqubits = 7
first_fixed = 5 ### first qubit to fix, fixed qubits must be continuous and final qubits with current shot recombination process
fixed_qubits = tuple(i for i in range(first_fixed, nqubits))
print('Defining fixed qubits as ' + str(fixed_qubits) + ' according to shot recombination specifications')
fixed_qubits = tuple(nqubits-1-q for q in fixed_qubits[::-1])
print('Reversing as ' + str(fixed_qubits) + ' because of Qiskit ordering')
ngates = 160
prob_range = [2e-2, 2e-1] ### somewhat realistic error range
#prob_range = [0.99999, 0.999999] ### make them errors almost certainly occur
#prob_range = [0.49, 0.51] ### probability range with greatest error set variance
#prob_range = [0.000001, 0.0000001] ### make them errors almost certainly not occur
nnoise_samples = 10
nshots = 1000
circuit_filename = 'random_circuit.qpy'
stim_script_filename = 'stim_script.stim'

gates_sampled = random_gate_sample(nqubits, ngates, prob_range) ### randomly sample the gates
build_circuit_and_script(nqubits, gates_sampled, circuit_filename, stim_script_filename) ### build and save the circuit and stim script
call1 = subprocess.call([sys.executable, 'stim_to_pts.py', stim_script_filename, os.path.join(current_dir, 'pts_output.py')])
call2 = subprocess.call([sys.executable, 'stim_to_be.py', stim_script_filename, os.path.join(current_dir, 'be_output.py'), str(nqubits)])
call3 = subprocess.call(['python pts_output.py ' + str(nnoise_samples) + ' ' + str(nshots)], shell=True)
call4 = subprocess.call(['python be_output.py ' + str(nqubits)], shell=True)
call5 = subprocess.call(['python plot_cudaq.py ' + str(nqubits)], shell=True)
with open('error_sets.pickle', 'rb') as file:
    noise_samples = pickle.load(file)
shots = get_noisy_shots2(circuit_filename, noise_samples, fixed_qubits, nshots)
recombined_shots = []
for error in shots:
    print('Error')
    print(error)
    for bitstring in error:
        recombined_shots += bitstring[1]
print('Total number of shots collected: ' + str(len(recombined_shots)))
plot_counts(recombined_shots, 'qiskit_histogram.png')

from utils_verification import load_cudaq_shots, verify_distributions

cudaq_shots = load_cudaq_shots(directory='shot_sets', nqubits=nqubits)
passed, tvd = verify_distributions(recombined_shots, cudaq_shots, threshold=0.15)
