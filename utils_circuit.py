# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import gc
import qiskit
import qiskit.qpy
import cupy
import time
from collections import Counter
import pandas
import matplotlib.pyplot as plt
from cuquantum.tensornet import contract, CircuitToEinsum, contract_path, Network
import itertools
import numpy as np


def estimate_contractions_batched_qubitShotFlex(nqubits, max_free_qubits, num_samples, nshots,
                                                proportional_sampling=False, verbose=True,
                                                shots_per_batch=None, qubits_per_batch=None):
    """
    Same as estimate_contractions_batched but with list-based batch layout.
    qubits_per_batch: list of qubit counts per batch (must sum to nqubits). If None, use uniform from max_free_qubits.
    shots_per_batch: list of shot counts per batch (length = num_batches). Used only when proportional_sampling is False.
    """
    if qubits_per_batch is not None:
        qubits_per_batch = list(qubits_per_batch)
        if sum(qubits_per_batch) != nqubits:
            raise ValueError(f"qubits_per_batch must sum to nqubits={nqubits}, got sum={sum(qubits_per_batch)}")
        ranges = [0] + np.cumsum(qubits_per_batch).tolist()
    else:
        ranges = list(range(0, nqubits, max_free_qubits))
        if ranges[-1] != nqubits:
            ranges.append(nqubits)
    num_batches = len(ranges) - 1
    if shots_per_batch is not None and not proportional_sampling:
        base_shots_per_batch = list(shots_per_batch)
        if len(base_shots_per_batch) != num_batches:
            raise ValueError(f"shots_per_batch length must be num_batches={num_batches}, got {len(base_shots_per_batch)}")
    else:
        base_shots_per_batch = [nshots] * num_batches

    min_estimate = num_samples * num_batches
    max_per_sample = 0
    qubits_traced = 0
    max_unique_at_level = base_shots_per_batch[0]
    for j in range(num_batches):
        batch_size = ranges[j + 1] - ranges[j]
        shots_j = base_shots_per_batch[j]
        if j == 0:
            max_per_sample += 1
            max_unique_at_level = min(shots_j, 2**batch_size)
        else:
            max_per_sample += max_unique_at_level
            if proportional_sampling:
                max_unique_at_level = min(nshots, 2**(qubits_traced + batch_size))
            else:
                max_unique_per_branch = min(shots_j, 2**batch_size)
                max_unique_at_level = min(
                    max_unique_at_level * max_unique_per_branch,
                    2**(qubits_traced + batch_size)
                )
        qubits_traced += batch_size
    max_estimate = num_samples * max_per_sample

    expected_per_sample = 0
    qubits_traced = 0
    expected_unique_at_level = base_shots_per_batch[0]
    for j in range(num_batches):
        batch_size = ranges[j + 1] - ranges[j]
        shots_j = base_shots_per_batch[j]
        if j == 0:
            expected_per_sample += 1
            output_dim = 2**batch_size
            if shots_j >= output_dim:
                expected_unique_at_level = output_dim
            else:
                expected_unique_at_level = int(output_dim * (1 - np.exp(-shots_j / output_dim)))
                expected_unique_at_level = max(1, min(shots_j, expected_unique_at_level))
        else:
            if proportional_sampling:
                expected_per_sample += expected_unique_at_level
                avg_shots_per_branch = nshots / expected_unique_at_level
                output_dim = 2**batch_size
                if avg_shots_per_branch >= output_dim:
                    unique_per_branch = output_dim
                else:
                    unique_per_branch = int(output_dim * (1 - np.exp(-avg_shots_per_branch / output_dim)))
                    unique_per_branch = max(1, unique_per_branch)
                expected_unique_at_level = min(
                    expected_unique_at_level * unique_per_branch,
                    2**(qubits_traced + batch_size), nshots
                )
            else:
                expected_per_sample += expected_unique_at_level
                output_dim = 2**batch_size
                if shots_j >= output_dim:
                    unique_per_branch = output_dim
                else:
                    unique_per_branch = int(output_dim * (1 - np.exp(-shots_j / output_dim)))
                    unique_per_branch = max(1, unique_per_branch)
                expected_unique_at_level = min(
                    expected_unique_at_level * unique_per_branch,
                    2**(qubits_traced + batch_size)
                )
        qubits_traced += batch_size
    expected_estimate = num_samples * expected_per_sample

    if verbose:
        qpb = [ranges[j+1]-ranges[j] for j in range(num_batches)]
        print(f"  .............................................................")
        print(f"** Contraction Estimate (Batched, qubitShotFlex) **")
        print(f"  Qubits: {nqubits}, max_free: {max_free_qubits}")
        print(f"  Batches: {num_batches}, ranges: {ranges}, qubits_per_batch: {qpb}")
        print(f"  Samples: {num_samples}, base_shots_per_batch: {base_shots_per_batch}, proportional_sampling: {proportional_sampling}")
        print(f"  Minimum contractions:  {min_estimate:,} in total for {num_samples} samples")
        print(f"  Expected contractions: {expected_estimate:,} in total for {num_samples} samples")
        print(f"  Maximum contractions:  {max_estimate:,} in total for {num_samples} samples")
        print(f"  .............................................................")
    return {'min': min_estimate, 'max': max_estimate, 'expected': expected_estimate, 'num_batches': num_batches, 'ranges': ranges}


def estimate_contractions_batched(nqubits, max_free_qubits, num_samples, nshots, 
                                   proportional_sampling=False, verbose=True,
                                   final_nshots=None):
    """
    Estimate the number of contract calls for the batched approach.
    
    The batched approach divides qubits into batches of max_free_qubits.
    For batch 0: 1 contraction per sample
    For batch j>0: depends on # unique bitstrings from previous batch
    """
    ranges = list(range(0, nqubits, max_free_qubits))
    if ranges[-1] != nqubits:
        ranges.append(nqubits)
    num_batches = len(ranges) - 1
    
    # Minimum: 1 contraction per batch per sample (if all samples collapse to same bitstring)
    min_estimate = num_samples * num_batches
    
    # Maximum: worst case where each shot gives unique bitstring
    max_per_sample = 0
    qubits_traced = 0
    max_unique_at_level = nshots  # Track max unique bitstrings from previous batches
    
    for j in range(num_batches):
        batch_size = ranges[j+1] - ranges[j]
        if j == 0:
            max_per_sample += 1
            # After batch 0: max unique is min(nshots, output_dim)
            max_unique_at_level = min(nshots, 2**batch_size)
        else:
            # Contractions needed = number of unique bitstrings from previous batches
            max_per_sample += max_unique_at_level
            
            if proportional_sampling:
                # With proportional sampling: total shots stay at nshots
                # Max unique stays capped at nshots
                max_unique_at_level = min(nshots, 2**(qubits_traced + batch_size))
            else:
                # Without proportional sampling: each branch samples nshots independently
                # Max unique can grow multiplicatively
                max_unique_per_branch = min(nshots, 2**batch_size)
                max_unique_at_level = min(
                    max_unique_at_level * max_unique_per_branch,
                    2**(qubits_traced + batch_size)  # Cap by output space
                )
        qubits_traced += batch_size
    max_estimate = num_samples * max_per_sample
    
    # Expected estimate depends on proportional_sampling
    expected_per_sample = 0
    qubits_traced = 0
    
    # Track expected unique bitstrings at each level for proportional case
    expected_unique_at_level = nshots  # Start with nshots
    
    for j in range(num_batches):
        batch_size = ranges[j+1] - ranges[j]
        if j == 0:
            expected_per_sample += 1
            # Estimate unique bitstrings after first batch
            output_dim = 2**batch_size
            # Birthday paradox: expected unique ~ output_dim * (1 - exp(-nshots/output_dim))
            if nshots >= output_dim:
                expected_unique_at_level = output_dim
            else:
                expected_unique_at_level = int(output_dim * (1 - np.exp(-nshots / output_dim)))
                expected_unique_at_level = max(1, min(nshots, expected_unique_at_level))
        else:
            if proportional_sampling:
                # With proportional sampling: each unique bitstring from prev batch
                # contributes 1 contraction, but the sub-sampling produces fewer unique
                # bitstrings because shots are split proportionally
                expected_per_sample += expected_unique_at_level
                
                # For next iteration: shots are distributed across branches
                # On average, each branch gets nshots / expected_unique_at_level shots
                avg_shots_per_branch = nshots / expected_unique_at_level
                output_dim = 2**batch_size
                
                # Each branch produces ~birthday_unique(avg_shots, output_dim) bitstrings
                # Total unique ≈ expected_unique_at_level * unique_per_branch
                # But capped by total output space
                if avg_shots_per_branch >= output_dim:
                    unique_per_branch = output_dim
                else:
                    unique_per_branch = int(output_dim * (1 - np.exp(-avg_shots_per_branch / output_dim)))
                    unique_per_branch = max(1, unique_per_branch)
                
                expected_unique_at_level = min(
                    expected_unique_at_level * unique_per_branch,
                    2**(qubits_traced + batch_size),
                    nshots  # Can't have more unique than total shots
                )
            else:
                # Without proportional sampling: each unique bitstring uses full nshots
                # This typically produces MORE unique bitstrings per branch
                expected_per_sample += expected_unique_at_level
                
                output_dim = 2**batch_size
                if nshots >= output_dim:
                    unique_per_branch = output_dim
                else:
                    unique_per_branch = int(output_dim * (1 - np.exp(-nshots / output_dim)))
                    unique_per_branch = max(1, unique_per_branch)
                
                # Each branch independently samples nshots, so total unique can grow multiplicatively
                # NOTE: No nshots cap here - each branch samples independently!
                expected_unique_at_level = min(
                    expected_unique_at_level * unique_per_branch,
                    2**(qubits_traced + batch_size)  # Only cap by output space size
                )
        
        qubits_traced += batch_size
    
    expected_estimate = num_samples * expected_per_sample
    
    if verbose:
        print(f"  .............................................................")
        print(f"** Contraction Estimate (Batched) **")
        print(f"  Qubits: {nqubits}, max_free: {max_free_qubits}")
        print(f"  Batches: {num_batches}, ranges: {ranges}")
        print(f"  Samples: {num_samples}, shots: {nshots}" + (f", final_nshots: {final_nshots}" if final_nshots else ""))
        print(f"  proportional_sampling: {proportional_sampling}") 
        print(f"  Minimum contractions:  {min_estimate:,} in total for {num_samples} samples")
        print(f"  Expected contractions: {expected_estimate:,} in total for {num_samples} samples")
        print(f"  Maximum contractions:  {max_estimate:,} in total for {num_samples} samples")
        print(f"  .............................................................")
    
    return {
        'min': min_estimate,
        'max': max_estimate,
        'expected': expected_estimate,
        'num_batches': num_batches,
        'ranges': ranges
    }


def estimate_contractions_nonbatched(num_samples, num_fixed_qubits, verbose=True):
    """
    Estimate the number of contract calls for the non-batched approach.
    
    The non-batched approach iterates over all 2^num_fixed_qubits combinations
    for each noise sample.
    """
    num_fixed_combinations = 2**num_fixed_qubits
    total = num_samples * num_fixed_combinations
    
    if verbose:
        print(f"=== Contraction Estimate (Non-Batched) ===")
        print(f"  Samples: {num_samples}")
        print(f"  Fixed qubits: {num_fixed_qubits}")
        print(f"  Fixed combinations: {num_fixed_combinations}")
        print(f"  ----------------------------------------")
        print(f"  Total contractions: {total:,}")
        print(f"  ==========================================")
    
    return {
        'total': total,
        'num_fixed_combinations': num_fixed_combinations
    }


def random_gate_sample(nqubits, ngates, prob_range, proportion_two_qubit=0.2, local_gates=True):
    one_qubit_gate_list = ['x', 'y', 'z', 'h', 't', 'rx'] ### Stim+T gates only for now, add others later
    two_qubit_gate_list = ['cx', 'cy', 'cz', 'ch', 'crx']
    one_qubit_noise_list = ['x_error', 'y_error', 'z_error'] ### Stim errors only for now, add others later
    two_qubit_noise_list = ['depolarize2_error'] ### Stim errors only for now, add others later
    gates_sampled = []

    two_qubit_counter = 0
    for gate_number in range(ngates):
        two_qubit = proportion_two_qubit > cupy.random.uniform()
        if two_qubit:
            gate, qubit = two_qubit_gate_list[cupy.random.randint(0, len(two_qubit_gate_list)).item()], cupy.random.randint(0, nqubits).item()
            two_qubit_counter += 1
        else:
            gate, qubit = one_qubit_gate_list[cupy.random.randint(0, len(one_qubit_gate_list)).item()], cupy.random.randint(0, nqubits).item()

        if (gate=='cx') or (gate=='cy') or (gate=='cz') or (gate=='ch') or (gate=='crx'):
            if local_gates:
                if qubit == (nqubits-1):
                    qubit = (qubit-1, qubit)
                else:
                    qubit = (qubit, qubit+1)
            else:
                eligible_qubits = [i for i in range(nqubits) if i != qubit]
                qubit = (qubit, cupy.random.choice(eligible_qubits, size=1).item())
        else:
            qubit = (qubit,)
        if (gate=='rx') or (gate=='crx'):
            param = cupy.random.uniform(0, 2*3.14).item()
        else:
            param = []
        gates_sampled.append((gate, qubit, param))

        if len(qubit) == 1:
            noise = one_qubit_noise_list[cupy.random.randint(0, len(one_qubit_noise_list)).item()]
        elif len(qubit) == 2:
            noise = two_qubit_noise_list[cupy.random.randint(0, len(two_qubit_noise_list)).item()]
        prob = cupy.random.uniform(prob_range[0], prob_range[1]).item()
        gates_sampled.append((noise, qubit, prob)) ### gate number is the same as ko_line as long as one error per each gate
    print('ASSUMING ONE ERROR PER COHERENT GATE! EXTEND THE CODE IF YOU DON\'T LIKE IT') ### gate number is ko_line

    return gates_sampled


def get_rdm_expr_gate_map(circuit, expr, fixed=None, return_projection_indices=False):
    """
    Return a dictionary mapping the gate index to the ket and bra operand indices.
    Args:
        circuit: A :class:`qiskit.QuantumCircuit` object.
        expr: The Einstein summation expression.
        fixed: A dictionary mapping the fixed qubits to the corresponding fixed states.
    Returns:
        A dictionary mapping the gate index to the ket and bra operand indices.
    """
    # note that the operands order can be described by the following:
    #   ket_qubit_tensors (vaccum), ket_gate_tensors (gates in forward order), ket_projection_tensors(dependent on fixed), 
    #   bra_projection_tensors (dependent on fixed), bra_gate_tensors (gates in inverse order), bra_qubit_tensors (vaccum)

    assert circuit.global_phase == 0 # if global phase is not 0, the assumption above is invalid
    num_qubits = len(circuit.qubits)
    num_input_operands = expr.count(',') + 1
    num_gates = len(circuit.data)
    num_projection_tensors = 0 if fixed is None else 2 * len(fixed)

    if num_input_operands != 2 * (num_gates + num_qubits) + num_projection_tensors:
        raise ValueError(f"Number of input operands {num_input_operands} does not match the number of gates {num_gates} and qubits {num_qubits} and projection tensors {num_projection_tensors}")

    gate_map = {}
    for i in range(num_gates):
        gate_map[f"ket_gate_{i}"] = i + num_qubits
        gate_map[f"bra_gate_{i}"] = num_input_operands - i - num_qubits - 1

    if return_projection_indices:
        return gate_map, [i for i in range(num_gates+num_qubits, num_gates+num_qubits+num_projection_tensors)]
    else:
        return gate_map
    if return_projection_indices:
        return gate_map, [i for i in range(num_gates+num_qubits, num_gates+num_qubits+num_projection_tensors)]
    else:
        return gate_map


print('SWAPPING GATE ORDER WHEN CONSTRUCTING CIRCUIT BECAUSE QISKIT')

def map_to_cudaq(i, n): ### swapped order because of qiskit
        return (n-1)-i

def build_circuit_and_script(nqubits, gates_sampled, circuit_filename, stim_script_filename):
    circuit, stim_script = qiskit.QuantumCircuit(nqubits), []

    for gate in gates_sampled:
        if gate[0] == 'x':
            line = 'X ' + str(gate[1][0])
            stim_script.append(line)
            line = 'circuit.'+gate[0]+'('+str(map_to_cudaq(gate[1][0], nqubits))+')' ### swapped order because of qiskit
            exec(line)
        elif gate[0] == 'y':
            line = 'Y ' + str(gate[1][0])
            stim_script.append(line)
            line = 'circuit.'+gate[0]+'('+str(map_to_cudaq(gate[1][0], nqubits))+')'
            exec(line)
        elif gate[0] == 'z':
            line = 'Z ' + str(gate[1][0])
            stim_script.append(line)
            line = 'circuit.'+gate[0]+'('+str(map_to_cudaq(gate[1][0], nqubits))+')'
            exec(line)
        elif gate[0] == 'h':
            line = 'H ' + str(gate[1][0])
            stim_script.append(line)
            line = 'circuit.'+gate[0]+'('+str(map_to_cudaq(gate[1][0], nqubits))+')'
            exec(line)
        elif gate[0] == 't':
            line = 'T ' + str(gate[1][0])
            stim_script.append(line)
            line = 'circuit.'+gate[0]+'('+str(map_to_cudaq(gate[1][0], nqubits))+')'
            exec(line)
        elif gate[0] == 'rx':
            line = 'RX(' + str(gate[2]) + ') ' + str(gate[1][0])
            stim_script.append(line)
            line = 'circuit.'+gate[0]+'('+str(gate[2])+', '+str(map_to_cudaq(gate[1][0], nqubits))+')'
            exec(line)
        elif gate[0] == 'cx':
            line = 'CX ' + str(gate[1][0]) + ' ' + str(gate[1][1])
            stim_script.append(line)
            line = 'circuit.'+gate[0]+'('+str(map_to_cudaq(gate[1][0], nqubits))+',' + str(map_to_cudaq(gate[1][1], nqubits)) + ')'
            exec(line)
        elif gate[0] == 'cy':
            line = 'CY ' + str(gate[1][0]) + ' ' + str(gate[1][1])
            stim_script.append(line)
            line = 'circuit.'+gate[0]+'('+str(map_to_cudaq(gate[1][0], nqubits))+',' + str(map_to_cudaq(gate[1][1], nqubits)) + ')'
            exec(line)
        elif gate[0] == 'cz':
            line = 'CZ ' + str(gate[1][0]) + ' ' + str(gate[1][1])
            stim_script.append(line)
            line = 'circuit.'+gate[0]+'('+str(map_to_cudaq(gate[1][0], nqubits))+',' + str(map_to_cudaq(gate[1][1], nqubits)) + ')'
            exec(line)
        elif gate[0] == 'ch':
            line = 'CH ' + str(gate[1][0]) + ' ' + str(gate[1][1])
            stim_script.append(line)
            line = 'circuit.'+gate[0]+'('+str(map_to_cudaq(gate[1][0], nqubits))+',' + str(map_to_cudaq(gate[1][1], nqubits)) + ')'
            exec(line)
        elif gate[0] == 'crx':
            line = 'CRX(' + str(gate[2]) + ') ' + str(gate[1][0]) + ' ' + str(gate[1][1])
            stim_script.append(line)
            line = 'circuit.'+gate[0]+'('+str(gate[2])+', '+str(map_to_cudaq(gate[1][0], nqubits))+',' + str(map_to_cudaq(gate[1][1], nqubits)) + ')'
            exec(line)
        elif gate[0] == 'x_error':
            line = 'X_ERROR(' + str(gate[2]) + ')' + ' ' + str(gate[1][0])
            stim_script.append(line)
        elif gate[0] == 'y_error':
            line = 'Y_ERROR(' + str(gate[2]) + ')' + ' ' + str(gate[1][0])
            stim_script.append(line)
        elif gate[0] == 'z_error':
            line = 'Z_ERROR(' + str(gate[2]) + ')' + ' ' + str(gate[1][0])
            stim_script.append(line)
        elif gate[0] == 'depolarize2_error':
            line = 'DEPOLARIZE2(' + str(gate[2]) + ')' + ' ' + str(gate[1][0]) + ' ' + str(gate[1][1])
            stim_script.append(line)
        else:
            print('UNRECOGNIZED GATE!')

    stim_script = "\n".join(stim_script)

    with open(circuit_filename, "wb") as file:
        qiskit.qpy.dump(circuit, file)
    with open(stim_script_filename, 'w') as f:
        f.write(stim_script)


def ket_contract(operand, error):
    if error.shape == (2,2):
        #return cupy.einsum('ab,bc->ac', operand, error)
        return cupy.einsum('ab,bc->ac', error, operand)
    elif error.shape == (4,4):
        return cupy.einsum('abcd,efab->efcd', operand, error.reshape(2,2,2,2))
    else:
        print('The ket_contract function cannot contract that shape')

def bra_contract(operand, error):
    if error.shape == (2,2):
        #return cupy.einsum('ab,bc->ac', error.T.conj(), operand) ### I think Hermitian conjugate is right here, but should nosetest
        return cupy.einsum('ab,bc->ac', operand, error.T.conj())
    elif error.shape == (4,4):
        return cupy.einsum('abcd,efab->efcd', error.T.conj().reshape(2,2,2,2), operand)
    else:
        print('The bra_contract function cannot contract that shape')


print('SWAPPING NOISE ORDER FOR TWO-QUBIT NOISE GATES BECAUSE QISKIT')

x = cupy.array([[0, 1], [1, 0]], dtype='complex128')
y = cupy.array([[0, -1j], [1j, 0]], dtype='complex128')
z = cupy.array([[1, 0], [0, -1]], dtype='complex128')
iden = cupy.array([[1, 0], [0, 1]], dtype='complex128')

def get_error_operand(noise_qubit, noise_int, dtype):
    if len(noise_qubit) == 1:
        if noise_int == 1:
            return x.astype(dtype)
        elif noise_int == 2:
            return y.astype(dtype)
        elif noise_int == 3:
            return z.astype(dtype)
        else:
            print('Invalid noise operator value')
    elif len(noise_qubit) == 2:
        if noise_int == 1:
            #return cupy.kron(iden, x).astype(dtype) ### swapped order because Qiskit
            return cupy.kron(x, iden).astype(dtype)
        elif noise_int == 2:
            #return cupy.kron(iden, y).astype(dtype)
            return cupy.kron(y, iden).astype(dtype)
        elif noise_int == 3:
            #return cupy.kron(iden, z).astype(dtype)
            return cupy.kron(z, iden).astype(dtype)
        elif noise_int == 4:
            #return cupy.kron(x, iden).astype(dtype)
            return cupy.kron(iden, x).astype(dtype)
        elif noise_int == 5:
            return cupy.kron(x, x).astype(dtype)
        elif noise_int == 6:
            #return cupy.kron(x, y).astype(dtype)
            return cupy.kron(y, x).astype(dtype)
        elif noise_int == 7:
            #return cupy.kron(x, z).astype(dtype)
            return cupy.kron(z, x).astype(dtype)
        elif noise_int == 8:
            #return cupy.kron(y, iden).astype(dtype)
            return cupy.kron(iden, y).astype(dtype)
        elif noise_int == 9:
            #return cupy.kron(y, x).astype(dtype)
            return cupy.kron(x, y).astype(dtype)
        elif noise_int == 10:
            return cupy.kron(y, y).astype(dtype)
        elif noise_int == 11:
            #return cupy.kron(y, z).astype(dtype)
            return cupy.kron(z, y).astype(dtype)
        elif noise_int == 12:
            #return cupy.kron(z, iden).astype(dtype)
            return cupy.kron(iden, z).astype(dtype)
        elif noise_int == 13:
            #return cupy.kron(z, x).astype(dtype)
            return cupy.kron(x, z).astype(dtype)
        elif noise_int == 14:
            #return cupy.kron(z, y).astype(dtype)
            return cupy.kron(y, z).astype(dtype)
        elif noise_int == 15:
            return cupy.kron(z, z).astype(dtype)
    else:
        print('Invalid noisy qubit operator number')


def get_noisy_shots(circuit_filename, noise_samples, nqubits, fixed_qubits, nshots, dtype):
    with open(circuit_filename, "rb") as handle:
        circuit = qiskit.qpy.load(handle)[0]

    converter = CircuitToEinsum(circuit, dtype=dtype, backend='cupy', options={'check_diagonal': False, 'decompose_gates': False})
    qubits = circuit.qubits
    where = tuple([qubits[i] for i in range(len(qubits)) if i not in fixed_qubits]) ### all of the non-fixed qubits
    output_dim = 2**len(where)
    fixed_qubit_inds = cupy.arange(len(fixed_qubits))
    free_qubits = [i for i in range(len(qubits)) if i not in fixed_qubits]
    free_qubit_inds, nfree = cupy.arange(len(free_qubits)), len(free_qubits)

    ### Build a list of dictionaries where each dictionary is a different bitstring assignment for the fixed qubits
    fixed_vals = [''.join(p) for p in itertools.product('01', repeat=len(fixed_qubits))] ### possible bitstring values for fixed qubits
    fixed_list = []
    for p in fixed_vals:
        fixed = {}
        for index in range(len(fixed_qubits)):
            fixed[qubits[fixed_qubits[index]]] = p[index]
        fixed_list.append(fixed)

    ### Get all exprs and operands. Each noise sample will iterate through all of them
    expr_list, operands_list = [], []
    if fixed_list == {}: ### if no fixed, then just get the one set of exprs and operands
        expr, operands = converter.reduced_density_matrix(where, fixed=fixed, lightcone=False)
        expr_list.append(expr)
        operands_list.append(operands)
    for fixed in fixed_list:
        expr, operands = converter.reduced_density_matrix(where, fixed=fixed, lightcone=False)
        expr_list.append(expr)
        operands_list.append(operands) ### rather than storing all operands, in future versions, change only fixed projectors
    gate_map = get_rdm_expr_gate_map(circuit, expr_list[0], fixed_list[0])
    path, info = contract_path(expr_list[0], *operands_list[0]) ### assuming same regardless of fixed value bitstring, correct otherwise

    noisy_sample_shots = []
    for index in range(len(noise_samples)): ### each of these is for a different noise set (sampled noise)

        error_sample_shots = []
        for expr, operands, fixed_val in zip(expr_list, operands_list, fixed_vals): ### each of these is for a different fixed bitstring
            fixed_val = fixed_val[::-1] ### reverse fixed values because of Qiskit ordering
            temp_operands = [operand.copy() for operand in operands] ### must be DEEP copy as you will change it for each error combination
            noise = noise_samples[index]
            _, errors, shots = noise
            for error in errors:
                gate_i, noise_qubits, noise_int = error
                error_operand = get_error_operand(noise_qubits, noise_int, dtype)
                ket_index, bra_index = gate_map['ket_gate_'+str(gate_i)], gate_map['bra_gate_'+str(gate_i)]
                temp_operands[ket_index] = ket_contract(temp_operands[ket_index], error_operand)
                temp_operands[bra_index] = bra_contract(temp_operands[bra_index], error_operand)
            out = contract(expr, *temp_operands, optimize={'path': path, 'slicing': info.slices}).reshape(output_dim, output_dim)
            out = cupy.clip(cupy.diagonal(out).real, 0, None)
            marginal_prob = cupy.sum(out)
            if (0 < marginal_prob):
                noisy_shots = cupy.random.choice(cupy.arange(len(out)), size=int(np.ceil(shots*marginal_prob)), p=out/marginal_prob)
                noisy_shots = noisy_shots.get() ### has to be done on CPU/Numpy because CuPy function malfunctioning, bug reported
                vectorized_binary_repr = np.vectorize(np.binary_repr)
                noisy_shots = vectorized_binary_repr(noisy_shots, width=nqubits).tolist()
                noisy_shots = [shots[::-1] for shots in noisy_shots] ### flipped shot order because classical convention, not Qiskit
                noisy_shots = [shots[0:nfree] + fixed_val for shots in noisy_shots] ### remove the bits for the fixed qubits
                error_sample_shots.append((marginal_prob, noisy_shots))

        noisy_sample_shots.append(error_sample_shots)


    return noisy_sample_shots


def sample_batched(tn, temp_operands, path, info, ranges_upper, local_shots, output_dim, prev_bits=None, full_rdm=True,
                   unique_sampling=False, is_final_batch=False, take_all_final=False, prev_weight=1.0,
                   return_joint_probability=False):
                """
                Sample bitstrings from the RDM diagonal.

                Args:
                    unique_sampling: If True, use replace=False to guarantee unique samples.
                    is_final_batch: If True, this is the last batch in the chain
                    take_all_final: If True AND unique_sampling=True AND is_final_batch, take ALL nonzero
                                    entries instead of sampling. Only valid when unique_sampling=True.
                    prev_weight: Cumulative weight from previous batches (only used when return_joint_probability=True).
                    return_joint_probability: If True and unique_sampling=True, compute and return joint probabilities
                                    across batches. If False, no weight computation or storage.

                Returns:
                    (shots_dict, weights_dict_or_None):
                    - shots_dict: {bitstring: multiplicity}. When unique_sampling=False, multiplicity is count.
                      When unique_sampling=True, multiplicity is 1 per bitstring.
                    - weights_dict_or_None: When unique_sampling=True and return_joint_probability=True,
                      dict of {bitstring: joint_probability}. Otherwise None.
                """
                nrdm = int(np.log2(output_dim))
                temp_operands = [cupy.ascontiguousarray(op) for op in temp_operands]
                tn.reset_operands(*temp_operands)
                if full_rdm:
                    # RDM format: output is d^2n, need to extract diagonal
                    out = tn.contract(release_workspace=True).reshape(output_dim, output_dim)
                    out = cupy.clip(cupy.diagonal(out).real, 0, None)
                else:
                    # Marginal probability format: output is already d^n (diagonal)
                    out = tn.contract(release_workspace=True).reshape(output_dim)
                    out = cupy.clip(out.real, 0, None)
                marginal_prob = cupy.sum(out)

                # Determine sampling strategy
                if unique_sampling and is_final_batch and take_all_final:
                    # Take-all-final: no random choice, do everything on GPU then transfer only small arrays
                    out_norm = out / marginal_prob
                    nonzero_mask = out_norm > 1e-15
                    noisy_shots_arr_gpu = cupy.where(nonzero_mask)[0]
                    if return_joint_probability:
                        weights = out_norm[nonzero_mask].get()
                    else:
                        weights = None
                    noisy_shots_arr = noisy_shots_arr_gpu.get()
                    compute_weights = return_joint_probability
                else:
                    marginal_prob_cpu = float(marginal_prob.get())
                    out_cpu = out.get() / marginal_prob_cpu
                    if unique_sampling:
                        num_nonzero = np.sum(out_cpu > 1e-15)
                        sample_size = min(int(np.ceil(local_shots)), num_nonzero)
                        if sample_size > 0:
                            noisy_shots_arr = np.random.choice(
                                len(out_cpu),
                                size=sample_size,
                                p=out_cpu,
                                replace=False
                            )
                            if return_joint_probability:
                                weights = out_cpu[noisy_shots_arr]
                        else:
                            noisy_shots_arr = np.array([], dtype=int)
                            if return_joint_probability:
                                weights = np.array([])
                        compute_weights = return_joint_probability
                    else:
                        noisy_shots_arr = np.random.choice(
                            len(out_cpu),
                            size=int(np.ceil(local_shots)),
                            p=out_cpu
                        )
                        weights = None
                        compute_weights = False

                vectorized_binary_repr = np.vectorize(np.binary_repr)
                noisy_shots = vectorized_binary_repr(noisy_shots_arr, width=nrdm).tolist()
                noisy_shots = [shots[::-1] for shots in noisy_shots]  # flipped shot order: classical convention, not Qiskit

                if unique_sampling:
                    # Multiplicity 1 per bitstring
                    if prev_bits:
                        shots_dict = {bitstring[0:ranges_upper] + prev_bits: 1 for bitstring in noisy_shots}
                    else:
                        shots_dict = {bitstring[0:ranges_upper]: 1 for bitstring in noisy_shots}
                    if compute_weights:
                        if prev_bits:
                            weights_dict = {bitstring[0:ranges_upper] + prev_bits: prev_weight * w
                                           for bitstring, w in zip(noisy_shots, weights)}
                        else:
                            weights_dict = {bitstring[0:ranges_upper]: prev_weight * w
                                           for bitstring, w in zip(noisy_shots, weights)}
                        return shots_dict, weights_dict
                    return shots_dict, None
                else:
                    # Original behavior: {bitstring: count}
                    shots_dict = Counter(noisy_shots)
                    if prev_bits:
                        shots_dict = {bitstring[0:ranges_upper] + prev_bits: shots_dict[bitstring] for bitstring in shots_dict}
                    else:
                        shots_dict = {bitstring[0:ranges_upper]: shots_dict[bitstring] for bitstring in shots_dict}
                    return shots_dict, None


proj0 = cupy.array([1.+0.j, 0.+0.j])
proj1 = cupy.array([0.+0.j, 1.+0.j])


def get_noisy_shots_batched(
    circuit_filename, 
    noise_samples, 
    nqubits, 
    max_free_qubits, 
    nshots, 
    dtype, 
    proportional_sampling=False,
    full_rdm=False,
    comm=None,
    rank=0,
    enable_profiling=False,
    verbose_level=0,
    colors=None,
    max_contractions=None,
    max_opt_cost=1e15,
    num_hyper_samples=None,
    lightcone=False,
    dumpnet_prefix=None,
    final_nshots=None,
    count_degenerate_shots=True,
    unique_sampling=False,
    take_all_final=False,
    return_joint_probability=False
):
    def sync_gpu():
        cupy.cuda.Device().synchronize()
    
    if colors is None:
        colors = {
            'COLOR1': '\033[1;92m',  
            'COLOR2': '\033[1;91m',  
            'RESET': '\033[0m'
        }
    
    COLOR1 = colors.get('COLOR1', '')
    COLOR2 = colors.get('COLOR2', '')
    RESET = colors.get('RESET', '')
    
    # Lightcone simplification is incompatible with noise injection because it removes
    # gates from the tensor network, breaking the gate_map used for error application
    if lightcone and len(noise_samples) > 0:
        if rank == 0:
            print(f"{COLOR2}[WARNING] Lightcone simplification is incompatible with noise injection.{RESET}", flush=True)
            print(f"{COLOR2}[WARNING] Disabling lightcone for this run.{RESET}", flush=True)
        lightcone = True
    
    # final_nshots only applies when proportional_sampling is False
    if final_nshots is not None and proportional_sampling:
        if rank == 0:
            print(f"{COLOR2}[WARNING] final_nshots is ignored when proportional_sampling=True.{RESET}", flush=True)
            print(f"{COLOR2}[WARNING] Set proportional_sampling=False to use final_nshots.{RESET}", flush=True)
        final_nshots = None  
    
    sync_gpu()
    t_load_start = time.perf_counter()
    with open(circuit_filename, "rb") as handle:
        circuit = qiskit.qpy.load(handle)[0]
    t_load_end = time.perf_counter()
    time_load_circuit = t_load_end - t_load_start
    
    if verbose_level >= 2 and rank == 0:
        print(f"[TIMING] Load circuit: {time_load_circuit:.4f}s", flush=True)


    sync_gpu()
    t_converter_start = time.perf_counter()
    converter = CircuitToEinsum(circuit, dtype=dtype, backend='cupy', options={'check_diagonal': False, 'decompose_gates': False})
    sync_gpu()
    t_converter_end = time.perf_counter()
    time_circuit_to_einsum = t_converter_end - t_converter_start
    
    if verbose_level >= 2 and rank == 0:
        print(f"[TIMING] CircuitToEinsum: {time_circuit_to_einsum:.4f}s", flush=True)
    
    qubits = circuit.qubits
    ranges = [i for i in range(0, nqubits, max_free_qubits)]
    if ranges[-1] != nqubits:
        ranges.append(nqubits)
    
    num_batches = len(ranges) - 1
    if verbose_level >= 2 and rank == 0:
        print(f"[INFO] Batched approach: {num_batches} batches, ranges={ranges}", flush=True)

    sync_gpu()
    t_expr_start = time.perf_counter()
    
    expr_list, operands_list, gate_map_list, projection_indices_list, path_list, info_list, tn_list, dummy_fixed_list, dim_rdm_list = [], [], [], [], [], [], [], [], []
    for j in range(len(ranges)-2):
        fixed = {}
        for index in range(ranges[j], ranges[j+1]):
            fixed[qubits[index]] = '0'
        dummy_fixed_list.append(fixed)
    for j in range(len(ranges)-1):
        # for lightconeCondition in [True, False]:
        #     if lightconeCondition:
        #         lightcone = True
        #     else:
        #         lightcone = False
            if dumpnet_prefix is not None:
                dumpnet_path = f'{dumpnet_prefix}_batch_{j}.txt'
                import os
                os.environ['CUTENSORNET_DUMPNET_PATH'] = dumpnet_path
            where = tuple([qubits[i] for i in range(ranges[j], ranges[j+1])])
            fixed = {}
            for i in range(j):
                fixed = fixed | dummy_fixed_list[i]
            if full_rdm:
                expr, operands = converter.reduced_density_matrix(where, fixed=fixed, lightcone=lightcone)
            else:
                expr, operands = converter.marginal_probability(where, fixed=fixed, lightcone=lightcone)
            gate_map, projection_indices = get_rdm_expr_gate_map(circuit, expr, fixed, return_projection_indices=True)
            ### flip first half of projection_indices
            ### flip them because Qiskit vs non-Qiskit order
            ### flip only the first half because the operand order is wrong for the second half of converter.reduced_density_matrix
            ### and by wrong it doesn't mirror/LIFO the projection operands for bra vs ket
            nqubits_projected = int(len(projection_indices)/2)
            projection_indices[0:nqubits_projected] = projection_indices[0:nqubits_projected][::-1]
            dim_rdm = 2**(len(where))
            _op_bytes = sum(op.nbytes for op in operands)
            _op_max = max(op.nbytes for op in operands) if operands else 0
            if verbose_level >= 2 and rank == 0:
                print(f"  Batch {j} operands: {len(operands)} tensors, {_op_bytes / 1e9:.3f} GB total, largest single tensor {_op_max / 1e6:.2f} MB", flush=True)
            tn = Network(expr, *operands)
            try:
                # Configure optimizer options with hyper-samples if specified
                if num_hyper_samples is not None:
                    optimize_opts = {'samples': num_hyper_samples}
                    path, info = tn.contract_path(optimize=optimize_opts)
                else:
                    path, info = tn.contract_path()
            except Exception as e:
                error_str = str(e)
                if 'ALL_HYPER_SAMPLES_FAILED' in error_str:
                    if rank == 0:
                        print(f"[ABORT] Batch {j} path optimization failed - ALL_HYPER_SAMPLES_FAILED", flush=True)
                        print(f"[ABORT] Network too complex for available memory. Skipping configuration.", flush=True)
                    return [], {
                        'aborted': True,
                        'abort_reason': 'ALL_HYPER_SAMPLES_FAILED',
                        'abort_batch': j,
                    }
                else:
                    # Re-raise unexpected errors
                    raise
            expr_list.append(expr), operands_list.append(operands), gate_map_list.append(gate_map), projection_indices_list.append(projection_indices), path_list.append(path), info_list.append(info), tn_list.append(tn), dim_rdm_list.append(dim_rdm)

    sync_gpu()
    t_expr_end = time.perf_counter()
    time_build_expr_operands = t_expr_end - t_expr_start
    if verbose_level >= 2 and rank == 0:
        _total_op_gb = sum(sum(op.nbytes for op in operands_list[k]) for k in range(len(operands_list)))
        print(f"[TIMING] Build expr/operands ({len(expr_list)} batches): {time_build_expr_operands:.4f}s, total operands {_total_op_gb / 1e9:.3f} GB", flush=True)
        for j, info in enumerate(info_list):
            print(f"  Batch {j}: largest_intermediate={info.largest_intermediate}, opt_cost={info.opt_cost}", flush=True)

    # Check if any batch exceeds the max_opt_cost threshold
    if max_opt_cost is not None:
        for j, info in enumerate(info_list):
            if info.opt_cost > max_opt_cost:
                if rank == 0:
                    print(f"[ABORT] Batch {j} opt_cost ({info.opt_cost:.2e}) exceeds threshold ({max_opt_cost:.2e})", flush=True)
                    print(f"[ABORT] Skipping this configuration to avoid blocking other experiments", flush=True)
                # Return early with empty results and abort info
                return [], {
                    'aborted': True,
                    'abort_reason': f'opt_cost {info.opt_cost:.2e} > {max_opt_cost:.2e}',
                    'abort_batch': j,
                    'opt_cost_batch0': float(info_list[0].opt_cost) if info_list else 0,
                    'time_build_expr_operands': time_build_expr_operands,
                }

    if verbose_level >= 2 and rank == 0:
        total_work = len(noise_samples) * num_batches
        print(f"[TIMING] Starting contractions: {len(noise_samples)} samples × {num_batches} batches = {total_work}", flush=True)
    
    sync_gpu()
    t_loop_start = time.perf_counter()
    
    # Sub-timing accumulators
    time_apply_errors_total = 0
    time_contract_total = 0
    time_sampling_total = 0
    num_contractions = 0
    
    # GPU timing events
    start_gpu = cupy.cuda.Event()
    end_gpu = cupy.cuda.Event()
    gpu_contract_times = []
    
    noisy_sample_shots = []
    noisy_sample_weights = [] if (unique_sampling and return_joint_probability) else None
    benchmark_stop = False  # Flag to stop early for benchmarking
    for index in range(len(noise_samples)): ### each of these is for a different noise set (sampled noise)
        if benchmark_stop:
            break
        error_sample_shots, noise, presampled_shots = [], noise_samples[index], {}
        presampled_weights = {} if (unique_sampling and return_joint_probability) else None
        _, errors, shots = noise

        for j in range(len(ranges)-1):
            if benchmark_stop:
                break
            expr, operands, gate_map, projection_indices, path, info, tn, dim_rdm = expr_list[j], operands_list[j], gate_map_list[j], projection_indices_list[j], path_list[j], info_list[j], tn_list[j], dim_rdm_list[j]
            
            # --- Apply errors ---
            sync_gpu()
            t_err_start = time.perf_counter()
            temp_operands = [operand.copy() for operand in operands] ### must be DEEP copy as you will change it for each error combination
            for error in errors:
                gate_i, noise_qubits, noise_int = error
                error_operand = get_error_operand(noise_qubits, noise_int, dtype)
                ket_index, bra_index = gate_map['ket_gate_'+str(gate_i)], gate_map['bra_gate_'+str(gate_i)]
                temp_operands[ket_index] = ket_contract(temp_operands[ket_index], error_operand)
                temp_operands[bra_index] = bra_contract(temp_operands[bra_index], error_operand)
            sync_gpu()
            time_apply_errors_total += time.perf_counter() - t_err_start

            if presampled_shots == {}:
                start_gpu.record()
                t_contract_start = time.perf_counter()
                
                total_batches = len(ranges) - 1
                is_final_batch = (total_batches == 1) or (j == total_batches - 1)
                
                presampled_shots, presampled_weights = sample_batched(
                    tn, temp_operands, path, info, ranges[j+1], shots, dim_rdm, 
                    full_rdm=full_rdm,
                    unique_sampling=unique_sampling,
                    is_final_batch=is_final_batch,
                    take_all_final=take_all_final,
                    prev_weight=1.0,
                    return_joint_probability=return_joint_probability
                )
                end_gpu.record()
                end_gpu.synchronize()
                gpu_time = cupy.cuda.get_elapsed_time(start_gpu, end_gpu) / 1000
                gpu_contract_times.append(gpu_time)
                time_contract_total += time.perf_counter() - t_contract_start
                num_contractions += 1
                
                if max_contractions is not None and num_contractions >= max_contractions:
                    if verbose_level >= 2 and rank == 0:
                        print(f"[BENCHMARK] Stopping after {num_contractions} contractions", flush=True)
                    benchmark_stop = True
                    break
            else:
                temp_shots = {}
                temp_weights = {} if (unique_sampling and return_joint_probability) else None
                is_last_batch = (j == len(ranges) - 2)
                for bitstring, bitstring_value in presampled_shots.items():
                    # bitstring_value is multiplicity (count when unique_sampling=False, 1 when unique_sampling=True)
                    if unique_sampling:
                        if is_last_batch and final_nshots is not None:
                            local_shots = final_nshots
                        else:
                            local_shots = shots
                        prev_weight = presampled_weights.get(bitstring, 1.0) if presampled_weights is not None else 1.0
                    elif proportional_sampling:
                        local_shots = bitstring_value
                        prev_weight = 1.0
                    elif is_last_batch and final_nshots is not None:
                        local_shots = final_nshots
                        prev_weight = 1.0
                    else:
                        if count_degenerate_shots:
                            local_shots = shots*bitstring_value
                        else:
                            local_shots = shots
                        prev_weight = 1.0
                    
                    for ind, bit in enumerate(bitstring):
                        if bit == '0':
                            temp_operands[projection_indices[ind]] = temp_operands[projection_indices[-(ind+1)]] = proj0
                        elif bit == '1':
                            temp_operands[projection_indices[ind]] = temp_operands[projection_indices[-(ind+1)]] = proj1
                    
                    start_gpu.record()
                    t_contract_start = time.perf_counter()
                    
                    out_shots, out_weights = sample_batched(
                        tn, temp_operands, path, info, ranges[j+1], local_shots, dim_rdm, 
                        prev_bits=bitstring, 
                        full_rdm=full_rdm,
                        unique_sampling=unique_sampling,
                        is_final_batch=is_last_batch,
                        take_all_final=take_all_final,
                        prev_weight=prev_weight,
                        return_joint_probability=return_joint_probability
                    )
                    temp_shots.update(out_shots)
                    del out_shots
                    if temp_weights is not None and out_weights is not None:
                        temp_weights.update(out_weights)
                        del out_weights
                    
                    end_gpu.record()
                    end_gpu.synchronize()
                    gpu_time = cupy.cuda.get_elapsed_time(start_gpu, end_gpu) / 1000
                    gpu_contract_times.append(gpu_time)
                    time_contract_total += time.perf_counter() - t_contract_start
                    num_contractions += 1
                    
                    if max_contractions is not None and num_contractions >= max_contractions:
                        if verbose_level >= 2 and rank == 0:
                            print(f"[BENCHMARK] Stopping after {num_contractions} contractions", flush=True)
                        benchmark_stop = True
                        break
                    
                presampled_shots = temp_shots
                presampled_weights = temp_weights if presampled_weights is not None else presampled_weights

        noisy_sample_shots.append(presampled_shots)
        if noisy_sample_weights is not None:
            noisy_sample_weights.append(presampled_weights if presampled_weights is not None else {})
        
        if verbose_level >= 1 and rank == 0 and (index + 1) % max(1, len(noise_samples) // 10) == 0:
            print(f"  Progress: {index + 1}/{len(noise_samples)} samples", flush=True)

        # Return freed contraction workspace to the driver after each sample to reduce pool fragmentation
        # (many allocations/frees over samples can fragment CuPy pool and cause OOM on a later contraction)
        try:
            cupy.get_default_memory_pool().free_all_blocks()
        except Exception:
            pass

    sync_gpu()
    t_loop_end = time.perf_counter()
    time_contraction_loop = t_loop_end - t_loop_start
    
    if verbose_level >= 2 and rank == 0:
        print(f"[TIMING] Contraction loop total: {time_contraction_loop:.4f}s", flush=True)


    if enable_profiling:
        gpu_total = sum(gpu_contract_times)
        if comm is not None:
            try:
                from mpi4py import MPI
                gpu_total = comm.allreduce(gpu_total, op=MPI.MAX)
            except ImportError:
                pass
        
        profiling_stats = {
            # Timing breakdown (in seconds)
            'time_load_circuit': time_load_circuit,
            'time_circuit_to_einsum': time_circuit_to_einsum,
            'time_build_expr_operands': time_build_expr_operands,
            'time_contraction_loop': time_contraction_loop,
            
            # Contraction loop sub-timings
            'time_apply_errors_total': time_apply_errors_total,
            'time_contract_cpu_total': time_contract_total, #this is the cumulative sum within the contraction loop measured in cpu
            'time_contract_gpu_total': gpu_total,
            
            # Per-contraction averages
            'time_per_contraction_loop_avg': time_contraction_loop / num_contractions if num_contractions > 0 else 0,
            'time_per_contract_gpu_avg': gpu_total / num_contractions if num_contractions > 0 else 0, #this is the average gpu time per contraction measured in gpu
            
            # Counts
            'num_noise_samples': len(noise_samples),
            'num_batches': num_batches,
            'num_contractions': num_contractions,
            'num_qubits': nqubits,
            'max_free_qubits': max_free_qubits,
            'nshots': nshots,
            'final_nshots': final_nshots if final_nshots is not None else nshots,
            
            # Memory/compute stats (from first batch)
            'largest_intermediate_batch0': info_list[0].largest_intermediate if info_list else 0,
            'opt_cost_batch0': float(info_list[0].opt_cost) if info_list else 0,
        }
        
        if verbose_level >= 3 and rank == 0:
            print(f"\n{'='*50}", flush=True)
            print(f"PROFILING SUMMARY (BATCHED)", flush=True)
            print(f"{'='*50}", flush=True)
            total = time_load_circuit + time_circuit_to_einsum + time_build_expr_operands + time_contraction_loop
            for key, val in profiling_stats.items():
                if key.startswith('time_') and not key.startswith('time_per_'):
                    pct = (val / total * 100) if total > 0 else 0
                    print(f"  {key:30s}: {val:8.4f}s ({pct:5.1f}%)", flush=True)
            print(f"{'='*50}\n", flush=True)
        
        if return_joint_probability and noisy_sample_weights is not None:
            return noisy_sample_shots, noisy_sample_weights, profiling_stats
        return noisy_sample_shots, profiling_stats

    if return_joint_probability and noisy_sample_weights is not None:
        return noisy_sample_shots, noisy_sample_weights
    return noisy_sample_shots


def get_noisy_shots_batched_qubitShotFlex(
    circuit_filename,
    noise_samples,
    nqubits,
    max_free_qubits,
    nshots,
    dtype,
    proportional_sampling=False,
    full_rdm=False,
    comm=None,
    rank=0,
    enable_profiling=False,
    verbose_level=0,
    colors=None,
    max_contractions=None,
    max_opt_cost=1e15,
    num_hyper_samples=None,
    lightcone=False,
    dumpnet_prefix=None,
    shots_per_batch=None,
    qubits_per_batch=None,
    count_degenerate_shots=True,
    unique_sampling=False,
    take_all_final=False,
    return_joint_probability=False,
    per_error_callback=None,
    sample_timeout=None,
):
    """Batched noisy shots with list-based qubits_per_batch and shots_per_batch (used only when proportional_sampling is False)."""
    def sync_gpu():
        cupy.cuda.Device().synchronize()

    if colors is None:
        colors = {'COLOR1': '\033[1;92m', 'COLOR2': '\033[1;91m', 'RESET': '\033[0m'}
    COLOR1 = colors.get('COLOR1', '')
    COLOR2 = colors.get('COLOR2', '')
    RESET = colors.get('RESET', '')

    if lightcone and len(noise_samples) > 0:
        if rank == 0:
            print(f"{COLOR2}[WARNING] Lightcone simplification is incompatible with noise injection.{RESET}", flush=True)
            print(f"{COLOR2}[WARNING] Disabling lightcone for this run.{RESET}", flush=True)
        lightcone = True

    if shots_per_batch is not None and proportional_sampling:
        if rank == 0:
            print(f"{COLOR2}[WARNING] shots_per_batch is ignored when proportional_sampling=True.{RESET}", flush=True)
            print(f"{COLOR2}[WARNING] Set proportional_sampling=False to use shots_per_batch.{RESET}", flush=True)
        shots_per_batch = None

    sync_gpu()
    t_load_start = time.perf_counter()
    with open(circuit_filename, "rb") as handle:
        circuit = qiskit.qpy.load(handle)[0]
    t_load_end = time.perf_counter()
    time_load_circuit = t_load_end - t_load_start
    if verbose_level >= 2 and rank == 0:
        print(f"[TIMING] Load circuit: {time_load_circuit:.4f}s", flush=True)

    sync_gpu()
    t_converter_start = time.perf_counter()
    converter = CircuitToEinsum(circuit, dtype=dtype, backend='cupy', options={'check_diagonal': False, 'decompose_gates': False})
    sync_gpu()
    t_converter_end = time.perf_counter()
    time_circuit_to_einsum = t_converter_end - t_converter_start
    if verbose_level >= 2 and rank == 0:
        print(f"[TIMING] CircuitToEinsum: {time_circuit_to_einsum:.4f}s", flush=True)

    qubits = circuit.qubits
    if qubits_per_batch is not None:
        qubits_per_batch = list(qubits_per_batch)
        if sum(qubits_per_batch) != nqubits:
            raise ValueError(f"qubits_per_batch must sum to nqubits={nqubits}, got sum={sum(qubits_per_batch)}")
        ranges = [0] + np.cumsum(qubits_per_batch).tolist()
    else:
        ranges = [i for i in range(0, nqubits, max_free_qubits)]
        if ranges[-1] != nqubits:
            ranges.append(nqubits)
    num_batches = len(ranges) - 1

    if proportional_sampling:
        if shots_per_batch is not None and rank == 0:
            print(f"{COLOR2}[WARNING] shots_per_batch is ignored when proportional_sampling=True.{RESET}", flush=True)
        base_shots_per_batch = [nshots] * num_batches
    else:
        if shots_per_batch is not None:
            base_shots_per_batch = list(shots_per_batch)
            if len(base_shots_per_batch) != num_batches:
                raise ValueError(f"shots_per_batch length must be num_batches={num_batches}, got {len(base_shots_per_batch)}")
        else:
           raise ValueError(f"shots_per_batch is required when proportional_sampling=False")

    if verbose_level >= 2 and rank == 0:
        print(f"[INFO] Batched approach (qubitShotFlex): {num_batches} batches, ranges={ranges}, base_shots_per_batch={base_shots_per_batch}", flush=True)

    sync_gpu()
    t_expr_start = time.perf_counter()
    expr_list, operands_list, gate_map_list, projection_indices_list, path_list, info_list, tn_list, dummy_fixed_list, dim_rdm_list = [], [], [], [], [], [], [], [], []
    for j in range(len(ranges) - 2):
        fixed = {}
        for index in range(ranges[j], ranges[j + 1]):
            fixed[qubits[index]] = '0'
        dummy_fixed_list.append(fixed)
    for j in range(len(ranges) - 1):
        if dumpnet_prefix is not None:
            import os
            dumpnet_path = f'{dumpnet_prefix}_batch_{j}.txt'
            os.environ['CUTENSORNET_DUMPNET_PATH'] = dumpnet_path
        where = tuple([qubits[i] for i in range(ranges[j], ranges[j + 1])])
        fixed = {}
        for i in range(j):
            fixed = fixed | dummy_fixed_list[i]
        if full_rdm:
            expr, operands = converter.reduced_density_matrix(where, fixed=fixed, lightcone=lightcone)
        else:
            expr, operands = converter.marginal_probability(where, fixed=fixed, lightcone=lightcone)
        gate_map, projection_indices = get_rdm_expr_gate_map(circuit, expr, fixed, return_projection_indices=True)
        nqubits_projected = int(len(projection_indices) / 2)
        projection_indices[0:nqubits_projected] = projection_indices[0:nqubits_projected][::-1]
        dim_rdm = 2 ** (len(where))
        # Diagnostic: operand memory per batch (to track memory bloat vs gate count)
        _op_bytes = sum(op.nbytes for op in operands)
        _op_max = max(op.nbytes for op in operands) if operands else 0
        if verbose_level >= 2 and rank == 0:
            print(f"  Batch {j} operands: {len(operands)} tensors, {_op_bytes / 1e9:.3f} GB total, largest single tensor {_op_max / 1e6:.2f} MB", flush=True)
        tn = Network(expr, *operands)
        try:
            if num_hyper_samples is not None:
                optimize_opts = {'samples': num_hyper_samples}
                path, info = tn.contract_path(optimize=optimize_opts)
            else:
                path, info = tn.contract_path()
        except Exception as e:
            error_str = str(e)
            if 'ALL_HYPER_SAMPLES_FAILED' in error_str:
                if rank == 0:
                    print(f"[ABORT] Batch {j} path optimization failed - ALL_HYPER_SAMPLES_FAILED", flush=True)
                    print(f"[ABORT] Network too complex for available memory. Skipping configuration.", flush=True)
                return [], {
                    'aborted': True, 'abort_reason': 'ALL_HYPER_SAMPLES_FAILED', 'abort_batch': j,
                }
            raise
        expr_list.append(expr)
        operands_list.append(operands)
        gate_map_list.append(gate_map)
        projection_indices_list.append(projection_indices)
        path_list.append(path)
        info_list.append(info)
        tn_list.append(tn)
        dim_rdm_list.append(dim_rdm)

    sync_gpu()
    t_expr_end = time.perf_counter()
    time_build_expr_operands = t_expr_end - t_expr_start
    if verbose_level >= 2 and rank == 0:
        _total_op_gb = sum(sum(op.nbytes for op in operands_list[k]) for k in range(len(operands_list)))
        print(f"[TIMING] Build expr/operands ({len(expr_list)} batches): {time_build_expr_operands:.4f}s, total operands {_total_op_gb / 1e9:.3f} GB", flush=True)
        for j, info in enumerate(info_list):
            print(f"  Batch {j}: largest_intermediate={info.largest_intermediate}, opt_cost={info.opt_cost}", flush=True)

    if max_opt_cost is not None:
        for j, info in enumerate(info_list):
            if info.opt_cost > max_opt_cost:
                if rank == 0:
                    print(f"[ABORT] Batch {j} opt_cost ({info.opt_cost:.2e}) exceeds threshold ({max_opt_cost:.2e})", flush=True)
                    print(f"[ABORT] Skipping this configuration to avoid blocking other experiments", flush=True)
                return [], {
                    'aborted': True,
                    'abort_reason': f'opt_cost {info.opt_cost:.2e} > {max_opt_cost:.2e}',
                    'abort_batch': j,
                    'opt_cost_batch0': float(info_list[0].opt_cost) if info_list else 0,
                    'time_build_expr_operands': time_build_expr_operands,
                }

    sync_gpu()
    gc.collect()
    try:
        cupy.get_default_memory_pool().free_all_blocks()
        cupy.get_default_pinned_memory_pool().free_all_blocks()
    except Exception:
        pass
    sync_gpu()
    free_mem, total_mem = cupy.cuda.Device().mem_info
    if verbose_level >= 1 and rank == 0:
        print(f"[GPU MEM] Before contraction loop: {free_mem/1e9:.2f} GB free / {total_mem/1e9:.2f} GB total "
              f"({(total_mem-free_mem)/1e9:.2f} GB in use)", flush=True)

    if verbose_level >= 2 and rank == 0:
        total_work = len(noise_samples) * num_batches
        print(f"[TIMING] Starting contractions: {len(noise_samples)} samples × {num_batches} batches = {total_work}", flush=True)

    sync_gpu()
    t_loop_start = time.perf_counter()
    time_apply_errors_total = 0
    time_contract_total = 0
    time_sampling_total = 0
    num_contractions = 0
    start_gpu = cupy.cuda.Event()
    end_gpu = cupy.cuda.Event()
    gpu_contract_times = []
    time_contract_per_batch = [0.0] * num_batches
    gpu_contract_per_batch = [[] for _ in range(num_batches)]
    num_contractions_per_batch = [0] * num_batches
    use_callback = per_error_callback is not None
    noisy_sample_shots = [] if not use_callback else None
    noisy_sample_weights = [] if (not use_callback and unique_sampling and return_joint_probability) else None
    benchmark_stop = False
    num_timed_out_samples = 0
    oom_error = False
    for index in range(len(noise_samples)):
        if benchmark_stop or oom_error:
            break
        t_sample_start = time.perf_counter()
        sample_timed_out = False
        presampled_shots = {}
        presampled_weights = {} if (unique_sampling and return_joint_probability) else None
        _, errors, _ = noise_samples[index]

        for j in range(len(ranges) - 1):
            if benchmark_stop or oom_error:
                break
            if sample_timeout is not None and (time.perf_counter() - t_sample_start) > sample_timeout:
                sample_timed_out = True
                if verbose_level >= 1 and rank == 0:
                    print(f"  [TIMEOUT] Error sample {index} exceeded {sample_timeout}s -- skipping", flush=True)
                break
            expr, operands, gate_map, projection_indices, path, info, tn, dim_rdm = expr_list[j], operands_list[j], gate_map_list[j], projection_indices_list[j], path_list[j], info_list[j], tn_list[j], dim_rdm_list[j]
            sync_gpu()
            t_err_start = time.perf_counter()
            temp_operands = [operand.copy() for operand in operands]
            for error in errors:
                gate_i, noise_qubits, noise_int = error
                error_operand = get_error_operand(noise_qubits, noise_int, dtype)
                ket_index, bra_index = gate_map['ket_gate_' + str(gate_i)], gate_map['bra_gate_' + str(gate_i)]
                temp_operands[ket_index] = ket_contract(temp_operands[ket_index], error_operand)
                temp_operands[bra_index] = bra_contract(temp_operands[bra_index], error_operand)
            sync_gpu()
            time_apply_errors_total += time.perf_counter() - t_err_start

            if presampled_shots == {}:
                start_gpu.record()
                t_contract_start = time.perf_counter()
                
                total_batches = len(ranges) - 1
                is_final_batch = (total_batches == 1) or (j == total_batches - 1)
                
                try:
                    presampled_shots, presampled_weights = sample_batched(
                        tn, temp_operands, path, info, ranges[j + 1], base_shots_per_batch[0], dim_rdm, 
                        full_rdm=full_rdm,
                        unique_sampling=unique_sampling,
                        is_final_batch=is_final_batch,
                        take_all_final=take_all_final,
                        prev_weight=1.0,
                        return_joint_probability=return_joint_probability
                    )
                except cupy.cuda.memory.OutOfMemoryError as e:
                    oom_error = True
                    if rank == 0:
                        print(f"  [OOM] GPU out of memory during contraction (sample {index}, batch {j}): {e}", flush=True)
                        print(f"  [OOM] Exiting PTSBE contraction loop (completed {index}/{len(noise_samples)} samples)", flush=True)
                    break
                end_gpu.record()
                end_gpu.synchronize()
                gpu_time = cupy.cuda.get_elapsed_time(start_gpu, end_gpu) / 1000
                wall_time = time.perf_counter() - t_contract_start
                gpu_contract_times.append(gpu_time)
                time_contract_total += wall_time
                num_contractions += 1
                time_contract_per_batch[j] += wall_time
                gpu_contract_per_batch[j].append(gpu_time)
                num_contractions_per_batch[j] += 1
                if max_contractions is not None and num_contractions >= max_contractions:
                    if verbose_level >= 2 and rank == 0:
                        print(f"[BENCHMARK] Stopping after {num_contractions} contractions", flush=True)
                    benchmark_stop = True
                    break
                if sample_timeout is not None and (time.perf_counter() - t_sample_start) > sample_timeout:
                    sample_timed_out = True
                    if verbose_level >= 1 and rank == 0:
                        print(f"  [TIMEOUT] Error sample {index} exceeded {sample_timeout}s after contraction -- skipping", flush=True)
                    break
            else:
                temp_shots = {}
                temp_weights = {} if (unique_sampling and return_joint_probability) else None
                is_last_batch = (j == len(ranges) - 2)
                for bitstring, bitstring_value in presampled_shots.items():
                    if sample_timeout is not None and (time.perf_counter() - t_sample_start) > sample_timeout:
                        sample_timed_out = True
                        if verbose_level >= 1 and rank == 0:
                            print(f"  [TIMEOUT] Error sample {index} exceeded {sample_timeout}s during batch {j} -- skipping", flush=True)
                        break
                    if unique_sampling:
                        local_shots = base_shots_per_batch[j]
                        prev_weight = presampled_weights.get(bitstring, 1.0) if presampled_weights is not None else 1.0
                    elif proportional_sampling:
                        local_shots = bitstring_value
                        prev_weight = 1.0
                    else:
                        if count_degenerate_shots:
                            local_shots = base_shots_per_batch[j]*bitstring_value
                        else:
                            local_shots = base_shots_per_batch[j]
                        prev_weight = 1.0
                    
                    for ind, bit in enumerate(bitstring):
                        if bit == '0':
                            temp_operands[projection_indices[ind]] = temp_operands[projection_indices[-(ind + 1)]] = proj0
                        elif bit == '1':
                            temp_operands[projection_indices[ind]] = temp_operands[projection_indices[-(ind + 1)]] = proj1
                    start_gpu.record()
                    t_contract_start = time.perf_counter()
                    try:
                        out_shots, out_weights = sample_batched(
                            tn, temp_operands, path, info, ranges[j + 1], local_shots, dim_rdm, 
                            prev_bits=bitstring, 
                            full_rdm=full_rdm,
                            unique_sampling=unique_sampling,
                            is_final_batch=is_last_batch,
                            take_all_final=take_all_final,
                            prev_weight=prev_weight,
                            return_joint_probability=return_joint_probability
                        )
                    except cupy.cuda.memory.OutOfMemoryError as e:
                        oom_error = True
                        if rank == 0:
                            print(f"  [OOM] GPU out of memory during contraction (sample {index}, batch {j}, bitstring loop): {e}", flush=True)
                            print(f"  [OOM] Exiting PTSBE contraction loop (completed {index}/{len(noise_samples)} samples)", flush=True)
                        break
                    temp_shots.update(out_shots)
                    del out_shots
                    if temp_weights is not None and out_weights is not None:
                        temp_weights.update(out_weights)
                        del out_weights
                    end_gpu.record()
                    end_gpu.synchronize()
                    gpu_time = cupy.cuda.get_elapsed_time(start_gpu, end_gpu) / 1000
                    wall_time = time.perf_counter() - t_contract_start
                    gpu_contract_times.append(gpu_time)
                    time_contract_total += wall_time
                    num_contractions += 1
                    time_contract_per_batch[j] += wall_time
                    gpu_contract_per_batch[j].append(gpu_time)
                    num_contractions_per_batch[j] += 1
                    if max_contractions is not None and num_contractions >= max_contractions:
                        if verbose_level >= 2 and rank == 0:
                            print(f"[BENCHMARK] Stopping after {num_contractions} contractions", flush=True)
                        benchmark_stop = True
                        break
                if sample_timed_out or oom_error:
                    break
                presampled_shots = temp_shots
                presampled_weights = temp_weights if presampled_weights is not None else presampled_weights

            try:
                cupy.get_default_memory_pool().free_all_blocks()
            except Exception:
                pass
            if verbose_level >= 2 and rank == 0:
                sync_gpu()
                free_mem_b, total_mem_b = cupy.cuda.Device().mem_info
                print(f"  [GPU MEM] After batch {j} pool cleanup: {free_mem_b/1e9:.2f} GB free / "
                      f"{total_mem_b/1e9:.2f} GB total ({(total_mem_b-free_mem_b)/1e9:.2f} GB in use)", flush=True)

        if oom_error:
            break
        if sample_timed_out:
            num_timed_out_samples = 1
            del presampled_shots, presampled_weights
            if rank == 0:
                print(f"  [TIMEOUT] Exiting PTSBE contraction loop -- error sample {index} "
                      f"exceeded {sample_timeout}s (completed {index}/{len(noise_samples)} samples)", flush=True)
            break
        elif use_callback:
            per_error_callback(index, presampled_shots, presampled_weights)
        else:
            noisy_sample_shots.append(presampled_shots)
            if noisy_sample_weights is not None:
                noisy_sample_weights.append(presampled_weights if presampled_weights is not None else {})
        del presampled_shots, presampled_weights
        if verbose_level >= 1 and rank == 0 and (index + 1) % max(1, len(noise_samples) // 10) == 0:
            print(f"  Progress: {index + 1}/{len(noise_samples)} samples", flush=True)

        gc.collect()
        try:
            cupy.get_default_memory_pool().free_all_blocks()
        except Exception:
            pass

    if oom_error:
        gc.collect()
        try:
            cupy.get_default_memory_pool().free_all_blocks()
            cupy.get_default_pinned_memory_pool().free_all_blocks()
        except Exception:
            pass

    sync_gpu()
    t_loop_end = time.perf_counter()
    time_contraction_loop = t_loop_end - t_loop_start
    if verbose_level >= 2 and rank == 0:
        print(f"[TIMING] Contraction loop total: {time_contraction_loop:.4f}s", flush=True)

    if enable_profiling:
        gpu_total = sum(gpu_contract_times)
        gpu_per_batch_totals = [sum(times) for times in gpu_contract_per_batch]
        if comm is not None:
            try:
                from mpi4py import MPI
                gpu_total = comm.allreduce(gpu_total, op=MPI.MAX)
            except ImportError:
                pass
        profiling_stats = {
            'time_load_circuit': time_load_circuit,
            'time_circuit_to_einsum': time_circuit_to_einsum,
            'time_build_expr_operands': time_build_expr_operands,
            'time_contraction_loop': time_contraction_loop,
            'time_apply_errors_total': time_apply_errors_total,
            'time_contract_cpu_total': time_contract_total,
            'time_contract_gpu_total': gpu_total,
            'time_per_contraction_loop_avg': time_contraction_loop / num_contractions if num_contractions > 0 else 0,
            'time_per_contract_gpu_avg': gpu_total / num_contractions if num_contractions > 0 else 0,
            'num_noise_samples': len(noise_samples),
            'num_batches': num_batches,
            'num_contractions': num_contractions,
            'num_qubits': nqubits,
            'max_free_qubits': max_free_qubits,
            'nshots': nshots,
            'largest_intermediate_batch0': info_list[0].largest_intermediate if info_list else 0,
            'opt_cost_batch0': float(info_list[0].opt_cost) if info_list else 0,
            'base_shots_per_batch': base_shots_per_batch,
            'num_timed_out_samples': num_timed_out_samples,
            'oom_error': oom_error,
            'time_contract_per_batch': time_contract_per_batch,
            'gpu_contract_per_batch': gpu_per_batch_totals,
            'num_contractions_per_batch': num_contractions_per_batch,
        }
        for b in range(num_batches):
            profiling_stats[f'time_contract_batch_{b}'] = time_contract_per_batch[b]
            profiling_stats[f'gpu_contract_batch_{b}'] = gpu_per_batch_totals[b]
            profiling_stats[f'num_contractions_batch_{b}'] = num_contractions_per_batch[b]

        if verbose_level >= 2 and rank == 0:
            print(f"\n[TIMING] Per-batch contraction breakdown ({num_batches} batches, summed across {len(noise_samples)} samples):", flush=True)
            for b in range(num_batches):
                pct = (time_contract_per_batch[b] / time_contraction_loop * 100) if time_contraction_loop > 0 else 0
                print(f"  Batch {b}: cpu={time_contract_per_batch[b]:.4f}s  gpu={gpu_per_batch_totals[b]:.4f}s  "
                      f"contractions={num_contractions_per_batch[b]}  ({pct:.1f}% of loop)", flush=True)

        if verbose_level >= 3 and rank == 0:
            print(f"\n{'='*50}", flush=True)
            print(f"PROFILING SUMMARY (BATCHED qubitShotFlex)", flush=True)
            print(f"{'='*50}", flush=True)
            total = time_load_circuit + time_circuit_to_einsum + time_build_expr_operands + time_contraction_loop
            for key, val in profiling_stats.items():
                if key.startswith('time_') and not key.startswith('time_per_') and isinstance(val, (int, float)):
                    pct = (val / total * 100) if total > 0 else 0
                    print(f"  {key:30s}: {val:8.4f}s ({pct:5.1f}%)", flush=True)
            for key, val in profiling_stats.items():
                if not key.startswith('time_') and not isinstance(val, list):
                    print(f"  {key:30s}: {val}", flush=True)
            print(f"{'='*50}\n", flush=True)
        if use_callback:
            if return_joint_probability:
                return None, None, profiling_stats
            return None, profiling_stats
        if return_joint_probability and noisy_sample_weights is not None:
            return noisy_sample_shots, noisy_sample_weights, profiling_stats
        return noisy_sample_shots, profiling_stats

    if use_callback:
        return (None, None) if return_joint_probability else (None, None)
    if return_joint_probability and noisy_sample_weights is not None:
        return noisy_sample_shots, noisy_sample_weights
    return noisy_sample_shots



def plot_counts(shots, histogram_file_name):
    counts = Counter(shots)
    counts = counts.most_common()
    counts = dict(sorted(counts))
    df = pandas.DataFrame.from_dict(counts, orient='index')
    df.plot(kind='bar')
    plt.savefig(histogram_file_name)

def network_to_txt(input_modes, input_extents, output_modes, qualifiers=None):

    int_modes = {}
    ind = 0
    
    for modes in input_modes:
        for m in modes:
            if m not in int_modes:
                int_modes[m] = ind
                ind += 1
    
    out_str = ""
    
    for i, (modes, extents) in enumerate(zip(input_modes, input_extents)):
        for m in modes:
            out_str += str(int_modes[m]) + " "
        out_str += "| "
        for e in extents:
            out_str += str(e) + " "
        
        if qualifiers and i < len(qualifiers):
            q = qualifiers[i]
            has_qual = q.get('is_constant') or q.get('requires_gradient') or q.get('is_conjugate')
            if has_qual:
                out_str += "| "
                if q.get('is_constant'):
                    out_str += "CONST "
                if q.get('requires_gradient'):
                    out_str += "GRAD "
                if q.get('is_conjugate'):
                    out_str += "CONJ "
        out_str += "\n"
    
    out_str += "---\n"
    
    for m in output_modes:
        if m in int_modes:
            out_str += str(int_modes[m]) + " "
    out_str += "\n"
    
    return out_str


def dump_network(expr, operands, filename='network_dump.txt', qualifiers=None):
    if '->' in expr:
        inputs_str, output_str = expr.split('->')
    else:
        inputs_str = expr
        output_str = ''
    
    input_modes = [list(s) for s in inputs_str.split(',')]
    output_modes = list(output_str)
    input_extents = [op.shape for op in operands]
    
    txt = network_to_txt(input_modes, input_extents, output_modes, qualifiers)
    
    with open(filename, 'w') as f:
        f.write(txt)
    
    print(f"Network dump written to: {filename}")


def create_network_and_path(expr, operands, dump_filename=None, record_time=False, path_filename=None, optimize=None, verbose_level=0, rank=0):
    result = {}
    
    t_net_start = time.perf_counter()
    tn = Network(expr, *operands)
    t_net_end = time.perf_counter()
    if verbose_level >= 2 and rank == 0:
        print(f"[TIMING] Network() creation: {t_net_end - t_net_start:.4f}s", flush=True)
    
    if dump_filename:
        dump_network(expr, operands, filename=dump_filename)
    
    if record_time:
        start_gpu = cupy.cuda.Event()
        end_gpu = cupy.cuda.Event()
        start_gpu.record()
        start_cpu = time.perf_counter()
    
    t_path_start = time.perf_counter()
    if verbose_level >= 2 and rank == 0:
        print("[TIMING] Starting contract_path()... (this can be SLOW for large networks)", flush=True)
    
    if optimize:
        path, info = tn.contract_path(optimize=optimize)
    else:
        path, info = tn.contract_path()
    
    t_path_end = time.perf_counter()
    if verbose_level >= 2 and rank == 0:
        print(f"[TIMING] contract_path(): {t_path_end - t_path_start:.4f}s", flush=True)

    if record_time:
        end_cpu = time.perf_counter()
        end_gpu.record()
        end_gpu.synchronize()
        result['gpu_time'] = cupy.cuda.get_elapsed_time(start_gpu, end_gpu) / 1000
        result['cpu_time'] = end_cpu - start_cpu
    
    if path_filename:
        with open(path_filename, "w") as f:
            print(path, file=f)
    
    result['network'] = tn
    result['path'] = path
    result['info'] = info
    
    return result
