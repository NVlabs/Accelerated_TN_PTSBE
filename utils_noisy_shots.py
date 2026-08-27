# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Unified utility for computing noisy shots from quantum circuits.

This module provides a single get_noisy_shots function that can be used by both
scaling.py and test_ptsbe.py with different configurations.
"""

import qiskit
import qiskit.qpy
import cupy
import numpy as np
import itertools
import time
from cuquantum.tensornet import CircuitToEinsum, contract, Network
from utils_circuit import (
    get_rdm_expr_gate_map, 
    ket_contract, 
    bra_contract, 
    get_error_operand, 
    create_network_and_path
)


def get_noisy_shots2(
    circuit_filename, 
    noise_samples, 
    fixed_qubits, 
    nshots,
    full_rdm=False,
    comm=None,
    rank=0,
    enable_profiling=False,
    verbose_level=0,
    colors=None,
    dtype='complex128',
    dumpnet_prefix=None
):
    """ 
    Args:
        circuit_filename: Path to .qpy circuit file
        noise_samples: List of noise samples, each containing (_, errors, shots)
        fixed_qubits: Tuple of fixed qubit indices
        nshots: Number of shots 
        full_rdm: If True, compute full RDM then extract diagonal 
                  If False, use marginal_probability 
        comm: MPI communicator (optional, for distributed runs)
        rank: MPI rank (default 0 for single-process)
        enable_profiling: If True, collect timing and workspace stats
        verbose_level: Verbosity level (0: quiet, 1: progress, 2: detailed)
        colors: Dict with color codes for output (optional)
        dtype: Data type for tensors (default 'complex128')
    
    Returns:
        If enable_profiling=False: noisy_sample_shots
        If enable_profiling=True: (noisy_sample_shots, profiling_stats)
        
        profiling_stats contains:
            - Timing breakdown (time_* keys in seconds)
            - Counts (num_* keys)
            - Memory stats (workspace_*, largest_intermediate, etc.)
    """
    from cuquantum.tensornet import Network
    
    # Helper for GPU sync before timing
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
    
    if dumpnet_prefix is not None:
        import os
        dumpnet_path = f'{dumpnet_prefix}.txt'
        os.environ['CUTENSORNET_DUMPNET_PATH'] = dumpnet_path
        
    # PHASE 1: LOAD CIRCUIT
    sync_gpu()
    t_load_start = time.perf_counter()
    with open(circuit_filename, "rb") as handle:
        circuit = qiskit.qpy.load(handle)[0]
    t_load_end = time.perf_counter()
    time_load_circuit = t_load_end - t_load_start
    
    if verbose_level >= 2 and rank == 0:
        print(f"[TIMING] Load circuit: {time_load_circuit:.4f}s", flush=True)

    # PHASE 2: CIRCUIT TO EINSUM CONVERSION
    sync_gpu()
    t_converter_start = time.perf_counter()
    converter = CircuitToEinsum(
        circuit, 
        dtype=dtype, 
        backend='cupy', 
        options={'check_diagonal': False, 'decompose_gates': False}
    )
    sync_gpu()
    t_converter_end = time.perf_counter()
    time_circuit_to_einsum = t_converter_end - t_converter_start
    
    if verbose_level >= 2 and rank == 0:
        print(f"[TIMING] CircuitToEinsum: {time_circuit_to_einsum:.4f}s", flush=True)
    
    qubits = circuit.qubits
    nqubits = len(qubits)
    where = tuple([qubits[i] for i in range(nqubits) if i not in fixed_qubits])
    output_dim = 2**len(where)
    free_qubits = [i for i in range(nqubits) if i not in fixed_qubits]
    nfree = len(free_qubits)

    # PHASE 3: BUILD FIXED QUBIT COMBINATIONS 
    sync_gpu()
    t_fixed_start = time.perf_counter()
    fixed_vals = [''.join(p) for p in itertools.product('01', repeat=len(fixed_qubits))]
    fixed_list = []
    for p in fixed_vals:
        fixed = {}
        for index in range(len(fixed_qubits)):
            fixed[qubits[fixed_qubits[index]]] = p[index]
        fixed_list.append(fixed)
    t_fixed_end = time.perf_counter()
    time_build_fixed_list = t_fixed_end - t_fixed_start
    
    if verbose_level >= 2 and rank == 0:
        print(f"[TIMING] Build fixed_list ({len(fixed_list)} combos): {time_build_fixed_list:.4f}s", flush=True)

    # PHASE 4: BUILD EXPR/OPERANDS LIST (THIS IS OFTEN THE BOTTLENECK)
    sync_gpu()
    t_expr_start = time.perf_counter()
    expr_list, operands_list = [], []
    
    # Note: The condition `if fixed_list == {}` is always False since fixed_list is a list
    # Keeping for backwards compatibility but this branch never executes
    if fixed_list == {}:
        if full_rdm:
            expr, operands = converter.reduced_density_matrix(where, fixed=fixed, lightcone=False)
        else:
            expr, operands = converter.marginal_probability(where, fixed=fixed, lightcone=False)
        expr_list.append(expr)
        operands_list.append(operands)
    
    for fixed in fixed_list:
        if full_rdm:
            expr, operands = converter.reduced_density_matrix(where, fixed=fixed, lightcone=False)
        else:
            expr, operands = converter.marginal_probability(where, fixed=fixed, lightcone=False)
        expr_list.append(expr)
        operands_list.append(operands)
    
    sync_gpu()
    t_expr_end = time.perf_counter()
    time_build_expr_operands = t_expr_end - t_expr_start
    
    if verbose_level >= 2 and rank == 0:
        print(f"[TIMING] Build expr/operands ({len(expr_list)} exprs): {time_build_expr_operands:.4f}s", flush=True)
    
    # PHASE 5: BUILD GATE MAP
    sync_gpu()
    t_gatemap_start = time.perf_counter()
    gate_map = get_rdm_expr_gate_map(circuit, expr_list[0], fixed_list[0])
    t_gatemap_end = time.perf_counter()
    time_build_gate_map = t_gatemap_end - t_gatemap_start
    
    if verbose_level >= 2 and rank == 0:
        print(f"[TIMING] Build gate_map: {time_build_gate_map:.4f}s", flush=True)

    # PHASE 6: PATH FINDING
    sync_gpu()
    t_path_start = time.perf_counter()
    
    net_kwargs = {}
    # if dump_network:
    #     net_kwargs = {
    #         'dump_filename': dump_filename, 
    #         'record_time': True, 
    #         'path_filename': path_filename
    #     }
    # elif enable_profiling:
    #     net_kwargs = {'record_time': True}
    if enable_profiling:
        net_kwargs = {'record_time': True}
    
    net_result = create_network_and_path(expr_list[0], operands_list[0], **net_kwargs)
    network, path, info = net_result['network'], net_result['path'], net_result['info']
    
    sync_gpu()
    t_path_end = time.perf_counter()
    time_path_finding = t_path_end - t_path_start
    
    if verbose_level >= 2 and rank == 0:
        print(f"[TIMING] Path finding: {time_path_finding:.4f}s", flush=True)
        print(f"  Workspace scratch: {network.workspace_scratch_size/1e9:.2f} GB", flush=True)
        print(f"  Largest intermediate: {info.largest_intermediate}", flush=True)
        print(f"  Opt cost: {info.opt_cost}", flush=True)
    

    
    # PHASE 8: MAIN CONTRACTION LOOP
    if verbose_level >= 2 and rank == 0:
        total_contractions = len(fixed_vals) * len(noise_samples)
        print(f"[TIMING] Starting contractions: {len(noise_samples)} samples × {len(fixed_vals)} combos = {total_contractions}", flush=True)
    
    sync_gpu()
    t_loop_start = time.perf_counter()
    
    noisy_sample_shots = []
    
    # Sub-timing accumulators for the loop
    time_apply_errors_total = 0
    time_reset_operands_total = 0
    time_contract_total = 0
    time_sampling_total = 0
    num_contractions = 0
    
    # GPU timing events
    start_gpu = cupy.cuda.Event()
    end_gpu = cupy.cuda.Event()
    gpu_contract_times = []
    
    for sample_idx in range(len(noise_samples)):
        error_sample_shots = []
        noise = noise_samples[sample_idx]
        _, errors, shots = noise
        
        for config_idx, (operands, fixed_val) in enumerate(zip(operands_list, fixed_vals)):
            fixed_val = fixed_val[::-1]
            
            # --- Apply errors ---
            sync_gpu()
            t_err_start = time.perf_counter()
            temp_operands = [operand.copy() for operand in operands]
            for error in errors:
                gate_i, noise_qubits, noise_int = error
                error_operand = get_error_operand(noise_qubits, noise_int, dtype)
                ket_index = gate_map['ket_gate_'+str(gate_i)]
                bra_index = gate_map['bra_gate_'+str(gate_i)]
                temp_operands[ket_index] = ket_contract(temp_operands[ket_index], error_operand)
                temp_operands[bra_index] = bra_contract(temp_operands[bra_index], error_operand)
            sync_gpu()
            time_apply_errors_total += time.perf_counter() - t_err_start
            
            # --- Reset operands ---
            sync_gpu()
            t_reset_start = time.perf_counter()
            temp_operands = [cupy.ascontiguousarray(op) for op in temp_operands]
            network.reset_operands(*temp_operands)
            sync_gpu()
            time_reset_operands_total += time.perf_counter() - t_reset_start
            
            # --- Contract (GPU timed) ---
            start_gpu.record()
            t_contract_start = time.perf_counter()
            
            if full_rdm:
                out = network.contract().reshape(output_dim, output_dim)
                out = cupy.clip(cupy.diagonal(out).real, 0, None)
            else:
                out = network.contract().reshape(output_dim)
                out = cupy.clip(out.real, 0, None)
            
            end_gpu.record()
            end_gpu.synchronize()
            gpu_time = cupy.cuda.get_elapsed_time(start_gpu, end_gpu) / 1000  # ms -> s
            gpu_contract_times.append(gpu_time)
            time_contract_total += time.perf_counter() - t_contract_start
            num_contractions += 1
            
            # --- Sampling ---
            sync_gpu()
            t_sample_start = time.perf_counter()
            marginal_prob = cupy.sum(out)
            if marginal_prob > 0:
                marginal_prob_cpu = float(marginal_prob.get())
                out_cpu = out.get() / marginal_prob_cpu
                noisy_shots_arr = np.random.choice(
                    len(out_cpu), 
                    size=int(np.ceil(shots * marginal_prob_cpu)), 
                    p=out_cpu
                )
                vectorized_binary_repr = np.vectorize(np.binary_repr)
                noisy_shots_list = vectorized_binary_repr(noisy_shots_arr, width=nqubits).tolist()
                noisy_shots_list = [s[::-1] for s in noisy_shots_list]
                noisy_shots_list = [s[0:nfree] + fixed_val for s in noisy_shots_list]
                error_sample_shots.append((marginal_prob, noisy_shots_list))
            sync_gpu()
            time_sampling_total += time.perf_counter() - t_sample_start
            
            # Cleanup
            del out, marginal_prob, temp_operands
            cupy.get_default_memory_pool().free_all_blocks()
 
        noisy_sample_shots.append(error_sample_shots)
        
        if verbose_level >= 1 and rank == 0 and (sample_idx + 1) % max(1, len(noise_samples) // 10) == 0:
            print(f"  Progress: {sample_idx + 1}/{len(noise_samples)} samples", flush=True)

    sync_gpu()
    t_loop_end = time.perf_counter()
    time_contraction_loop = t_loop_end - t_loop_start
    
    if verbose_level >= 2 and rank == 0:
        print(f"[TIMING] Contraction loop total: {time_contraction_loop:.4f}s", flush=True)

    # CLEANUP
    del network
    cupy.get_default_memory_pool().free_all_blocks()

    # BUILD PROFILING STATS
    if enable_profiling:
        # Aggregate GPU times across MPI ranks if applicable
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
            'time_build_fixed_list': time_build_fixed_list,
            'time_build_expr_operands': time_build_expr_operands,
            'time_build_gate_map': time_build_gate_map,
            'time_path_finding': time_path_finding,
            'time_contraction_loop': time_contraction_loop,
            
            # Contraction loop sub-timings
            'time_apply_errors': time_apply_errors_total,
            'time_reset_operands': time_reset_operands_total,
            'time_contract_cpu': time_contract_total,
            'time_contract_gpu': gpu_total,
            'time_sampling': time_sampling_total,
            
            # Per-contraction averages
            'time_per_contraction_avg': time_contraction_loop / num_contractions if num_contractions > 0 else 0,
            'time_per_contract_gpu_avg': gpu_total / num_contractions if num_contractions > 0 else 0,
            
            # Counts
            'num_noise_samples': len(noise_samples),
            'num_fixed_combinations': len(fixed_vals),
            'num_contractions': num_contractions,
            'num_qubits': nqubits,
            'num_free_qubits': nfree,
            'num_fixed_qubits': len(fixed_qubits),
            
            # Memory/compute stats
            'workspace_scratch_size_gb': network.workspace_scratch_size / 1e9,
            'largest_intermediate': info.largest_intermediate,
            'opt_cost': float(info.opt_cost),
            
            # Path finding sub-times (if available)
            'time_path_cpu': net_result.get('cpu_time', 0),
            'time_path_gpu': net_result.get('gpu_time', 0),
        }
        
        if verbose_level >= 2 and rank == 0:
            print(f"\n{'='*50}", flush=True)
            print(f"PROFILING SUMMARY", flush=True)
            print(f"{'='*50}", flush=True)
            total = (time_load_circuit + time_circuit_to_einsum + time_build_fixed_list + 
                    time_build_expr_operands + time_build_gate_map + time_path_finding + 
                    time_contraction_loop)
            for key, val in profiling_stats.items():
                if key.startswith('time_') and not key.startswith('time_per_'):
                    pct = (val / total * 100) if total > 0 else 0
                    print(f"  {key:30s}: {val:8.4f}s ({pct:5.1f}%)", flush=True)
            print(f"{'='*50}\n", flush=True)
        
        return noisy_sample_shots, profiling_stats
    
    return noisy_sample_shots
