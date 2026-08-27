# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
PTSBE vs CUDA-Q Comparison Script (Averaged over Multiple Circuits)

This script runs both CUDA-Q and PTSBE on multiple random circuits and reports
averaged timing measurements for both, allowing robust comparison.

Usage:
    python scaling_comparison_avg.py --nqubits 40 --ngates 800 --first_fixed 24 --ptsbe_nshots 100 --num_circuits 5
"""

import sys
import os
import time
import subprocess
import argparse
import glob
import shutil
import re
import json
import pickle
import gc
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# WAR: Monkeypatch for Qiskit circuit compatibility
from cuquantum.tensornet._internal import circuit_parser_utils_qiskit
from qiskit.circuit import Measure

def remove_measurements(circuit):
    for instruction in circuit.data:
        if isinstance(instruction.operation, Measure): 
            raise ValueError('the input circuit can not contain measurements')
    return circuit

circuit_parser_utils_qiskit.remove_measurements = remove_measurements

from utils_circuit import build_circuit_and_script, random_gate_sample, get_noisy_shots_batched_qubitShotFlex, estimate_contractions_batched_qubitShotFlex
from collections import Counter

# Color codes for output
BOLD_GREEN = '\033[1;92m'
BOLD_YELLOW = '\033[1;93m'
BOLD_CYAN = '\033[1;96m'
BOLD_RED = '\033[1;91m'
BOLD_MAGENTA = '\033[1;95m'
COLOR1 = BOLD_GREEN
COLOR2 = BOLD_RED
COLOR3 = BOLD_CYAN
RESET = '\033[0m'

dtype = 'complex128'


def cleanup_files():
    """Remove generated files from previous runs."""
    patterns = ['*.png', 'stim*', '*.pickle', '*output.py', 'pts.py', '*.qpy', 
                'cudaq_output.py', '*.json']
    for pattern in patterns:
        for f in glob.glob(pattern):
            try:
                os.remove(f)
            except OSError:
                pass


def parse_cudaq_output(output_text):
    timing_stats = {}
    
    patterns = {
        'time_setup': r'Setup time:\s+([\d.]+)s',
        'time_warmup': r'Warmup/JIT time:\s+([\d.]+)s',
        'time_sample': r'Sample time:\s+([\d.]+)s',
        'time_total': r'Total time:\s+([\d.]+)s',
        'time_per_shot': r'Per-shot time:\s+([\d.]+)s',
    }
    
    for key, pattern in patterns.items():
        match = re.search(pattern, output_text)
        if match:
            timing_stats[key] = float(match.group(1))
    
    distinct_match = re.search(r'Total number of distinct CUDA-Q shots collected:\s+(\d+)', output_text)
    total_match = re.search(r'Total number of overall CUDA-Q shots collected:\s+(\d+)', output_text)
    
    if distinct_match:
        timing_stats['num_distinct_shots'] = int(distinct_match.group(1))
    if total_match:
        timing_stats['num_total_shots'] = int(total_match.group(1))
    
    return timing_stats


def main():
    parser = argparse.ArgumentParser(description='Run PTSBE vs CUDA-Q comparison on the same circuit.')
    
    # Circuit parameters
    parser.add_argument('--nqubits', type=int, default=20, help='Number of qubits')
    parser.add_argument('--ngates', type=int, default=160, help='Number of gates')
    parser.add_argument('--first_fixed', type=int, default=10, help='First fixed qubit (determines batching)')
    parser.add_argument('--ptsbe_nshots', type=int, default=100, help='Number of shots per noise sample for PTSBE')
    parser.add_argument('--final_nshots', type=int, default=100, help='Number of shots for last batch (if None, uses ptsbe_nshots). Required for batch_shots_policy=alternate.')
    parser.add_argument('--final_batch_qubits', type=int, default=None,
                        help='Qubits in last batch only. Required for batch_qubit_policy=alternate; if None with alternate, auto-set to nqubits %% first_fixed (or first_fixed if 0).')
    parser.add_argument('--batch_shots_policy', type=str, choices=['uniform', 'alternate', 'custom'], default='alternate',
                        help='Shot allocation across batches (when proportional_sampling=False). uniform: same ptsbe_nshots every batch; alternate: non-final batches use ptsbe_nshots, final uses --final_nshots; custom: use --shots_per_batch.')
    parser.add_argument('--shots_per_batch', type=str, default=None,
                        help='Comma-separated shot count per batch, e.g. "100,100,1000". Required when batch_shots_policy=custom; length must equal num_batches.')
    parser.add_argument('--batch_qubit_policy', type=str, choices=['uniform', 'alternate', 'custom'], default='alternate',
                        help='Qubits per batch: uniform (first_fixed each); alternate: non-final use first_fixed, final uses --final_batch_qubits; custom: use --qubits_per_batch.')
    parser.add_argument('--qubits_per_batch', type=str, default=None,
                        help='Comma-separated qubit count per batch, e.g. "6,10,8". Required when qubits_per_batch=custom; must sum to nqubits.')
    parser.add_argument('--nnoise_samples', type=int, default=10, help='Number of noise samples for PTSBE')
    parser.add_argument('--cudaq_nshots', type=int, default=1000, help='Number of shots for CUDA-Q')
    parser.add_argument('--prob_range_min', type=float, default=0.02, help='Min error probability')
    parser.add_argument('--prob_range_max', type=float, default=0.2, help='Max error probability')
    parser.add_argument('--proportion_two_qubit', type=float, default=0.2, help='Proportion of two-qubit gates')
    parser.add_argument('--local_gates', action='store_true', help='Use local gates only')
    parser.add_argument('--proportional_sampling', action='store_true', help='Use proportional sampling')
    parser.add_argument('--smart-opt-off', type=str, choices=['cudaq', 'ptsbe', 'both'], default=None,help='Disable smart contraction path caching for: cudaq, ptsbe, or both')

    
    # PTSBE-specific parameters
    parser.add_argument('--full_rdm', action='store_true', help='Use full RDM computation')
    parser.add_argument('--max_contractions', type=int, default=None, help='Limit contractions for benchmarking')
    parser.add_argument('--num_hyper_samples', type=int, default=None, help='Number of hyper-samples for path optimizer')
    parser.add_argument('--lightcone', action='store_true', help='Enable lightcone simplification')
    parser.add_argument('--max_opt_cost', type=float, default=1e15, help='Maximum opt_cost threshold; abort if exceeded (default: 1e15)')
    parser.add_argument('--dumpnet_prefix', type=str, default=None, help='Prefix path for CUTENSORNET_DUMPNET_PATH (if set, enables network dumping)')
    
    # File parameters
    parser.add_argument('--circuit_filename', type=str, default='random_circuit.qpy', help='Circuit filename')
    parser.add_argument('--stim_script_filename', type=str, default='stim_script.stim', help='Stim script filename')
    parser.add_argument('--work_dir', type=str, default=None, help='Working directory for intermediates (default: current dir). Use for parallel runs.')
    parser.add_argument('--circuit_dir', type=str, default=None,
                        help='Directory with pre-generated circuits (from generate_circuits.py). '
                             'When set, Phase 1 loads circuit_{id}.qpy/.stim instead of generating.')
    parser.add_argument('--circuit_id', type=int, default=None,
                        help='0-based circuit index to load from --circuit_dir. Required when --circuit_dir is set.')
    
    # Output parameters
    parser.add_argument('--verbose_level', type=int, default=1, help='Verbose level (0: quiet, 1: normal, 2: detailed)')
    parser.add_argument('--json_output', type=str, default=None, help='Save results to JSON file')
    parser.add_argument('--skip_cudaq', action='store_true', help='Skip CUDA-Q run (PTSBE only)')
    parser.add_argument('--skip_ptsbe', action='store_true', help='Skip PTSBE run (CUDA-Q only)')
    
    # Averaging parameters
    parser.add_argument('--num_circuits', type=int, default=1, help='Number of random circuits to test and average')
    parser.add_argument('--no_count_degenerate_shots', action='store_true', help='Do not count degenerate shots')
    parser.add_argument('--unique_sampling', action='store_true',
                        help='Sample unique bitstrings per batch (no duplicates). Default: False.')
    parser.add_argument('--take_all_final', action='store_true',
                        help='On final batch, take all nonzero outcomes instead of sampling (only with --unique_sampling). Default: False.')
    parser.add_argument('--joint_probability', action='store_true',
                        help='Compute and return joint probabilities per bitstring (only when unique_sampling is used). Default: False.')
    parser.add_argument('--include_total_time', action='store_true',
                        help='Use total/wall-clock time for comparison and throughput (CUDA-Q total, PTSBE load+convert+build+contraction). Default: False uses CUDA-Q sample time and PTSBE contraction loop only.')
    parser.add_argument('--no-mem-opt', action='store_true',
                        help='Disable PTSBE memory optimization: return full per-error shots (and joint_weights) list instead of aggregating via callback. Use for debugging or when per-error data is needed.')
    parser.add_argument('--cudaq_timeout', type=float, default=None,
                        help='Timeout in seconds for the CUDA-Q subprocess. If exceeded, CUDA-Q is skipped for this circuit. Default: no limit.')
    parser.add_argument('--ptsbe_sample_timeout', type=float, default=None,
                        help='Timeout in seconds per PTSBE error sample. Samples exceeding this are skipped. Default: no limit.')
    
    args = parser.parse_args()

    if args.circuit_dir is not None and args.circuit_id is None:
        parser.error("--circuit_dir requires --circuit_id (0-based index of the circuit to load).")
    if args.circuit_id is not None and args.circuit_dir is None:
        parser.error("--circuit_id requires --circuit_dir.")
    if args.circuit_dir is not None:
        args.circuit_dir = os.path.abspath(args.circuit_dir)
        args.num_circuits = 1
        config_path = os.path.join(args.circuit_dir, 'config.json')
        if os.path.isfile(config_path):
            with open(config_path, 'r') as _f:
                gen_cfg = json.load(_f)
            if gen_cfg.get('nqubits') != args.nqubits:
                parser.error(
                    f"--nqubits {args.nqubits} does not match circuit_dir config "
                    f"(generated with nqubits={gen_cfg['nqubits']})")
            if gen_cfg.get('ngates') != args.ngates:
                parser.error(
                    f"--ngates {args.ngates} does not match circuit_dir config "
                    f"(generated with ngates={gen_cfg['ngates']})")
            if args.circuit_id >= gen_cfg.get('num_circuits', 0):
                parser.error(
                    f"--circuit_id {args.circuit_id} out of range "
                    f"(circuit_dir has {gen_cfg['num_circuits']} circuits, valid: 0..{gen_cfg['num_circuits']-1})")
        else:
            print(f"WARNING: No config.json in {args.circuit_dir}, skipping parameter validation", flush=True)

    if args.work_dir is not None:
        os.makedirs(args.work_dir, exist_ok=True)
        os.chdir(args.work_dir)

    # Auto-set final_batch_qubits for alternate policy if not provided
    if args.batch_qubit_policy == 'alternate' and args.final_batch_qubits is None:
        rem = args.nqubits % args.first_fixed
        args.final_batch_qubits = rem if rem > 0 else args.first_fixed

    # Parse shots_per_batch for custom policy

    qubits_per_batch_list = None
    if args.batch_qubit_policy == 'custom' and (args.qubits_per_batch is None or not args.qubits_per_batch.strip()):
        parser.error("--batch_qubit_policy=custom requires --qubits_per_batch (e.g. --qubits_per_batch 6,10,8).")
    if args.batch_qubit_policy == 'custom' and args.qubits_per_batch:
        qubits_per_batch_list = [int(x.strip()) for x in args.qubits_per_batch.split(',')]
        if sum(qubits_per_batch_list) != args.nqubits:
            parser.error(
                f"--qubits_per_batch must sum to nqubits={args.nqubits}, got {qubits_per_batch_list} (sum={sum(qubits_per_batch_list)})"
            )
    if args.batch_qubit_policy == 'alternate' and args.final_batch_qubits is None:
        parser.error("--batch_qubit_policy=alternate requires --final_batch_qubits.")
    if args.batch_qubit_policy == 'alternate' and args.final_batch_qubits is not None:
        rest = args.nqubits - args.final_batch_qubits
        if rest < 0:
            parser.error(f"batch_qubit_policy=alternate: (nqubits - final_batch_qubits) = {rest} must be >= 0")
        num_non_final_batches = rest // args.first_fixed
        rem = rest % args.first_fixed
        if rem > 0:
            qubits_per_batch_list = [args.first_fixed] * num_non_final_batches + [rem] + [args.final_batch_qubits]
        else:
            qubits_per_batch_list = [args.first_fixed] * num_non_final_batches + [args.final_batch_qubits]
    if args.batch_qubit_policy == 'uniform':
        qubits_per_batch_list = [args.first_fixed] * (args.nqubits // args.first_fixed)
        if args.nqubits % args.first_fixed > 0:
            qubits_per_batch_list.append(args.nqubits % args.first_fixed)

    num_batches = len(qubits_per_batch_list)

    shots_per_batch_list = None
    if args.batch_shots_policy == 'custom' and (args.shots_per_batch is None or not args.shots_per_batch.strip()):
        parser.error("--batch_shots_policy=custom requires --shots_per_batch (e.g. --shots_per_batch 100,100,1000).")
    if args.batch_shots_policy == 'custom' and args.shots_per_batch:
        shots_per_batch_list = [int(x.strip()) for x in args.shots_per_batch.split(',')]
        if len(shots_per_batch_list) != num_batches:
            parser.error(f"--shots_per_batch length ({len(shots_per_batch_list)}) must match num_batches={num_batches}")
    if args.batch_shots_policy == 'alternate' and args.final_nshots is None:
        parser.error("--batch_shots_policy=alternate requires --final_nshots to be set.")
    if args.batch_shots_policy == 'alternate' and args.final_nshots is not None:
        shots_per_batch_list = [args.ptsbe_nshots] * (num_batches - 1) + [args.final_nshots]
    if args.batch_shots_policy == 'uniform':
        shots_per_batch_list = [args.ptsbe_nshots] * num_batches


    
    # Derived parameters
    prob_range = [args.prob_range_min, args.prob_range_max]
    max_free_qubits = args.first_fixed
    num_fixed = args.nqubits - args.first_fixed

    circuit_filename = args.circuit_filename
    stim_script_filename = args.stim_script_filename
    
    if args.verbose_level >= 1:
        print(f"\n{'='*70}")
        print(f"{COLOR3}PTSBE vs CUDA-Q COMPARISON (AVERAGED){RESET}")
        print(f"{'='*70}")
        print(f"\n{COLOR1}Configuration:{RESET}")
        print(f"  num_circuits:         {args.num_circuits}")
        print(f"  nqubits:              {args.nqubits}")
        print(f"  ngates:               {args.ngates}")
        print(f"  cudaq_nshots:         {args.cudaq_nshots}")
        print(f"  ptsbe_nshots:         {args.ptsbe_nshots}")
        print(f"  final_nshots:         {args.final_nshots if args.final_nshots else args.ptsbe_nshots} (PTSBE last batch)")
        print(f"  batch_shots_policy:   {args.batch_shots_policy}" + (f", shots_per_batch: {shots_per_batch_list}" if shots_per_batch_list else ""))
        print(f"  batch_qubit_policy:   {args.batch_qubit_policy}" + (f", qubits_per_batch: {qubits_per_batch_list}" if qubits_per_batch_list else ""))
        print(f"  first_fixed:          {args.first_fixed}")
        print(f"  max_free_qubits:      {max_free_qubits}")
        print(f"  num_fixed:            {num_fixed if num_fixed > 0 else 'None'} -> {f'2^{num_fixed} = {2**num_fixed} combinations' if num_fixed > 0 else '1 combination'}")
        print(f"  nnoise_samples:       {args.nnoise_samples}")
        print(f"  prob_range:           {prob_range}")
        print(f"  proportion_two_qubit: {args.proportion_two_qubit}")
        print(f"  local_gates:          {args.local_gates}")
        print(f"  smart_opt_off:        {args.smart_opt_off if args.smart_opt_off else 'None (smart opt enabled)'}")
        print(f"  num_hyper_samples:    {args.num_hyper_samples}")
        print(f"  no_count_degenerate_shots: {args.no_count_degenerate_shots}")
        if args.circuit_dir is not None:
            print(f"  circuit_dir:          {args.circuit_dir}")
            print(f"  circuit_id:           {args.circuit_id}")
        print(f"{'='*70}\n")
    
    # Store results from all circuits
    all_circuit_results = []
    
    # Loop over random circuits
    for circuit_idx in range(args.num_circuits):
        # Print circuit banner
        if args.verbose_level >= 1:
            print(f"\n{'#'*70}")
            print(f"{'#'*20} CIRCUIT {circuit_idx + 1} / {args.num_circuits} {'#'*20}")
            print(f"{'#'*70}\n")
        
        # Initialize results for this circuit
        results = {
            'config': {
                'nqubits': args.nqubits,
                'ngates': args.ngates,
                'cudaq_nshots': args.cudaq_nshots,
                'ptsbe_nshots': args.ptsbe_nshots,
                'final_nshots': args.final_nshots if args.final_nshots else args.ptsbe_nshots,
                'final_batch_qubits': args.final_batch_qubits,
                'batch_shots_policy': args.batch_shots_policy,
                'shots_per_batch': shots_per_batch_list,
                'batch_qubit_policy': args.batch_qubit_policy,
                'qubits_per_batch': qubits_per_batch_list,
                'first_fixed': args.first_fixed,
                'max_free_qubits': max_free_qubits,
                'num_fixed': num_fixed,
                'nnoise_samples': args.nnoise_samples,
                'prob_range': prob_range,
                'proportion_two_qubit': args.proportion_two_qubit,
                'smart_opt_off': args.smart_opt_off,
                'circuit_idx': args.circuit_id if args.circuit_id is not None else circuit_idx,
                'no_count_degenerate_shots': args.no_count_degenerate_shots,
            },
            'cudaq': {},
            'ptsbe': {},
            'comparison': {}
        }
        
        # PHASE 1: BUILD SHARED CIRCUIT (or load pre-generated)
        if args.verbose_level >= 1:
            print(f"{COLOR1}[PHASE 1] {'Loading' if args.circuit_dir else 'Building'} shared circuit...{RESET}")
        
        cleanup_files()
        
        t_circuit_start = time.perf_counter()

        if args.circuit_dir is not None:
            src_qpy = os.path.join(args.circuit_dir, f'circuit_{args.circuit_id}.qpy')
            src_stim = os.path.join(args.circuit_dir, f'circuit_{args.circuit_id}.stim')
            for src in (src_qpy, src_stim):
                if not os.path.isfile(src):
                    raise FileNotFoundError(f"Pre-generated circuit file not found: {src}")
            shutil.copy2(src_qpy, circuit_filename)
            shutil.copy2(src_stim, stim_script_filename)
            if args.verbose_level >= 1:
                print(f"  Loaded circuit {args.circuit_id} from {args.circuit_dir}")
            results['config']['circuit_dir'] = args.circuit_dir
            results['config']['circuit_id'] = args.circuit_id
        else:
            gates_sampled = random_gate_sample(
                args.nqubits, 
                args.ngates, 
                prob_range, 
                proportion_two_qubit=args.proportion_two_qubit, 
                local_gates=args.local_gates
            )
            build_circuit_and_script(
                args.nqubits, 
                gates_sampled, 
                circuit_filename, 
                stim_script_filename
            )
        
        t_circuit_end = time.perf_counter()
        results['time_build_circuit'] = t_circuit_end - t_circuit_start
        
        if args.verbose_level >= 1:
            print(f"  Circuit {'loaded' if args.circuit_dir else 'built'} in {results['time_build_circuit']:.4f}s")
        

        # PHASE 2: RUN CUDA-Q
        if not args.skip_cudaq:
            if args.verbose_level >= 1:
                print(f"\n{COLOR1}[PHASE 2] Running CUDA-Q...{RESET}")
            
            # Convert Stim to CUDA-Q
            script_dir = os.path.dirname(os.path.abspath(__file__))
            stim_to_cudaq_path = os.path.join(os.path.dirname(script_dir), 'stim_to_cudaq.py')
            cudaq_output_file = 'cudaq_output.py'
            
            print(f"Running: python {stim_to_cudaq_path} {stim_script_filename} {cudaq_output_file}")
            t_convert_start = time.perf_counter()
            convert_cmd = f'python {stim_to_cudaq_path} {stim_script_filename} {cudaq_output_file}'
            subprocess.run(convert_cmd, shell=True, capture_output=True, text=True)
            t_convert_end = time.perf_counter()
            results['cudaq']['time_stim_to_cudaq'] = t_convert_end - t_convert_start
            
            # Run CUDA-Q
            t_cudaq_start = time.perf_counter()
            cudaq_cmd = f'python {cudaq_output_file} {args.nqubits} {args.cudaq_nshots}'
            
            cudaq_env = os.environ.copy()
            if args.smart_opt_off in ['cudaq', 'both']:
                cudaq_env['CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_CACHE_REUSE_NRUNS'] = '1000000'
                cudaq_env['CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_SMART_OPTION'] = '0'
                cudaq_env['CUDAQ_TENSORNET_NUM_HYPER_SAMPLES'] = '1' if args.num_hyper_samples is None else str(args.num_hyper_samples)
            
            try:
                cudaq_result = subprocess.run(cudaq_cmd, shell=True, capture_output=True, text=True,
                                              env=cudaq_env, timeout=args.cudaq_timeout)
            except subprocess.TimeoutExpired:
                t_cudaq_end = time.perf_counter()
                elapsed = t_cudaq_end - t_cudaq_start
                print(f"{COLOR2}[TIMEOUT] CUDA-Q exceeded {args.cudaq_timeout}s limit "
                      f"(ran {elapsed:.1f}s) -- skipping{RESET}", flush=True)
                results['cudaq']['time_execution_wallclock'] = elapsed
                results['cudaq']['error'] = f'timeout after {elapsed:.1f}s (limit={args.cudaq_timeout}s)'
                results['cudaq']['timed_out'] = True
                cudaq_result = None
            else:
                t_cudaq_end = time.perf_counter()
                results['cudaq']['time_execution_wallclock'] = t_cudaq_end - t_cudaq_start

            if cudaq_result is not None and cudaq_result.returncode == 0:
                cudaq_timing = parse_cudaq_output(cudaq_result.stdout)
                results['cudaq'].update(cudaq_timing)
                
                if args.verbose_level >= 1:
                    print(cudaq_result.stdout)
            elif cudaq_result is not None:
                print(f"{COLOR2}[ERROR] CUDA-Q failed:{RESET}")
                print(cudaq_result.stderr)
                results['cudaq']['error'] = cudaq_result.stderr

        # Reclaim GPU memory before PTSBE (subprocess exit may not free driver memory immediately)
        gc.collect()
        try:
            import cupy as cp
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
            cp.cuda.Device().synchronize()
            free_mem, total_mem = cp.cuda.Device().mem_info
            if args.verbose_level >= 1:
                print(f"[GPU MEM] After CUDA-Q cleanup: {free_mem/1e9:.2f} GB free / {total_mem/1e9:.2f} GB total "
                      f"({(total_mem-free_mem)/1e9:.2f} GB in use)", flush=True)
        except Exception:
            pass

        # PHASE 3: RUN PTSBE
        if not args.skip_ptsbe:
            if args.verbose_level >= 1:
                print(f"\n{COLOR1}[PHASE 3] Running PTSBE...{RESET}")
            
            # Convert Stim to PTS for noise sampling
            script_dir = os.path.dirname(os.path.abspath(__file__))
            stim_to_pts_path = os.path.join(os.path.dirname(script_dir), 'stim_to_pts.py')
            pts_output_file = 'pts_output.py'
            
            t_pts_convert_start = time.perf_counter()
            convert_cmd = f'python {stim_to_pts_path} {args.stim_script_filename} {pts_output_file}'
            subprocess.run(convert_cmd, shell=True, capture_output=True, text=True)
            t_pts_convert_end = time.perf_counter()
            results['ptsbe']['time_stim_to_pts'] = t_pts_convert_end - t_pts_convert_start
            
            # Run PTS to generate noise samples
            t_pts_sample_start = time.perf_counter()
            pts_cmd = f'python {pts_output_file} {args.nnoise_samples} {args.ptsbe_nshots}'
            subprocess.run(pts_cmd, shell=True, capture_output=True, text=True)
            t_pts_sample_end = time.perf_counter()
            results['ptsbe']['time_pts_sampling'] = t_pts_sample_end - t_pts_sample_start
            
            # Load noise samples
            with open('error_sets.pickle', 'rb') as file:
                noise_samples = pickle.load(file)
            
            if args.verbose_level >= 1:
                print(f"  Loaded {len(noise_samples)} noise samples")
            
            ptsbe_env_var_set = False
            if args.smart_opt_off in ['ptsbe', 'both']:
                os.environ['CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_CACHE_REUSE_NRUNS'] = '1000000'
                os.environ['CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_SMART_OPTION'] = '0'
                ptsbe_env_var_set = True
            
            # Print contraction estimate (before timing starts)
            if args.verbose_level >= 1:
                estimate_contractions_batched_qubitShotFlex(
                    args.nqubits, max_free_qubits, len(noise_samples), args.ptsbe_nshots, 
                    proportional_sampling=args.proportional_sampling, verbose=True,
                    shots_per_batch=shots_per_batch_list,
                    qubits_per_batch=qubits_per_batch_list
                )
            
            # Run PTSBE: with callback (default) to avoid storing full shots list; use --no-mem-opt to get full list
            ptsbe_agg = None
            if not args.no_mem_opt:
                ptsbe_agg = {
                    'recombined': Counter(),
                    'total_shots': 0,
                    'per_error_unique': 0,
                }
                if args.joint_probability:
                    ptsbe_agg['joint_weights'] = []  # list of per-error weight dicts; use as needed later

                def _ptsbe_per_error_cb(_index, presampled_shots, presampled_weights):
                    ptsbe_agg['recombined'].update(presampled_shots)
                    ptsbe_agg['total_shots'] += sum(presampled_shots.values())
                    ptsbe_agg['per_error_unique'] += len(presampled_shots)
                    if presampled_weights is not None and 'joint_weights' in ptsbe_agg:
                        ptsbe_agg['joint_weights'].append(presampled_weights)

            t_ptsbe_start = time.perf_counter()
            ptsbe_result = get_noisy_shots_batched_qubitShotFlex(
                args.circuit_filename,
                noise_samples,
                args.nqubits,
                max_free_qubits,
                args.ptsbe_nshots,
                dtype,
                proportional_sampling=args.proportional_sampling,
                full_rdm=args.full_rdm,
                enable_profiling=True,
                verbose_level=args.verbose_level,
                max_contractions=args.max_contractions,
                num_hyper_samples=args.num_hyper_samples,
                lightcone=args.lightcone,
                shots_per_batch=shots_per_batch_list,
                qubits_per_batch=qubits_per_batch_list,
                count_degenerate_shots=not args.no_count_degenerate_shots,
                unique_sampling=args.unique_sampling,
                take_all_final=args.take_all_final,
                return_joint_probability=args.joint_probability,
                per_error_callback=None if args.no_mem_opt else _ptsbe_per_error_cb,
                sample_timeout=args.ptsbe_sample_timeout,
            )
            t_ptsbe_end = time.perf_counter()
            if args.joint_probability:
                shots, joint_weights, internal_stats = ptsbe_result
            else:
                shots, internal_stats = ptsbe_result
            
            # Restore environment if we modified it
            if ptsbe_env_var_set:
                del os.environ['CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_CACHE_REUSE_NRUNS']
                del os.environ['CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_SMART_OPTION']
            
            results['ptsbe']['time_execution'] = t_ptsbe_end - t_ptsbe_start
            

            if internal_stats:
                results['ptsbe'].update(internal_stats)

            if shots is None and ptsbe_agg is not None:
                results['ptsbe']['num_distinct_shots'] = len(ptsbe_agg['recombined'])
                results['ptsbe']['num_total_shots'] = ptsbe_agg['total_shots']
                results['ptsbe']['num_unique_per_error'] = ptsbe_agg['per_error_unique']
                # Clear aggregation so shot data is not retained across circuits (avoids accumulation)
                ptsbe_agg['recombined'].clear()
                if 'joint_weights' in ptsbe_agg:
                    ptsbe_agg['joint_weights'].clear()
            else:
                recombined_shots, total_shots, per_error_unique_shots = Counter({}), 0, 0
                for error in shots:
                    per_error_unique_shots += len(error.values())
                    recombined_shots = recombined_shots + Counter(error)
                    total_shots += sum(error.values())
                results['ptsbe']['num_distinct_shots'] = len(recombined_shots)
                results['ptsbe']['num_total_shots'] = total_shots
                results['ptsbe']['num_unique_per_error'] = per_error_unique_shots
            
            if args.verbose_level >= 1:
                print(f"\n{COLOR1}PTSBE Results:{RESET}")
                print(f"  Total number of distinct PTSBE shots collected:       {results['ptsbe']['num_distinct_shots']}")
                print(f"  Total number of overall PTSBE shots collected:          {results['ptsbe']['num_total_shots']}")
                print(f"  Total number of distinct PTSBE pieces of labeled training data collected:     {results['ptsbe']['num_unique_per_error']}")
        

        # Check for timeouts or OOM before comparison
        cudaq_timed_out = results['cudaq'].get('timed_out', False)
        ptsbe_timed_out = results['ptsbe'].get('num_timed_out_samples', 0) > 0
        ptsbe_oom = results['ptsbe'].get('oom_error', False)

        if cudaq_timed_out or ptsbe_timed_out or ptsbe_oom:
            skip_reasons = []
            if cudaq_timed_out:
                skip_reasons.append(f"CUDA-Q timed out ({results['cudaq'].get('error', '')})")
            if ptsbe_timed_out:
                skip_reasons.append(f"PTSBE sample timed out (limit={args.ptsbe_sample_timeout}s)")
            if ptsbe_oom:
                skip_reasons.append("PTSBE ran out of GPU memory")
            reason_str = '; '.join(skip_reasons)
            results['comparison']['skipped'] = True
            results['comparison']['skip_reason'] = reason_str
            if args.verbose_level >= 1:
                print(f"\n{BOLD_YELLOW}[SKIPPED] Circuit {circuit_idx + 1} comparison skipped: {reason_str}{RESET}")
                print(f"  Moving on to the next circuit...", flush=True)
            all_circuit_results.append(results)
            gc.collect()
            try:
                import cupy as cp
                cp.get_default_memory_pool().free_all_blocks()
                cp.get_default_pinned_memory_pool().free_all_blocks()
            except Exception:
                pass
            continue

        # PHASE 4: PER-CIRCUIT COMPARISON
        if args.include_total_time:
            cudaq_total = results['cudaq'].get('time_total')
            ptsbe_total = results['ptsbe'].get('time_execution')
            if not args.skip_cudaq and cudaq_total is None:
                raise ValueError("CUDA-Q time_total not found (required when --include_total_time is set).")
            if not args.skip_ptsbe and ptsbe_total is None:
                raise ValueError("PTSBE time_execution not found (required when --include_total_time is set).")
        else:
            cudaq_total = results['cudaq'].get('time_sample')
            ptsbe_total = results['ptsbe'].get('time_contraction_loop')
            if not args.skip_cudaq and cudaq_total is None:
                raise ValueError("CUDA-Q time_sample not found (required when --include_total_time is not set).")
            if not args.skip_ptsbe and ptsbe_total is None:
                raise ValueError("PTSBE time_contraction_loop not found (required when --include_total_time is not set).")

        cudaq_total = cudaq_total or 0
        ptsbe_total = ptsbe_total or 0

        if cudaq_total > 0 and ptsbe_total > 0:
            speedup = cudaq_total / ptsbe_total
            results['comparison']['speedup'] = speedup
            results['comparison']['cudaq_total'] = cudaq_total
            results['comparison']['ptsbe_total'] = ptsbe_total
            
            cudaq_total_shots = results['cudaq'].get('num_total_shots', 0)
            ptsbe_total_shots = results['ptsbe'].get('num_total_shots', 0)

            cudaq_unique_total_shots = results['cudaq'].get('num_distinct_shots', 0)
            ptsbe_unique_total_shots = results['ptsbe'].get('num_distinct_shots', 0)
            
            if cudaq_total_shots > 0 and ptsbe_total_shots > 0:
                cudaq_shots_per_second = cudaq_total_shots / cudaq_total
                ptsbe_shots_per_second = ptsbe_total_shots / ptsbe_total
                throughput_advantage = ptsbe_shots_per_second / cudaq_shots_per_second

                cudaq_unique_shots_per_second = cudaq_unique_total_shots / cudaq_total
                ptsbe_unique_shots_per_second = ptsbe_unique_total_shots / ptsbe_total
                unique_throughput_advantage = ptsbe_unique_shots_per_second / cudaq_unique_shots_per_second

                results['comparison']['unique_throughput_advantage'] = unique_throughput_advantage
                results['comparison']['cudaq_unique_shots_per_second'] = cudaq_unique_shots_per_second
                results['comparison']['ptsbe_unique_shots_per_second'] = ptsbe_unique_shots_per_second
                results['comparison']['throughput_advantage'] = throughput_advantage
                results['comparison']['cudaq_shots_per_second'] = cudaq_shots_per_second
                results['comparison']['ptsbe_shots_per_second'] = ptsbe_shots_per_second

        if ptsbe_total > 0 and args.skip_cudaq:
            ptsbe_total_shots = results['ptsbe'].get('num_total_shots', 0)
            ptsbe_unique_total_shots = results['ptsbe'].get('num_distinct_shots', 0)
            results['comparison']['ptsbe_total'] = ptsbe_total
            if ptsbe_total_shots > 0:
                results['comparison']['ptsbe_shots_per_second'] = ptsbe_total_shots / ptsbe_total
            if ptsbe_unique_total_shots > 0:
                results['comparison']['ptsbe_unique_shots_per_second'] = ptsbe_unique_total_shots / ptsbe_total

        if cudaq_total > 0 and args.skip_ptsbe:
            cudaq_total_shots = results['cudaq'].get('num_total_shots', 0)
            cudaq_unique_total_shots = results['cudaq'].get('num_distinct_shots', 0)
            results['comparison']['cudaq_total'] = cudaq_total
            if cudaq_total_shots > 0:
                results['comparison']['cudaq_shots_per_second'] = cudaq_total_shots / cudaq_total
            if cudaq_unique_total_shots > 0:
                results['comparison']['cudaq_unique_shots_per_second'] = cudaq_unique_total_shots / cudaq_total

        if args.verbose_level >= 1:
            circuit_timing_note = " (CUDA-Q: sample time; PTSBE: contraction loop only.)" if not args.include_total_time else ""
            print(f"\n{COLOR1}Circuit {circuit_idx + 1} Summary:{RESET}{circuit_timing_note}")
            if not args.skip_cudaq:
                print(f"  CUDA-Q total: {cudaq_total:.4f}s")
            if not args.skip_ptsbe:
                print(f"  PTSBE total: {ptsbe_total:.4f}s")
            if cudaq_total > 0 and not args.skip_cudaq:
                cudaq_shots_per_second = results['comparison'].get('cudaq_shots_per_second', 0)
                cudaq_unique_shots_per_second = results['comparison'].get('cudaq_unique_shots_per_second', 0)
                print(f"  CUDA-Q shots/s: {cudaq_shots_per_second:.2f}")
                print(f"  CUDA-Q unique shots/s: {cudaq_unique_shots_per_second:.2f}")
            if ptsbe_total > 0 and not args.skip_ptsbe:
                ptsbe_shots_per_second = results['comparison'].get('ptsbe_shots_per_second', 0)
                ptsbe_unique_shots_per_second = results['comparison'].get('ptsbe_unique_shots_per_second', 0)
                print(f"  PTSBE shots/s: {ptsbe_shots_per_second:.2f}")
                print(f"  PTSBE unique shots/s: {ptsbe_unique_shots_per_second:.2f}")
                print(f"  num_contractions: {results['ptsbe'].get('num_contractions', 0)}")
            if cudaq_total > 0 and ptsbe_total > 0:
                print(f"  Speedup: {results['comparison'].get('speedup', 0):.2f}x")
                print(f"  Throughput advantage: {results['comparison'].get('throughput_advantage', 0):.2f}x")
                print(f"  Unique throughput advantage: {results['comparison'].get('unique_throughput_advantage', 0):.2f}x")
        
        # Store this circuit's results
        all_circuit_results.append(results)

        # Reclaim GPU memory after each circuit so the next circuit starts with a clean pool
        # (avoids accumulation / fragmentation across circuits that can cause OOM on later circuits)
        gc.collect()
        try:
            import cupy as cp
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
        except Exception:
            pass

    # END OF CIRCUIT LOOP
    
    # PHASE 5: COMPUTE AND DISPLAY AVERAGED RESULTS
    if args.verbose_level >= 1:
        print(f"\n{'='*70}")
        print(f"{COLOR3}AVERAGED RESULTS (N={args.num_circuits} circuits){RESET}")
        print(f"{'='*70}")
    
    # Compute averages
    def compute_averages(all_results, key_path):
        """Extract values from nested dict path and compute mean/std."""
        values = []
        for result in all_results:
            obj = result
            for key in key_path:
                obj = obj.get(key, {}) if isinstance(obj, dict) else None
                if obj is None:
                    break
            if obj is not None and isinstance(obj, (int, float)):
                values.append(obj)
        if values:
            return {'mean': np.mean(values), 'std': np.std(values), 'values': values}
        return None
    
    # Key metrics to average
    averaged_results = {
        'config': all_circuit_results[0]['config'] if all_circuit_results else {},
        'num_circuits': args.num_circuits,
        'cudaq': {},
        'ptsbe': {},
        'comparison': {},
        'individual_results': all_circuit_results
    }
    
    # CUDA-Q metrics
    cudaq_metrics = ['time_total', 'time_setup', 'time_sample', 'num_total_shots', 'num_distinct_shots']
    for metric in cudaq_metrics:
        avg = compute_averages(all_circuit_results, ['cudaq', metric])
        if avg:
            averaged_results['cudaq'][f'{metric}_mean'] = avg['mean']
            averaged_results['cudaq'][f'{metric}_std'] = avg['std']
    
    # PTSBE metrics
    ptsbe_metrics = ['time_execution', 'time_load_circuit', 'time_circuit_to_einsum', 
                     'time_build_expr_operands', 'time_contraction_loop', 
                     'time_contract_gpu_total', 'time_apply_errors_total',
                     'num_total_shots', 'num_distinct_shots', 'num_unique_per_error', 'num_contractions']
    for metric in ptsbe_metrics:
        avg = compute_averages(all_circuit_results, ['ptsbe', metric])
        if avg:
            averaged_results['ptsbe'][f'{metric}_mean'] = avg['mean']
            averaged_results['ptsbe'][f'{metric}_std'] = avg['std']

    ptsbe_num_batches = 0
    if all_circuit_results:
        ptsbe_num_batches = all_circuit_results[0].get('ptsbe', {}).get('num_batches', 0)
    for b in range(ptsbe_num_batches):
        for prefix in ['time_contract_batch_', 'gpu_contract_batch_', 'num_contractions_batch_']:
            metric = f'{prefix}{b}'
            avg = compute_averages(all_circuit_results, ['ptsbe', metric])
            if avg:
                averaged_results['ptsbe'][f'{metric}_mean'] = avg['mean']
                averaged_results['ptsbe'][f'{metric}_std'] = avg['std']
    
    # Comparison metrics
    comparison_metrics = ['speedup', 'throughput_advantage', 'cudaq_shots_per_second', 'ptsbe_shots_per_second', 'cudaq_unique_shots_per_second', 'ptsbe_unique_shots_per_second', 'unique_throughput_advantage']
    for metric in comparison_metrics:
        avg = compute_averages(all_circuit_results, ['comparison', metric])
        if avg:
            averaged_results['comparison'][f'{metric}_mean'] = avg['mean']
            averaged_results['comparison'][f'{metric}_std'] = avg['std']
    
    # Print averaged results
    if args.verbose_level >= 1:
        if args.include_total_time:
            cudaq_time_mean = averaged_results['cudaq'].get('time_total_mean', 0)
            cudaq_time_std = averaged_results['cudaq'].get('time_total_std', 0)
            ptsbe_time_mean = averaged_results['ptsbe'].get('time_execution_mean', 0)
            ptsbe_time_std = averaged_results['ptsbe'].get('time_execution_std', 0)
        else:
            cudaq_time_mean = averaged_results['cudaq'].get('time_sample_mean', 0)
            cudaq_time_std = averaged_results['cudaq'].get('time_sample_std', 0)
            ptsbe_time_mean = averaged_results['ptsbe'].get('time_contraction_loop_mean', 0)
            ptsbe_time_std = averaged_results['ptsbe'].get('time_contraction_loop_std', 0)
        
        timing_note = " (CUDA-Q: sample time; PTSBE: contraction loop only.)" if not args.include_total_time else ""
        print(f"\n{COLOR1}Timing Comparison (Averaged) (Note this makes sense only when proportional_sampling = true and nnoise_samples = 1):{RESET}{timing_note}")
        print(f"  CUDA-Q total:         {cudaq_time_mean:>10.4f}s +/- {cudaq_time_std:.4f}s")
        print(f"  PTSBE total:          {ptsbe_time_mean:>10.4f}s +/- {ptsbe_time_std:.4f}s")
        print(f"  ─────────────────────────────────")
        
        speedup_mean = averaged_results['comparison'].get('speedup_mean', 0)
        speedup_std = averaged_results['comparison'].get('speedup_std', 0)
        if speedup_mean > 1:
            print(f"  {COLOR1}SPEEDUP:              {speedup_mean:>10.2f}x +/- {speedup_std:.2f}x (PTSBE faster){RESET}")
        else:
            print(f"  {COLOR2}SPEEDUP:              {speedup_mean:>10.2f}x +/- {speedup_std:.2f}x (CUDA-Q faster){RESET}")
        
        throughput_mean = averaged_results['comparison'].get('throughput_advantage_mean', 0)
        throughput_std = averaged_results['comparison'].get('throughput_advantage_std', 0)
        print(f"\n{COLOR1}Throughput Comparison (Averaged):{RESET}{timing_note}")
        cudaq_shots_mean = averaged_results['comparison'].get('cudaq_shots_per_second_mean', 0)
        cudaq_shots_std = averaged_results['comparison'].get('cudaq_shots_per_second_std', 0)
        ptsbe_shots_mean = averaged_results['comparison'].get('ptsbe_shots_per_second_mean', 0)
        ptsbe_shots_std = averaged_results['comparison'].get('ptsbe_shots_per_second_std', 0)
        print(f"  CUDA-Q shots/s:       {cudaq_shots_mean:>10.2f} +/- {cudaq_shots_std:.2f}")
        print(f"  PTSBE shots/s:        {ptsbe_shots_mean:>10.2f} +/- {ptsbe_shots_std:.2f}")
        print(f"  ─────────────────────────────────")
        if throughput_mean > 1:
            print(f"  {COLOR1}THROUGHPUT ADVANTAGE: {throughput_mean:>10.2f}x +/- {throughput_std:.2f}x (PTSBE faster){RESET}")
        else:
            print(f"  {COLOR2}THROUGHPUT ADVANTAGE: {throughput_mean:>10.2f}x +/- {throughput_std:.2f}x (CUDA-Q faster){RESET}")

        unique_throughput_mean = averaged_results['comparison'].get('unique_throughput_advantage_mean', 0)
        unique_throughput_std = averaged_results['comparison'].get('unique_throughput_advantage_std', 0)
        print(f"\n{COLOR1}Unique Throughput Comparison (Averaged):{RESET}{timing_note}")
        cudaq_unique_shots_mean = averaged_results['comparison'].get('cudaq_unique_shots_per_second_mean', 0)
        cudaq_unique_shots_std = averaged_results['comparison'].get('cudaq_unique_shots_per_second_std', 0)
        ptsbe_unique_shots_mean = averaged_results['comparison'].get('ptsbe_unique_shots_per_second_mean', 0)
        ptsbe_unique_shots_std = averaged_results['comparison'].get('ptsbe_unique_shots_per_second_std', 0)
        print(f"  CUDA-Q unique shots/s: {cudaq_unique_shots_mean:>10.2f} +/- {cudaq_unique_shots_std:.2f}")
        print(f"  PTSBE unique shots/s: {ptsbe_unique_shots_mean:>10.2f} +/- {ptsbe_unique_shots_std:.2f}")
        print(f"  ─────────────────────────────────")
        if unique_throughput_mean > 1:
            print(f"  {COLOR1}UNIQUE THROUGHPUT ADVANTAGE: {unique_throughput_mean:>10.2f}x +/- {unique_throughput_std:.2f}x (PTSBE faster){RESET}")
        else:
            print(f"  {COLOR2}UNIQUE THROUGHPUT ADVANTAGE: {unique_throughput_mean:>10.2f}x +/- {unique_throughput_std:.2f}x (CUDA-Q faster){RESET}")

        # Detailed timing breakdown (averaged)
        if not args.skip_cudaq and not args.skip_ptsbe:
            print(f"\n{COLOR1}Detailed Timing Breakdown (Averaged):{RESET}")
            print(f"\n  {'CUDA-Q':<35} {'PTSBE':<35}")
            print(f"  {'-'*35} {'-'*35}")
            
            cudaq_setup_mean = averaged_results['cudaq'].get('time_setup_mean', 0)
            cudaq_sample_mean = averaged_results['cudaq'].get('time_sample_mean', 0)
            
            ptsbe_load_mean = averaged_results['ptsbe'].get('time_load_circuit_mean', 0)
            ptsbe_convert_mean = averaged_results['ptsbe'].get('time_circuit_to_einsum_mean', 0)
            ptsbe_build_mean = averaged_results['ptsbe'].get('time_build_expr_operands_mean', 0)
            ptsbe_contract_mean = averaged_results['ptsbe'].get('time_contraction_loop_mean', 0)
            ptsbe_contract_gpu_mean = averaged_results['ptsbe'].get('time_contract_gpu_total_mean', 0)
            ptsbe_apply_errors_mean = averaged_results['ptsbe'].get('time_apply_errors_total_mean', 0)
            ptsbe_num_contractions_mean = averaged_results['ptsbe'].get('num_contractions_mean', 0)
            ptsbe_num_contractions_std = averaged_results['ptsbe'].get('num_contractions_std', 0)
            
            print(f"  {'Setup:':<27} {cudaq_setup_mean:>6.4f}s" + f"{'   Load+Convert:':<35}       {ptsbe_load_mean + ptsbe_convert_mean:>6.4f}s")
            print(f"  {'Sample:':<27} {cudaq_sample_mean:>6.4f}s" + f"{' ' * 2} {'Build path:':<37}  {ptsbe_build_mean:>6.4f}s")
            print(f"  {'':<27} {'':>6}    {'Contraction loop:':<35}    {ptsbe_contract_mean:>6.4f}s")
            print(f"  {'':<27} {'':>6}    {'  (contract + apply errors):':<37}  {ptsbe_contract_gpu_mean:>6.4f}s + {ptsbe_apply_errors_mean:>6.4f}s")
            print(f"  {'':<27} {'':>6}    {'Num contractions:':<35}    {ptsbe_num_contractions_mean:>6.0f} +/- {ptsbe_num_contractions_std:.0f}")

            if ptsbe_num_batches > 0:
                print(f"\n{COLOR1}  Per-Batch Contraction Breakdown (Averaged across {args.num_circuits} circuits):{RESET}")
                print(f"  {'Batch':<8} {'CPU time (s)':<20} {'GPU time (s)':<20} {'Contractions':<20} {'% of loop'}")
                print(f"  {'-'*8} {'-'*20} {'-'*20} {'-'*20} {'-'*12}")
                for b in range(ptsbe_num_batches):
                    cpu_mean = averaged_results['ptsbe'].get(f'time_contract_batch_{b}_mean', 0)
                    cpu_std = averaged_results['ptsbe'].get(f'time_contract_batch_{b}_std', 0)
                    gpu_mean = averaged_results['ptsbe'].get(f'gpu_contract_batch_{b}_mean', 0)
                    gpu_std = averaged_results['ptsbe'].get(f'gpu_contract_batch_{b}_std', 0)
                    nc_mean = averaged_results['ptsbe'].get(f'num_contractions_batch_{b}_mean', 0)
                    nc_std = averaged_results['ptsbe'].get(f'num_contractions_batch_{b}_std', 0)
                    pct = (cpu_mean / ptsbe_contract_mean * 100) if ptsbe_contract_mean > 0 else 0
                    print(f"  {b:<8} {cpu_mean:>7.4f} +/- {cpu_std:<7.4f}  {gpu_mean:>7.4f} +/- {gpu_std:<7.4f}  {nc_mean:>6.0f} +/- {nc_std:<6.0f}  {pct:>5.1f}%")
    
    print(f"\n{'='*70}\n")
    
    # Save results
    if args.json_output:
        def make_serializable(obj):
            if isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [make_serializable(v) for v in obj]
            elif isinstance(obj, (int, float, str, bool, type(None))):
                return obj
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return str(obj)
        
        with open(args.json_output, 'w') as f:
            json.dump(make_serializable(averaged_results), f, indent=2)
        
        if args.verbose_level >= 1:
            print(f"{COLOR1}Results saved to: {args.json_output}{RESET}\n")
    
    return averaged_results


if __name__ == '__main__':
    main()
