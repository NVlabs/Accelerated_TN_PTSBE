#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Proportional3: 100q/600g, bs=10 (uniform), ptsbe_nshots=10000, nnoise_samples=1.
"""

import subprocess
import json
import os
import sys
import argparse
import time

print("run_data_collection (proportional3) started", flush=True)
try:
    _nvidia = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
    print(_nvidia.stdout, flush=True)
    if _nvidia.returncode != 0 and _nvidia.stderr:
        print(_nvidia.stderr, flush=True)
except Exception as e:
    print(f"nvidia-smi failed: {e}", flush=True)

SHARED_ARGS = [
    '--num_circuits', '1',
    '--nnoise_samples', '1',
    '--cudaq_nshots', '1',
    '--smart-opt-off', 'both',
    '--verbose_level', '3',
    '--proportional_sampling',
    '--batch_qubit_policy', 'custom',
    '--batch_shots_policy', 'uniform',
    '--cudaq_timeout', '4500',
    '--ptsbe_sample_timeout', '50000',
    '--skip_cudaq',
]

RUNS = [
    {'nqubits': 100, 'ngates': 600, 'num_hyper_samples': 100,
     'ptsbe_nshots': 10000, 'qubits_per_batch': '10,10,10,10,10,10,10,10,10,10'},
]


def run_single(run_index, params, output_dir, scaling_script, scaling_dir,
               project_root, circuit_dir, circuit_id, dry_run=False):
    run_label = run_index
    output_file = os.path.abspath(
        os.path.join(output_dir, f'run_{run_label}_circuit_{circuit_id}.json'))
    work_dir = os.path.abspath(
        os.path.join(output_dir, f'work_circuit_{circuit_id}'))
    os.makedirs(work_dir, exist_ok=True)

    cmd = (
        [sys.executable, scaling_script]
        + SHARED_ARGS
        + ['--nqubits', str(params['nqubits']),
           '--ngates', str(params['ngates']),
           '--num_hyper_samples', str(params['num_hyper_samples']),
           '--ptsbe_nshots', str(params['ptsbe_nshots']),
           '--qubits_per_batch', params['qubits_per_batch'],
           '--circuit_dir', circuit_dir,
           '--circuit_id', str(circuit_id),
           '--work_dir', work_dir,
           '--json_output', output_file]
    )

    print(f"\n{'='*70}")
    total = len(RUNS)
    print(f"RUN {run_label}/{total} | circuit_id={circuit_id}: "
          f"nqubits={params['nqubits']} ngates={params['ngates']} "
          f"ptsbe_nshots={params['ptsbe_nshots']}")
    print(f"  qubits_per_batch={params['qubits_per_batch']}")
    print(f"{'='*70}")
    print(f"Command: {' '.join(cmd)}")
    print(f"Output: {output_file}")

    if dry_run:
        print("[DRY RUN] Skipping execution")
        return None

    env = os.environ.copy()
    env['PYTHONPATH'] = project_root + os.pathsep + env.get('PYTHONPATH', '')
    start = time.time()
    result = subprocess.run(cmd, cwd=scaling_dir, env=env, capture_output=False, text=True)
    elapsed = time.time() - start
    print(f"Completed in {elapsed:.1f}s (returncode={result.returncode})")

    if result.returncode == 0 and os.path.exists(output_file):
        with open(output_file, 'r') as f:
            data = json.load(f)
        data['_run_label'] = run_label
        data['_params'] = params
        data['_circuit_id'] = circuit_id
        return data
    return None


def run_all(output_dir, circuit_dir, dry_run=False):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    scaling_dir = os.path.abspath(os.path.join(script_dir, '..', '..', '..', '..'))
    project_root = os.path.dirname(scaling_dir)
    scaling_script = os.path.join(scaling_dir, 'scaling_comparison_avg.py')

    circuit_dir = os.path.abspath(circuit_dir)
    config_path = os.path.join(circuit_dir, 'config.json')
    if os.path.isfile(config_path):
        with open(config_path, 'r') as f:
            gen_cfg = json.load(f)
        num_circuits = gen_cfg['num_circuits']
        print(f"Auto-detected {num_circuits} circuits from {config_path}")
    else:
        import glob as _glob
        num_circuits = len(_glob.glob(os.path.join(circuit_dir, 'circuit_*.qpy')))
        if num_circuits == 0:
            raise FileNotFoundError(f"No circuit_*.qpy files found in {circuit_dir}")
        print(f"No config.json found; detected {num_circuits} circuit files in {circuit_dir}")

    if num_circuits > 10:
        print(f"Capping num_circuits from {num_circuits} to 10")
        num_circuits = 10

    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Scaling script: {scaling_script}")
    print(f"Working dir:    {scaling_dir}")
    print(f"Output dir:     {output_dir}")
    print(f"Circuit dir:    {circuit_dir}")
    print(f"Num circuits:   {num_circuits}")
    print(f"Total runs:     {len(RUNS)} config(s) x {num_circuits} circuit(s) = {len(RUNS) * num_circuits}")

    all_results = []
    for circuit_id in range(num_circuits):
        for i, params in enumerate(RUNS):
            run_index = i + 1
            result = run_single(
                run_index, params, output_dir, scaling_script, scaling_dir,
                project_root, circuit_dir, circuit_id, dry_run=dry_run
            )
            if result is not None:
                all_results.append(result)

    combined_path = os.path.join(output_dir, 'benchmark_combined.json')
    if not dry_run and all_results:
        with open(combined_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\nCombined results: {combined_path}")

    return all_results


def print_summary(all_results):
    if not all_results:
        print("No results to summarize.")
        return
    print(f"\n{'='*100}")
    print("PROPORTIONAL SAMPLING 3 SUMMARY")
    print(f"{'='*100}")
    print(f"{'Run':>4} | {'nq':>4} | {'ng':>4} | {'nshots':>7} | {'ptsbe_time':>12} | {'contractions':>12} | {'throughput_adv':>14}")
    print("-" * 100)
    for r in all_results:
        label = r.get('_run_label', '?')
        p = r.get('_params', {})
        nq, ng = p.get('nqubits', ''), p.get('ngates', '')
        nshots = p.get('ptsbe_nshots', '')
        ptsbe = r.get('ptsbe', {})
        comp = r.get('comparison', {})
        t = ptsbe.get('time_execution_mean')
        nc = ptsbe.get('num_contractions_mean')
        ta = comp.get('throughput_advantage_mean')
        t = f"{t:.2f}s" if isinstance(t, (int, float)) else str(t)
        nc = f"{nc:,.0f}" if isinstance(nc, (int, float)) else str(nc)
        ta = f"{ta:.2f}x" if isinstance(ta, (int, float)) else str(ta)
        print(f"{label:>4} | {nq:>4} | {ng:>4} | {nshots:>7} | {t:>12} | {nc:>12} | {ta:>14}")
    print(f"{'='*100}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Run 100q/600g proportional3 (10bs, 10000nshots)')
    default_circuit_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), 'circuits', '100q_600g')
    parser.add_argument('--circuit-dir', type=str, default=default_circuit_dir,
                        help='Directory with pre-generated circuits.')
    parser.add_argument('--dry-run', action='store_true', help='Print commands only')
    parser.add_argument('--output-dir', type=str, default='output_data_collection_1',
                        help='Output directory for JSON results')
    parser.add_argument('--summary-only', type=str, default=None,
                        help='Print summary from existing benchmark_combined.json')
    args = parser.parse_args()

    if args.summary_only:
        with open(args.summary_only, 'r') as f:
            all_results = json.load(f)
        print_summary(all_results)
        return

    all_results = run_all(args.output_dir, args.circuit_dir, dry_run=args.dry_run)
    if not args.dry_run:
        print_summary(all_results)


if __name__ == '__main__':
    main()
