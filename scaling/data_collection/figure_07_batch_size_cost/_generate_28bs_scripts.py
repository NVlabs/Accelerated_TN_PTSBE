#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate all run scripts for nfbs=28 batch size experiments."""

import os
import stat

BASE = os.path.dirname(os.path.abspath(__file__))

CONFIGS = [
    {'label': '100q_600g', 'nqubits': 100, 'ngates': 600,
     'qubits_per_batch': '28,28,16,28'},
]

def make_shots(qpb_str, num_2shots):
    """Build shots_per_batch: last `num_2shots` non-final batches get 2, rest get 1, final=100."""
    batches = qpb_str.split(',')
    n = len(batches)
    shots = ['1'] * (n - 1) + ['100']
    for i in range(num_2shots):
        shots[n - 2 - i] = '2'
    return ','.join(shots)

PYTHON_TEMPLATE = '''#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Data collection for {nqubits}-qubit / {ngates}-gate circuits (100 hyper-samples, fbs=28).

Iterates over all pre-generated circuits in --circuit-dir, running each one
through scaling_comparison_avg.py individually.

Usage:
    python {py_name} --circuit-dir /path/to/circuits [--dry-run] [--output-dir DIR]
    python {py_name} --circuit-dir /path/to/circuits --summary-only output_dir/benchmark_combined.json
"""

import subprocess
import json
import os
import sys
import argparse
import time

print("run_data_collection_1.py started", flush=True)
try:
    _nvidia = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
    print(_nvidia.stdout, flush=True)
    if _nvidia.returncode != 0 and _nvidia.stderr:
        print(_nvidia.stderr, flush=True)
except Exception as e:
    print(f"nvidia-smi failed: {{e}}", flush=True)

# ============================================================================
# DATA COLLECTION: March4 repeat with fbs=28
# ============================================================================

SHARED_ARGS = [
    '--num_circuits', '1',
    '--nnoise_samples', '10',
    '--cudaq_nshots', '1',
    '--smart-opt-off', 'both',
    '--verbose_level', '3',
    '--unique_sampling',
    '--take_all_final',
    '--batch_shots_policy', 'custom',
    '--batch_qubit_policy', 'custom',
    '--cudaq_timeout', '4500',
    '--ptsbe_sample_timeout', '3600',
    '--skip_cudaq',
]

RUNS = [
    {{'nqubits': {nqubits}, 'ngates': {ngates}, 'num_hyper_samples': 100,
   'shots_per_batch': '{shots_per_batch}', 'qubits_per_batch': '{qubits_per_batch}'}},
]


def run_single(run_index, params, output_dir, scaling_script, scaling_dir,
               project_root, circuit_dir, circuit_id, dry_run=False):
    """Run one data-collection configuration for a single circuit. Returns result dict or None."""
    run_label = run_index
    output_file = os.path.abspath(
        os.path.join(output_dir, f'run_{{run_label}}_circuit_{{circuit_id}}.json'))
    work_dir = os.path.abspath(
        os.path.join(output_dir, f'work_circuit_{{circuit_id}}'))
    os.makedirs(work_dir, exist_ok=True)

    cmd = (
        [sys.executable, scaling_script]
        + SHARED_ARGS
        + ['--nqubits', str(params['nqubits']),
           '--ngates', str(params['ngates']),
           '--num_hyper_samples', str(params['num_hyper_samples']),
           '--shots_per_batch', params['shots_per_batch'],
           '--qubits_per_batch', params['qubits_per_batch'],
           '--circuit_dir', circuit_dir,
           '--circuit_id', str(circuit_id),
           '--work_dir', work_dir,
           '--json_output', output_file]
    )

    print(f"\\n{{'='*70}}")
    total = len(RUNS)
    print(f"RUN {{run_label}}/{{total}} | circuit_id={{circuit_id}}: "
          f"nqubits={{params['nqubits']}} ngates={{params['ngates']}} "
          f"num_hyper_samples={{params['num_hyper_samples']}}")
    print(f"  shots_per_batch={{params['shots_per_batch']}}")
    print(f"  qubits_per_batch={{params['qubits_per_batch']}}")
    print(f"{{'='*70}}")
    print(f"Command: {{' '.join(cmd)}}")
    print(f"Output: {{output_file}}")

    if dry_run:
        print("[DRY RUN] Skipping execution")
        return None

    env = os.environ.copy()
    env['PYTHONPATH'] = project_root + os.pathsep + env.get('PYTHONPATH', '')
    start = time.time()
    result = subprocess.run(cmd, cwd=scaling_dir, env=env, capture_output=False, text=True)
    elapsed = time.time() - start
    print(f"Completed in {{elapsed:.1f}}s (returncode={{result.returncode}})")

    if result.returncode == 0 and os.path.exists(output_file):
        with open(output_file, 'r') as f:
            data = json.load(f)
        data['_run_label'] = run_label
        data['_params'] = params
        data['_circuit_id'] = circuit_id
        return data
    return None


def run_all(output_dir, circuit_dir, dry_run=False):
    """Run data-collection run(s) over all pre-generated circuits. Returns list of result dicts."""
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
        print(f"Auto-detected {{num_circuits}} circuits from {{config_path}}")
    else:
        import glob as _glob
        num_circuits = len(_glob.glob(os.path.join(circuit_dir, 'circuit_*.qpy')))
        if num_circuits == 0:
            raise FileNotFoundError(f"No circuit_*.qpy files found in {{circuit_dir}}")
        print(f"No config.json found; detected {{num_circuits}} circuit files in {{circuit_dir}}")

    if num_circuits > 10:
        print(f"Capping num_circuits from {{num_circuits}} to 10")
        num_circuits = 10

    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Scaling script: {{scaling_script}}")
    print(f"Working dir:    {{scaling_dir}}")
    print(f"Output dir:     {{output_dir}}")
    print(f"Circuit dir:    {{circuit_dir}}")
    print(f"Num circuits:   {{num_circuits}}")
    print(f"Total runs:     {{len(RUNS)}} config(s) x {{num_circuits}} circuit(s) = {{len(RUNS) * num_circuits}}")

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
        print(f"\\nCombined results: {{combined_path}}")

    return all_results


def print_summary(all_results):
    """Print a short summary table of the runs."""
    if not all_results:
        print("No results to summarize.")
        return
    print(f"\\n{{'='*100}}")
    print("DATA COLLECTION 1 SUMMARY")
    print(f"{{'='*100}}")
    print(f"{{'Run':>4}} | {{'nq':>4}} | {{'ng':>4}} | {{'hyper':>6}} | {{'ptsbe_time':>12}} | {{'contractions':>12}} | {{'speedup':>10}} | {{'throughput_adv':>14}}")
    print("-" * 100)
    for r in all_results:
        label = r.get('_run_label', '?')
        p = r.get('_params', {{}})
        nq, ng, hyper = p.get('nqubits', ''), p.get('ngates', ''), p.get('num_hyper_samples', '')
        ptsbe = r.get('ptsbe', {{}})
        comp = r.get('comparison', {{}})
        t = ptsbe.get('time_execution_mean')
        nc = ptsbe.get('num_contractions_mean')
        sp = comp.get('speedup_mean')
        ta = comp.get('throughput_advantage_mean')
        t = f"{{t:.2f}}s" if isinstance(t, (int, float)) else str(t)
        nc = f"{{nc:,.0f}}" if isinstance(nc, (int, float)) else str(nc)
        sp = f"{{sp:.2f}}x" if isinstance(sp, (int, float)) else str(sp)
        ta = f"{{ta:.2f}}x" if isinstance(ta, (int, float)) else str(ta)
        print(f"{{label:>4}} | {{nq:>4}} | {{ng:>4}} | {{hyper:>6}} | {{t:>12}} | {{nc:>12}} | {{sp:>10}} | {{ta:>14}}")
    print(f"{{'='*100}}\\n")


def main():
    parser = argparse.ArgumentParser(
        description='Run {nqubits}q/{ngates}g data collection over pre-generated circuits')
    default_circuit_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), 'circuits', '{circuit_subdir}')
    parser.add_argument('--circuit-dir', type=str, default=default_circuit_dir,
                        help='Directory with pre-generated circuits (from generate_circuits.py). '
                             'Default: data_collection/circuits/{circuit_subdir}')
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
'''

SHELL_TEMPLATE = '''#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Get script directory (POSIX compatible)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Source shared public container configuration
. "$SCRIPT_DIR/../../../container_config.sh"

# Start container (uses NV_GPU env var if set, otherwise 'all')
start_container "${{NV_GPU:-all}}"

echo "Running experiment..."
docker exec $CONTAINER_ID bash -l -c '
  cd /workspace/ptsbe/scaling/data_collection/figure_07_batch_size_cost/{config_label}/python/
  echo "=== cuQuantum Environment Ready ==="
  echo "cuTensorNet version: $(python -c "from cuquantum.bindings import cutensornet as cutn; print(cutn.get_version())" 2>/dev/null)"
  
  echo ""
  echo "Setup complete!"
  echo "=== Starting Parameter Sweep ===" 
 
  rm -f *.png stim* *.pickle *output.py pts.py *.qpy utils_circuit.py
  cp ../../../../../stim_to_pts.py $PWD
  cp ../../../../../pts.py $PWD
  cp ../../../../../utils_circuit.py $PWD
  echo "Copy done, starting Python..."
  PYTHONUNBUFFERED=1 python -u {py_name}
'

cleanup_container
'''

CRUN_TEMPLATE = '''#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

SCRIPT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"

nvidia-smi
cd "$SCRIPT_DIR"
sh "$SCRIPT_DIR/{shell_name}" >> "$SCRIPT_DIR/../{output_txt}" 2>&1
'''

BS = 28
POLICIES = {'2p1': 1, '2p2': 2, '2p3': 3}

created = []

for cfg in CONFIGS:
    label = cfg['label']
    nq = cfg['nqubits']
    ng = cfg['ngates']
    qpb = cfg['qubits_per_batch']
    circuit_subdir = f"{nq}q_{ng}g"

    for pol_name, num_2 in POLICIES.items():
        spb = make_shots(qpb, num_2)
        file_tag = f"{label}_100hs_{BS}bs_{pol_name}_ptsbe"
        py_name = f"run_data_collection_{file_tag}.py"
        shell_name = f"run_data_collection_{file_tag}.sh"
        crun_name = f"run_crun_{file_tag}.sh"
        output_txt = f"{file_tag}.txt"

        py_path = os.path.join(BASE, label, 'python', py_name)
        py_content = PYTHON_TEMPLATE.format(
            nqubits=nq, ngates=ng, py_name=py_name,
            shots_per_batch=spb, qubits_per_batch=qpb,
            circuit_subdir=circuit_subdir)
        with open(py_path, 'w') as f:
            f.write(py_content)
        os.chmod(py_path, os.stat(py_path).st_mode | stat.S_IEXEC)

        sh_path = os.path.join(BASE, label, shell_name)
        sh_content = SHELL_TEMPLATE.format(config_label=label, py_name=py_name)
        with open(sh_path, 'w') as f:
            f.write(sh_content)
        os.chmod(sh_path, os.stat(sh_path).st_mode | stat.S_IEXEC)

        crun_path = os.path.join(BASE, label, crun_name)
        crun_content = CRUN_TEMPLATE.format(
            config_label=label, shell_name=shell_name, output_txt=output_txt)
        with open(crun_path, 'w') as f:
            f.write(crun_content)
        os.chmod(crun_path, os.stat(crun_path).st_mode | stat.S_IEXEC)

        created.append(file_tag)
        print(f"  Created: {py_name}, {shell_name}, {crun_name}")

print(f"\nGenerated {len(created)} configurations ({len(created)*3} files total).")
