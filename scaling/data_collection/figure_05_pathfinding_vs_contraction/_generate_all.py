#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate all run scripts for figure_05_pathfinding_vs_contraction.

Sweep: [50, 75, 100, 150, 200] qubits x [200, 400, 600, 800, 1000] gates
Fixed: hypersamples=1, nfbs=10, fbs=28, 1 noise sample, 10 repeats per circuit.
"""

import os
import stat
import textwrap

QUBITS = [50, 75, 100, 150, 200]
GATES = [200, 400, 600, 800, 1000]
HS = 1
NFBS = 10
FBS = 28
NUM_REPEATS = 10

BASE = os.path.dirname(os.path.abspath(__file__))
FIGURE_DIR = "figure_05_pathfinding_vs_contraction"


def batch_config(nqubits):
    """Compute qubits_per_batch and shots_per_batch for nfbs=10, fbs=28."""
    remaining = nqubits - FBS
    full = remaining // NFBS
    leftover = remaining % NFBS
    qpb = [NFBS] * full + ([leftover] if leftover > 0 else []) + [FBS]
    num_batches = len(qpb)
    spb = [1] * (num_batches - 1) + [100]
    return ','.join(str(x) for x in qpb), ','.join(str(x) for x in spb)


def write_py(q, g, subdir):
    qpb, spb = batch_config(q)
    tag = f"{q}q_{g}g_1hs_{NFBS}nfbs_{FBS}fbs"
    content = textwrap.dedent(f'''\
#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Data collection for {q}-qubit / {g}-gate circuits (1 hyper-sample, {NFBS}nfbs_{FBS}fbs).

Each of the 10 pre-generated circuits is run {NUM_REPEATS} times (1 noise sample each)
to allow averaging of contraction and path-finding times.
"""

import subprocess
import json
import os
import sys
import argparse
import time

print("run_data_collection started", flush=True)
try:
    _nvidia = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
    print(_nvidia.stdout, flush=True)
    if _nvidia.returncode != 0 and _nvidia.stderr:
        print(_nvidia.stderr, flush=True)
except Exception as e:
    print(f"nvidia-smi failed: {{e}}", flush=True)

SHARED_ARGS = [
    '--num_circuits', '1',
    '--nnoise_samples', '1',
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

NUM_REPEATS = {NUM_REPEATS}
RUNS = [
    {{'nqubits': {q}, 'ngates': {g}, 'num_hyper_samples': {HS},
     'shots_per_batch': '{spb}', 'qubits_per_batch': '{qpb}'}}
] * NUM_REPEATS


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
    print(f"RUN {{run_label}}/{{NUM_REPEATS}} | circuit_id={{circuit_id}}: "
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


def run_all(output_dir, circuit_dir, dry_run=False, start_circuit=0):
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

    remaining = num_circuits - start_circuit
    print(f"Scaling script: {{scaling_script}}")
    print(f"Working dir:    {{scaling_dir}}")
    print(f"Output dir:     {{output_dir}}")
    print(f"Circuit dir:    {{circuit_dir}}")
    print(f"Num circuits:   {{num_circuits}} (starting from {{start_circuit}})")
    print(f"Num repeats:    {{NUM_REPEATS}}")
    print(f"Total runs:     {{NUM_REPEATS}} repeat(s) x {{remaining}} circuit(s) = {{NUM_REPEATS * remaining}}")

    all_results = []
    for circuit_id in range(start_circuit, num_circuits):
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
    print("DATA COLLECTION SUMMARY")
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
        description='Run {q}q/{g}g data collection (1 hypersample, {NUM_REPEATS} repeats) over pre-generated circuits')
    default_circuit_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), 'circuits', '{q}q_{g}g')
    parser.add_argument('--circuit-dir', type=str, default=default_circuit_dir,
                        help='Directory with pre-generated circuits (from generate_circuits.py). '
                             'Default: data_collection/circuits/{q}q_{g}g')
    parser.add_argument('--dry-run', action='store_true', help='Print commands only')
    parser.add_argument('--output-dir', type=str, default='output_data_collection',
                        help='Output directory for JSON results')
    parser.add_argument('--start-circuit', type=int, default=0,
                        help='Circuit ID to start from (skip earlier circuits)')
    parser.add_argument('--summary-only', type=str, default=None,
                        help='Print summary from existing benchmark_combined.json')
    args = parser.parse_args()

    if args.summary_only:
        with open(args.summary_only, 'r') as f:
            all_results = json.load(f)
        print_summary(all_results)
        return

    all_results = run_all(args.output_dir, args.circuit_dir, dry_run=args.dry_run,
                          start_circuit=args.start_circuit)
    if not args.dry_run:
        print_summary(all_results)


if __name__ == '__main__':
    main()
''')
    py_dir = os.path.join(subdir, 'python')
    os.makedirs(py_dir, exist_ok=True)
    path = os.path.join(py_dir, f'run_data_collection_{tag}.py')
    with open(path, 'w') as f:
        f.write(content)
    os.chmod(path, os.stat(path).st_mode | stat.S_IXUSR)
    return path


def write_sh(q, g, subdir):
    tag = f"{q}q_{g}g_1hs_{NFBS}nfbs_{FBS}fbs"
    content = textwrap.dedent(f'''\
#!/bin/bash
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
  cd /workspace/ptsbe/scaling/data_collection/{FIGURE_DIR}/{q}q_{g}g/python/
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
  PYTHONUNBUFFERED=1 python -u run_data_collection_{tag}.py
'

cleanup_container
''')
    path = os.path.join(subdir, f'run_data_collection_{tag}.sh')
    with open(path, 'w') as f:
        f.write(content)
    os.chmod(path, os.stat(path).st_mode | stat.S_IXUSR)
    return path


def write_crun(q, g, subdir):
    tag = f"{q}q_{g}g_1hs_{NFBS}nfbs_{FBS}fbs"
    content = textwrap.dedent(f'''\
#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

SCRIPT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"

nvidia-smi
cd "$SCRIPT_DIR"
sh "$SCRIPT_DIR/run_data_collection_{tag}.sh" >> "$SCRIPT_DIR/../{tag}_ptsbe.txt" 2>&1
''')
    path = os.path.join(subdir, f'run_crun_{tag}.sh')
    with open(path, 'w') as f:
        f.write(content)
    os.chmod(path, os.stat(path).st_mode | stat.S_IXUSR)
    return path


def main():
    created = []
    for q in QUBITS:
        for g in GATES:
            subdir = os.path.join(BASE, f'{q}q_{g}g')
            os.makedirs(os.path.join(subdir, 'python'), exist_ok=True)

            py = write_py(q, g, subdir)
            sh = write_sh(q, g, subdir)
            cr = write_crun(q, g, subdir)
            created.extend([py, sh, cr])
            print(f"  [{q}q {g}g] py, sh, crun")

    print(f"\nCreated {len(created)} files ({len(QUBITS)*len(GATES)} configs)")


if __name__ == '__main__':
    main()
