#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Pre-generate random circuits for independent PTSBE / CUDA-Q runs.

Produces N circuit files (.qpy + .stim) in an output directory along with a
config.json that records the generation parameters for reproducibility.

Usage:
    python generate_circuits.py --nqubits 50 --ngates 300 --num_circuits 10 \
        --output_dir circuits_50q_300g/
"""

import sys
import os
import json
import argparse
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils_circuit import random_gate_sample, build_circuit_and_script


def main():
    parser = argparse.ArgumentParser(
        description='Pre-generate random quantum circuits (.qpy + .stim)')
    parser.add_argument('--nqubits', type=int, required=True)
    parser.add_argument('--ngates', type=int, required=True)
    parser.add_argument('--num_circuits', type=int, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--prob_range_min', type=float, default=0.02)
    parser.add_argument('--prob_range_max', type=float, default=0.2)
    parser.add_argument('--proportion_two_qubit', type=float, default=0.2)
    parser.add_argument('--local_gates', action='store_true')
    args = parser.parse_args()

    prob_range = [args.prob_range_min, args.prob_range_max]
    os.makedirs(args.output_dir, exist_ok=True)

    config = {
        'nqubits': args.nqubits,
        'ngates': args.ngates,
        'num_circuits': args.num_circuits,
        'prob_range_min': args.prob_range_min,
        'prob_range_max': args.prob_range_max,
        'proportion_two_qubit': args.proportion_two_qubit,
        'local_gates': args.local_gates,
    }

    print(f"Generating {args.num_circuits} circuits "
          f"(nqubits={args.nqubits}, ngates={args.ngates}) "
          f"into {args.output_dir}")

    t_start = time.perf_counter()
    for i in range(args.num_circuits):
        qpy_path = os.path.join(args.output_dir, f'circuit_{i}.qpy')
        stim_path = os.path.join(args.output_dir, f'circuit_{i}.stim')

        gates_sampled = random_gate_sample(
            args.nqubits,
            args.ngates,
            prob_range,
            proportion_two_qubit=args.proportion_two_qubit,
            local_gates=args.local_gates,
        )

        build_circuit_and_script(args.nqubits, gates_sampled, qpy_path, stim_path)
        print(f"  [{i+1}/{args.num_circuits}] {qpy_path}  {stim_path}")

    t_end = time.perf_counter()

    config['generation_time_s'] = round(t_end - t_start, 4)
    config_path = os.path.join(args.output_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"\nDone in {t_end - t_start:.2f}s.  Config saved to {config_path}")


if __name__ == '__main__':
    main()
