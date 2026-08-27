# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Utility functions for verifying tensor network results against reference implementations.
"""
import numpy as np
import os
from collections import Counter


def load_cudaq_shots(directory='shot_sets', nqubits=7):
    shot_strings = []
    files = os.listdir(directory)
    
    for filename in files:
        shots = np.load(os.path.join(directory, filename), allow_pickle=True).item()
        for key, val in shots.items():
            key_str = str(key)
            key_str = '0' * (nqubits - len(key_str)) + key_str
            shot_strings.extend([key_str] * val)
    
    return shot_strings


def compute_tvd(shots1, shots2):
    ### TVD = 0.5 * sum(|p1(x) - p2(x)|) for all bitstrings x
    counts1 = Counter(shots1)
    counts2 = Counter(shots2)
    
    total1 = sum(counts1.values())
    total2 = sum(counts2.values())
    
    if total1 == 0 or total2 == 0:
        return 1.0 
    
    all_keys = set(counts1.keys()) | set(counts2.keys())
    
    tvd = 0.0
    for key in all_keys:
        p1 = counts1.get(key, 0) / total1
        p2 = counts2.get(key, 0) / total2
        tvd += abs(p1 - p2)
    
    return tvd / 2.0


def verify_distributions(tn_shots, cudaq_shots, threshold=0.15, verbose=True):
    tvd = compute_tvd(tn_shots, cudaq_shots)
    passed = tvd < threshold
    
    if verbose:
        print(f"\n{'='*50}")
        print(f"VERIFICATION RESULTS")
        print(f"{'='*50}")
        print(f"Tensor Network shots: {len(tn_shots)}")
        print(f"CUDA-Q reference shots: {len(cudaq_shots)}")
        print(f"Total Variation Distance: {tvd:.4f}")
        print(f"Threshold: {threshold}")
        if passed:
            print(f"PASS: Distributions match within threshold")
        else:
            print(f"FAIL: Distributions differ beyond threshold")
        print(f"{'='*50}\n")
    
    return passed, tvd
