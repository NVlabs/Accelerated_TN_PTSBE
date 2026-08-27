# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import sys
import os
import numpy as np
import pandas
from collections import Counter
import matplotlib.pyplot as plt
from itertools import islice


nqubits = int(sys.argv[1])
directory = 'shot_sets'
use_gzip = False
use_int_strings = True
if use_gzip:
    import pickle
    import gzip

np.set_printoptions(linewidth=np.inf)

unrolled_shots = []
files = os.listdir(directory)
for filename in files:
    if use_gzip:
        with gzip.open(directory+'/'+filename, 'rb') as f:
            shots = pickle.load(f)
    else:
        shots = np.load(directory+'/'+filename, allow_pickle=True).item()
    for key, val in zip(shots.keys(), shots.values()):
        if use_int_strings:
            key = str(key)
            key = '0'*(nqubits-len(key)) + key
        for _ in range(val):
            unrolled_shots.append(np.array([[int(c) for c in key]]))
unrolled_shots = np.stack(unrolled_shots).reshape((-1, nqubits))
sample_size = unrolled_shots.shape[0]

shot_strings = [''.join([str(bit) for bit in shot]) for shot in unrolled_shots]
counts = Counter(shot_strings)
counts = counts.most_common()
counts = dict(sorted(counts))
df = pandas.DataFrame.from_dict(counts, orient='index')
df.plot(kind='bar')
plt.savefig('cudaq_histogram.png')
