# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
rm *.pickle
rm *output.py
rm pts.py
rm *.qpy
rm plot_cudaq.py
rm utils_circuit.py

cp ../stim_to_cudaq.py $PWD
cp ../stim_to_pts.py $PWD
cp ../pts.py $PWD
cp ../plot_cudaq.py $PWD
cp ../utils_circuit.py $PWD

python time_batched_ptsbe.py
