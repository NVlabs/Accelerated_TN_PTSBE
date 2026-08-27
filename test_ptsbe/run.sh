# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
rm *.png
rm *.pickle
rm *output.py
rm pts.py
rm shot_sets/*
rm *.qpy
rm plot_cudaq.py
rm utils_circuit.py

cp ../stim_to_pts.py $PWD
cp ../stim_to_be.py $PWD
cp ../pts.py $PWD
cp ../plot_cudaq.py $PWD
cp ../utils_circuit.py $PWD

python test_ptsbe.py
