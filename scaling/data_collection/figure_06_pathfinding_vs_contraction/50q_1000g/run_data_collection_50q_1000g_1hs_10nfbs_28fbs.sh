#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Get script directory (POSIX compatible)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Source shared public container configuration
. "$SCRIPT_DIR/../../../container_config.sh"

# Start container (uses NV_GPU env var if set, otherwise 'all')
start_container "${NV_GPU:-all}"

echo "Running experiment..."
docker exec $CONTAINER_ID bash -l -c '
  cd /workspace/ptsbe/scaling/data_collection/figure_06_pathfinding_vs_contraction/50q_1000g/python/
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
  PYTHONUNBUFFERED=1 python -u run_data_collection_50q_1000g_1hs_10nfbs_28fbs.py
'

cleanup_container
