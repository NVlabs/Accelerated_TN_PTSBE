#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Get script directory (POSIX compatible)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Source shared public container configuration
. "$SCRIPT_DIR/../../../container_config.sh"

# Start container (uses NV_GPU env var if set, otherwise 'all')
start_container "${NV_GPU:-all}"

echo "Generating 200q_1200g circuits..."
docker exec $CONTAINER_ID bash -l -c '
  cd /workspace/ptsbe/scaling
  python generate_circuits.py --nqubits 200 --ngates 1200 --num_circuits 10 --output_dir data_collection/circuits/200q_1200g
'

cleanup_container
