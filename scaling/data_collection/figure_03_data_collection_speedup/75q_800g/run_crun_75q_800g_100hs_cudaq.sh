#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

nvidia-smi
export CIRCUIT_ID=2
cd "$SCRIPT_DIR"
sh "$SCRIPT_DIR/run_data_collection_75q_800g_100hs_cudaq.sh" >> "$SCRIPT_DIR/../75q_800g_100hs_cudaq.txt" 2>&1
