#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

nvidia-smi
cd "$SCRIPT_DIR"
sh "$SCRIPT_DIR/run_data_collection_200q_1000g_100hs_10bs_proportional_10nshots.sh" >> "$SCRIPT_DIR/../200q_1000g_100hs_10bs_proportional_10nshots_ptsbe.txt" 2>&1
