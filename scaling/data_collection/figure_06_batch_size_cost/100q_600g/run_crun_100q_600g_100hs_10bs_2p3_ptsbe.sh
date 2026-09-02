#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

nvidia-smi
cd "$SCRIPT_DIR"
sh "$SCRIPT_DIR/run_data_collection_100q_600g_100hs_10bs_2p3_ptsbe.sh" >> "$SCRIPT_DIR/../100q_600g_100hs_10bs_2p3_ptsbe.txt" 2>&1
