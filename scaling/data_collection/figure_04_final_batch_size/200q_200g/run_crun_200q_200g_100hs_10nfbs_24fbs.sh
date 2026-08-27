#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

nvidia-smi
export CIRCUIT_ID=2
cd "$SCRIPT_DIR"
sh "$SCRIPT_DIR/run_data_collection_200q_200g_100hs_10nfbs_24fbs.sh" >> "$SCRIPT_DIR/../200q_200g_100hs_10nfbs_24fbs_ptsbe.txt" 2>&1
