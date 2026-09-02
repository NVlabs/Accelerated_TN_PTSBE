#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

nvidia-smi
cd "$SCRIPT_DIR"
sh "$SCRIPT_DIR/run_data_collection_75q_800g_1hs_10nfbs_28fbs.sh" >> "$SCRIPT_DIR/../75q_800g_1hs_10nfbs_28fbs_ptsbe.txt" 2>&1
