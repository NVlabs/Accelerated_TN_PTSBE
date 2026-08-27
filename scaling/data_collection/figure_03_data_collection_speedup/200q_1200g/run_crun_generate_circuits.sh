#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

nvidia-smi
cd "$SCRIPT_DIR"
sh "$SCRIPT_DIR/generate_circuits_200q_1200g.sh" >> "$SCRIPT_DIR/generate_circuits.log" 2>&1
