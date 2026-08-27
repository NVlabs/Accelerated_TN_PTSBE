#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="${PTSBE_IMAGE:-ptsbe-public:local}"

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "Building $IMAGE..."
  docker build -t "$IMAGE" "$REPO_ROOT"
fi

TTY_ARGS=()
if [[ -t 0 && -t 1 ]]; then
  TTY_ARGS=(-it)
fi

if [[ $# -eq 0 ]]; then
  set -- /bin/bash
fi

docker run --rm \
  "${TTY_ARGS[@]}" \
  --gpus all \
  --entrypoint /usr/bin/env \
  --user "$(id -u):$(id -g)" \
  -e HOME=/tmp \
  -e XDG_CONFIG_HOME=/tmp \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -v "$REPO_ROOT:/workspace/ptsbe" \
  -w /workspace/ptsbe \
  "$IMAGE" \
  "$@"

