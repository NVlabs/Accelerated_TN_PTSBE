#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Common public container configuration for paper data-collection scripts.

CONFIG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_PATH="$(cd "$CONFIG_DIR/.." && pwd)"
CONTAINER_PROJECT_PATH="/workspace/ptsbe"
IMAGE="${PTSBE_IMAGE:-ptsbe-public:local}"

start_container() {
  local gpu_device="${1:-all}"
  local gpu_args
  local host_uid
  local host_gid

  host_uid="$(id -u)"
  host_gid="$(id -g)"

  if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "Building $IMAGE..."
    docker build -t "$IMAGE" "$PROJECT_PATH"
  fi

  if [[ "$gpu_device" == "all" ]]; then
    gpu_args=(--gpus all)
  else
    gpu_args=(--gpus "device=$gpu_device")
  fi

  echo "Starting public cuQuantum container..."
  CONTAINER_ID=$(docker run -d \
    "${gpu_args[@]}" \
    --entrypoint /usr/bin/env \
    --user "$host_uid:$host_gid" \
    -e HOME=/tmp \
    -e XDG_CONFIG_HOME=/tmp \
    -e PYTHONDONTWRITEBYTECODE=1 \
    --name "ptsbe_session_$$" \
    -v "$PROJECT_PATH:$CONTAINER_PROJECT_PATH" \
    -w "$CONTAINER_PROJECT_PATH" \
    "$IMAGE" \
    tail -f /dev/null)

  echo "Container started: $CONTAINER_ID"
  export CONTAINER_ID
}

CONTAINER_SETUP_COMMANDS='
  cd /workspace/ptsbe/scaling/
  echo "=== cuQuantum Environment Ready ==="
  echo "cuTensorNet version: $(python -c "from cuquantum.bindings import cutensornet as cutn; print(cutn.get_version())" 2>/dev/null)"
'

cleanup_container() {
  if [[ -n "${CONTAINER_ID:-}" ]]; then
    echo "Removing container..."
    docker rm -f "$CONTAINER_ID"
    echo "Container removed."
  fi
}
