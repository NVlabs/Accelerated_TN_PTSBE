# Optimized Tensor Network PTSBE

This repository contains the implementation and experiment artifacts for **“Accelerating Quantum Tensor Network Simulations with Unified Path Variations and Non-Degenerate Batched Sampling”** ([arXiv:2604.08467](https://arxiv.org/abs/2604.08467)).

The code implements optimized Pre-Trajectory Sampling with Batched Execution (PTSBE) using NVIDIA cuQuantum/cuTensorNet and compares against CUDA-Q trajectory sampling.

## Environment

The paper results used:

- NVIDIA H100 80GB GPUs
- CUDA 12.9
- cuQuantum 26.1.0 / cuTensorNet 2.11.0
- CuPy 13.6.0
- CUDA-Q 0.13.0
- Qiskit 2.2.3

Large paper configurations may require H100-class GPU memory.

## Docker Quick Start

Requirements: Docker, NVIDIA Container Toolkit, and an NVIDIA GPU.

```bash
docker build -t ptsbe-public:local .
bash run_setup.sh
```

The launcher mounts this repository at `/workspace/ptsbe`. It uses only the public NGC cuQuantum appliance and public PyPI packages.

See [`DOCKER_BUILD_GUIDE.md`](DOCKER_BUILD_GUIDE.md) for build details, command execution, tests, and troubleshooting.

## Core Source

- `utils_circuit.py`: optimized tensor-network construction, path reuse, and batched sampling
- `pts.py`: PTS noise operators and sampling helpers
- `stim_to_pts.py`: Stim-to-PTS code generation
- `stim_to_cudaq.py`: Stim-to-CUDA-Q code generation
- `scaling/scaling_comparison_avg.py`: canonical paper benchmark driver
- `scaling/generate_circuits.py`: reference-circuit generator

## Paper Data and Figures

Reference circuits are stored in `scaling/data_collection/circuits/`. Circuit IDs 0–9 are the instances used in the paper runs.

- `figure_03_data_collection_speedup`: non-proportional PTSBE/CUDA-Q speedup
- `figure_04_final_batch_size`: final-batch-size sweep
- `figure_05_proportional_speedup`: proportional-sampling speedup
- `figure_06_pathfinding_vs_contraction`: contraction and path-finding costs
- `figure_07_batch_size_cost`: per-batch cost versus batch size

Each figure directory contains the retained PDF/PNG, plotting script, benchmark output logs, and per-configuration execution scripts.

See [`REPRODUCING_FIGURES.md`](REPRODUCING_FIGURES.md) for the figure-by-figure campaign and plotting commands.

The `.txt` files preserve captured benchmark output with internal filesystem and container-registry identifiers sanitized for public release. Numerical benchmark output is unchanged.

## Run a Paper Configuration

Run a retained configuration directly from the host:

```bash
bash scaling/data_collection/figure_03_data_collection_speedup/100q_600g/run_data_collection_100q_600g_100hs_10nfbs_28fbs.sh
```

Set `NV_GPU` to select a GPU or `PTSBE_IMAGE` to override the Docker image tag.

## Manual Validation Workflows

Small core checks:

```bash
bash run_setup.sh python test_gate_map/test_gate_map.py
bash run_setup.sh python test_contract/test_contract.py
```

End-to-end and timing workflows:

```bash
bash run_setup.sh bash -lc 'cd test_ptsbe && bash run.sh'
bash run_setup.sh bash -lc 'cd test_batched_ptsbe && bash run.sh'
bash run_setup.sh bash -lc 'cd time_batched_ptsbe && bash run.sh'
```

All five workflows were validated with the pinned public Docker image.

## Citation

Citation metadata for the associated manuscript is provided in [`CITATION.cff`](CITATION.cff).

## Contributions

This project is currently not accepting contributions.
