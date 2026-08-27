# Public Docker Setup

The Docker image reproduces the software versions used for the paper with public NVIDIA and PyPI packages. It does not require NVIDIA-internal source repositories, registries, or credentials.

## Requirements

- Linux with an NVIDIA GPU
- Docker
- NVIDIA Container Toolkit
- Internet access to NGC and PyPI

Using the NGC base image requires acceptance of the NVIDIA cuQuantum appliance license displayed by the container.

The Dockerfile pins both the appliance tag and its image digest so that builds use the validated base-image contents.

## Build

From the repository root:

```bash
docker build -t ptsbe-public:local .
```

The build verifies:

- pip 26.2.1
- cuQuantum 26.1.0
- cuTensorNet 2.11.0
- CuPy 13.6.0
- CUDA-Q 0.13.0
- Qiskit 2.2.3
- NumPy 1.26.4
- pandas 2.3.2
- Matplotlib 3.10.6
- SciPy 1.16.2
- nvmath-python 0.7.0
- mpi4py 4.1.0

## Open an Interactive Container

```bash
bash run_setup.sh
```

`run_setup.sh` builds the image if it is missing, mounts the repository at `/workspace/ptsbe`, exposes all GPUs, and removes the container when the shell exits.

To use a differently tagged image:

```bash
PTSBE_IMAGE=my-image:tag bash run_setup.sh
```

## Run a Command

Pass a command after the launcher:

```bash
bash run_setup.sh python test_gate_map/test_gate_map.py
```

The command runs from `/workspace/ptsbe`.

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

All five workflows were validated with the pinned image.

## Paper Data-Collection Scripts

Each retained experiment configuration contains:

- `python/run_data_collection_*.py`: benchmark parameters and orchestration
- `run_data_collection_*.sh`: launches the public container and runs the Python driver
- `run_crun_*.sh`: optional scheduler wrapper that captures output

Run a configuration directly from the host:

```bash
bash scaling/data_collection/figure_03_data_collection_speedup/100q_600g/run_data_collection_100q_600g_100hs_10nfbs_28fbs.sh
```

The shared container configuration is:

```text
scaling/container_config.sh
```

Set `NV_GPU` to select a specific GPU:

```bash
NV_GPU=0 bash path/to/run_data_collection_*.sh
```

Set `PTSBE_IMAGE` to override the image tag.

## MPI

The retained paper runs distribute independent jobs/error sets externally and do not use intra-process cuTensorNet MPI. `CUDA_PATH`, `MPI_PATH`, and `CUTENSORNET_COMM_LIB` do not need to be set manually.

The cuQuantum base image configures the runtime CUDA library path automatically.

## Troubleshooting

Verify Docker GPU access:

```bash
docker run --rm --gpus all \
  nvcr.io/nvidia/cuda:12.9.1-base-ubuntu24.04 \
  nvidia-smi
```

Rebuild after changing dependencies:

```bash
docker build --no-cache -t ptsbe-public:local .
```

Inspect installed versions:

```bash
bash run_setup.sh python -c \
  "import cudaq, cupy, qiskit; from cuquantum.bindings import cutensornet; print(cudaq.__version__, cupy.__version__, qiskit.__version__, cutensornet.get_version())"
```
