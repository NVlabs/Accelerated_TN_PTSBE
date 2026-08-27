# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
FROM nvcr.io/nvidia/cuquantum-appliance:25.09-cuda12.9.1-devel-ubuntu24.04-x86_64@sha256:77f06b148feb48ebd9a804f7a7aaa3f488ad7415ec084f1dc9505a9d3ce31fe1

SHELL ["/bin/bash", "-lc"]

ENV PATH="/opt/conda/envs/cuquantum/bin:${PATH}"

USER root

RUN python -m pip install --no-cache-dir --upgrade "pip==26.2.1" \
    && python -m pip install --no-cache-dir \
        "cuquantum-cu12==26.1.0" \
        "cuquantum-python-cu12==26.1.0" \
        "cupy-cuda12x==13.6.0" \
        "cuda-quantum-cu12==0.13.0" \
        "nvmath-python==0.7.0" \
        "qiskit==2.2.3" \
        "numpy==1.26.4" \
        "pandas==2.3.2" \
        "matplotlib==3.10.6" \
        "scipy==1.16.2" \
        "mpi4py==4.1.0"

RUN python - <<'PY'
import cudaq
import cupy
from importlib import metadata
import matplotlib
import mpi4py
import numpy
import nvmath
import pandas
import qiskit
import scipy
from cuquantum.bindings import cutensornet

assert cupy.__version__ == "13.6.0"
assert metadata.version("cuda-quantum-cu12") == "0.13.0"
assert matplotlib.__version__ == "3.10.6"
assert mpi4py.__version__ == "4.1.0"
assert numpy.__version__ == "1.26.4"
assert nvmath.__version__ == "0.7.0"
assert pandas.__version__ == "2.3.2"
assert metadata.version("pip") == "26.2.1"
assert qiskit.__version__ == "2.2.3"
assert scipy.__version__ == "1.16.2"
assert cutensornet.get_version() == 21100
print(f"CUDA-Q: {cudaq.__version__}")
print(f"CuPy: {cupy.__version__}")
print(f"Qiskit: {qiskit.__version__}")
print(f"cuTensorNet: {cutensornet.get_version()}")
PY

RUN mkdir -p /workspace/ptsbe \
    && chown -R cuquantum:cuquantum /workspace

USER cuquantum

WORKDIR /workspace/ptsbe

CMD ["/bin/bash"]
