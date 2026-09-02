# Third-Party Notices

The source code authored for this project is licensed under Apache-2.0 as described in the root [`LICENSE`](LICENSE) file. Third-party components remain subject to their own license terms.

This repository does not vendor or redistribute the container image or Python packages listed below. The [`Dockerfile`](Dockerfile) retrieves them from NVIDIA NGC and PyPI when a user builds the environment. Copyright, attribution, and license notices delivered with each downloaded image or package remain applicable and are not replaced by this document.

## Container Base

| Component | Version | License terms | License and source |
|---|---:|---|---|
| NVIDIA cuQuantum Appliance | 25.09, CUDA 12.9.1, Ubuntu 24.04 | NVIDIA Software License and component-specific terms and notices delivered with the image | [NGC catalog](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/cuquantum-appliance) |

The appliance contains CUDA, conda, Ubuntu packages, and other transitive components. Those components are governed by the license and attribution notices included in the appliance. Public availability of the image does not mean that every component is open-source software.

## Build Tooling

| Component | Version constraint | Purpose | License | Public source |
|---|---:|---|---|---|
| pip | 26.2.1 | Python package installer | MIT | [pypa/pip](https://github.com/pypa/pip) |

## Direct Python Dependencies

| Package | Version | Purpose | License | License and source |
|---|---:|---|---|---|
| cuquantum-cu12 | 26.1.0 | cuQuantum C-library distribution, including cuTensorNet 2.11.0 | NVIDIA Software License | [cuQuantum license](https://docs.nvidia.com/cuda/cuquantum/latest/license.html) |
| cuquantum-python-cu12 | 26.1.0 | Python bindings for cuQuantum | BSD-3-Clause | [NVIDIA/cuQuantum](https://github.com/NVIDIA/cuQuantum) |
| cupy-cuda12x | 13.6.0 | GPU array backend | MIT | [cupy/cupy](https://github.com/cupy/cupy) |
| cuda-quantum-cu12 | 0.13.0 | CUDA-Q reference trajectory implementation | Apache-2.0 | [NVIDIA/cuda-quantum](https://github.com/NVIDIA/cuda-quantum) |
| nvmath-python | 0.7.0 | cuQuantum numerical runtime dependency | Apache-2.0 | [NVIDIA/nvmath-python](https://github.com/NVIDIA/nvmath-python) |
| qiskit | 2.2.3 | Circuit intermediate representation | Apache-2.0 | [Qiskit/qiskit](https://github.com/Qiskit/qiskit) |
| numpy | 1.26.4 | Core numerical arrays | BSD-3-Clause | [numpy/numpy](https://github.com/numpy/numpy) |
| pandas | 2.3.2 | Histogram and benchmark-data handling | BSD-3-Clause | [pandas-dev/pandas](https://github.com/pandas-dev/pandas) |
| matplotlib | 3.10.6 | Figure and histogram generation | Matplotlib License | [license](https://matplotlib.org/stable/project/license.html) · [source](https://github.com/matplotlib/matplotlib) |
| scipy | 1.16.2 | Geometric statistics for Figure 6 | BSD-3-Clause | [scipy/scipy](https://github.com/scipy/scipy) |
| mpi4py | 4.1.0 | CUDA-Q baseline and optional MPI support | BSD-3-Clause | [mpi4py/mpi4py](https://github.com/mpi4py/mpi4py) |

## Transitive and Bundled Components

The container base and direct Python dependencies install or bundle additional components. Examples include CUDA runtime libraries, numerical libraries, Python support packages, fonts, and native libraries distributed inside binary wheels. These components retain the copyright, attribution, and license files supplied by their respective distributors.

Users can inspect the notices and package metadata delivered with a built image. The direct versions selected by this project are recorded above; transitive versions may be resolved by the package installer at build time.

## Verification

The direct runtime package versions above are selected in `Dockerfile` and checked during the image build where an importable version attribute is available. Package names, versions, and license metadata were also checked against the installed Python distributions in the validated image.
