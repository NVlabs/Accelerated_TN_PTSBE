# Reproducing the Paper Figures

This guide describes how to regenerate the paper figures from the retained benchmark logs and how to rerun the underlying GPU campaigns using the original per-configuration scripts.

## Reproduction Scope

There are two supported workflows:

1. **Regenerate plots from retained data.** This is fast, does not rerun the GPU benchmarks, and reproduces the numerical values shown by the plotting scripts.
2. **Rerun benchmark campaigns.** This requires NVIDIA GPU hardware and substantial execution time. Each configuration has its own script; there is no consolidated campaign runner.

The exact circuit instances used for the paper are retained in `scaling/data_collection/circuits/`. Do not regenerate the circuits when validating the paper results.

GPU timings are sensitive to hardware, system load, path optimization, and sampling randomness. Fresh runs should reproduce the reported trends but are not expected to be bit-for-bit or timing-identical to the retained logs.

## Environment Setup

Requirements:

- Linux with an NVIDIA GPU
- Docker
- NVIDIA Container Toolkit
- Access to NVIDIA NGC and PyPI

Build the validated image from the repository root:

```bash
docker build -t ptsbe-public:local .
```

Set `NV_GPU` to select a GPU or `PTSBE_IMAGE` to use another local image tag.

## Capturing Fresh Benchmark Logs

Run all collection scripts with `bash`, not `sh`.

The `run_data_collection_*.sh` scripts write benchmark output to standard output. The plotting scripts read `.txt` files with fixed names from each figure directory. To replace a retained data point in a disposable clone, first preserve the reference log and then redirect the fresh run to the expected filename:

```bash
cp scaling/data_collection/figure_03_data_collection_speedup/100q_600g_100hs_10nfbs_28fbs_ptsbe.txt /tmp/reference_100q_600g_ptsbe.txt

bash scaling/data_collection/figure_03_data_collection_speedup/100q_600g/run_data_collection_100q_600g_100hs_10nfbs_28fbs.sh \
  > scaling/data_collection/figure_03_data_collection_speedup/100q_600g_100hs_10nfbs_28fbs_ptsbe.txt 2>&1
```

Use a disposable clone or worktree for campaign reruns because fresh logs and generated JSON files modify the working tree. The retained `run_crun_*.sh` files are capture wrappers from the original campaign and are not required for reproduction.

## Figure 3: Non-Proportional Data-Collection Speedup

Configurations:

- Qubits: 50, 75, 100, 150, and 200
- Gates: 200, 400, 600, 800, and 1000
- Additional configuration: 200 qubits and 1200 gates

For each configuration directory, run both the PTSBE and CUDA-Q scripts. Example:

```bash
bash scaling/data_collection/figure_03_data_collection_speedup/100q_600g/run_data_collection_100q_600g_100hs_10nfbs_28fbs.sh

bash scaling/data_collection/figure_03_data_collection_speedup/100q_600g/run_data_collection_100q_600g_100hs_cudaq.sh
```

There are 26 circuit-size configurations and 52 primary collection scripts.

Regenerate Figure 3 from the logs:

```bash
bash run_setup.sh python scaling/data_collection/figure_03_data_collection_speedup/plot_final_throughput_advantage.py
```

## Figure 4: Final-Batch-Size Sweep

Configurations use 200 qubits, gate counts 200–1000, and final batch sizes 24, 26, and 28.

For each gate count, run the final-batch-size 24 and 26 scripts. Example:

```bash
bash scaling/data_collection/figure_04_final_batch_size/200q_600g/run_data_collection_200q_600g_100hs_10nfbs_24fbs.sh

bash scaling/data_collection/figure_04_final_batch_size/200q_600g/run_data_collection_200q_600g_100hs_10nfbs_26fbs.sh
```

The final-batch-size 28 data are the corresponding Figure 3 PTSBE runs. When rerunning the complete campaign, copy each fresh 200-qubit Figure 3 result into the matching Figure 4 filename. Example:

```bash
cp scaling/data_collection/figure_03_data_collection_speedup/200q_600g_100hs_10nfbs_28fbs_ptsbe.txt \
  scaling/data_collection/figure_04_final_batch_size/200q_600g_100hs_10nfbs_28fbs_ptsbe.txt
```

Regenerate Figure 4:

```bash
bash run_setup.sh python scaling/data_collection/figure_04_final_batch_size/plot_final_fbs_sweep.py
```

## Figure 5: Proportional-Sampling Speedup

Configurations:

- 100 qubits and 600 gates
- 200 qubits and 1000 gates
- Shot counts: 10, 100, 1000, and 10000

Run all four PTSBE shot-count scripts in each configuration directory. Example:

```bash
bash scaling/data_collection/figure_05_proportional_speedup/100q_600g/run_data_collection_100q_600g_100hs_10bs_proportional_1000nshots.sh
```

Figure 5 uses the matching CUDA-Q baselines from Figure 3. Copy fresh Figure 3 CUDA-Q logs into the Figure 5 directory before plotting a fully rerun campaign:

```bash
cp scaling/data_collection/figure_03_data_collection_speedup/100q_600g_100hs_cudaq.txt \
  scaling/data_collection/figure_05_proportional_speedup/100q_600g_100hs_cudaq.txt

cp scaling/data_collection/figure_03_data_collection_speedup/200q_1000g_100hs_cudaq.txt \
  scaling/data_collection/figure_05_proportional_speedup/200q_1000g_100hs_cudaq.txt
```

Regenerate Figure 5:

```bash
bash run_setup.sh python scaling/data_collection/figure_05_proportional_speedup/plot_throughput_advantage.py
```

## Figure 6: Path-Finding Versus Contraction Cost

Configurations cover five qubit counts and five gate counts:

- Qubits: 50, 75, 100, 150, and 200
- Gates: 200, 400, 600, 800, and 1000

Run the single collection script in each of the 25 configuration directories. Example:

```bash
bash scaling/data_collection/figure_06_pathfinding_vs_contraction/100q_600g/run_data_collection_100q_600g_1hs_10nfbs_28fbs.sh
```

Each script runs the ten retained circuits with ten timing repeats. The Figure 6 plot uses RUN 1 from each circuit, matching the one-measurement-per-circuit analysis used by the other paper figures.

The two proportional-sampling stars are read from the 10000-shot Figure 5 logs, so rerun Figure 5 first when rebuilding those points.

Regenerate Figure 6:

```bash
bash run_setup.sh python scaling/data_collection/figure_06_pathfinding_vs_contraction/plot_qubit_gate_sweep.py
```

## Figure 7: Batch-Size Cost

Figure 7 uses the 100-qubit, 600-gate circuit set.

Run every combination of:

- Batch size: 2, 5, 10, 15, 20, 24, and 28
- Sampling policy: `2p1`, `2p2`, and `2p3`

Example:

```bash
bash scaling/data_collection/figure_07_batch_size_cost/100q_600g/run_data_collection_100q_600g_100hs_10bs_2p1_ptsbe.sh
```

There are 21 primary configurations. Files containing `_RESTART` record recovery of an interrupted original run and are not part of a fresh campaign.

Regenerate Figure 7:

```bash
bash run_setup.sh python scaling/data_collection/figure_07_batch_size_cost/plot_contraction_cost_scatter.py
```

## Expected Failures and Resource Requirements

The complete campaign requires hundreds of serial H100 GPU-hours. Some large or highly connected configurations can exceed optimizer-cost, memory, or timeout limits. These failures are represented in the retained logs and may recur during a fresh campaign.

Plots regenerated from fresh runs may differ slightly in timing values, error bars, PDF metadata, or raster layout while preserving the reported performance trends.
