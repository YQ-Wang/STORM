# STORM

## A Principled Statistical Framework for Analyzing Spatial Patterns in Spatially Resolved Multi-Omics

STORM identifies spatially variable features (SVFs), quantifies spatial
dependency with interpretable effect sizes, compares spatial patterns across
biological conditions, and supports power and sample-size calculations for
single-sample and multi-sample spatial studies.

This repository provides a standalone Python package equivalent to
[CastleLi/STORM](https://github.com/CastleLi/STORM).

## Features

- Approximate and finite-sample STORM tests
- Dense NumPy, pandas DataFrame, and SciPy sparse expression matrices
- Batched CPU execution that preserves sparse inputs
- Optional batched PyTorch CUDA execution
- Reusable immutable exact k-NN graphs for repeated analyses
- Treatment-versus-control and treatment-versus-treatment comparisons
- Power analysis for all three tests

The Python API maps all six upstream functions:

| Python | CastleLi/STORM | Purpose |
| --- | --- | --- |
| `storm` | `storm` | Detect and quantify spatial patterns |
| `stormtrt` | `stormtrt` | Compare significant-pattern prevalence |
| `storm2trt` | `storm2trt` | Compare spatial effect sizes |
| `power_storm` | `power_storm` | Single-sample power analysis |
| `powertrt` | `powertrt` | Treatment-versus-control power analysis |
| `power2trt` | `power2trt` | Two-treatment power analysis |

## Installation

The PyPI distribution name is `storm-omics`; the Python import
is `storm_omics`. Python identifiers cannot contain hyphens. The shorter
`storm` distribution name belongs to an unrelated
Canonical ORM package. Until the first PyPI release, install from a source
checkout:

```bash
python -m pip install .
```

For local development:

```bash
# Run from the root of a source checkout.
python -m pip install -e .
```

### GPU support

Install a CUDA-enabled PyTorch build appropriate for your operating system,
driver, and CUDA runtime using the
[official PyTorch installer](https://pytorch.org/get-started/locally/), then
install STORM:

```bash
python -m pip install ".[gpu]"
```

GPU execution is opt-in with `use_gpu=True`. If PyTorch or CUDA is not
available, STORM uses the CPU. If a CUDA operation fails, STORM emits a
`RuntimeWarning` and recomputes on the CPU.

## Spatial Pattern Test

The expression matrix must have shape `M x N`: spots in rows and features in
columns. Coordinates must have shape `M x D`.

```python
import pandas as pd
import storm_omics

data = pd.read_csv("spatial_data.csv")
coords = data[["x", "y", "z"]].to_numpy()
expression = data.drop(columns=["x", "y", "z"])

result = storm_omics.storm(
    coords,
    expression,
    k_nn=50,
    approx=True,
    use_gpu=False,
)
```

`storm` returns a DataFrame with:

- `gene_names`: DataFrame column names, or generated names for array inputs
- `p_values`: two-sided p-values for spatial dependency
- `effect_size`: spatial effect size, `1 - S2 / S0`

Example output:

| gene_names | p_values | effect_size |
| --- | ---: | ---: |
| `GeneA` | 0.0001 | 0.065 |
| `GeneB` | 0.0123 | 0.041 |

`k_nn` defaults to 50 for the approximation and 100 for the finite-sample
calculation. It is capped at `M - 1`. Constant features receive a neutral
p-value of 1 and effect size of 0.

To use CUDA:

```python
result_gpu = storm_omics.storm(coords, expression, k_nn=50, use_gpu=True)
```

On the CPU, sparse inputs remain sparse throughout each feature batch. CUDA
uses a sparse neighbor graph but transfers dense feature batches, with the
batch size chosen from currently available VRAM. Large matrices remain
float32; column reductions use bounded partial sums accumulated in float64 for
CPU/GPU agreement without full-size float64 GPU copies.

### Reference performance

Measured June 19, 2026 with Python 3.13, PyTorch 2.10.0, CUDA 12.8, an NVIDIA
GeForce RTX 5070 Ti (15.9 GB), `k_nn=50`, `approx=True`, and the included real
2,308-spot by 10,000-feature dataset tiled with small coordinate jitter:

| Copies | Spots | CPU | GPU | Speedup | CPU-run peak | GPU-run host peak | GPU peak VRAM |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1x | 2,308 | 0.243 s | 0.092 s | 2.64x | 185 MB | 156 MB | 193 MB |
| 2x | 4,616 | 0.476 s | 0.160 s | 2.97x | 265 MB | 248 MB | 259 MB |
| 3x | 6,924 | 0.717 s | 0.232 s | 3.09x | 357 MB | 340 MB | 260 MB |
| 4x | 9,232 | 0.903 s | 0.298 s | 3.03x | 449 MB | 440 MB | 262 MB |

All four CPU/GPU comparisons had identical `p < 0.05` calls and maximum
p-value differences below `4.5e-5`. Host peak allocations are measured inside
`storm` and exclude the already-loaded input DataFrame and interpreter. GPU
memory is PyTorch peak allocated VRAM. Timings are median of three; memory-mode
timings include tracing overhead and are therefore not shown here.

### Reusing the exact neighbor graph

When several expression matrices share identical coordinates and neighbor
count, prepare the graph once:

```python
graph = storm_omics.prepare_storm_graph(coords, k_nn=50)

result_a = storm_omics.storm(coords, expression_a, graph=graph)
result_b = storm_omics.storm(coords, expression_b, graph=graph, use_gpu=True)
```

`prepare_storm_graph` runs the same exact directed k-NN construction as the
ordinary call. The graph is immutable, `storm` verifies that its coordinates
and neighbor count match, and exact-mode's graph-only scaling constant is
cached lazily. Reuse therefore changes only setup cost, not any statistic.

On the 4x real dataset, median repeated-call time decreased from 0.925 s to
0.871 s on CPU and from 0.360 s to 0.231 s on GPU. Cold graph preparation took
0.198 s, so it paid for itself after about four CPU analyses or two GPU
analyses in this environment.

## Group Comparisons

### Treatment versus control

`stormtrt` implements upstream STORM's one-sided pooled two-proportion test and
uses the same default per-sample significance threshold of 0.05.

```python
import pandas as pd
import storm_omics

data = pd.DataFrame({
    "Group": ["Control"] * 4 + ["Treatment"] * 4,
    "Pvalue": [0.40, 0.20, 0.01, 0.30, 0.01, 0.02, 0.03, 0.20],
})

comparison = storm_omics.stormtrt(data, control="Control", sig_level=0.05)
```

### Two treatment groups

```python
data = pd.DataFrame({
    "Group": ["A"] * 3 + ["B"] * 3,
    "EffectSize": [0.10, 0.11, 0.09, 0.02, 0.03, 0.01],
    "M": [3000] * 6,
    "K": [50] * 6,
})

comparison = storm_omics.storm2trt(data, alternative="two.sided")
```

## Power Analysis

Exactly one parameter described as unknown must be `None`.

```python
import storm_omics

# Single-sample STORM: solve for number of spots.
single = storm_omics.power_storm(
    es=0.06, n=None, power=0.80, sig_level=0.05
)

# Treatment versus control: solve for samples per control group.
control = storm_omics.powertrt(
    nsample=None,
    power=0.90,
    sig_level=0.05,
    ratio=1,
    power_single=0.80,
    sig_level_single=0.05,
)

# Two treatments: solve for group-1 sample size.
two_treatments = storm_omics.power2trt(
    delta=0.04,
    phi=4 / (50 * 3000),
    psi1=0.010,
    psi2=0.005,
    nsample=None,
    ratio=2,
    sig_level=0.05,
    power=0.80,
)
```

Power calculations return continuous sample-size estimates. Round up before
using them as study sizes.

## Tests and Benchmarks

```bash
python -m pytest
python test/benchmark_cpu_gpu.py --multipliers 1 2 3 4 --reps 3
python test/benchmark_cpu_gpu.py --multipliers 4 --reps 5 --reuse-graph
python test/benchmark_memory.py --multipliers 1 2 3 4
```

Benchmark results depend on data density, feature count, neighbor count, CPU,
GPU, CUDA/PyTorch versions, and available memory. Run the included scripts on
the target system rather than relying on timings from another machine.

## Upstream Reproduction

The upstream repository contains the manuscript reproduction scripts for data
processing, simulations, benchmark analyses, figures, and supplementary
analyses in its
[`Reproduction`](https://github.com/CastleLi/STORM/tree/main/Reproduction)
directory. Those R and manuscript-specific scripts are not duplicated in this
Python distribution.

## Release Process

Git tags beginning with `v` build and publish both wheel and source
distributions through PyPI Trusted Publishing. Configure the GitHub `pypi`
environment and a pending publisher for `storm-omics` before the first
release.

## License

This Python distribution uses GNU General Public License v3.0. See
[LICENSE](LICENSE). Upstream CastleLi/STORM is distributed under GPL version 2
or later, which permits redistribution under GPLv3.

## Citation

When using this implementation, cite the STORM method and the upstream
[CastleLi/STORM](https://github.com/CastleLi/STORM) software. The upstream
manuscript citation is still listed as pending and should be added here once
its final bibliographic record is available.

STORM authors: Jinpu Li
([ORCID 0000-0002-6656-2896](https://orcid.org/0000-0002-6656-2896)) and
Yiqing Wang.
