"""
Memory & speed benchmark for STORM storm.

Usage:
    python test/benchmark_memory.py [--multipliers 1 2 4 8 16]
"""

import argparse
import os
import sys
import time
import tracemalloc
import warnings

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from storm.storm import gpu_enabled, storm

COORD_JITTER_STD = 1e-3
DEFAULT_CSV = os.path.join(os.path.dirname(__file__), "test_data", "scenario1_RW1_3-5_1")

gpu_available = False
torch = None
if gpu_enabled:
    try:
        import torch as _torch
        torch = _torch
        gpu_available = torch.cuda.is_available()
    except ImportError:
        pass


def build_dataset(df_base, multiplier, rng):
    coord_cols = ["x", "y", "z"]
    exp_cols = [c for c in df_base.columns if c not in coord_cols]
    base_coords = df_base[coord_cols].to_numpy(dtype=np.float64)
    base_exp = df_base[exp_cols]
    tiles_coords = []
    tiles_exp = []
    xy_span = np.ptp(base_coords[:, :2], axis=0)
    tile_spacing = np.maximum(xy_span, 1.0) + 10.0
    grid_cols = int(np.ceil(np.sqrt(multiplier)))
    for i in range(multiplier):
        row_idx, col_idx = divmod(i, grid_cols)
        offset = np.array(
            [col_idx * tile_spacing[0], row_idx * tile_spacing[1], 0.0],
            dtype=np.float64,
        )
        jitter = np.zeros_like(base_coords)
        jitter[:, :2] = rng.normal(0, COORD_JITTER_STD, size=(base_coords.shape[0], 2))
        tiles_coords.append(base_coords + offset + jitter)
        tiles_exp.append(base_exp)
    return np.vstack(tiles_coords), pd.concat(tiles_exp, ignore_index=True)


def reset_gpu_stats():
    if gpu_available:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()


def measure_call(fn, *args, **kwargs):
    import gc
    gc.collect()
    if gpu_available:
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    tracemalloc.start()
    reset_gpu_stats()

    t0 = time.perf_counter()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = fn(*args, **kwargs)
        if gpu_available:
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        _, peak_cpu = tracemalloc.get_traced_memory()
        peak_gpu = torch.cuda.max_memory_allocated() / 1024**2 if gpu_available else 0.0
        tracemalloc.stop()
        return result, elapsed, peak_cpu / 1024**2, peak_gpu
    except Exception as e:
        tracemalloc.stop()
        return None, None, None, str(e)


def run_benchmark(df_base, multipliers, k_nn=50):
    rng = np.random.default_rng(42)

    print(f"\n{'='*100}")
    print(
        f"  {'Func':<6} {'Cells':>8} {'Genes':>6}  {'Time(s)':>8}  "
        f"{'CPU Peak(MB)':>13}  {'GPU Peak(MB)':>13}  {'Status':>8}"
    )
    print(f"{'='*100}")

    results = []
    for mult in multipliers:
        coords, exp_df = build_dataset(df_base, mult, rng)
        n_cells, n_genes = coords.shape[0], exp_df.shape[1]
        ek = min(k_nn, n_cells - 1)

        res, t, cpu_mb, gpu_mb = measure_call(
            storm, coords, exp_df, k_nn=ek, approx=True, use_gpu=False
        )
        status = "OK" if res is not None else f"FAIL({gpu_mb})"
        print(
            f"  {'storm':<6} {n_cells:>8,} {n_genes:>6,}  {t if t else 0:>8.3f}  "
            f"{cpu_mb if cpu_mb else 0:>13.1f}  {'N/A':>13}  {status:>8}  [CPU]"
        )
        results.append(
            dict(
                func="storm", device="CPU", n_cells=n_cells, n_genes=n_genes,
                time_s=t, cpu_peak_mb=cpu_mb, gpu_peak_mb=None, status=status,
            )
        )

        if gpu_available:
            res, t, cpu_mb, gpu_mb = measure_call(
                storm, coords, exp_df, k_nn=ek, approx=True, use_gpu=True
            )
            status = "OK" if res is not None else "OOM"
            gpu_str = f"{gpu_mb:>13.1f}" if isinstance(gpu_mb, (int, float)) else f"{'OOM':>13}"
            print(
                f"  {'storm':<6} {n_cells:>8,} {n_genes:>6,}  {t if t else 0:>8.3f}  "
                f"{cpu_mb if cpu_mb else 0:>13.1f}  {gpu_str}  {status:>8}  [GPU]"
            )
            results.append(
                dict(
                    func="storm", device="GPU", n_cells=n_cells, n_genes=n_genes,
                    time_s=t, cpu_peak_mb=cpu_mb,
                    gpu_peak_mb=gpu_mb if isinstance(gpu_mb, (int, float)) else None,
                    status=status,
                )
            )

        print(f"  {'-'*94}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Memory & speed benchmark for STORM")
    parser.add_argument("--csv", default=DEFAULT_CSV)
    parser.add_argument("--multipliers", nargs="+", type=int, default=[1, 2, 4, 8, 16])
    parser.add_argument("--k-nn", type=int, default=50)
    args = parser.parse_args()

    print("\n  STORM Memory & Speed Benchmark")
    print(f"  GPU: {'available' if gpu_available else 'not available'}")
    if gpu_available:
        print(f"  GPU Device: {torch.cuda.get_device_name(0)}")
        print(f"  GPU Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    df_base = pd.read_csv(args.csv)
    print(f"  Base dataset: {df_base.shape[0]:,} cells x {df_base.shape[1] - 3:,} genes")
    print(f"  Multipliers: {args.multipliers}")

    run_benchmark(df_base, args.multipliers, k_nn=args.k_nn)


if __name__ == "__main__":
    main()
