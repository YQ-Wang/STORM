"""
Benchmark: CPU vs GPU performance for STORM storm.

Usage (from the STORM package root):
    python test/benchmark_cpu_gpu.py
    python test/benchmark_cpu_gpu.py --csv path/to/data.csv
"""

import argparse
import os
import sys
import time
import warnings
from typing import Optional

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from storm_omics.storm import (  # noqa: E402
    gpu_backend,
    gpu_enabled,
    prepare_storm_graph,
    storm,
)

DEFAULT_CSV = os.path.join(os.path.dirname(__file__), "test_data", "scenario1_RW1_3-5_1")
COORD_JITTER_STD = 1e-3
N_REPS = 2
STORM_K_NN = 50
_P_VALUE_ATOL = 1e-3
_EFFECT_SIZE_ATOL = 1e-5


def build_dataset(df_base: pd.DataFrame, multiplier: int, rng: np.random.Generator):
    coord_cols = ["x", "y", "z"]
    exp_cols = [c for c in df_base.columns if c not in coord_cols]
    base_coords = df_base[coord_cols].to_numpy(dtype=np.float64)
    base_exp = df_base[exp_cols]
    xy_span = np.ptp(base_coords[:, :2], axis=0)
    tile_spacing = np.maximum(xy_span, 1.0) + 10.0
    grid_cols = int(np.ceil(np.sqrt(multiplier)))

    tiles_coords = []
    tiles_exp = []
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


def _time_call(fn, *args, warmup: bool = False, **kwargs) -> Optional[float]:
    if warmup:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fn(*args, **kwargs)
        except Exception:
            pass

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            t0 = time.perf_counter()
            fn(*args, **kwargs)
            return time.perf_counter() - t0
    except Exception as e:
        print(f"      ERROR: {e}")
        return None


def _check_consistency(
    cpu_vals: np.ndarray, gpu_vals: np.ndarray, *, atol: float
) -> dict:
    abs_diff = np.abs(cpu_vals - gpu_vals)
    max_diff = float(abs_diff.max())
    mean_diff = float(abs_diff.mean())
    if cpu_vals.std() == 0 or gpu_vals.std() == 0:
        corr = float("nan")
    else:
        corr = float(np.corrcoef(cpu_vals, gpu_vals)[0, 1])
    consistent = np.allclose(cpu_vals, gpu_vals, rtol=1e-4, atol=atol)
    return dict(max_diff=max_diff, mean_diff=mean_diff, corr=corr, consistent=consistent)


def benchmark_storm(
    df_base: pd.DataFrame,
    multipliers: list[int],
    k_nn: int = STORM_K_NN,
    reuse_graph: bool = False,
):
    print("\n" + "=" * 90)
    print(f"  BENCHMARK: storm (spatial pattern test, k_nn={k_nn}, approx=True)")
    print("=" * 90)
    print(
        f"  {'Cells':>8}  {'Genes':>6}  {'CPU (s)':>10}  {'GPU (s)':>10}  {'Speedup':>8}  "
        f"{'Corr':>8}  {'MaxDiff':>10}  {'OK?':>4}"
    )
    print("-" * 90)

    rng = np.random.default_rng(99)
    results = []

    for mult in multipliers:
        coords, exp_df = build_dataset(df_base, mult, rng)
        n_cells, n_genes = coords.shape[0], exp_df.shape[1]
        effective_k = min(k_nn, n_cells - 1)
        graph = (
            prepare_storm_graph(coords, k_nn=effective_k)
            if reuse_graph
            else None
        )
        storm_kwargs = (
            {"graph": graph} if graph is not None else {"k_nn": effective_k}
        )

        cpu_result = None
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cpu_result = storm(
                    coords, exp_df, approx=True, use_gpu=False, **storm_kwargs
                )
        except Exception as e:
            print(f"      CPU result ERROR: {e}")

        cpu_times = []
        for _ in range(N_REPS):
            t = _time_call(
                storm, coords, exp_df, approx=True, use_gpu=False, **storm_kwargs
            )
            if t is not None:
                cpu_times.append(t)
        cpu_best = float(np.median(cpu_times)) if cpu_times else None

        gpu_result = None
        gpu_best = None
        if gpu_enabled:
            _time_call(
                storm,
                coords,
                exp_df,
                approx=True,
                use_gpu=True,
                warmup=True,
                **storm_kwargs,
            )
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    gpu_result = storm(
                        coords, exp_df, approx=True, use_gpu=True, **storm_kwargs
                    )
            except Exception as e:
                print(f"      GPU result ERROR: {e}")

            gpu_times = []
            for _ in range(N_REPS):
                t = _time_call(
                    storm,
                    coords,
                    exp_df,
                    approx=True,
                    use_gpu=True,
                    **storm_kwargs,
                )
                if t is not None:
                    gpu_times.append(t)
            gpu_best = float(np.median(gpu_times)) if gpu_times else None

        cons = None
        if cpu_result is not None and gpu_result is not None:
            cons_pval = _check_consistency(
                cpu_result["p_values"].to_numpy(),
                gpu_result["p_values"].to_numpy(),
                atol=_P_VALUE_ATOL,
            )
            cons_eff = _check_consistency(
                cpu_result["effect_size"].to_numpy(),
                gpu_result["effect_size"].to_numpy(),
                atol=_EFFECT_SIZE_ATOL,
            )
            significance_matches = np.array_equal(
                cpu_result["p_values"].to_numpy() < 0.05,
                gpu_result["p_values"].to_numpy() < 0.05,
            )
            cons = dict(
                corr=min(cons_pval["corr"], cons_eff["corr"]),
                max_diff=max(cons_pval["max_diff"], cons_eff["max_diff"]),
                consistent=(
                    cons_pval["consistent"]
                    and cons_eff["consistent"]
                    and significance_matches
                ),
            )

        speedup_str = f"{cpu_best / gpu_best:.2f}x" if (cpu_best and gpu_best) else "N/A"
        cpu_str = f"{cpu_best:.3f}" if cpu_best is not None else "ERR"
        gpu_str = (
            f"{gpu_best:.3f}"
            if gpu_best is not None
            else ("N/A" if not gpu_enabled else "ERR")
        )
        corr_str = f"{cons['corr']:.6f}" if cons else "N/A"
        diff_str = f"{cons['max_diff']:.2e}" if cons else "N/A"
        ok_str = ("PASS" if cons["consistent"] else "FAIL") if cons else "N/A"

        print(
            f"  {n_cells:>8,}  {n_genes:>6,}  {cpu_str:>10}  {gpu_str:>10}  {speedup_str:>8}  "
            f"{corr_str:>8}  {diff_str:>10}  {ok_str:>4}"
        )

        results.append(
            dict(
                func="storm",
                n_cells=n_cells,
                n_genes=n_genes,
                k_nn=effective_k,
                cpu_s=cpu_best,
                gpu_s=gpu_best,
                speedup=cpu_best / gpu_best if (cpu_best and gpu_best) else None,
                corr=cons["corr"] if cons else None,
                max_diff=cons["max_diff"] if cons else None,
                consistent=cons["consistent"] if cons else None,
            )
        )

    return results


def print_summary(all_results: list[dict]):
    print("\n" + "=" * 90)
    print("  SUMMARY")
    print("=" * 90)
    print(
        f"  {'Function':<10}  {'Cells':>8}  {'Genes':>6}  "
        f"{'CPU (s)':>10}  {'GPU (s)':>10}  {'Speedup':>8}  "
        f"{'Corr':>8}  {'MaxDiff':>10}  {'Consistent':>10}"
    )
    print("-" * 90)
    all_consistent = True
    for r in all_results:
        cpu_str = f"{r['cpu_s']:.3f}" if r["cpu_s"] is not None else "ERR"
        gpu_str = f"{r['gpu_s']:.3f}" if r["gpu_s"] is not None else "N/A"
        sp_str = f"{r['speedup']:.2f}x" if r["speedup"] is not None else "N/A"
        corr_str = f"{r['corr']:.6f}" if r["corr"] is not None else "N/A"
        diff_str = f"{r['max_diff']:.2e}" if r["max_diff"] is not None else "N/A"
        ok_str = ("PASS" if r["consistent"] else "FAIL") if r["consistent"] is not None else "N/A"
        if r["consistent"] is False:
            all_consistent = False
        print(
            f"  {r['func']:<10}  {r['n_cells']:>8,}  {r['n_genes']:>6,}  "
            f"{cpu_str:>10}  {gpu_str:>10}  {sp_str:>8}  "
            f"{corr_str:>8}  {diff_str:>10}  {ok_str:>10}"
        )
    print("=" * 90)
    if any(r["consistent"] is not None for r in all_results):
        verdict = "ALL CONSISTENT [OK]" if all_consistent else "SOME INCONSISTENCIES DETECTED [!!]"
        print(f"  Consistency verdict: {verdict}")
        print(
            "  Consistency rule: allclose(p, atol=1e-3), "
            "allclose(effect, atol=1e-5), and matching p<0.05 calls"
        )
        print("=" * 90)


def main():
    global N_REPS
    parser = argparse.ArgumentParser(description="Benchmark STORM CPU vs GPU performance.")
    parser.add_argument("--csv", default=DEFAULT_CSV)
    parser.add_argument("--multipliers", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--k-nn", type=int, default=STORM_K_NN)
    parser.add_argument("--reps", type=int, default=N_REPS)
    parser.add_argument(
        "--reuse-graph",
        action="store_true",
        help="Prepare the exact k-NN graph once and exclude it from call timings.",
    )
    args = parser.parse_args()
    N_REPS = args.reps

    print("\n" + "=" * 72)
    print("  STORM CPU vs GPU Benchmark")
    print("=" * 72)
    print(f"  CSV     : {args.csv}")
    gpu_status = (
        f"enabled ({gpu_backend})" if gpu_enabled else "disabled (CPU only)"
    )
    print(f"  GPU     : {gpu_status}")
    print(f"  Reps    : median of {N_REPS}")
    print(f"  Graph   : {'reused' if args.reuse_graph else 'rebuilt per call'}")
    print(f"  Multipliers (x base rows): {args.multipliers}")

    if not os.path.exists(args.csv):
        print(f"\nERROR: CSV file not found: {args.csv}")
        sys.exit(1)

    df_base = pd.read_csv(args.csv)
    print(f"Base dataset: {df_base.shape[0]:,} cells x {df_base.shape[1] - 3:,} genes")

    print_summary(
        benchmark_storm(
            df_base,
            args.multipliers,
            k_nn=args.k_nn,
            reuse_graph=args.reuse_graph,
        )
    )

    if not gpu_enabled:
        print("\n  NOTE: GPU was not available; only CPU timings reported.")
        print("  Install 'storm-omics[gpu]' to enable GPU benchmarks.")


if __name__ == "__main__":
    main()
