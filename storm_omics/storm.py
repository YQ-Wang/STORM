"""
@author: lijinp yiqingwang

Statistical Test for spatial patterns using k-nearest neighbors (STORM).
"""

import warnings
from dataclasses import dataclass, field
from numbers import Integral
from typing import Optional, Union

import numpy as np
import pandas as pd  # type: ignore
from scipy.sparse import (  # type: ignore
    csr_matrix,
    identity,
    isspmatrix_csr,
    issparse,
)
from scipy.stats import norm  # type: ignore

gpu_enabled = True
gpu_backend = "torch"

try:
    import torch  # type: ignore
    if not torch.cuda.is_available():
        gpu_enabled = False
        gpu_backend = None
except ImportError:
    gpu_enabled = False
    gpu_backend = None


def _validate_coords(coords: np.ndarray) -> np.ndarray:
    coords_array = np.asarray(coords)
    if coords_array.ndim != 2 or coords_array.shape[1] == 0:
        raise ValueError("coords must be a non-empty two-dimensional array")
    if not np.issubdtype(coords_array.dtype, np.number):
        raise ValueError("coords must contain only finite numeric values")
    if not np.isfinite(coords_array).all():
        raise ValueError("coords must contain only finite numeric values")
    if coords_array.shape[0] < 2:
        raise ValueError("coords must contain at least 2 spots")
    return coords_array


def _normalize_k_nn(k_nn: int, n_spots: int) -> int:
    if isinstance(k_nn, bool) or not isinstance(k_nn, Integral) or k_nn < 1:
        raise ValueError("k_nn must be a positive integer")
    return min(int(k_nn), n_spots - 1)


@dataclass(frozen=True)
class StormGraph:
    """Immutable exact k-NN graph reusable across STORM calculations."""

    coords: np.ndarray = field(repr=False)
    patches_raw: csr_matrix = field(repr=False)
    k_nn: int
    _test_constant: Optional[float] = field(default=None, init=False, repr=False)

    @property
    def n_spots(self) -> int:
        return self.coords.shape[0]

    def exact_test_constant(self) -> float:
        constant = self._test_constant
        if constant is None:
            constant = _exact_test_constant(
                self.patches_raw, self.n_spots, self.k_nn
            )
            object.__setattr__(self, "_test_constant", constant)
        return constant


def prepare_storm_graph(coords: np.ndarray, k_nn: int = 50) -> StormGraph:
    """Build the exact directed k-NN graph used by upstream STORM.

    Reuse the returned object only with the same coordinates. Passing it to
    :func:`storm` skips neighbor search and sparse graph construction without
    changing any statistic.
    """
    from sklearn.neighbors import NearestNeighbors

    coords_array = _validate_coords(coords)
    n_spots = coords_array.shape[0]
    k_nn = _normalize_k_nn(k_nn, n_spots)
    nn = NearestNeighbors(n_neighbors=k_nn + 1, algorithm="auto", n_jobs=-1)
    knn_candidates = nn.fit(coords_array).kneighbors(
        coords_array, return_distance=False
    )
    is_self = knn_candidates == np.arange(n_spots)[:, None]
    neighbor_order = np.argsort(is_self, axis=1, kind="stable")
    knn_indices = np.take_along_axis(
        knn_candidates, neighbor_order, axis=1
    )[:, :k_nn]

    indptr = np.arange(0, (n_spots + 1) * k_nn, k_nn, dtype=np.int64)
    patches_raw = csr_matrix(
        (
            np.ones(n_spots * k_nn, dtype=np.float32),
            knn_indices.ravel().astype(np.int64, copy=False),
            indptr,
        ),
        shape=(n_spots, n_spots),
        dtype=np.float32,
    )
    patches_raw.sort_indices()

    coords_copy = np.array(coords_array, copy=True)
    coords_copy.setflags(write=False)
    patches_raw.data.setflags(write=False)
    patches_raw.indices.setflags(write=False)
    patches_raw.indptr.setflags(write=False)
    return StormGraph(coords=coords_copy, patches_raw=patches_raw, k_nn=k_nn)


def _exact_test_constant(patches_raw: csr_matrix, n_spots: int, k_nn: int) -> float:
    """Return the finite-sample scaling constant used by upstream STORM."""
    patches_raw_t = patches_raw.T.tocsr()
    w_mat_n1 = patches_raw @ patches_raw_t
    w_n2 = float(w_mat_n1.multiply(w_mat_n1).sum()) - n_spots * k_nn**2
    # w_n3/w_n4 sum w_mat_n1 and patches_raw_t over the support of patches_raw.
    # Masking with an elementwise product is mathematically identical to upstream's
    # ``w_mat_n1[PatchesCells_Raw > 0]`` while avoiding slow CSR fancy indexing and
    # the large intermediate index arrays it allocates.
    w_n3 = float(w_mat_n1.multiply(patches_raw).sum())
    w_n4 = float(patches_raw.multiply(patches_raw_t).sum())
    denominator = 4 * k_nn**3 + (
        2 * w_n2 - 8 * k_nn * w_n3 + 4 * k_nn**2 * w_n4
    ) / n_spots
    return float(
        np.sqrt(n_spots * k_nn**2 * (k_nn + 1) ** 2 / denominator)
    )


def _centered_patches(patches_raw: csr_matrix, n_spots: int, k_nn: int) -> csr_matrix:
    """Return upstream's ``PatchesCells = PatchesCells_Raw - K_NN * Diagonal``.

    Folding the centroid term into the neighbor matrix lets each feature batch use
    a single sparse matmul instead of a separate scaled subtraction. The identity is
    built with ``patches_raw``'s dtype so the result stays float32 rather than
    being upcast to float64.
    """
    centered = patches_raw - k_nn * identity(
        n_spots, dtype=patches_raw.dtype, format="csr"
    )
    centered.sort_indices()
    return centered


def _gpu_batch_size(n_spots: int, n_genes: int) -> int:
    """Choose a conservative batch size from currently available GPU memory."""
    free_bytes, _ = torch.cuda.mem_get_info()
    # Smaller batches reduce host copies and CUDA SpMM workspace while also
    # performing better on the included real dataset.
    bytes_per_matrix = min(int(free_bytes * 0.10), 64 * 1024**2)
    genes = bytes_per_matrix // (np.dtype(np.float32).itemsize * n_spots)
    return min(n_genes, max(1, int(genes)))


def _gpu_column_sum_float64(values) -> "torch.Tensor":
    """Accumulate columns accurately without a full float64 CUDA temporary."""
    total = torch.zeros(
        values.shape[1], dtype=torch.float64, device=values.device
    )
    for start in range(0, values.shape[0], 512):
        partial = values[start:start + 512].sum(dim=0)
        total.add_(partial.to(dtype=torch.float64))
    return total


def storm(
    coords: np.ndarray,
    exp_mat: Union[np.ndarray, pd.DataFrame, csr_matrix],
    k_nn: Optional[int] = None,
    approx: bool = True,
    use_gpu: bool = False,
    *,
    graph: Optional[StormGraph] = None,
) -> pd.DataFrame:
    """
    Statistical test for identifying spatial patterns using k-nearest neighbors.

    This function computes a test statistic based on the spatial autocorrelation
    of gene expression values using a k-NN graph structure.

    Args:
        coords: An M x D numpy array of D-dimensional coordinates for M spots.
        exp_mat: An M x N expression matrix (spots x genes). Can be numpy array,
                 pandas DataFrame, or scipy sparse matrix.
        k_nn: Number of nearest neighbors. If None, defaults to 50 when
              approx=True and 100 when approx=False.
        approx: Use approximation for test statistic variance (default True).
                If False, computes exact variance which is slower.
        use_gpu: Whether to use GPU acceleration (default False).
        graph: Optional graph returned by :func:`prepare_storm_graph`. It must
               match ``coords`` and ``k_nn`` exactly and changes only graph
               construction cost, never the STORM statistic.

    Returns:
        A pandas DataFrame with columns:
        - gene_names: Gene identifiers
        - p_values: Two-sided p-values for spatial dependency
        - effect_size: Effect size of spatial dependency (1 - S2/S0)
    """
    coords = _validate_coords(coords)

    exp_mat_array: Optional[np.ndarray] = None
    exp_mat_sparse: Optional[csr_matrix] = None
    if isinstance(exp_mat, pd.DataFrame):
        gene_names = exp_mat.columns.astype(str).tolist()
        exp_mat_array = exp_mat.to_numpy(dtype=np.float32)
    elif issparse(exp_mat):
        if exp_mat.ndim != 2:
            raise ValueError("exp_mat must be two-dimensional")
        gene_names = [f"Gene_{i}" for i in range(exp_mat.shape[1])]
        exp_mat_sparse = (
            exp_mat if isspmatrix_csr(exp_mat) else exp_mat.tocsr()
        ).astype(np.float32, copy=False)
        if not np.isfinite(exp_mat_sparse.data).all():
            raise ValueError("exp_mat must contain only finite values")
    else:
        exp_mat_array = np.asarray(exp_mat, dtype=np.float32)
        if exp_mat_array.ndim != 2:
            raise ValueError("exp_mat must be two-dimensional")
        gene_names = [f"Gene_{i}" for i in range(exp_mat_array.shape[1])]

    if exp_mat_array is not None and not np.isfinite(exp_mat_array).all():
        raise ValueError("exp_mat must contain only finite values")

    n_spots = coords.shape[0]
    if graph is None:
        if k_nn is None:
            k_nn = 50 if approx else 100
        graph = prepare_storm_graph(coords, k_nn=k_nn)
    else:
        if not isinstance(graph, StormGraph):
            raise TypeError("graph must be a StormGraph returned by prepare_storm_graph")
        if not np.array_equal(coords, graph.coords):
            raise ValueError("graph coordinates do not match coords")
        if k_nn is not None and _normalize_k_nn(k_nn, n_spots) != graph.k_nn:
            raise ValueError("graph k_nn does not match k_nn")
    k_nn = graph.k_nn
    patches_raw = graph.patches_raw

    exp_rows = (
        exp_mat_sparse.shape[0]
        if exp_mat_sparse is not None
        else exp_mat_array.shape[0]
    )
    n_genes = (
        exp_mat_sparse.shape[1]
        if exp_mat_sparse is not None
        else exp_mat_array.shape[1]
    )
    if exp_rows != n_spots:
        raise ValueError(
            f"Expression matrix rows ({exp_rows}) must match coords rows ({n_spots})"
        )
    if n_genes == 0:
        return pd.DataFrame(columns=["gene_names", "p_values", "effect_size"])

    def _dense_batch(start_idx: int, end_idx: int) -> np.ndarray:
        if exp_mat_sparse is not None:
            values = exp_mat_sparse[:, start_idx:end_idx].toarray()
        else:
            values = exp_mat_array[:, start_idx:end_idx]
        return np.ascontiguousarray(values, dtype=np.float32)

    test_const = None if approx else graph.exact_test_constant()

    gpu_succeeded = False
    if use_gpu and gpu_enabled:
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", message="Sparse CSR tensor support is in beta state"
                )
                patches_centered_raw = _centered_patches(patches_raw, n_spots, k_nn)
                patches_gpu = torch.sparse_csr_tensor(
                    torch.from_numpy(
                        np.array(
                            patches_centered_raw.indptr, dtype=np.int64, copy=True
                        )
                    ).cuda(),
                    torch.from_numpy(
                        np.array(
                            patches_centered_raw.indices, dtype=np.int64, copy=True
                        )
                    ).cuda(),
                    torch.from_numpy(patches_centered_raw.data.copy()).cuda(),
                    size=patches_centered_raw.shape,
                    device="cuda",
                )
            batch_size = _gpu_batch_size(n_spots, n_genes)
            ft_tscores_list = []
            effect_size_list = []
            scale = np.sqrt(k_nn * n_spots / 4) if approx else test_const

            for start_idx in range(0, n_genes, batch_size):
                end_idx = min(start_idx + batch_size, n_genes)
                exp_batch_gpu = torch.from_numpy(
                    _dense_batch(start_idx, end_idx)
                ).cuda()
                diff_ik = torch.sparse.mm(patches_gpu, exp_batch_gpu)
                ft_s2 = _gpu_column_sum_float64(diff_ik.square_())
                ft_mean = _gpu_column_sum_float64(exp_batch_gpu) / n_spots
                ft_var = (
                    _gpu_column_sum_float64(exp_batch_gpu.square_()) / n_spots
                    - ft_mean.square_()
                )
                ft_s0 = n_spots * k_nn * (k_nn + 1) * ft_var
                ratio = torch.where(
                    ft_s0 > 0, ft_s2 / ft_s0, torch.ones_like(ft_s0)
                )
                scores = (ratio - 1) * scale
                ft_tscores_list.append(scores.cpu().numpy())
                effect_size_list.append((1 - ratio).cpu().numpy())

            ft_tscores_np = np.concatenate(ft_tscores_list)
            effect_size = np.concatenate(effect_size_list)
            gpu_succeeded = True
        except Exception as exc:
            warnings.warn(
                f"GPU computation failed; using CPU instead: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )

    if not gpu_succeeded:
        patches_centered = _centered_patches(patches_raw, n_spots, k_nn)
        batch_size = max(1, 5_000_000 // n_spots)
        ft_tscores_list = []
        effect_size_list = []
        scale = np.sqrt(k_nn * n_spots / 4) if approx else test_const

        for start_idx in range(0, n_genes, batch_size):
            end_idx = min(start_idx + batch_size, n_genes)
            if exp_mat_sparse is not None:
                exp_batch = exp_mat_sparse[:, start_idx:end_idx]
                diff_ik = patches_centered @ exp_batch
                diff_ik.data **= 2
                ft_s2 = np.asarray(
                    diff_ik.sum(axis=0, dtype=np.float64)
                ).ravel()
                ft_mean = np.asarray(
                    exp_batch.mean(axis=0, dtype=np.float64)
                ).ravel()
                exp_batch_sq = exp_batch.copy()
                exp_batch_sq.data **= 2
                ft_var = (
                    np.asarray(
                        exp_batch_sq.mean(axis=0, dtype=np.float64)
                    ).ravel()
                    - ft_mean**2
                )
            else:
                # A contiguous batch lets the sparse matmul skip an internal copy
                # and speeds up the column reductions below.
                exp_batch = np.ascontiguousarray(
                    exp_mat_array[:, start_idx:end_idx]
                )
                diff_ik = patches_centered @ exp_batch
                np.square(diff_ik, out=diff_ik)
                ft_s2 = diff_ik.sum(axis=0, dtype=np.float64)
                ft_mean = exp_batch.mean(axis=0, dtype=np.float64)
                # Reuse the dense difference buffer for the second moment.
                np.square(exp_batch, out=diff_ik)
                ft_var = (
                    diff_ik.sum(axis=0, dtype=np.float64) / n_spots
                    - ft_mean**2
                )

            ft_s0 = n_spots * k_nn * (k_nn + 1) * np.maximum(ft_var, 0)
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = np.divide(
                    ft_s2,
                    ft_s0,
                    out=np.ones_like(ft_s2, dtype=np.result_type(ft_s2, float)),
                    where=ft_s0 > 0,
                )
            ft_tscores_list.append((ratio - 1) * scale)
            effect_size_list.append(1 - ratio)

        ft_tscores_np = np.concatenate(ft_tscores_list)
        effect_size = np.concatenate(effect_size_list)

    ft_pvalues = 2 * norm.sf(np.abs(ft_tscores_np))

    return pd.DataFrame({
        'gene_names': gene_names,
        'p_values': ft_pvalues,
        'effect_size': effect_size
    })


def stormtrt(
    data: pd.DataFrame,
    control: str,
    sig_level: float = 0.05,
    *,
    conf_level: Optional[float] = None,
) -> dict:
    """
    Two-sample proportion test for spatial pattern significance between groups.

    Conducts a one-sided test comparing the proportion of significant spatial
    patterns in treatment vs control groups.

    Args:
        data: A DataFrame with columns 'Group' and 'Pvalue'.
        control: Name of the control group in the 'Group' column.
        sig_level: Significance threshold for each sample (default 0.05),
                   matching upstream STORM's ``sig.level`` argument.
        conf_level: Deprecated compatibility alias. ``conf_level=0.95`` is
                    equivalent to ``sig_level=0.05``.

    Returns:
        A dictionary containing:
        - statistic: Z test statistic
        - p_value: One-sided p-value (H1: treatment > control)
        - conf_int: Confidence interval for the difference
        - estimate: Proportions in control and treatment groups
        - method: Description of the test
    """
    if 'Group' not in data.columns or 'Pvalue' not in data.columns:
        raise ValueError("data must contain columns 'Group' and 'Pvalue'")

    if conf_level is not None:
        if not 0 < conf_level < 1:
            raise ValueError("conf_level must be between 0 and 1")
        if sig_level != 0.05:
            raise ValueError("Specify only sig_level or conf_level, not both")
        warnings.warn(
            "conf_level is deprecated; use sig_level instead",
            DeprecationWarning,
            stacklevel=2,
        )
        sig_level = 1 - conf_level
    if not 0 < sig_level < 1:
        raise ValueError("sig_level must be between 0 and 1")

    data = data.copy()
    data['Sig'] = (data['Pvalue'] < sig_level).astype(int)

    groups = data['Group'].unique()
    if control not in groups:
        raise ValueError(f"Control group '{control}' not found in 'Group' column")

    groups = list(groups)
    trt_groups = [g for g in groups if g != control]
    if len(trt_groups) != 1:
        raise ValueError("Exactly two groups (control + treatment) are required")
    trt = trt_groups[0]

    control_data = data[data['Group'] == control]
    trt_data = data[data['Group'] == trt]

    x1, n1 = control_data['Sig'].sum(), len(control_data)
    x2, n2 = trt_data['Sig'].sum(), len(trt_data)

    p1 = x1 / n1
    p2 = x2 / n2
    difference = p2 - p1

    # Upstream STORM uses the null-pooled variance for the test statistic.
    pooled = (x1 + x2) / (n1 + n2)
    se_pooled = np.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    if se_pooled > 0:
        z = difference / se_pooled
    elif difference > 0:
        z = np.inf
    elif difference < 0:
        z = -np.inf
    else:
        z = 0.0
    pval = norm.sf(z)

    # The interval uses the usual unpooled standard error under H1.
    se_difference = np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    ci = (difference - norm.ppf(1 - sig_level) * se_difference, np.inf)

    return {
        'statistic': z,
        'parameter': {'n1': n1, 'n2': n2},
        'p_value': pval,
        'conf_int': ci,
        'estimate': {'prop_control': p1, 'prop_treatment': p2},
        'null_value': 0,
        'alternative': 'greater',
        'method': 'One-sided two-sample test for proportions (treatment > control)',
        'data_name': f"Group = {control} vs {trt}"
    }


def storm2trt(
    data: pd.DataFrame,
    alternative: str = "two.sided",
    conf_level: float = 0.95
) -> dict:
    """
    Welch's t-test on effect sizes between two treatment groups.

    Args:
        data: A DataFrame with columns 'Group', 'EffectSize', 'M', 'K'.
              - Group: grouping variable with exactly 2 distinct values
              - EffectSize: numeric effect size values
              - M: number of coordinates in each sample
              - K: number of neighbors used in the test
        alternative: One of "two.sided", "less", or "greater".
        conf_level: Confidence level for the interval (default 0.95).

    Returns:
        A dictionary containing test results similar to R's htest object.
    """
    from scipy.stats import t as t_dist

    if not 0 < conf_level < 1:
        raise ValueError("conf_level must be between 0 and 1")

    required_cols = ['Group', 'EffectSize', 'M', 'K']
    if not all(col in data.columns for col in required_cols):
        raise ValueError(f"data must contain columns: {required_cols}")

    if alternative not in ['two.sided', 'less', 'greater']:
        raise ValueError("alternative must be 'two.sided', 'less', or 'greater'")

    groups = data['Group'].unique()
    if len(groups) != 2:
        raise ValueError("Group must contain exactly 2 distinct values")

    if (data['K'] <= 0).any() or (data['M'] <= 0).any():
        raise ValueError("M and K must contain positive values")
    phi = (4 / (data['K'] * data['M'])).mean()

    g1 = data[data['Group'] == groups[0]]
    g2 = data[data['Group'] == groups[1]]

    n1, n2 = len(g1), len(g2)
    if n1 < 2 or n2 < 2:
        raise ValueError(
            f"Each group must contain at least 2 samples to compute valid group effect variances, "
            f"but found N1={n1} and N2={n2}."
        )

    m1, m2 = g1['EffectSize'].mean(), g2['EffectSize'].mean()
    delta = m1 - m2

    psi1, psi2 = g1['EffectSize'].var(ddof=1), g2['EffectSize'].var(ddof=1)

    var1 = phi + psi1
    var2 = phi + psi2

    se = np.sqrt(var1 / n1 + var2 / n2)

    df = (var1 / n1 + var2 / n2) ** 2 / (
        (var1 ** 2) / (n1 ** 2 * (n1 - 1)) + (var2 ** 2) / (n2 ** 2 * (n2 - 1))
    )

    tstat = delta / se if se > 0 else 0

    alpha = 1 - conf_level
    if alternative == "two.sided":
        pval = 2 * t_dist.sf(abs(tstat), df)
        tcrit = t_dist.ppf(1 - alpha / 2, df)
        ci = (delta - tcrit * se, delta + tcrit * se)
    elif alternative == "greater":
        pval = t_dist.sf(tstat, df)
        tcrit = t_dist.ppf(conf_level, df)
        ci = (delta - tcrit * se, np.inf)
    else:  # less
        pval = t_dist.cdf(tstat, df)
        tcrit = t_dist.ppf(conf_level, df)
        ci = (-np.inf, delta + tcrit * se)

    return {
        'statistic': tstat,
        'parameter': {'df': df},
        'p_value': pval,
        'conf_int': ci,
        'estimate': {'mean_group1': m1, 'mean_group2': m2},
        'null_value': 0,
        'alternative': alternative,
        'method': 'Comparisons of Effect Sizes Between Two Treatment Groups'
    }


def power_storm(
    es: Optional[float] = None,
    n: Optional[int] = None,
    power: Optional[float] = None,
    sig_level: float = 0.05,
    noise: float = 0,
    k: Optional[int] = None
) -> dict:
    """
    Power calculation for STORM method.

    Exactly one of es, n, power, or sig_level must be None.

    Args:
        es: Effect size.
        n: Sample size.
        power: Desired power (0-1).
        sig_level: Significance level (0-1, default 0.05).
        noise: Noise factor (default 0).
        k: Number of nearest neighbors; default is floor(sqrt(n)).

    Returns:
        A dictionary with computed power analysis parameters.
    """
    from scipy.optimize import brentq

    none_count = sum(x is None for x in [es, n, power, sig_level])
    if none_count != 1:
        raise ValueError("Exactly one of es, n, power, sig_level must be None")
    solving_for_effect = es is None

    if sig_level is not None and not (0 < sig_level < 1):
        raise ValueError("sig_level must be between 0 and 1")
    if power is not None and not (0 < power < 1):
        raise ValueError("power must be between 0 and 1")
    if es is not None and es <= 0:
        raise ValueError("es must be positive")
    if n is not None and n <= 0:
        raise ValueError("n must be positive")
    if k is not None and (isinstance(k, bool) or not isinstance(k, Integral) or k < 1):
        raise ValueError("k must be a positive integer")
    if noise <= -1:
        raise ValueError("noise must be greater than -1")

    if n is not None and k is None:
        k = int(np.floor(np.sqrt(n)))

    if n is None:
        if k is None:
            z_crit = norm.ppf(sig_level / 2) + norm.ppf(1 - power)
            n = (4 / (es / (1 + noise)) ** 2 * z_crit ** 2) ** (2 / 3)
            k = int(np.floor(np.sqrt(n)))
        else:
            z_crit = norm.ppf(sig_level / 2) + norm.ppf(1 - power)
            n = 4 / (k * (es / (1 + noise)) ** 2) * z_crit ** 2
    elif power is None:
        ncp = np.sqrt(k * n / 4) * (es / (1 + noise))
        z_alpha = -norm.ppf(sig_level / 2)
        power = 1 - norm.cdf(z_alpha - ncp) + norm.cdf(-z_alpha - ncp)
    elif sig_level is None:
        def alpha_eq(alpha):
            z_alpha2 = norm.ppf(1 - alpha / 2)
            ncp = np.sqrt(k * n / 4) * (es / (1 + noise))
            return 1 - norm.cdf(z_alpha2 - ncp) + norm.cdf(-z_alpha2 - ncp) - power
        sig_level = brentq(alpha_eq, 1e-10, 1 - 1e-10)
    elif es is None:
        # Match upstream R exactly: this branch recomputes K from n even when
        # the caller supplied K_NN.
        k_local = int(np.floor(np.sqrt(n)))
        def es_eq(effect):
            term = np.sqrt(k_local * n / 4) * (effect / (1 + noise))
            z_alpha2 = norm.ppf(1 - sig_level / 2)
            return 1 - norm.cdf(z_alpha2 - term) + norm.cdf(-z_alpha2 - term) - power
        es = brentq(es_eq, 1e-8, 1)

    if k > np.sqrt(n):
        warnings.warn("k should be no greater than sqrt(n)", UserWarning, stacklevel=2)

    return {
        'n': n,
        'effect_size': es,
        'sig_level': sig_level,
        'power': power,
        'noise': noise,
        'k': int(np.floor(np.sqrt(n))) if solving_for_effect else k
    }


def _exact_power_calc(
    nsample: int, ratio: float, p1: float, p2: float, alpha: float
) -> float:
    """Calculate exact power for the small-sample treatment comparison."""
    from scipy.stats import binom

    n1 = int(nsample)
    n2 = round(n1 * ratio)
    x1 = np.arange(n1 + 1)[:, None]
    x2 = np.arange(n2 + 1)[None, :]
    phat1 = x1 / n1
    phat2 = x2 / n2
    se = np.sqrt(phat1 * (1 - phat1) / n1 + phat2 * (1 - phat2) / n2)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.divide(phat2 - phat1, se, out=np.zeros_like(se), where=se > 0)
    p_values = norm.sf(z)
    p_values[(se == 0) & (phat2 > phat1)] = 0
    p_values[(se == 0) & (phat2 <= phat1)] = 1
    probabilities = binom.pmf(x1, n1, p1) * binom.pmf(x2, n2, p2)
    return float(probabilities[p_values < alpha].sum())


def powertrt(
    nsample: Optional[int] = None,
    power: Optional[float] = None,
    sig_level: Optional[float] = 0.05,
    ratio: float = 1,
    power_single: float = 0.8,
    sig_level_single: float = 0.05,
    exact: bool = False,
) -> dict:
    """Power analysis for treatment-versus-control STORM comparisons.

    Exactly one of ``nsample``, ``power``, and ``sig_level`` must be ``None``.
    This is the Python equivalent of upstream STORM's ``powertrt`` function.
    """
    from scipy.optimize import brentq

    if sum(x is None for x in [nsample, power, sig_level]) != 1:
        raise ValueError("Exactly one of nsample, power, and sig_level must be None")
    for name, value in [
        ("power", power),
        ("sig_level", sig_level),
        ("power_single", power_single),
        ("sig_level_single", sig_level_single),
    ]:
        if value is not None and not 0 < value < 1:
            raise ValueError(f"{name} must be between 0 and 1")
    if nsample is not None and nsample < 1:
        raise ValueError("nsample must be positive")
    if ratio <= 0:
        raise ValueError("ratio must be positive")
    if power_single == sig_level_single:
        raise ValueError("power_single and sig_level_single must differ")

    method = (
        "Comparison Between Control and Treatment Groups (Equal Size)"
        if ratio == 1
        else "Comparison Between Control and Treatment Groups (Unequal Size)"
    )
    pooled = (sig_level_single + power_single) / 2
    pooled_scale = np.sqrt(2 * pooled * (1 - pooled))
    alternative_scale = np.sqrt(
        sig_level_single * (1 - sig_level_single) / ratio
        + power_single * (1 - power_single)
    )

    if nsample is None:
        z_alpha = norm.ppf(1 - sig_level)
        z_beta = norm.ppf(power)
        nsample = (
            (z_alpha * pooled_scale + z_beta * alternative_scale) ** 2
            / (power_single - sig_level_single) ** 2
        )
    elif power is None:
        if exact:
            power = _exact_power_calc(
                int(nsample), ratio, sig_level_single, power_single, sig_level
            )
        else:
            z_beta = (
                np.sqrt(nsample) * (power_single - sig_level_single)
                - norm.ppf(1 - sig_level) * pooled_scale
            ) / alternative_scale
            power = norm.cdf(z_beta)
    else:
        def alpha_eq(alpha: float) -> float:
            z_beta = (
                np.sqrt(nsample) * (power_single - sig_level_single)
                - norm.ppf(1 - alpha) * pooled_scale
            ) / alternative_scale
            return norm.cdf(z_beta) - power

        sig_level = brentq(alpha_eq, 1e-6, 0.5)

    return {
        "nsample": nsample,
        "power": power,
        "sig_level": sig_level,
        "ratio": ratio,
        "power_single": power_single,
        "sig_level_single": sig_level_single,
        "method": method,
        "exact": exact,
    }


def power2trt(
    delta: Optional[float] = None,
    phi: Optional[float] = None,
    psi1: Optional[float] = None,
    psi2: Optional[float] = None,
    nsample: Optional[float] = None,
    ratio: Optional[float] = None,
    sig_level: Optional[float] = None,
    power: Optional[float] = None,
    alternative: str = "two.sided",
) -> dict:
    """Power analysis for comparisons between two treatment groups."""
    from scipy.optimize import brentq
    from scipy.stats import nct, t as t_dist

    if alternative not in {"two.sided", "less", "greater"}:
        raise ValueError("alternative must be 'two.sided', 'less', or 'greater'")
    if sum(x is None for x in [delta, nsample, ratio, sig_level, power]) != 1:
        raise ValueError(
            "Exactly one of delta, nsample, ratio, sig_level, and power must be None"
        )
    if phi is None or psi1 is None or psi2 is None:
        raise ValueError("phi, psi1, and psi2 must all be provided")
    if phi + psi1 <= 0 or phi + psi2 <= 0:
        raise ValueError("phi + psi1 and phi + psi2 must be positive")
    if nsample is not None and nsample <= 1:
        raise ValueError("nsample must be greater than 1")
    if ratio is not None and ratio <= 0:
        raise ValueError("ratio must be positive")
    if sig_level is not None and not 0 < sig_level < 1:
        raise ValueError("sig_level must be between 0 and 1")
    if power is not None and not 0 < power < 1:
        raise ValueError("power must be between 0 and 1")

    var1 = phi + psi1
    var2 = phi + psi2

    def power_function(d: float, n: float, r: float, alpha: float) -> float:
        n2 = n * r
        if n <= 1 or n2 <= 1:
            return np.nan
        se = np.sqrt(var1 / n + var2 / n2)
        df = (var1 / n + var2 / n2) ** 2 / (
            var1**2 / (n**2 * (n - 1))
            + var2**2 / (n2**2 * (n2 - 1))
        )
        ncp = d / se
        if alternative == "two.sided":
            critical = t_dist.ppf(1 - alpha / 2, df)
            result = nct.cdf(-critical, df, ncp) + nct.sf(critical, df, ncp)
            fallback = norm.cdf(-critical - ncp) + norm.sf(critical - ncp)
        elif alternative == "greater":
            critical = t_dist.ppf(1 - alpha, df)
            result = nct.sf(critical, df, ncp)
            fallback = norm.sf(critical - ncp)
        else:
            critical = t_dist.ppf(alpha, df)
            result = nct.cdf(critical, df, ncp)
            fallback = norm.cdf(critical - ncp)
        return float(result if np.isfinite(result) else fallback)

    if power is None:
        power = power_function(delta, nsample, ratio, sig_level)
    elif delta is None:
        bounds = (-1e8, -1e-8) if alternative == "less" else (1e-8, 1e8)
        delta = brentq(
            lambda value: power_function(value, nsample, ratio, sig_level) - power,
            *bounds,
        )
    elif nsample is None:
        nsample = brentq(
            lambda value: power_function(delta, value, ratio, sig_level) - power,
            2.000001,
            1e7,
        )
    elif ratio is None:
        ratio = brentq(
            lambda value: power_function(delta, nsample, value, sig_level) - power,
            max(1.000001 / nsample, 1e-3),
            1e3,
        )
    else:
        sig_level = brentq(
            lambda value: power_function(delta, nsample, ratio, value) - power,
            1e-10,
            1 - 1e-10,
        )

    return {
        "delta": delta,
        "phi": phi,
        "psi1": psi1,
        "psi2": psi2,
        "nsample": nsample,
        "ratio": ratio,
        "sig_level": sig_level,
        "power": power,
        "alternative": alternative,
        "method": "Welch Two-Sample t-test power calculation (Satterthwaite df)",
    }
