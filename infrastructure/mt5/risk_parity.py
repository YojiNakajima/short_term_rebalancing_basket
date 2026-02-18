from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


def ewma_lambda_from_half_life(half_life_bars: float) -> float:
    """Convert EWMA half-life (in bars) to lambda.

    EWMA update:
        S_t = lambda * S_{t-1} + (1-lambda) * r_t r_t^T
    """

    h = float(half_life_bars)
    if not math.isfinite(h) or h <= 0:
        raise ValueError("half_life_bars must be positive")
    return float(math.exp(math.log(0.5) / h))


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return float(sum(float(x) * float(y) for x, y in zip(a, b)))


def _mat_vec(m: Sequence[Sequence[float]], v: Sequence[float]) -> List[float]:
    out: List[float] = []
    for row in m:
        out.append(_dot(row, v))
    return out


def _outer(v: Sequence[float]) -> List[List[float]]:
    vv: List[List[float]] = []
    for x in v:
        vx: List[float] = []
        fx = float(x)
        for y in v:
            vx.append(fx * float(y))
        vv.append(vx)
    return vv


def _zeros(n: int) -> List[List[float]]:
    return [[0.0 for _ in range(n)] for _ in range(n)]


def _is_square(m: Sequence[Sequence[float]]) -> bool:
    try:
        n = len(m)
    except Exception:
        return False
    return all(isinstance(r, Sequence) and len(r) == n for r in m)


def covariance_to_correlation(cov: Sequence[Sequence[float]]) -> List[List[float]]:
    if not _is_square(cov):
        raise ValueError("cov must be a square matrix")
    n = len(cov)
    diag = [float(cov[i][i]) for i in range(n)]
    std = []
    for v in diag:
        if not math.isfinite(v) or v <= 0:
            std.append(float("nan"))
        else:
            std.append(math.sqrt(v))

    corr = _zeros(n)
    for i in range(n):
        for j in range(n):
            si = std[i]
            sj = std[j]
            if not (math.isfinite(si) and math.isfinite(sj)) or si <= 0 or sj <= 0:
                corr[i][j] = float("nan")
                continue
            corr[i][j] = float(cov[i][j]) / (si * sj)
    return corr


def ewma_covariance(
    returns: Sequence[Sequence[float]],
    *,
    half_life_bars: float,
    ridge: float = 0.0,
) -> List[List[float]]:
    """EWMA covariance from a sequence of return vectors.

    returns: T x N
    """

    if not returns:
        raise ValueError("returns is empty")

    n = len(returns[0])
    if n <= 0:
        raise ValueError("returns must have positive dimension")
    for r in returns:
        if len(r) != n:
            raise ValueError("returns must have consistent dimension")

    lam = ewma_lambda_from_half_life(half_life_bars)
    one_minus = 1.0 - lam

    # Initialize with the first observation outer product.
    s = _outer([float(x) for x in returns[0]])
    for r in returns[1:]:
        rr = _outer([float(x) for x in r])
        for i in range(n):
            row_s = s[i]
            row_rr = rr[i]
            for j in range(n):
                row_s[j] = (lam * row_s[j]) + (one_minus * row_rr[j])

    ridge_f = float(ridge)
    if math.isfinite(ridge_f) and ridge_f > 0:
        # Scale ridge by average variance (to be dimensionless across assets).
        diag = [float(s[i][i]) for i in range(n)]
        finite_diag = [v for v in diag if math.isfinite(v) and v > 0]
        scale = float(sum(finite_diag) / len(finite_diag)) if finite_diag else 1.0
        add = ridge_f * scale
        for i in range(n):
            s[i][i] = float(s[i][i]) + add

    return s


def risk_parity_weights(
    cov: Sequence[Sequence[float]],
    *,
    max_iter: int = 5000,
    tol: float = 1e-8,
    step: float = 0.5,
    min_weight: float = 1e-12,
) -> List[float]:
    """Long-only Equal Risk Contribution (ERC) weights.

    Pure Python multiplicative update. Assumes cov is (regularized) PSD/PD.
    Returns weights that sum to 1.
    """

    if not _is_square(cov):
        raise ValueError("cov must be a square matrix")
    n = len(cov)
    if n == 0:
        raise ValueError("cov must be non-empty")

    # Start from uniform strictly-positive weights.
    w = [1.0 / float(n) for _ in range(n)]
    step_f = float(step)
    if not math.isfinite(step_f) or step_f <= 0:
        step_f = 0.5
    tol_f = float(tol)
    if not math.isfinite(tol_f) or tol_f <= 0:
        tol_f = 1e-8

    eps = float(min_weight) if math.isfinite(float(min_weight)) and float(min_weight) > 0 else 1e-12

    for _ in range(int(max_iter)):
        m = _mat_vec(cov, w)  # Sigma * w
        c = _dot(w, m)  # w^T Sigma w
        if not math.isfinite(c) or c <= 0:
            raise ValueError("invalid portfolio variance during ERC solve")

        target = c / float(n)
        max_rel_err = 0.0

        # multiplicative update
        for i in range(n):
            wi = float(w[i])
            mi = float(m[i])
            g = wi * mi
            if not math.isfinite(g) or g <= 0:
                # Keep it positive; will be corrected by normalization.
                w[i] = eps
                continue

            ratio = target / g
            if math.isfinite(ratio) and ratio > 0:
                w[i] = wi * (ratio ** step_f)
            else:
                w[i] = eps

            rel_err = abs(g - target) / target
            if rel_err > max_rel_err:
                max_rel_err = rel_err

        s = sum(w)
        if not math.isfinite(s) or s <= 0:
            raise ValueError("invalid weight sum during ERC solve")
        w = [max(float(x) / s, eps) for x in w]
        s2 = sum(w)
        w = [float(x) / s2 for x in w]

        if max_rel_err < tol_f:
            return [float(x) for x in w]

    # Return best effort; caller can decide to accept or fallback.
    return [float(x) for x in w]


def log_returns_from_aligned_closes(
    close_by_symbol: Dict[str, Sequence[float]],
    *,
    symbols: Sequence[str],
    abs_clip: Optional[float] = None,
) -> List[List[float]]:
    """Compute close-to-close log returns from already aligned close arrays.

    close arrays must all have the same length and correspond to the same timestamps.
    Output: (T-1) x N
    """

    sym_list = list(symbols)
    if not sym_list:
        return []

    closes = [list(close_by_symbol[s]) for s in sym_list]
    t = len(closes[0])
    if t < 2:
        return []
    for c in closes:
        if len(c) != t:
            raise ValueError("close series are not aligned")

    clip = float(abs_clip) if abs_clip is not None else None
    if clip is not None and (not math.isfinite(clip) or clip <= 0):
        clip = None

    out: List[List[float]] = []
    for k in range(1, t):
        row: List[float] = []
        for i in range(len(sym_list)):
            p0 = float(closes[i][k - 1])
            p1 = float(closes[i][k])
            if not (math.isfinite(p0) and math.isfinite(p1)) or p0 <= 0 or p1 <= 0:
                raise ValueError("non-finite or non-positive close")
            r = math.log(p1 / p0)
            if clip is not None:
                r = max(-clip, min(clip, r))
            row.append(float(r))
        out.append(row)
    return out


@dataclass(frozen=True)
class CorrelationEstimationResult:
    symbols: List[str]
    cov: List[List[float]]
    corr: List[List[float]]
    weights: List[float]
