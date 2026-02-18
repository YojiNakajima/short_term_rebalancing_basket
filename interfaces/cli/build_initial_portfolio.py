from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

from config import settings
from infrastructure.dB.sqlite3.portfolio import persist_initial_portfolio_run
from infrastructure.mt5.connect import Mt5ConnectionError, load_mt5_config
from infrastructure.mt5.symbols_report import SymbolSnapshot, collect_target_symbol_snapshots
from infrastructure.mt5.risk_parity import (
    CorrelationEstimationResult,
    covariance_to_correlation,
    ewma_covariance,
    log_returns_from_aligned_closes,
    risk_parity_weights,
)

try:
    import MetaTrader5 as mt5
except ImportError:  # pragma: no cover
    mt5 = None


def _safe_last_error(module: Any) -> Optional[Any]:
    try:
        return module.last_error()
    except Exception:
        return None


def _safe_input(prompt: str) -> Optional[str]:
    try:
        return input(prompt)
    except EOFError:
        return None


def _stdin_is_interactive() -> bool:
    try:
        if not sys.stdin:
            return False
        if getattr(sys.stdin, "closed", False):
            return False

        # Normal terminal execution.
        if hasattr(sys.stdin, "isatty") and bool(sys.stdin.isatty()):
            return True

        # Some IDE consoles may report isatty()==False but still accept interactive input.
        if os.getenv("PYCHARM_HOSTED"):
            return True
        if os.getenv("VSCODE_PID"):
            return True

        return False
    except Exception:
        return False


@dataclass(frozen=True)
class PlannedPosition:
    symbol: str
    atr_pct: Optional[float]
    weight: Optional[float]
    risk_usd: Optional[float]
    risk_per_lot_at_1atr: Optional[float]
    ideal_lot: Optional[float]
    lot: float
    # Entry direction for the planned position.
    # Currently the initial portfolio builder always plans long entries.
    direction: str = "BUY"
    volume_min: Optional[float] = None
    volume_step: Optional[float] = None
    volume_max: Optional[float] = None
    usd_per_lot: Optional[float] = None
    usd_nominal: Optional[float] = None
    reason: Optional[str] = None


def inverse_volatility_weights(rows: Iterable[SymbolSnapshot]) -> dict[str, float]:
    valid: list[tuple[str, float]] = []
    for r in rows:
        if not r.exists:
            continue
        if r.atr_pct is None:
            continue
        try:
            v = float(r.atr_pct)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(v) or v <= 0:
            continue
        valid.append((r.symbol, v))

    raw = {sym: 1.0 / v for sym, v in valid}
    s = sum(raw.values())
    if s <= 0 or not math.isfinite(s):
        return {}
    return {sym: w / s for sym, w in raw.items()}


def _timeframe_from_settings_value(value: Any, *, module: Any) -> Optional[int]:
    """Map settings.TIME_FRAME (e.g. 4 for 4H) to MT5 timeframe constant."""

    if value is None:
        return None

    try:
        v = int(value)
    except (TypeError, ValueError):
        return None

    h1 = getattr(module, "TIMEFRAME_H1", None)
    h4 = getattr(module, "TIMEFRAME_H4", None)

    # If caller already passed MT5 constant, accept it as-is.
    if v in {h1, h4}:
        return v

    mapping = {
        1: h1,
        4: h4,
    }
    return mapping.get(v)


def _rate_value(rec: Any, key: str) -> float:
    # Support numpy records (rec['close']), dicts, and SimpleNamespace/objects.
    try:
        return float(rec[key])
    except Exception:
        return float(getattr(rec, key))


def _normalize_rates_order(rates: Any) -> Any:
    try:
        n = len(rates)
    except Exception:
        return rates
    if n <= 1:
        return rates
    try:
        t0 = _rate_value(rates[0], "time")
        t1 = _rate_value(rates[-1], "time")
    except Exception:
        return rates
    if t0 <= t1:
        return rates
    # Reverse (oldest -> newest)
    try:
        return list(reversed(list(rates)))
    except Exception:
        return rates


def _fetch_symbol_close_series_by_time(
    symbol: str,
    *,
    timeframe_const: int,
    count: int,
    module: Any,
) -> dict[int, float]:
    rates = module.copy_rates_from_pos(symbol, timeframe_const, 1, int(count))
    if rates is None:
        raise RuntimeError(f"copy_rates_from_pos returned None for {symbol}")
    rs = _normalize_rates_order(rates)
    out: dict[int, float] = {}
    for rec in rs:
        t = int(_rate_value(rec, "time"))
        c = float(_rate_value(rec, "close"))
        out[t] = c
    return out


def estimate_erc_from_mt5(
    symbols: list[str],
    *,
    settings_module: Any,
    module: Any,
) -> CorrelationEstimationResult:
    """Estimate EWMA covariance/correlation from MT5 history and compute ERC weights."""

    timeframe_const = _timeframe_from_settings_value(getattr(settings_module, "TIME_FRAME", None), module=module)
    if timeframe_const is None:
        raise RuntimeError("Unsupported TIME_FRAME for correlation estimation")

    lookback = int(getattr(settings_module, "CORR_LOOKBACK_BARS", 1200))
    half_life = float(getattr(settings_module, "CORR_EWMA_HALF_LIFE_BARS", 90))
    abs_clip = getattr(settings_module, "CORR_LOG_RETURN_ABS_CLIP", None)
    ridge = float(getattr(settings_module, "CORR_COV_RIDGE", 0.0))

    max_iter = int(getattr(settings_module, "ERC_MAX_ITER", 5000))
    tol = float(getattr(settings_module, "ERC_TOL", 1e-8))
    step = float(getattr(settings_module, "ERC_STEP", 0.5))

    # Fetch a bit more to survive missing bars; alignment will take the intersection.
    fetch_count = max(lookback + 1, lookback + 50)

    by_time: dict[str, dict[int, float]] = {}
    for s in symbols:
        by_time[s] = _fetch_symbol_close_series_by_time(
            s, timeframe_const=timeframe_const, count=fetch_count, module=module
        )

    # Align by intersection of timestamps.
    times = set(by_time[symbols[0]].keys())
    for s in symbols[1:]:
        times &= set(by_time[s].keys())
    aligned_times = sorted(times)

    if len(aligned_times) < 2:
        raise RuntimeError("Not enough aligned history to estimate correlation")

    # Keep the most recent lookback+1 closes (so we get lookback returns).
    aligned_times = aligned_times[-(lookback + 1) :]
    close_by_symbol = {
        s: [float(by_time[s][t]) for t in aligned_times]
        for s in symbols
    }

    returns = log_returns_from_aligned_closes(
        close_by_symbol, symbols=symbols, abs_clip=abs_clip
    )
    if len(returns) < 2:
        raise RuntimeError("Not enough returns to estimate covariance")

    cov = ewma_covariance(returns, half_life_bars=half_life, ridge=ridge)
    corr = covariance_to_correlation(cov)
    w = risk_parity_weights(cov, max_iter=max_iter, tol=tol, step=step)
    return CorrelationEstimationResult(symbols=list(symbols), cov=cov, corr=corr, weights=w)


def _round_to_volume_constraints(
    ideal_lot: float,
    *,
    volume_min: Optional[float],
    volume_step: Optional[float],
    volume_max: Optional[float],
) -> float:
    if volume_min is None or volume_step is None:
        return float(ideal_lot)

    vmin = float(volume_min)
    step = float(volume_step)
    vmax = float(volume_max) if volume_max is not None else None

    if step <= 0 or not math.isfinite(step):
        return max(float(ideal_lot), vmin)

    lot = max(float(ideal_lot), vmin)
    if vmax is not None:
        lot = min(lot, vmax)

    # Align to step increments starting from vmin.
    n = (lot - vmin) / step
    if not math.isfinite(n):
        return vmin

    n_round = int(round(n))
    lot_rounded = vmin + (n_round * step)

    if lot_rounded < vmin:
        lot_rounded = vmin
    if vmax is not None and lot_rounded > vmax:
        lot_rounded = vmax

    # Avoid float artifacts (e.g. 0.30000000004)
    prec = max(0, int(round(-math.log10(step))) if step < 1 else 0)
    return float(round(lot_rounded, prec + 2))


def build_initial_portfolio_from_snapshots(
    rows: list[SymbolSnapshot],
    *,
    tpv: float,
    risk_pct: float,
    weights: Optional[dict[str, float]] = None,
) -> list[PlannedPosition]:
    weights = dict(weights) if weights is not None else inverse_volatility_weights(rows)
    total_risk_usd = float(tpv) * float(risk_pct)

    plan: list[PlannedPosition] = []
    for r in rows:
        vmin = r.volume_min
        min_lot = float(vmin) if vmin is not None else 0.0

        # Always include the symbol.
        if r.symbol not in weights:
            usd_nominal = (
                float(r.usd_per_lot) * min_lot if r.usd_per_lot is not None else None
            )
            plan.append(
                PlannedPosition(
                    symbol=r.symbol,
                    atr_pct=r.atr_pct,
                    weight=None,
                    risk_usd=None,
                    risk_per_lot_at_1atr=None,
                    ideal_lot=None,
                    lot=min_lot,
                    volume_min=r.volume_min,
                    volume_step=r.volume_step,
                    volume_max=r.volume_max,
                    usd_per_lot=r.usd_per_lot,
                    usd_nominal=usd_nominal,
                    reason="fallback_min_lot",
                )
            )
            continue

        w = float(weights[r.symbol])
        risk_i = total_risk_usd * w

        # Use lot_for_base/usd_for_base_lot as a "unit".
        if (
            r.atr_pct is None
            or r.lot_for_base is None
            or r.usd_for_base_lot is None
            or r.atr_pct <= 0
        ):
            usd_nominal = (
                float(r.usd_per_lot) * min_lot if r.usd_per_lot is not None else None
            )
            plan.append(
                PlannedPosition(
                    symbol=r.symbol,
                    atr_pct=r.atr_pct,
                    weight=w,
                    risk_usd=risk_i,
                    risk_per_lot_at_1atr=None,
                    ideal_lot=None,
                    lot=min_lot,
                    volume_min=r.volume_min,
                    volume_step=r.volume_step,
                    volume_max=r.volume_max,
                    usd_per_lot=r.usd_per_lot,
                    usd_nominal=usd_nominal,
                    reason="invalid_inputs_min_lot",
                )
            )
            continue

        # 1 unit (lot_for_base) risk at 1ATR (using ATR%)
        risk_per_unit = float(r.usd_for_base_lot) * (float(r.atr_pct) / 100.0)
        if risk_per_unit <= 0 or not math.isfinite(risk_per_unit):
            usd_nominal = (
                float(r.usd_per_lot) * min_lot if r.usd_per_lot is not None else None
            )
            plan.append(
                PlannedPosition(
                    symbol=r.symbol,
                    atr_pct=r.atr_pct,
                    weight=w,
                    risk_usd=risk_i,
                    risk_per_lot_at_1atr=None,
                    ideal_lot=None,
                    lot=min_lot,
                    volume_min=r.volume_min,
                    volume_step=r.volume_step,
                    volume_max=r.volume_max,
                    usd_per_lot=r.usd_per_lot,
                    usd_nominal=usd_nominal,
                    reason="invalid_risk_per_unit_min_lot",
                )
            )
            continue

        units = risk_i / risk_per_unit
        ideal_lot = float(units) * float(r.lot_for_base)

        lot = _round_to_volume_constraints(
            ideal_lot,
            volume_min=r.volume_min,
            volume_step=r.volume_step,
            volume_max=r.volume_max,
        )
        if lot < min_lot:
            lot = min_lot

        usd_nominal = float(r.usd_per_lot) * float(lot) if r.usd_per_lot is not None else None

        risk_per_lot_at_1atr = None
        if r.usd_for_base_lot is not None and r.lot_for_base is not None:
            usd_per_1lot_approx = float(r.usd_for_base_lot) / float(r.lot_for_base)
            risk_per_lot_at_1atr = usd_per_1lot_approx * (float(r.atr_pct) / 100.0)

        plan.append(
            PlannedPosition(
                symbol=r.symbol,
                atr_pct=r.atr_pct,
                weight=w,
                risk_usd=risk_i,
                risk_per_lot_at_1atr=risk_per_lot_at_1atr,
                ideal_lot=ideal_lot,
                lot=float(lot),
                volume_min=r.volume_min,
                volume_step=r.volume_step,
                volume_max=r.volume_max,
                usd_per_lot=r.usd_per_lot,
                usd_nominal=usd_nominal,
                reason=None,
            )
        )

    return plan


def _format_portfolio_table(plan: list[PlannedPosition]) -> str:
    def s(v: Any) -> str:
        if v is None:
            return "-"
        # NOTE: show all numeric values with 2 decimal places.
        # (Avoid treating bool as int.)
        if isinstance(v, bool):
            return str(v)
        if isinstance(v, (int, float)):
            fv = float(v)
            if not math.isfinite(fv):
                return "-"
            return f"{fv:.2f}"
        return str(v)

    header = [
        "symbol",
        "direction",
        "atr_pct",
        "weight",
        "risk_usd",
        "ideal_lot",
        "lot",
        "usd_nominal",
        "reason",
    ]

    rows: list[list[str]] = [header]
    for p in plan:
        rows.append(
            [
                p.symbol,
                str(p.direction),
                s(p.atr_pct),
                s(p.weight),
                s(p.risk_usd),
                s(p.ideal_lot),
                s(p.lot),
                s(p.usd_nominal),
                s(p.reason),
            ]
        )

    widths = [max(len(r[i]) for r in rows) for i in range(len(header))]

    def fmt_row(r: list[str]) -> str:
        return " | ".join(v.ljust(widths[i]) for i, v in enumerate(r))

    sep = "-+-".join("-" * w for w in widths)
    lines = [fmt_row(rows[0]), sep]
    lines.extend(fmt_row(r) for r in rows[1:])
    return "\n".join(lines)


def _normalize_direction(direction: str) -> str:
    d = str(direction).strip().upper()
    if d in {"BUY", "LONG", "L"}:
        return "BUY"
    if d in {"SELL", "SHORT", "S"}:
        return "SELL"
    raise ValueError(f"Unsupported direction: {direction!r}")


def build_mt5_market_order_requests(
    plan: list[PlannedPosition],
    *,
    module: Any,
    deviation: int = 20,
    magic: Optional[int] = None,
    comment: str = "initial_portfolio",
) -> list[dict[str, Any]]:
    """Build MT5 market order request dicts from the planned positions.

    Note: This function does not call order_send(). It only creates request payloads.
    """

    if module is None:
        raise RuntimeError("MT5 module is required")

    requests: list[dict[str, Any]] = []
    for p in plan:
        lot = float(p.lot)
        if not math.isfinite(lot) or lot <= 0:
            continue

        direction = _normalize_direction(p.direction)

        # Ensure symbol is available in Market Watch.
        if hasattr(module, "symbol_select"):
            ok = bool(module.symbol_select(p.symbol, True))
            if not ok:
                raise RuntimeError(f"symbol_select failed for {p.symbol}")

        tick = module.symbol_info_tick(p.symbol)
        if tick is None:
            raise RuntimeError(f"symbol_info_tick returned None for {p.symbol}")

        ask = getattr(tick, "ask", None)
        bid = getattr(tick, "bid", None)
        if direction == "BUY":
            price = float(ask) if ask is not None else None
            order_type = getattr(module, "ORDER_TYPE_BUY", None)
        else:
            price = float(bid) if bid is not None else None
            order_type = getattr(module, "ORDER_TYPE_SELL", None)

        if price is None or not math.isfinite(float(price)):
            raise RuntimeError(
                f"Invalid tick price for {p.symbol}: bid={bid!r} ask={ask!r}"
            )
        if order_type is None:
            raise RuntimeError("MT5 ORDER_TYPE_* constants are not available")

        req: dict[str, Any] = {
            "action": getattr(module, "TRADE_ACTION_DEAL", None),
            "symbol": p.symbol,
            "volume": lot,
            "type": order_type,
            "price": float(price),
            "deviation": int(deviation),
            "comment": str(comment),
        }

        if magic is not None:
            req["magic"] = int(magic)

        # Optional fields - add only if available.
        if hasattr(module, "ORDER_TIME_GTC"):
            req["type_time"] = getattr(module, "ORDER_TIME_GTC")

        # Prefer the broker/symbol suggested filling_mode when available.
        # NOTE: Do not force a default here; some brokers reject unsupported filling modes.
        filling: Optional[int] = None
        info = None
        if hasattr(module, "symbol_info"):
            info = module.symbol_info(p.symbol)
        if info is not None and hasattr(info, "filling_mode") and info.filling_mode is not None:
            try:
                filling = int(info.filling_mode)
            except Exception:
                filling = None
        if filling is not None:
            req["type_filling"] = filling

        if req["action"] is None:
            raise RuntimeError("MT5 TRADE_ACTION_DEAL constant is not available")

        requests.append(req)

    return requests


def execute_mt5_market_orders(
    requests: list[dict[str, Any]],
    *,
    module: Any,
) -> list[Any]:
    if module is None:
        raise RuntimeError("MT5 module is required")
    if not hasattr(module, "order_send"):
        raise RuntimeError("MT5 order_send() is required")

    results: list[Any] = []

    ok_codes = {
        getattr(module, "TRADE_RETCODE_DONE", None),
        getattr(module, "TRADE_RETCODE_PLACED", None),
        getattr(module, "TRADE_RETCODE_DONE_PARTIAL", None),
    }
    ok_codes.discard(None)

    invalid_fill_codes = {
        getattr(module, "TRADE_RETCODE_INVALID_FILL", None),
        10030,  # Observed in practice for "Unsupported filling mode".
    }
    invalid_fill_codes.discard(None)

    filling_candidates: list[Optional[int]] = []
    for name in (
        "ORDER_FILLING_RETURN",
        "ORDER_FILLING_IOC",
        "ORDER_FILLING_FOK",
        "ORDER_FILLING_BOC",
    ):
        if hasattr(module, name):
            try:
                filling_candidates.append(int(getattr(module, name)))
            except Exception:
                continue
    # As a last resort, try omitting type_filling.
    filling_candidates.append(None)

    def is_ok(res: Any) -> bool:
        if not ok_codes:
            return True
        if not hasattr(res, "retcode"):
            return True
        return res.retcode in ok_codes

    def is_unsupported_filling(res: Any) -> bool:
        rc = getattr(res, "retcode", None)
        if rc in invalid_fill_codes:
            return True
        comment = str(getattr(res, "comment", ""))
        c = comment.lower()
        return ("unsupported" in c and "filling" in c) or ("unsupported filling mode" in c)

    for req in requests:
        res = module.order_send(req)
        if res is None:
            raise RuntimeError(f"order_send returned None for request={req!r}")

        if is_ok(res):
            results.append(res)
            continue

        # Auto-fallback for unsupported filling modes.
        if is_unsupported_filling(res):
            original_filling = req.get("type_filling", None)
            recovered = False
            last_res = res

            for filling in filling_candidates:
                # Skip if it would be the same request.
                if filling is None:
                    if "type_filling" not in req:
                        continue
                else:
                    if req.get("type_filling") == filling:
                        continue

                new_req = dict(req)
                if filling is None:
                    new_req.pop("type_filling", None)
                else:
                    new_req["type_filling"] = filling

                retry_res = module.order_send(new_req)
                if retry_res is None:
                    last_res = retry_res
                    continue
                last_res = retry_res

                if is_ok(retry_res):
                    # Update the request in-place so callers see the actually used filling mode.
                    req.clear()
                    req.update(new_req)
                    results.append(retry_res)
                    recovered = True
                    break

                # NOTE: Do not stop early if the retry failed for a different reason.
                # Some brokers return different retcodes/comments for unsupported filling modes.
                # Continue trying remaining candidates, including omitting type_filling.

            if recovered:
                continue

            raise RuntimeError(
                "order_send failed (unsupported filling mode): "
                f"original_type_filling={original_filling!r} last_result={last_res}"
            )

        if ok_codes and hasattr(res, "retcode") and res.retcode not in ok_codes:
            raise RuntimeError(f"order_send failed: retcode={res.retcode} result={res}")

        results.append(res)

    return results


def trade_planned_positions_on_mt5(
    plan: list[PlannedPosition],
    *,
    env_file: Optional[os.PathLike[str] | str] = None,
    module: Any = mt5,
    deviation: int = 20,
    magic: Optional[int] = None,
    comment: str = "initial_portfolio",
    dry_run: bool = False,
) -> tuple[list[dict[str, Any]], list[Any]]:
    """Connect to MT5 and (optionally) send market orders based on the plan.

    Returns: (requests, results)
      - requests: order_send request payloads built from the plan
      - results: order_send results (empty when dry_run=True)
    """

    if module is None:
        raise RuntimeError("MT5 module is required")
    if not hasattr(module, "initialize") or not hasattr(module, "shutdown"):
        raise RuntimeError("MT5 initialize()/shutdown() are required")

    config = load_mt5_config(env_file)

    initialized = False
    try:
        initialized = bool(
            module.initialize(
                path=config.path,
                login=config.login,
                password=config.password,
                server=config.server,
                portable=config.portable,
            )
        )
        if not initialized:
            raise Mt5ConnectionError(
                f"MT5 initialize failed. last_error={_safe_last_error(module)}"
            )

        requests = build_mt5_market_order_requests(
            plan,
            module=module,
            deviation=deviation,
            magic=magic,
            comment=comment,
        )

        if dry_run:
            return requests, []

        results = execute_mt5_market_orders(requests, module=module)
        return requests, results
    finally:
        if initialized:
            module.shutdown()


def save_portfolio_plots(
    plan: list[PlannedPosition],
    *,
    output_dir: os.PathLike[str] | str,
    title_prefix: str = "",
) -> list[Path]:
    # Backward-compatible wrapper: keep "save to PNG" behavior.
    return render_portfolio_plots(
        plan,
        output_dir=output_dir,
        title_prefix=title_prefix,
        show=False,
        force_agg=True,
    )


def render_portfolio_plots(
    plan: list[PlannedPosition],
    *,
    title_prefix: str = "",
    output_dir: Optional[os.PathLike[str] | str] = None,
    show: bool = False,
    force_agg: bool = False,
    corr_symbols: Optional[list[str]] = None,
    corr_matrix: Optional[list[list[float]]] = None,
    weights_title: str = "Weights",
) -> list[Path]:
    """Render portfolio plots.

    - When show=True, figures are displayed (PyCharm SciView can capture them).
    - When output_dir is provided, figures are saved as PNG.
    """

    try:
        import matplotlib

        if force_agg and not show:
            # Ensure headless-safe backend for file-only output.
            matplotlib.use("Agg")

        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "matplotlib is required for plotting. Install dependencies from requirements.txt"
        ) from e

    out: Optional[Path]
    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
    else:
        out = None

    symbols = [p.symbol for p in plan]
    weights = [float(p.weight) if p.weight is not None else 0.0 for p in plan]
    lots = [float(p.lot) for p in plan]
    risks = [float(p.risk_usd) if p.risk_usd is not None else 0.0 for p in plan]

    saved: list[Path] = []
    figs = []

    def maybe_save(fig, filename: str) -> None:
        if out is None:
            return
        p = out / filename
        fig.savefig(p)
        saved.append(p)

    # Weights bar chart
    fig, ax = plt.subplots(figsize=(max(8, len(symbols) * 0.6), 4.5))
    ax.bar(symbols, weights)
    ax.set_ylabel("weight")
    ax.set_title(f"{title_prefix}{weights_title}".strip())
    ax.tick_params(axis="x", rotation=45, labelsize=9)
    fig.tight_layout()
    maybe_save(fig, "weights_bar.png")
    figs.append(fig)

    # Correlation heatmap (optional)
    if corr_symbols is not None and corr_matrix is not None:
        fig, ax = plt.subplots(figsize=(max(7, len(corr_symbols) * 0.6), max(6, len(corr_symbols) * 0.6)))
        im = ax.imshow(corr_matrix, vmin=-1.0, vmax=1.0, cmap="coolwarm")
        ax.set_xticks(list(range(len(corr_symbols))))
        ax.set_yticks(list(range(len(corr_symbols))))
        ax.set_xticklabels(corr_symbols, rotation=45, ha="right", fontsize=9)
        ax.set_yticklabels(corr_symbols, fontsize=9)
        ax.set_title(f"{title_prefix}Correlation (EWMA)".strip())
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        maybe_save(fig, "corr_heatmap.png")
        figs.append(fig)

    # Lots bar chart
    fig, ax = plt.subplots(figsize=(max(8, len(symbols) * 0.6), 4.5))
    ax.bar(symbols, lots)
    ax.set_ylabel("lot")
    ax.set_title(f"{title_prefix}Planned Lots".strip())
    ax.tick_params(axis="x", rotation=45, labelsize=9)
    fig.tight_layout()
    maybe_save(fig, "lots_bar.png")
    figs.append(fig)

    # Risk bar chart
    fig, ax = plt.subplots(figsize=(max(8, len(symbols) * 0.6), 4.5))
    ax.bar(symbols, risks)
    ax.set_ylabel("risk_usd")
    ax.set_title(f"{title_prefix}Risk Allocation (USD)".strip())
    ax.tick_params(axis="x", rotation=45, labelsize=9)
    fig.tight_layout()
    maybe_save(fig, "risk_bar.png")
    figs.append(fig)

    # Pie (weights)
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    # Avoid matplotlib error when all zeros
    if sum(weights) > 0:
        ax.pie(weights, labels=symbols, autopct="%1.1f%%", textprops={"fontsize": 9})
    ax.set_title(f"{title_prefix}Weights".strip())
    fig.tight_layout()
    maybe_save(fig, "weights_pie.png")
    figs.append(fig)

    if show:
        # In PyCharm, plt.show() typically renders into SciView/tool window.
        plt.show()
        for f in figs:
            plt.close(f)
    else:
        for f in figs:
            plt.close(f)

    return saved


def build_initial_portfolio(
    env_file: Optional[os.PathLike[str] | str] = None,
    *,
    settings_module: Any = settings,
    module: Any = mt5,
    collector: Callable[..., tuple[list[SymbolSnapshot], Optional[float]]] = collect_target_symbol_snapshots,
) -> tuple[list[PlannedPosition], float, float, float]:
    rows, equity = collector(env_file, settings_module=settings_module, module=module)
    if equity is None:
        raise RuntimeError("Account equity is required to build initial portfolio")

    tpv_leverage = getattr(settings_module, "TPV_LEVERAGE", None)
    if tpv_leverage is None:
        raise RuntimeError("settings.TPV_LEVERAGE is required")

    risk_pct = getattr(settings_module, "RISK_PCT", None)
    if risk_pct is None:
        raise RuntimeError("settings.RISK_PCT is required")

    tpv = float(equity) * float(tpv_leverage)
    plan = build_initial_portfolio_from_snapshots(rows, tpv=tpv, risk_pct=float(risk_pct))
    return plan, float(equity), float(tpv), float(risk_pct)


def build_initial_portfolio_with_diagnostics(
    env_file: Optional[os.PathLike[str] | str] = None,
    *,
    settings_module: Any = settings,
    module: Any = mt5,
    collector: Callable[..., tuple[list[SymbolSnapshot], Optional[float]]] = collect_target_symbol_snapshots,
) -> tuple[list[PlannedPosition], float, float, float, Optional[CorrelationEstimationResult], str]:
    """Build portfolio and (when possible) compute ERC/correlation diagnostics.

    Returns: (plan, equity, tpv, risk_pct, corr_result, weights_title)
    """

    rows, plan, equity, tpv, risk_pct, corr_result, weights_title = build_initial_portfolio_run_with_diagnostics(
        env_file,
        settings_module=settings_module,
        module=module,
        collector=collector,
    )
    return plan, equity, tpv, risk_pct, corr_result, weights_title


def build_initial_portfolio_run_with_diagnostics(
    env_file: Optional[os.PathLike[str] | str] = None,
    *,
    settings_module: Any = settings,
    module: Any = mt5,
    collector: Callable[..., tuple[list[SymbolSnapshot], Optional[float]]] = collect_target_symbol_snapshots,
) -> tuple[
    list[SymbolSnapshot],
    list[PlannedPosition],
    float,
    float,
    float,
    Optional[CorrelationEstimationResult],
    str,
]:
    """Build portfolio and keep input snapshots for persistence.

    Returns: (rows, plan, equity, tpv, risk_pct, corr_result, weights_title)
    """

    rows, equity = collector(env_file, settings_module=settings_module, module=module)
    if equity is None:
        raise RuntimeError("Account equity is required to build initial portfolio")

    tpv_leverage = getattr(settings_module, "TPV_LEVERAGE", None)
    if tpv_leverage is None:
        raise RuntimeError("settings.TPV_LEVERAGE is required")

    risk_pct = getattr(settings_module, "RISK_PCT", None)
    if risk_pct is None:
        raise RuntimeError("settings.RISK_PCT is required")

    tpv = float(equity) * float(tpv_leverage)

    symbols = [r.symbol for r in rows if r.exists]
    corr_result: Optional[CorrelationEstimationResult] = None
    weights_title = "Inverse-Vol Weights"
    weights: Optional[dict[str, float]] = None

    try:
        if (
            module is not None
            and hasattr(module, "initialize")
            and hasattr(module, "shutdown")
            and hasattr(module, "copy_rates_from_pos")
            and len(symbols) >= 2
        ):
            config = load_mt5_config(env_file)
            initialized = False
            try:
                initialized = bool(
                    module.initialize(
                        path=config.path,
                        login=config.login,
                        password=config.password,
                        server=config.server,
                        portable=config.portable,
                    )
                )
                if not initialized:
                    raise Mt5ConnectionError(
                        f"MT5 initialize failed. last_error={_safe_last_error(module)}"
                    )

                corr_result = estimate_erc_from_mt5(
                    symbols,
                    settings_module=settings_module,
                    module=module,
                )
                w = {s: float(corr_result.weights[i]) for i, s in enumerate(corr_result.symbols)}
                if w and math.isfinite(sum(w.values())):
                    weights = w
                    weights_title = "ERC Weights"
            finally:
                if initialized:
                    module.shutdown()
    except Exception:
        corr_result = None

    plan = build_initial_portfolio_from_snapshots(
        rows,
        tpv=tpv,
        risk_pct=float(risk_pct),
        weights=weights,
    )
    return rows, plan, float(equity), float(tpv), float(risk_pct), corr_result, weights_title


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build initial portfolio lots using inverse volatility (ATR%) and risk allocation"
    )
    parser.add_argument(
        "--env",
        default=None,
        help="Path to .env file (default: project root .env)",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Do not generate matplotlib plots",
    )
    parser.add_argument(
        "--plot-mode",
        choices=["auto", "show", "save", "show+save"],
        default="auto",
        help=(
            "How to render plots: auto=show in PyCharm (PYCHARM_HOSTED) else save, "
            "show=display only, save=save PNG only, show+save=both"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="plots",
        help="Directory to save plots when plot-mode includes save (default: plots)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build and print MT5 market order requests without sending orders",
    )
    parser.add_argument(
        "--trade",
        action="store_true",
        help="Send MT5 market orders based on the displayed plan (requires confirmation unless --yes)",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip confirmation prompt when using --trade",
    )
    parser.add_argument(
        "--deviation",
        type=int,
        default=20,
        help="Max price deviation in points for market orders (default: 20)",
    )
    parser.add_argument(
        "--magic",
        type=int,
        default=None,
        help="MT5 magic number to set on orders (default: not set)",
    )
    parser.add_argument(
        "--comment",
        default="initial_portfolio",
        help="MT5 order comment (default: initial_portfolio)",
    )
    args = parser.parse_args(argv)

    rows, plan, equity, tpv, risk_pct, corr_result, weights_title = build_initial_portfolio_run_with_diagnostics(
        args.env
    )

    run_id, db_path = persist_initial_portfolio_run(
        rows,
        plan,
        env_file=args.env,
        time_frame=getattr(settings, "TIME_FRAME", None),
        atr_period=getattr(settings, "ATR_PERIOD", None),
        tpv_leverage=getattr(settings, "TPV_LEVERAGE", None),
        risk_pct=risk_pct,
        equity=equity,
        tpv=tpv,
        weights_title=weights_title,
        mt5_order_comment=str(args.comment),
    )

    print("Initial portfolio (risk allocation):")
    print(f"Saved to SQLite: {db_path} (run_id={run_id})")
    print(f"Account Equity: {equity:,.2f}")
    print(
        f"Target Portfolio Value (Equity * TPV_LEVERAGE({float(settings.PV_LEVERAGE):.2f})): {tpv:,.2f}"
    )
    print(f"RISK_PCT: {risk_pct:.2f}")
    print(f"Total Risk Budget (TPV * RISK_PCT): {tpv * risk_pct:,.2f}")
    print()
    print(_format_portfolio_table(plan))

    if args.trade and args.dry_run:
        raise SystemExit("--trade and --dry-run cannot be used together")

    print()
    print("MT5 orders (preview):")
    print("symbol | direction | lot")
    print("------ | --------- | ---")
    for p in plan:
        lot = float(p.lot)
        if not math.isfinite(lot) or lot <= 0:
            continue
        print(f"{p.symbol} | {_normalize_direction(p.direction)} | {lot:.2f}")

    trade = bool(args.trade)
    dry_run = bool(args.dry_run)
    skip_confirm = bool(args.yes)

    if not trade and not dry_run and _stdin_is_interactive():
        print()
        print("Choose action:")
        print("  - Type 'YES' to TRADE (send market orders)")
        print("  - Type 'DRY' to DRY-RUN (print order requests only)")
        print("  - Press Enter to exit")
        choice = _safe_input("Choice [Enter/YES/DRY]: ")
        if choice is not None:
            c = choice.strip().upper()
            if c in {"YES", "TRADE", "T"}:
                trade = True
                skip_confirm = True
            elif c in {"DRY", "DRY-RUN", "DRYRUN", "D"}:
                dry_run = True

    if trade or dry_run:
        mode_label = "DRY-RUN" if dry_run else "TRADE"
        print()
        print(f"MT5 orders ({mode_label}):")

        if trade and not skip_confirm:
            print()
            confirm = _safe_input("Type 'YES' to send market orders to the connected account: ")
            if confirm is None:
                print("No interactive input available. Re-run with --trade --yes to skip confirmation.")
                return 2
            if confirm.strip().upper() != "YES":
                print("Aborted.")
                return 0

        requests, results = trade_planned_positions_on_mt5(
            plan,
            env_file=args.env,
            module=mt5,
            deviation=int(args.deviation),
            magic=args.magic,
            comment=str(args.comment),
            dry_run=bool(dry_run),
        )

        print()
        if dry_run:
            print("Order requests:")
            for r in requests:
                print(r)
        else:
            print("Order results:")
            for res in results:
                print(res)

    if not args.no_plot:
        mode = str(args.plot_mode)

        show = False
        save = False
        if mode == "auto":
            # PyCharm run configurations usually set PYCHARM_HOSTED=1.
            show = bool(os.getenv("PYCHARM_HOSTED"))
            save = not show
        elif mode == "show":
            show = True
        elif mode == "save":
            save = True
        elif mode == "show+save":
            show = True
            save = True

        saved = render_portfolio_plots(
            plan,
            output_dir=args.output_dir if save else None,
            show=show,
            title_prefix=f"TPV={tpv:,.0f} / RISK_PCT={risk_pct:.2f} - ",
            force_agg=save and not show,
            corr_symbols=corr_result.symbols if corr_result is not None else None,
            corr_matrix=corr_result.corr if corr_result is not None else None,
            weights_title=weights_title,
        )

        if saved:
            print()
            print("Saved plots:")
            for p in saved:
                print(f"- {p}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())