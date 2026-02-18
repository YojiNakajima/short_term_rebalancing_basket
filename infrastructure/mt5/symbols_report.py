from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
import math
from typing import Any, Optional, Sequence

from config import settings
from infrastructure.mt5.connect import Mt5ConnectionError, load_mt5_config

try:
    import MetaTrader5 as mt5
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "MetaTrader5 package is required. Install dependencies from requirements.txt"
    ) from e


@dataclass(frozen=True)
class SymbolSnapshot:
    symbol: str
    exists: bool
    # MT5 bid corresponds to the "sell" price.
    sell: Optional[float] = None
    atr: Optional[float] = None
    atr_pct: Optional[float] = None
    contract_size: Optional[float] = None
    currency_profit: Optional[str] = None
    usd_pair_symbol: Optional[str] = None
    usd_pair_sell: Optional[float] = None
    usd_per_lot: Optional[float] = None
    volume_min: Optional[float] = None
    volume_step: Optional[float] = None
    volume_max: Optional[float] = None

    # Lot adjusted to match the base USD value (derived from XAUUSD 0.01 lot)
    lot_for_base: Optional[float] = None
    usd_for_base_lot: Optional[float] = None
    usd_diff_to_base: Optional[float] = None
    error: Optional[str] = None
    usd_pair_error: Optional[str] = None
    atr_error: Optional[str] = None


BASE_SYMBOL = "XAUUSD"
BASE_LOT = 0.01


def get_enabled_target_symbols(*, settings_module: Any = settings) -> list[str]:
    target_symbols = getattr(settings_module, "target_symbols", {})
    if not isinstance(target_symbols, dict):
        return []

    # Use only symbols explicitly marked as True
    return [symbol for symbol, enabled in target_symbols.items() if enabled is True]


def _safe_last_error(module: Any) -> Optional[Any]:
    try:
        return module.last_error()
    except Exception:
        return None


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


def _timeframe_label(value: Any) -> str:
    try:
        v = int(value)
    except (TypeError, ValueError):
        return str(value)

    if v == 1:
        return "1H"
    if v == 4:
        return "4H"
    return str(value)


def _rate_value(rec: Any, key: str) -> float:
    # Support numpy records (rec['high']), dicts, and SimpleNamespace/objects.
    try:
        return float(rec[key])
    except Exception:
        return float(getattr(rec, key))


def _normalize_rates_order(rates: Sequence[Any]) -> list[Any]:
    # MT5 may return rates in either chronological order depending on wrapper/version.
    # If time is available, sort to ensure old -> new.
    try:
        times = []
        for r in rates:
            try:
                t = r["time"]
            except Exception:
                t = getattr(r, "time")
            times.append(float(t))
    except Exception:
        return list(rates)

    if len(times) < 2:
        return list(rates)

    # If already old->new, keep as-is.
    if times[0] <= times[-1]:
        return list(rates)

    return [r for _, r in sorted(zip(times, rates), key=lambda x: x[0])]


def _calculate_atr_and_close_from_rates(
    rates: Sequence[Any], *, period: int
) -> tuple[Optional[float], Optional[float], Optional[str]]:
    if period <= 0:
        return None, None, "ATR period must be positive"

    try:
        n = len(rates)
    except Exception:
        return None, None, "rates is not a sized sequence"

    need = period + 1
    if n < need:
        return None, None, f"not enough rates. need={need} got={n}"

    rs = _normalize_rates_order(rates)

    trs: list[float] = []
    for i in range(1, len(rs)):
        high = _rate_value(rs[i], "high")
        low = _rate_value(rs[i], "low")
        prev_close = _rate_value(rs[i - 1], "close")
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        trs.append(tr)

    if len(trs) < period:
        return None, None, f"not enough TR values. need={period} got={len(trs)}"

    # Wilder's smoothing (RMA). First value is SMA(TR, period), then recursive update.
    atr = sum(trs[:period]) / float(period)
    for tr in trs[period:]:
        atr = ((atr * float(period - 1)) + tr) / float(period)

    if not math.isfinite(atr):
        return None, None, "ATR is not finite"

    last_close = _rate_value(rs[-1], "close")
    if not math.isfinite(last_close) or last_close <= 0:
        return None, None, "close is not finite or non-positive"

    return atr, last_close, None


def _fetch_atr_values(
    symbol: str,
    *,
    atr_period: int,
    time_frame: Any,
    module: Any,
) -> tuple[Optional[float], Optional[float], Optional[str]]:
    timeframe_const = _timeframe_from_settings_value(time_frame, module=module)
    if timeframe_const is None:
        return None, None, f"unsupported TIME_FRAME={time_frame!r}"

    # Request additional bars for more stable Wilder smoothing when history is available.
    count = max(int(atr_period) + 1, int(atr_period) + 50)
    try:
        # start_pos=1 to avoid the currently forming (incomplete) bar.
        rates = module.copy_rates_from_pos(symbol, timeframe_const, 1, count)
    except Exception as e:
        return None, None, f"copy_rates_from_pos failed: {e}"

    if rates is None:
        return (
            None,
            None,
            f"copy_rates_from_pos returned None. last_error={_safe_last_error(module)}",
        )

    return _calculate_atr_and_close_from_rates(rates, period=int(atr_period))


def _select_symbol_if_needed(symbol: str, info: Any, *, module: Any) -> tuple[Any, Optional[str]]:
    """Ensure the symbol is visible in MarketWatch if possible."""

    visible = getattr(info, "visible", None)
    if visible is not False:
        return info, None

    try:
        selected_ok = bool(module.symbol_select(symbol, True))
    except Exception as e:
        return info, f"symbol_select failed: {e}"

    if not selected_ok:
        select_error: Optional[str] = (
            f"symbol_select returned False. last_error={_safe_last_error(module)}"
        )
    else:
        select_error = None

    refreshed = module.symbol_info(symbol)
    if refreshed is not None:
        info = refreshed

    return info, select_error


def _fetch_sell_price(symbol: str, *, module: Any) -> tuple[Optional[float], Optional[str]]:
    tick = module.symbol_info_tick(symbol)
    if tick is None:
        return None, f"symbol_info_tick returned None. last_error={_safe_last_error(module)}"

    bid = getattr(tick, "bid", None)
    if bid is None:
        return None, f"tick.bid is None. last_error={_safe_last_error(module)}"

    return bid, None


def _get_contract_size(info: Any) -> Optional[float]:
    for attr in ("trade_contract_size", "contract_size"):
        v = getattr(info, attr, None)
        if v is not None:
            return v
    return None


def _float_precision_from_step(step: float) -> int:
    if step <= 0:
        return 8

    # Try to infer decimal places from the step size.
    # Example: 0.01 -> 2, 0.1 -> 1, 1.0 -> 0
    s = f"{step:.16f}".rstrip("0").rstrip(".")
    if "." not in s:
        return 0
    return min(8, len(s.split(".", 1)[1]))


def _round_lot(lot: float, *, step: Optional[float]) -> float:
    if step is None:
        return lot
    try:
        p = _float_precision_from_step(float(step))
    except Exception:
        return lot
    return round(lot, p)


def _nearest_lot_for_target_usd(
    *,
    target_usd: float,
    usd_per_lot: float,
    volume_min: float,
    volume_step: float,
    volume_max: Optional[float],
) -> Optional[float]:
    if usd_per_lot <= 0 or volume_step <= 0 or volume_min <= 0:
        return None

    ideal = target_usd / usd_per_lot
    if ideal <= volume_min:
        lot = volume_min
        if volume_max is not None:
            lot = min(lot, volume_max)
        return _round_lot(lot, step=volume_step)

    steps = (ideal - volume_min) / volume_step
    if not math.isfinite(steps):
        return None

    n_floor = int(math.floor(steps))
    n_ceil = int(math.ceil(steps))

    candidates: list[float] = []
    for n in {n_floor - 1, n_floor, n_ceil, n_ceil + 1}:
        if n < 0:
            continue
        lot = volume_min + (n * volume_step)
        if lot < volume_min:
            continue
        if volume_max is not None and lot > volume_max:
            continue
        candidates.append(lot)

    if not candidates:
        if volume_max is not None and volume_max >= volume_min:
            return _round_lot(volume_max, step=volume_step)
        return _round_lot(volume_min, step=volume_step)

    def score(lot: float) -> tuple[float, float]:
        usd = usd_per_lot * lot
        return (abs(usd - target_usd), lot)

    best = min(candidates, key=score)
    return _round_lot(best, step=volume_step)


def adjust_lots_to_base(
    rows: list[SymbolSnapshot],
    *,
    base_symbol: str = BASE_SYMBOL,
    base_lot: float = BASE_LOT,
) -> list[SymbolSnapshot]:
    base = next((r for r in rows if r.symbol == base_symbol), None)
    if base is None:
        return rows

    base_lot_used = base_lot
    if base.volume_min is not None:
        base_lot_used = max(base_lot_used, float(base.volume_min))
    if base.volume_step is not None and base.volume_min is not None and base_lot_used > base.volume_min:
        # Align to step increments starting from volume_min.
        n = (base_lot_used - float(base.volume_min)) / float(base.volume_step)
        base_lot_used = float(base.volume_min) + (math.ceil(n) * float(base.volume_step))
        base_lot_used = _round_lot(base_lot_used, step=float(base.volume_step))

    base_usd = None
    if base.usd_per_lot is not None:
        base_usd = float(base.usd_per_lot) * float(base_lot_used)

    adjusted: list[SymbolSnapshot] = []
    for r in rows:
        lot_for_base: Optional[float]
        usd_for_base_lot: Optional[float]
        usd_diff_to_base: Optional[float]

        if r.symbol == base_symbol:
            lot_for_base = float(base_lot_used)
            usd_for_base_lot = float(base_usd) if base_usd is not None else None
            usd_diff_to_base = 0.0 if base_usd is not None else None
        else:
            lot_for_base = None
            usd_for_base_lot = None
            usd_diff_to_base = None

            if (
                base_usd is not None
                and r.usd_per_lot is not None
                and r.volume_min is not None
                and r.volume_step is not None
            ):
                lot_for_base = _nearest_lot_for_target_usd(
                    target_usd=float(base_usd),
                    usd_per_lot=float(r.usd_per_lot),
                    volume_min=float(r.volume_min),
                    volume_step=float(r.volume_step),
                    volume_max=float(r.volume_max) if r.volume_max is not None else None,
                )
                if lot_for_base is not None:
                    usd_for_base_lot = float(r.usd_per_lot) * float(lot_for_base)
                    usd_diff_to_base = usd_for_base_lot - float(base_usd)

        adjusted.append(
            SymbolSnapshot(
                symbol=r.symbol,
                exists=r.exists,
                sell=r.sell,
                atr=r.atr,
                atr_pct=r.atr_pct,
                contract_size=r.contract_size,
                currency_profit=r.currency_profit,
                usd_pair_symbol=r.usd_pair_symbol,
                usd_pair_sell=r.usd_pair_sell,
                usd_per_lot=r.usd_per_lot,
                volume_min=r.volume_min,
                volume_step=r.volume_step,
                volume_max=r.volume_max,
                lot_for_base=lot_for_base,
                usd_for_base_lot=usd_for_base_lot,
                usd_diff_to_base=usd_diff_to_base,
                error=r.error,
                usd_pair_error=r.usd_pair_error,
                atr_error=r.atr_error,
            )
        )

    return adjusted


def _resolve_usd_pair_candidates(currency: str) -> list[str]:
    ccy = currency.strip().upper()
    if ccy == "USD":
        return []

    # Heuristic to match typical FX market conventions.
    if ccy in {"EUR", "GBP", "AUD", "NZD"}:
        return [f"{ccy}USD", f"USD{ccy}"]

    return [f"USD{ccy}", f"{ccy}USD"]


def _fetch_usd_pair_sell(currency: str, *, module: Any) -> tuple[Optional[str], Optional[float], Optional[str]]:
    candidates = _resolve_usd_pair_candidates(currency)
    for sym in candidates:
        info = module.symbol_info(sym)
        if info is None:
            continue

        info, select_error = _select_symbol_if_needed(sym, info, module=module)
        sell, tick_error = _fetch_sell_price(sym, module=module)
        err = select_error or tick_error
        return sym, sell, err

    return (
        None,
        None,
        f"usd pair not found for currency_profit={currency!r}. tried={candidates}. last_error={_safe_last_error(module)}",
    )


def _calc_usd_per_lot(
    *,
    sell: Optional[float],
    contract_size: Optional[float],
    currency_profit: Optional[str],
    usd_pair_symbol: Optional[str],
    usd_pair_sell: Optional[float],
) -> Optional[float]:
    if sell is None or contract_size is None:
        return None

    ccy = (str(currency_profit).strip().upper() if currency_profit is not None else "")
    if not ccy:
        return None

    base = sell * contract_size
    if ccy == "USD":
        return base

    if not usd_pair_symbol or usd_pair_sell is None:
        return None

    pair = str(usd_pair_symbol).strip().upper()
    if pair == f"{ccy}USD" or (pair.startswith(ccy) and pair.endswith("USD")):
        return base * usd_pair_sell

    if pair == f"USD{ccy}" or (pair.startswith("USD") and pair.endswith(ccy)):
        if usd_pair_sell == 0:
            return None
        return base / usd_pair_sell

    return None


def fetch_symbol_snapshot(
    symbol: str,
    *,
    module: Any = mt5,
    atr_period: Optional[int] = None,
    time_frame: Optional[Any] = None,
) -> SymbolSnapshot:
    info = module.symbol_info(symbol)
    if info is None:
        return SymbolSnapshot(
            symbol=symbol,
            exists=False,
            error=f"symbol not found. last_error={_safe_last_error(module)}",
        )

    info, select_error = _select_symbol_if_needed(symbol, info, module=module)
    sell, tick_error = _fetch_sell_price(symbol, module=module)
    error = select_error or tick_error

    currency_profit = getattr(info, "currency_profit", None)
    usd_pair_symbol = None
    usd_pair_sell = None
    usd_pair_error = None
    if currency_profit and str(currency_profit).strip().upper() != "USD":
        usd_pair_symbol, usd_pair_sell, usd_pair_error = _fetch_usd_pair_sell(
            str(currency_profit), module=module
        )

    contract_size = _get_contract_size(info)
    usd_per_lot = _calc_usd_per_lot(
        sell=sell,
        contract_size=contract_size,
        currency_profit=currency_profit,
        usd_pair_symbol=usd_pair_symbol,
        usd_pair_sell=usd_pair_sell,
    )

    atr = None
    atr_pct = None
    atr_error = None
    if atr_period is not None and time_frame is not None:
        atr_raw, last_close, atr_error = _fetch_atr_values(
            symbol,
            atr_period=int(atr_period),
            time_frame=time_frame,
            module=module,
        )
        if atr_error is None and atr_raw is not None and last_close is not None:
            atr = float(atr_raw)
            atr_pct = (float(atr_raw) / float(last_close)) * 100.0

    return SymbolSnapshot(
        symbol=symbol,
        exists=True,
        sell=sell,
        atr=atr,
        atr_pct=atr_pct,
        contract_size=contract_size,
        currency_profit=currency_profit,
        usd_pair_symbol=usd_pair_symbol,
        usd_pair_sell=usd_pair_sell,
        usd_per_lot=usd_per_lot,
        volume_min=getattr(info, "volume_min", None),
        volume_step=getattr(info, "volume_step", None),
        volume_max=getattr(info, "volume_max", None),
        error=error,
        usd_pair_error=usd_pair_error,
        atr_error=atr_error,
    )


def collect_target_symbol_snapshots(
    env_file: Optional[os.PathLike[str] | str] = None,
    *,
    settings_module: Any = settings,
    module: Any = mt5,
) -> tuple[list[SymbolSnapshot], Optional[float]]:
    symbols = get_enabled_target_symbols(settings_module=settings_module)
    if not symbols:
        return [], None

    if BASE_SYMBOL not in symbols:
        symbols = [BASE_SYMBOL, *symbols]

    config = load_mt5_config(env_file)
    initialized = False
    try:
        initialized = module.initialize(
            path=config.path,
            login=config.login,
            password=config.password,
            server=config.server,
            portable=config.portable,
        )
        if not initialized:
            raise Mt5ConnectionError(
                f"MT5 initialize failed. last_error={_safe_last_error(module)}"
            )

        account_info = module.account_info()
        equity = getattr(account_info, "equity", None)

        atr_period = getattr(settings_module, "ATR_PERIOD", None)
        time_frame = getattr(settings_module, "TIME_FRAME", None)

        rows = [
            fetch_symbol_snapshot(
                symbol,
                module=module,
                atr_period=int(atr_period) if atr_period is not None else None,
                time_frame=time_frame,
            )
            for symbol in symbols
        ]
        return adjust_lots_to_base(rows), equity
    finally:
        if initialized:
            module.shutdown()


def _format_table(rows: list[SymbolSnapshot]) -> str:
    headers = [
        "symbol",
        "sell",
        "atr_pct",
        "contract_size",
        "currency_profit",
        "usd_pair",
        "usd_pair_sell",
        "usd_per_lot",
        "min_lot",
        "lot_step",
        "lot_for_base",
        "usd_for_base_lot",
        "usd_diff_to_base",
    ]

    def s(v: Any) -> str:
        if v is None:
            return ""

        if isinstance(v, float):
            return f"{v:.10g}"
        return str(v)

    data = [
        [
            r.symbol,
            s(r.sell),
            s(r.atr_pct),
            s(r.contract_size),
            s(r.currency_profit),
            s(r.usd_pair_symbol),
            s(r.usd_pair_sell),
            s(r.usd_per_lot),
            s(r.volume_min),
            s(r.volume_step),
            s(r.lot_for_base),
            s(r.usd_for_base_lot),
            s(r.usd_diff_to_base),
        ]
        for r in rows
    ]

    widths = [len(h) for h in headers]
    for row in data:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def fmt_row(row: list[str]) -> str:
        return " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(row))

    lines = [fmt_row(headers), "-+-".join("-" * w for w in widths)]
    lines.extend(fmt_row(row) for row in data)
    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Connect to MT5 and print target symbols info (including USD per lot and lot adjusted to XAUUSD 0.01 lot base)"
        )
    )
    parser.add_argument(
        "--env",
        default=None,
        help="Path to .env file (default: project root .env)",
    )
    args = parser.parse_args(argv)

    rows, equity = collect_target_symbol_snapshots(args.env)
    if not rows:
        print("No enabled target symbols.")
        return 0

    print("MT5 target symbols info:")
    if equity is not None:
        tpv_label = "TPV_LEVERAGE"
        tpv_multiplier = getattr(settings, "TPV_LEVERAGE", None)
        if tpv_multiplier is None:
            # Backward compatibility for older settings.
            tpv_label = "TPV"
            tpv_multiplier = getattr(settings, "TPV", 1.0)

        target_portfolio_value = equity * float(tpv_multiplier)
        print(f"Account Equity: {equity:,.2f}")
        print(
            f"Target Portfolio Value (Equity * {tpv_label}({tpv_multiplier})): {target_portfolio_value:,.2f}"
        )
        print()

    atr_period = getattr(settings, "ATR_PERIOD", None)
    time_frame = getattr(settings, "TIME_FRAME", None)
    if atr_period is not None and time_frame is not None:
        print(
            f"ATR% (Normalized ATR = 100 * ATR / Close, period={atr_period}, timeframe={_timeframe_label(time_frame)}):"
        )
        print()

    print(_format_table(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
