from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Any, Optional

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
    contract_size: Optional[float] = None
    currency_profit: Optional[str] = None
    usd_pair_symbol: Optional[str] = None
    usd_pair_sell: Optional[float] = None
    usd_per_lot: Optional[float] = None
    volume_min: Optional[float] = None
    volume_step: Optional[float] = None
    error: Optional[str] = None
    usd_pair_error: Optional[str] = None


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


def fetch_symbol_snapshot(symbol: str, *, module: Any = mt5) -> SymbolSnapshot:
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

    return SymbolSnapshot(
        symbol=symbol,
        exists=True,
        sell=sell,
        contract_size=contract_size,
        currency_profit=currency_profit,
        usd_pair_symbol=usd_pair_symbol,
        usd_pair_sell=usd_pair_sell,
        usd_per_lot=usd_per_lot,
        volume_min=getattr(info, "volume_min", None),
        volume_step=getattr(info, "volume_step", None),
        error=error,
        usd_pair_error=usd_pair_error,
    )


def collect_target_symbol_snapshots(
    env_file: Optional[os.PathLike[str] | str] = None,
    *,
    settings_module: Any = settings,
    module: Any = mt5,
) -> list[SymbolSnapshot]:
    symbols = get_enabled_target_symbols(settings_module=settings_module)
    if not symbols:
        return []

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

        return [fetch_symbol_snapshot(symbol, module=module) for symbol in symbols]
    finally:
        if initialized:
            module.shutdown()


def _format_table(rows: list[SymbolSnapshot]) -> str:
    headers = [
        "symbol",
        "sell",
        "contract_size",
        "currency_profit",
        "usd_pair",
        "usd_pair_sell",
        "usd_per_lot",
        "min_lot",
        "lot_step",
    ]

    def s(v: Any) -> str:
        if v is None:
            return ""
        return str(v)

    data = [
        [
            r.symbol,
            s(r.sell),
            s(r.contract_size),
            s(r.currency_profit),
            s(r.usd_pair_symbol),
            s(r.usd_pair_sell),
            s(r.usd_per_lot),
            s(r.volume_min),
            s(r.volume_step),
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
            "Connect to MT5 and print target symbols info (sell/contract size/currency/usd pair/min lot/lot step)"
        )
    )
    parser.add_argument(
        "--env",
        default=None,
        help="Path to .env file (default: project root .env)",
    )
    args = parser.parse_args(argv)

    rows = collect_target_symbol_snapshots(args.env)
    if not rows:
        print("No enabled target symbols.")
        return 0

    print("MT5 target symbols info:")
    print(_format_table(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
