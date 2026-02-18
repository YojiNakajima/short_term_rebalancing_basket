from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Callable, Optional

from config import settings
from infrastructure.dB.sqlite3.daily_metrics import (
    upsert_daily_atr,
    upsert_daily_pv_tpv,
    upsert_daily_symbol_price,
)
from infrastructure.dB.sqlite3.global_settings import (
    now_utc_iso,
    set_global_running,
    today_ymd_local,
    upsert_daily_equity,
)
from infrastructure.dB.sqlite3.initialize_db import initialize_db, load_sqlite_db_path
from infrastructure.mt5.connect import Mt5ConnectionError, load_mt5_config
from infrastructure.mt5.symbols_report import fetch_symbol_snapshot, get_enabled_target_symbols
from infrastructure.slack.webhook import send_slack_message

try:
    import MetaTrader5 as mt5
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "MetaTrader5 package is required. Install dependencies from requirements.txt"
    ) from e


def initialize_once(
    *,
    env_file: Optional[os.PathLike[str] | str] = None,
    db_path: Optional[os.PathLike[str] | str] = None,
    module: Any = mt5,
    settings_module: Any = settings,
    slack_sender: Optional[Callable[..., Any]] = send_slack_message,
) -> dict[str, Any]:
    resolved_db_path = Path(db_path) if db_path is not None else load_sqlite_db_path(env_file)
    initialize_db(resolved_db_path)

    ymd = today_ymd_local()
    fetched_at = now_utc_iso()

    # 1) global_running=True
    set_global_running(True, env_file=env_file, db_path=resolved_db_path)

    # 2) Fetch equity from MT5 account + daily snapshots
    config = load_mt5_config(env_file)
    initialized = False
    equity_f: float
    symbol_snapshots: dict[str, Any] = {}
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
            last_error = None
            try:
                last_error = module.last_error()
            except Exception:
                last_error = None
            raise Mt5ConnectionError(f"MT5 initialize failed. last_error={last_error}")

        info = module.account_info()
        if info is None:
            last_error = None
            try:
                last_error = module.last_error()
            except Exception:
                last_error = None
            raise Mt5ConnectionError(
                f"MT5 account_info() returned None. last_error={last_error}"
            )

        equity = getattr(info, "equity", None)
        if equity is None:
            raise RuntimeError("MT5 account_info.equity is missing")

        equity_f = float(equity)

        # Target symbols (enabled=True)
        symbols = get_enabled_target_symbols(settings_module=settings_module)
        atr_period = getattr(settings_module, "ATR_PERIOD", None)
        time_frame = getattr(settings_module, "TIME_FRAME", None)
        if atr_period is None:
            raise RuntimeError("Missing settings.ATR_PERIOD")
        if time_frame is None:
            raise RuntimeError("Missing settings.TIME_FRAME")

        for symbol in symbols:
            symbol_snapshots[symbol] = fetch_symbol_snapshot(
                symbol,
                atr_period=int(atr_period),
                time_frame=time_frame,
                module=module,
            )

    finally:
        if initialized:
            try:
                module.shutdown()
            except Exception:
                pass

    # 3) Persist daily metrics
    upsert_daily_equity(
        ymd=ymd,
        equity=equity_f,
        fetched_at=fetched_at,
        env_file=env_file,
        db_path=resolved_db_path,
    )

    pv_leverage = getattr(settings_module, "PV_LEVERAGE", None)
    tpv_leverage = getattr(settings_module, "TPV_LEVERAGE", None)
    leverage = getattr(settings_module, "LEVERAGE", None)
    if pv_leverage is None:
        raise RuntimeError("Missing settings.PV_LEVERAGE")
    if tpv_leverage is None:
        raise RuntimeError("Missing settings.TPV_LEVERAGE")
    if leverage is None:
        raise RuntimeError("Missing settings.LEVERAGE")

    pv = float(equity_f) * float(pv_leverage)
    tpv = float(equity_f) * float(pv_leverage) * float(leverage) * float(tpv_leverage)
    upsert_daily_pv_tpv(
        ymd=ymd,
        equity=equity_f,
        pv=pv,
        tpv=tpv,
        pv_leverage=float(pv_leverage),
        leverage=float(leverage),
        tpv_leverage=float(tpv_leverage),
        fetched_at=fetched_at,
        env_file=env_file,
        db_path=resolved_db_path,
    )

    # Persist symbol daily snapshots
    price_error_count = 0
    atr_error_count = 0
    time_frame = getattr(settings_module, "TIME_FRAME", None)
    atr_period = getattr(settings_module, "ATR_PERIOD", None)
    for symbol, snap in symbol_snapshots.items():
        sell = getattr(snap, "sell", None)
        price_error = getattr(snap, "error", None)
        if sell is None or price_error:
            price_error_count += 1
        upsert_daily_symbol_price(
            ymd=ymd,
            symbol=symbol,
            sell=sell,
            fetched_at=fetched_at,
            error=price_error,
            env_file=env_file,
            db_path=resolved_db_path,
        )

        atr = getattr(snap, "atr", None)
        atr_pct = getattr(snap, "atr_pct", None)
        atr_error = getattr(snap, "atr_error", None)
        if atr_pct is None or atr_error:
            atr_error_count += 1
        upsert_daily_atr(
            ymd=ymd,
            symbol=symbol,
            atr=atr,
            atr_pct=atr_pct,
            time_frame=None if time_frame is None else int(time_frame),
            atr_period=None if atr_period is None else int(atr_period),
            fetched_at=fetched_at,
            error=atr_error,
            env_file=env_file,
            db_path=resolved_db_path,
        )

    slack_error: Optional[str] = None
    if slack_sender is not None:
        try:
            n_symbols = len(symbol_snapshots)
            text = "\n".join(
                [
                    "[daily_initialize] completed",
                    f"ymd={ymd} fetched_at={fetched_at}",
                    f"global_running=ON",
                    f"equity={equity_f:.2f}",
                    f"PV={pv:.2f} (PV_LEVERAGE={float(pv_leverage)})",
                    f"TPV={tpv:.2f} (PV_LEVERAGE={float(pv_leverage)} LEVERAGE={float(leverage)} TPV_LEVERAGE={float(tpv_leverage)})",
                    f"symbols={n_symbols} price_errors={price_error_count} atr_errors={atr_error_count}",
                ]
            )
            try:
                slack_sender(text, env_file=env_file)
            except TypeError:
                slack_sender(text)
        except Exception as e:
            slack_error = str(e)

    return {
        "db_path": str(resolved_db_path),
        "ymd": ymd,
        "equity": equity_f,
        "fetched_at": fetched_at,
        "pv": pv,
        "tpv": tpv,
        "symbol_count": len(symbol_snapshots),
        "price_error_count": price_error_count,
        "atr_error_count": atr_error_count,
        "slack_error": slack_error,
    }


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Daily initialize (global_running ON + equity/PV/TPV + symbol price/ATR% snapshots + Slack notify)"
    )
    parser.add_argument(
        "--env",
        default=None,
        help="Path to .env file (default: project root .env)",
    )
    parser.add_argument(
        "--db",
        default=None,
        help="Path to sqlite db (overrides SQLITE_DB_PATH)",
    )
    args = parser.parse_args(argv)

    out = initialize_once(env_file=args.env, db_path=args.db, module=mt5)
    print("[OK] initialize completed")
    print(f"db_path={out['db_path']}")
    print(
        f"ymd={out['ymd']} equity={out['equity']:.2f} PV={out['pv']:.2f} TPV={out['tpv']:.2f} fetched_at={out['fetched_at']}"
    )
    if out.get("slack_error"):
        print(f"[WARN] slack notification failed: {out['slack_error']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
