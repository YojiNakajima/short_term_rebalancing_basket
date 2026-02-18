from __future__ import annotations

import argparse
import logging
import math
import os
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from config import settings
from infrastructure.dB.sqlite3.daily_metrics import (
    read_daily_atr,
    read_daily_pv_tpv,
    read_daily_symbol_price,
)
from infrastructure.dB.sqlite3.global_settings import (
    read_global_running,
    set_global_running,
    today_ymd_local,
)
from infrastructure.dB.sqlite3.initialize_db import initialize_db, load_sqlite_db_path
from infrastructure.dB.sqlite3.portfolio import (
    read_latest_portfolio_run,
    read_planned_positions_for_run,
)
from infrastructure.dB.sqlite3.rebalance import (
    RebalanceState,
    create_rebalance_run,
    ensure_rebalance_state_rows,
    persist_rebalance_actions,
    read_rebalance_state_map,
    upsert_rebalance_state,
)
from infrastructure.mt5.connect import Mt5ConnectionError, load_mt5_config
from infrastructure.slack.webhook import send_slack_message
from interfaces.cli.build_initial_portfolio import (
    PlannedPosition,
    _round_to_volume_constraints,
    build_mt5_market_order_requests,
    execute_mt5_market_orders,
)

try:
    import MetaTrader5 as mt5
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "MetaTrader5 package is required. Install dependencies from requirements.txt"
    ) from e


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _default_rebalance_log_path() -> Path:
    # project_root / logs / rebalance_portfolio.log
    return Path(__file__).resolve().parents[2] / "logs" / "rebalance_portfolio.log"


def _resolve_rebalance_log_path(
    log_path: Optional[os.PathLike[str] | str],
    *,
    settings_module: Any,
) -> Path:
    if log_path is not None:
        return Path(log_path)
    configured = getattr(settings_module, "REBALANCE_LOG_PATH", None)
    if configured:
        return Path(configured)
    return _default_rebalance_log_path()


def _get_rebalance_logger(*, log_path: Path) -> logging.Logger:
    """Return a file logger configured with local timezone timestamps.

    Best-effort: logger setup failures should not abort trading.
    """

    log_path = log_path.resolve()

    logger_name = f"rebalance_portfolio:{str(log_path)}"
    logger = logging.getLogger(logger_name)

    # Avoid duplicate handlers when called multiple times.
    for h in list(getattr(logger, "handlers", [])):
        try:
            if isinstance(h, logging.FileHandler) and Path(h.baseFilename).resolve() == log_path:
                return logger
        except Exception:
            continue

    log_path.parent.mkdir(parents=True, exist_ok=True)

    handler = logging.FileHandler(str(log_path), encoding="utf-8")
    handler.setLevel(logging.INFO)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S%z",
        )
    )

    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.addHandler(handler)
    return logger


def _append_run_log(*, text: str, log_path: Path) -> None:
    """Write one execution log entry (standard logging format).

    Includes log level and local timezone timestamp.
    Best-effort: logging failures should not abort trading.
    """

    try:
        logger = _get_rebalance_logger(log_path=log_path)
        lines = text.splitlines() or [""]
        for line in lines:
            logger.info(line)
        logger.info("-" * 80)
    except Exception:
        pass
    finally:
        # Close handlers to avoid keeping file handles open (important on Windows).
        try:
            resolved = log_path.resolve()
        except Exception:
            resolved = log_path
        try:
            for h in list(getattr(logger, "handlers", [])):
                try:
                    if isinstance(h, logging.FileHandler) and Path(h.baseFilename).resolve() == resolved:
                        logger.removeHandler(h)
                        try:
                            h.flush()
                        except Exception:
                            pass
                        try:
                            h.close()
                        except Exception:
                            pass
                except Exception:
                    continue
        except Exception:
            pass


def _safe_last_error(module: Any) -> Optional[Any]:
    try:
        return module.last_error()
    except Exception:
        return None


def _pos_type_buy(module: Any) -> int:
    return int(getattr(module, "POSITION_TYPE_BUY", 0))


def _pos_type_sell(module: Any) -> int:
    return int(getattr(module, "POSITION_TYPE_SELL", 1))


def _direction_to_pos_type(direction: str, *, module: Any) -> int:
    d = (direction or "").strip().upper()
    if d == "SELL":
        return _pos_type_sell(module)
    return _pos_type_buy(module)


def _opposite_direction(direction: str) -> str:
    d = (direction or "").strip().upper()
    return "BUY" if d == "SELL" else "SELL"


def _is_rebalance_comment(comment: Any) -> bool:
    if comment is None:
        return False
    try:
        c = str(comment)
    except Exception:
        return False
    return c.startswith("rebalance_")


def _fetch_sell_price(symbol: str, *, module: Any) -> Optional[float]:
    try:
        module.symbol_select(symbol, True)
    except Exception:
        # ignore
        pass
    tick = None
    try:
        tick = module.symbol_info_tick(symbol)
    except Exception:
        tick = None
    if tick is None:
        return None
    bid = getattr(tick, "bid", None)
    if bid is None:
        return None
    try:
        return float(bid)
    except Exception:
        return None


@dataclass(frozen=True)
class SymbolExposure:
    symbol: str
    base_direction: str
    base_volume: float
    base_avg_entry: Optional[float]
    hedge_volume: float
    net_volume_signed: float
    profit: float


def _aggregate_positions_for_symbol(
    positions: list[Any],
    *,
    symbol: str,
    base_direction: str,
    module: Any,
) -> SymbolExposure:
    buy_type = _pos_type_buy(module)
    sell_type = _pos_type_sell(module)

    base_type = _direction_to_pos_type(base_direction, module=module)
    hedge_type = sell_type if base_type == buy_type else buy_type

    base_vol = 0.0
    base_px_vol = 0.0
    hedge_vol = 0.0
    net_signed = 0.0
    profit = 0.0

    for p in positions:
        if str(getattr(p, "symbol", "")) != symbol:
            continue

        ptype = getattr(p, "type", None)
        try:
            ptype_i = int(ptype)
        except Exception:
            continue

        vol = getattr(p, "volume", None)
        try:
            v = float(vol)
        except Exception:
            continue

        if ptype_i == buy_type:
            net_signed += v
        elif ptype_i == sell_type:
            net_signed -= v

        pr = getattr(p, "profit", 0.0)
        try:
            profit += float(pr)
        except Exception:
            pass

        c = getattr(p, "comment", None)

        if ptype_i == base_type and not _is_rebalance_comment(c):
            price_open = getattr(p, "price_open", None)
            try:
                px = float(price_open)
            except Exception:
                px = None
            # Always count base volume. price_open may be missing on mocked positions.
            base_vol += v
            if px is not None and math.isfinite(px):
                base_px_vol += px * v
        elif ptype_i == hedge_type and _is_rebalance_comment(c):
            hedge_vol += v

    avg_entry = None
    if base_vol > 0 and base_px_vol > 0:
        avg_entry = base_px_vol / base_vol

    return SymbolExposure(
        symbol=symbol,
        base_direction=str(base_direction).upper(),
        base_volume=float(base_vol),
        base_avg_entry=float(avg_entry) if avg_entry is not None else None,
        hedge_volume=float(hedge_vol),
        net_volume_signed=float(net_signed),
        profit=float(profit),
    )


def _price_diff_pct(
    *,
    price_now: float,
    avg_entry: float,
    base_direction: str,
) -> Optional[float]:
    if avg_entry <= 0 or not math.isfinite(avg_entry) or not math.isfinite(price_now):
        return None
    if str(base_direction).upper() == "SELL":
        return ((avg_entry - price_now) / avg_entry) * 100.0
    return ((price_now - avg_entry) / avg_entry) * 100.0


def _r_value(*, price_diff_pct: float, atr_pct_ref: float) -> Optional[float]:
    if atr_pct_ref is None:
        return None
    a = float(atr_pct_ref)
    if a <= 0 or not math.isfinite(a):
        return None
    if not math.isfinite(price_diff_pct):
        return None
    return float(price_diff_pct) / a


def _tp_stage_from_r(
    r: Optional[float], *, thresholds: tuple[float, float, float]
) -> int:
    """Map r (= daily_change_pct / daily_atr_pct) to TP stage (0..3)."""

    if r is None or not math.isfinite(r):
        return 0

    th1, th2, th3 = (float(thresholds[0]), float(thresholds[1]), float(thresholds[2]))
    if r >= th3:
        return 3
    if r >= th2:
        return 2
    if r >= th1:
        return 1
    return 0


def _target_hedge_ratio(stage: int, *, amounts: tuple[float, float, float]) -> float:
    """Cumulative hedge ratio (0..1) for the given TP stage.

    stage=1 => amount_1st
    stage=2 => amount_1st + amount_2nd
    stage=3 => amount_1st + amount_2nd + amount_3rd
    """

    a1, a2, a3 = (float(amounts[0]), float(amounts[1]), float(amounts[2]))
    if stage <= 0:
        return 0.0
    if stage == 1:
        return max(0.0, min(1.0, a1))
    if stage == 2:
        return max(0.0, min(1.0, a1 + a2))
    return max(0.0, min(1.0, a1 + a2 + a3))


def _compute_drift(
    *,
    current_weights: dict[str, float],
    target_weights: dict[str, float],
) -> float:
    s = 0.0
    keys = set(target_weights.keys()) | set(current_weights.keys())
    for k in keys:
        tw = float(target_weights.get(k, 0.0) or 0.0)
        cw = float(current_weights.get(k, 0.0) or 0.0)
        s += abs(cw - tw)
    return 0.5 * s


def _fmt_or_dash(v: Any, *, digits: int = 2) -> str:
    if v is None:
        return "-"
    try:
        fv = float(v)
    except Exception:
        return str(v)
    if not math.isfinite(fv):
        return "-"
    return f"{fv:.{digits}f}"


def _fmt_pct_or_dash(v: Any, *, digits: int = 2) -> str:
    s = _fmt_or_dash(v, digits=digits)
    return "-" if s == "-" else f"{s}%"


def _box_table(
    header: list[str],
    rows: list[list[str]],
    *,
    right_align: Optional[set[int]] = None,
) -> str:
    right_align = set(right_align or set())

    all_rows = [header] + rows
    widths = [0] * len(header)
    for r in all_rows:
        for i, c in enumerate(r):
            widths[i] = max(widths[i], len(str(c)))

    def hline(left: str, mid: str, right: str) -> str:
        parts = ["─" * (w + 2) for w in widths]
        return left + mid.join(parts) + right

    def fmt_row(r: list[str]) -> str:
        cells = []
        for i, c in enumerate(r):
            t = str(c)
            if i in right_align:
                cells.append(t.rjust(widths[i]))
            else:
                cells.append(t.ljust(widths[i]))
        return "│ " + " │ ".join(cells) + " │"

    out = []
    out.append(hline("┌", "┬", "┐"))
    out.append(fmt_row(header))
    out.append(hline("├", "┼", "┤"))
    for r in rows:
        out.append(fmt_row(r))
    out.append(hline("└", "┴", "┘"))
    return "\n".join(out)


def _plain_table(
    header: list[str],
    rows: list[list[str]],
    *,
    right_align: Optional[set[int]] = None,
) -> str:
    """Format rows into a simple fixed-width table (ASCII-friendly).

    Slack's default font is proportional; use this inside a code block
    (```...```) to keep alignment.
    """

    right_align = set(right_align or set())
    all_rows = [header] + rows
    widths = [0] * len(header)
    for r in all_rows:
        for i, c in enumerate(r):
            widths[i] = max(widths[i], len(str(c)))

    def fmt_row(r: list[str]) -> str:
        parts: list[str] = []
        for i, c in enumerate(r):
            t = str(c)
            if i in right_align:
                parts.append(t.rjust(widths[i]))
            else:
                parts.append(t.ljust(widths[i]))
        return " ".join(parts)

    header_line = fmt_row(header)
    sep_line = " ".join("-" * w for w in widths)
    out = [header_line, sep_line]
    for r in rows:
        out.append(fmt_row(r))
    return "\n".join(out)


def _next_tp_threshold_r(r: Optional[float]) -> Optional[float]:
    if r is None or not math.isfinite(r):
        return 1.0
    if r < 1.0:
        return 1.0
    if r < 1.6:
        return 1.6
    if r < 2.4:
        return 2.4
    return None


def _settings_float(settings_module: Any, name: str) -> float:
    v = getattr(settings_module, name, None)
    if v is None:
        raise RuntimeError(f"Missing settings.{name}")
    f = float(v)
    if not math.isfinite(f):
        raise RuntimeError(f"Invalid settings.{name}={v!r}")
    return f


def _rebalance_portfolio_once_v2(
    *,
    env_file: Optional[os.PathLike[str] | str] = None,
    db_path: Optional[os.PathLike[str] | str] = None,
    settings_module: Any = settings,
    module: Any = mt5,
    trade: bool = False,
    slack_notify: bool = True,
    log_path: Optional[os.PathLike[str] | str] = None,
) -> dict[str, Any]:
    resolved_db_path = Path(db_path) if db_path is not None else load_sqlite_db_path(env_file)
    initialize_db(resolved_db_path)

    log_path_resolved = _resolve_rebalance_log_path(log_path, settings_module=settings_module)

    ymd = today_ymd_local()
    daily = read_daily_pv_tpv(ymd=ymd, env_file=env_file, db_path=resolved_db_path)
    if daily is None:
        raise RuntimeError(f"Missing daily_pv_tpv for ymd={ymd}. Run daily initialize first.")

    daily_pv = float(daily.get("pv") or 0.0)
    daily_tpv = float(daily.get("tpv") or 0.0)
    if daily_pv <= 0 or daily_tpv <= 0:
        raise RuntimeError(f"Invalid daily PV/TPV for ymd={ymd}: pv={daily_pv} tpv={daily_tpv}")

    latest_run = read_latest_portfolio_run(resolved_db_path)
    if latest_run is None:
        raise RuntimeError(f"No portfolio_runs found in db: {resolved_db_path}")
    portfolio_run_id = int(latest_run["id"])

    plan_map = read_planned_positions_for_run(resolved_db_path, portfolio_run_id)
    symbols = sorted(plan_map.keys())
    if not symbols:
        raise RuntimeError("No planned positions found. Build initial portfolio first.")

    ensure_rebalance_state_rows(
        portfolio_run_id,
        symbols,
        env_file=env_file,
        db_path=resolved_db_path,
    )
    state_map, _ = read_rebalance_state_map(
        portfolio_run_id,
        env_file=env_file,
        db_path=resolved_db_path,
    )

    metrics_base: dict[str, Any] = {
        "ymd": ymd,
        "daily_pv": daily_pv,
        "daily_tpv": daily_tpv,
    }

    global_running = read_global_running(env_file=env_file, db_path=resolved_db_path, default=True)
    mode_label = "TRADE" if trade else "DRY-RUN"
    if not bool(global_running):
        # Persist a run record for audit.
        try:
            create_rebalance_run(
                portfolio_run_id=portfolio_run_id,
                tpv_ref=float(daily_tpv),
                tpv_now=None,
                tpv_return=None,
                drift=None,
                max_contrib=None,
                triggered=False,
                dry_run=not bool(trade),
                error="SKIPPED: global_running=false",
                metrics={
                    **metrics_base,
                    "skipped": True,
                    "reason": "global_running=false",
                    "global_running": False,
                },
                env_file=env_file,
                db_path=resolved_db_path,
            )
        except Exception:
            pass

        text = "\n".join(
            [
                f"[rebalance] skipped ({mode_label})",
                "reason=global_running=false",
                f"ymd={ymd}",
                f"portfolio_run_id={portfolio_run_id}",
                "",
                "actions:",
                "```(none)```",
            ]
        )
        _append_run_log(text=text, log_path=log_path_resolved)
        if slack_notify:
            try:
                send_slack_message(text, env_file=env_file)
            except Exception:
                pass
        return {
            "db_path": str(resolved_db_path),
            "portfolio_run_id": portfolio_run_id,
            "ymd": ymd,
            "skipped": True,
            "global_running": False,
            "actions": [],
        }

    pv_leverage = _settings_float(settings_module, "PV_LEVERAGE")
    cb_th = _settings_float(settings_module, "CIRCUIT_BREAKER_THRESHOLD")
    tp_thresholds = (
        _settings_float(settings_module, "TP_ATR_1ST"),
        _settings_float(settings_module, "TP_ATR_2ND"),
        _settings_float(settings_module, "TP_ATR_3RD"),
    )
    tp_amounts = (
        _settings_float(settings_module, "TP_AMOUNT_1ST"),
        _settings_float(settings_module, "TP_AMOUNT_2ND"),
        _settings_float(settings_module, "TP_AMOUNT_3RD"),
    )

    config = load_mt5_config(env_file)
    initialized = False
    positions: list[Any] = []
    actions: list[dict[str, Any]] = []
    metrics: dict[str, Any] = {
        **metrics_base,
        "pv_leverage": pv_leverage,
        "circuit_breaker_threshold": cb_th,
    }

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
            raise Mt5ConnectionError(f"MT5 initialize failed. last_error={_safe_last_error(module)}")

        info = module.account_info()
        if info is None:
            raise Mt5ConnectionError(f"MT5 account_info returned None. last_error={_safe_last_error(module)}")
        equity = getattr(info, "equity", None)
        if equity is None:
            raise RuntimeError("MT5 account_info.equity is missing")
        equity_f = float(equity)
        pv_now = equity_f * pv_leverage
        metrics["equity_now"] = equity_f
        metrics["pv_now"] = pv_now

        raw_positions = module.positions_get()
        if raw_positions:
            positions = list(raw_positions)

        if not positions:
            # Persist a run record for audit.
            try:
                create_rebalance_run(
                    portfolio_run_id=portfolio_run_id,
                    tpv_ref=float(daily_tpv),
                    tpv_now=float(pv_now),
                    tpv_return=(float(pv_now) / float(daily_tpv)) - 1.0,
                    drift=None,
                    max_contrib=None,
                    triggered=False,
                    dry_run=not bool(trade),
                    error="SKIPPED: no_positions",
                    metrics={
                        **metrics,
                        "skipped": True,
                        "reason": "no_positions",
                        "positions_count": 0,
                    },
                    env_file=env_file,
                    db_path=resolved_db_path,
                )
            except Exception:
                pass

            text = "\n".join(
                [
                    f"[rebalance] skipped ({mode_label})",
                    "reason=no_positions",
                    f"ymd={ymd}",
                    f"pv_now={pv_now:.2f} daily_pv={daily_pv:.2f} daily_tpv={daily_tpv:.2f}",
                    "",
                    "actions:",
                    "```(none)```",
                ]
            )
            _append_run_log(text=text, log_path=log_path_resolved)
            if slack_notify:
                try:
                    send_slack_message(text, env_file=env_file)
                except Exception:
                    pass
            return {
                "db_path": str(resolved_db_path),
                "portfolio_run_id": portfolio_run_id,
                "ymd": ymd,
                "metrics": metrics,
                "triggered": False,
                "actions": [],
            }

        circuit_pv = daily_pv * cb_th
        metrics["circuit_pv"] = circuit_pv

        # 1) Circuit breaker
        if pv_now < circuit_pv:
            close_reqs: list[dict[str, Any]] = []
            for p in positions:
                sym = getattr(p, "symbol", None)
                if not sym:
                    continue
                ptype = getattr(p, "type", None)
                vol = getattr(p, "volume", None)
                if vol is None:
                    continue
                v = float(vol)
                if not math.isfinite(v) or v <= 0:
                    continue
                d = "SELL" if ptype == _pos_type_buy(module) else "BUY"

                # Build a close request (best-effort). If ticket is available, use it.
                if hasattr(module, "symbol_select"):
                    _ = bool(module.symbol_select(str(sym), True))

                tick = module.symbol_info_tick(str(sym))
                if tick is None:
                    continue
                ask = getattr(tick, "ask", None)
                bid = getattr(tick, "bid", None)
                if d == "BUY":
                    price = float(ask) if ask is not None else None
                    order_type = getattr(module, "ORDER_TYPE_BUY", None)
                else:
                    price = float(bid) if bid is not None else None
                    order_type = getattr(module, "ORDER_TYPE_SELL", None)
                if price is None or order_type is None:
                    continue

                req: dict[str, Any] = {
                    "action": getattr(module, "TRADE_ACTION_DEAL", None),
                    "symbol": str(sym),
                    "volume": float(v),
                    "type": order_type,
                    "price": float(price),
                    "deviation": 20,
                    "comment": "circuit_breaker_close",
                }

                ticket = getattr(p, "ticket", None)
                if ticket is not None:
                    try:
                        req["position"] = int(ticket)
                    except Exception:
                        pass

                if hasattr(module, "ORDER_TIME_GTC"):
                    req["type_time"] = getattr(module, "ORDER_TIME_GTC")

                filling: Optional[int] = None
                info = None
                if hasattr(module, "symbol_info"):
                    info = module.symbol_info(str(sym))
                if info is not None and hasattr(info, "filling_mode") and info.filling_mode is not None:
                    try:
                        filling = int(info.filling_mode)
                    except Exception:
                        filling = None
                if filling is not None:
                    req["type_filling"] = filling

                close_reqs.append(req)
                actions.append(
                    {
                        "symbol": str(sym),
                        "action": "CLOSE",
                        "direction": d,
                        "volume": v,
                        "stage": None,
                        "r": None,
                        "price": float(price),
                        "atr_pct_ref": None,
                        "reason": "circuit_breaker",
                        "comment": "circuit_breaker_close",
                    }
                )

            if trade and close_reqs:
                _ = execute_mt5_market_orders(close_reqs, module=module)
                set_global_running(False, env_file=env_file, db_path=resolved_db_path)

            run_id, _ = create_rebalance_run(
                portfolio_run_id=portfolio_run_id,
                tpv_ref=float(daily_tpv),
                tpv_now=float(pv_now),
                tpv_return=(float(pv_now) / float(daily_tpv)) - 1.0,
                drift=None,
                max_contrib=None,
                triggered=True,
                dry_run=not bool(trade),
                metrics=metrics,
                env_file=env_file,
                db_path=resolved_db_path,
            )
            if actions:
                persist_rebalance_actions(
                    run_id,
                    actions,
                    env_file=env_file,
                    db_path=resolved_db_path,
                )

            actions_text = "(none)"
            if actions:
                actions_text = "\n".join(
                    [
                        f"- {a['symbol']} {a['action']} {a['direction']} vol={a['volume']:.2f}"
                        for a in actions
                    ]
                )
            text = "\n".join(
                [
                    f"[rebalance] circuit_breaker ({mode_label})",
                    f"ymd={ymd}",
                    f"pv_now={pv_now:.2f} daily_pv={daily_pv:.2f} cb_th={cb_th:.3f} cb_pv={circuit_pv:.2f}",
                    "",
                    "actions:",
                    f"```{actions_text}```",
                    "",
                    "note: global_running set to OFF" if trade else "note: dry-run",
                ]
            )
            _append_run_log(text=text, log_path=log_path_resolved)
            if slack_notify:
                try:
                    send_slack_message(text, env_file=env_file)
                except Exception:
                    pass
            return {
                "db_path": str(resolved_db_path),
                "portfolio_run_id": portfolio_run_id,
                "ymd": ymd,
                "metrics": metrics,
                "triggered": True,
                "actions": actions,
            }

        # 2) PV <= TPV => no-op
        if pv_now <= daily_tpv:
            run_id, _ = create_rebalance_run(
                portfolio_run_id=portfolio_run_id,
                tpv_ref=float(daily_tpv),
                tpv_now=float(pv_now),
                tpv_return=(float(pv_now) / float(daily_tpv)) - 1.0,
                drift=None,
                max_contrib=None,
                triggered=False,
                dry_run=not bool(trade),
                metrics=metrics,
                env_file=env_file,
                db_path=resolved_db_path,
            )
            _ = run_id
            text = "\n".join(
                [
                    f"[rebalance] no_op ({mode_label})",
                    f"ymd={ymd}",
                    f"pv_now={pv_now:.2f} daily_tpv={daily_tpv:.2f}",
                    "",
                    "actions:",
                    "```(none)```",
                ]
            )
            _append_run_log(text=text, log_path=log_path_resolved)
            if slack_notify:
                try:
                    send_slack_message(text, env_file=env_file)
                except Exception:
                    pass
            return {
                "db_path": str(resolved_db_path),
                "portfolio_run_id": portfolio_run_id,
                "ymd": ymd,
                "metrics": metrics,
                "triggered": False,
                "actions": [],
            }

        # 3) PV > TPV => staged TP hedge
        hedge_plan: list[PlannedPosition] = []
        table_rows: list[list[str]] = []
        for sym in symbols:
            plan = plan_map.get(sym) or {}
            base_dir = str(plan.get("direction") or "BUY").upper()
            exp = _aggregate_positions_for_symbol(
                positions,
                symbol=str(sym),
                base_direction=base_dir,
                module=module,
            )
            if float(exp.base_volume) <= 0:
                continue

            ref_price_row = read_daily_symbol_price(
                ymd=ymd,
                symbol=str(sym),
                env_file=env_file,
                db_path=resolved_db_path,
            )
            atr_row = read_daily_atr(
                ymd=ymd,
                symbol=str(sym),
                env_file=env_file,
                db_path=resolved_db_path,
            )
            ref_price = None if not ref_price_row else ref_price_row.get("sell")
            atr_pct = None if not atr_row else atr_row.get("atr_pct")

            price_now = _fetch_sell_price(str(sym), module=module)
            diff_pct = None
            r = None
            stage_reached = 0
            if price_now is not None and ref_price is not None:
                diff_pct = _price_diff_pct(
                    price_now=float(price_now),
                    avg_entry=float(ref_price),
                    base_direction=base_dir,
                )
                if diff_pct is not None and atr_pct is not None:
                    r = _r_value(price_diff_pct=float(diff_pct), atr_pct_ref=float(atr_pct))
                    stage_reached = _tp_stage_from_r(r, thresholds=tp_thresholds)

            prev_state = state_map.get(str(sym))
            prev_stage = int(prev_state.tp_stage) if prev_state is not None else 0
            prev_hedged = float(prev_state.hedged_volume) if prev_state is not None else 0.0

            new_stage = prev_stage + 1 if stage_reached > prev_stage else prev_stage
            target_ratio = _target_hedge_ratio(new_stage, amounts=tp_amounts)
            target_hedge_vol = float(exp.base_volume) * float(target_ratio)
            to_hedge = max(0.0, float(target_hedge_vol) - float(prev_hedged))

            to_hedge_rounded = 0.0
            if to_hedge > 0:
                to_hedge_rounded = _round_to_volume_constraints(
                    float(to_hedge),
                    volume_min=plan.get("volume_min"),
                    volume_step=plan.get("volume_step"),
                    volume_max=plan.get("volume_max"),
                )

            if new_stage > prev_stage and to_hedge_rounded > 0:
                hedge_direction = _opposite_direction(base_dir)
                hedge_plan.append(
                    PlannedPosition(
                        symbol=str(sym),
                        atr_pct=None,
                        weight=None,
                        risk_usd=None,
                        risk_per_lot_at_1atr=None,
                        ideal_lot=None,
                        lot=float(to_hedge_rounded),
                        direction=str(hedge_direction),
                        volume_min=plan.get("volume_min"),
                        volume_step=plan.get("volume_step"),
                        volume_max=plan.get("volume_max"),
                        usd_per_lot=None,
                        usd_nominal=None,
                        reason="tp_hedge",
                    )
                )
                actions.append(
                    {
                        "symbol": str(sym),
                        "action": "HEDGE",
                        "direction": str(hedge_direction),
                        "volume": float(to_hedge_rounded),
                        "stage": int(new_stage),
                        "r": float(r) if r is not None else None,
                        "price": float(price_now) if price_now is not None else None,
                        "atr_pct_ref": float(atr_pct) if atr_pct is not None else None,
                        "reason": "tp_stage_reached",
                        "comment": f"tp_stage_{new_stage}",
                    }
                )
                upsert_rebalance_state(
                    RebalanceState(
                        portfolio_run_id=int(portfolio_run_id),
                        symbol=str(sym),
                        tp_stage=int(new_stage),
                        hedged_volume=float(prev_hedged) + float(to_hedge_rounded),
                        last_peak_price=None,
                        updated_at=_now_utc_iso(),
                    ),
                    env_file=env_file,
                    db_path=resolved_db_path,
                )

            table_rows.append(
                [
                    str(sym),
                    str(base_dir),
                    _fmt_or_dash(exp.base_volume, digits=2),
                    _fmt_or_dash(prev_hedged, digits=2),
                    _fmt_or_dash(ref_price, digits=5),
                    _fmt_or_dash(price_now, digits=5),
                    _fmt_pct_or_dash(diff_pct, digits=2),
                    _fmt_pct_or_dash(atr_pct, digits=2),
                    _fmt_or_dash(r, digits=2),
                    f"{prev_stage}->{new_stage}",
                    _fmt_or_dash(to_hedge_rounded, digits=2),
                ]
            )

        reqs = []
        if hedge_plan:
            reqs = build_mt5_market_order_requests(hedge_plan, module=module, comment="rebalance_tp_hedge")
            if trade:
                _ = execute_mt5_market_orders(reqs, module=module)

        run_id, _ = create_rebalance_run(
            portfolio_run_id=portfolio_run_id,
            tpv_ref=float(daily_tpv),
            tpv_now=float(pv_now),
            tpv_return=(float(pv_now) / float(daily_tpv)) - 1.0,
            drift=None,
            max_contrib=None,
            triggered=bool(actions),
            dry_run=not bool(trade),
            metrics=metrics,
            env_file=env_file,
            db_path=resolved_db_path,
        )
        if actions:
            persist_rebalance_actions(run_id, actions, env_file=env_file, db_path=resolved_db_path)

        header = ["sym", "dir", "base", "hedged", "ref", "now", "diff%", "ATR%", "R", "stage", "hedge"]
        table = _plain_table(header, table_rows, right_align={2, 3, 4, 5, 6, 7, 8, 10}) if table_rows else "(none)"

        actions_text = "(none)"
        if actions:
            actions_text = "\n".join(
                [
                    f"- {a['symbol']} {a['action']} {a['direction']} vol={a['volume']:.2f} stage={a.get('stage')} r={_fmt_or_dash(a.get('r'), digits=2)}"
                    for a in actions
                ]
            )

        text = "\n".join(
            [
                f"[rebalance] tp_check ({mode_label})",
                f"ymd={ymd}",
                f"pv_now={pv_now:.2f} daily_pv={daily_pv:.2f} daily_tpv={daily_tpv:.2f}",
                "",
                "symbols:",
                f"```{table}```",
                "",
                "actions:",
                f"```{actions_text}```",
            ]
        )
        _append_run_log(text=text, log_path=log_path_resolved)
        if slack_notify:
            try:
                send_slack_message(text, env_file=env_file)
            except Exception:
                pass
        return {
            "db_path": str(resolved_db_path),
            "portfolio_run_id": portfolio_run_id,
            "ymd": ymd,
            "metrics": metrics,
            "triggered": bool(actions),
            "actions": actions,
        }
    finally:
        if initialized:
            try:
                module.shutdown()
            except Exception:
                pass


def rebalance_portfolio_once(
    *,
    env_file: Optional[os.PathLike[str] | str] = None,
    db_path: Optional[os.PathLike[str] | str] = None,
    settings_module: Any = settings,
    module: Any = mt5,
    trade: bool = False,
    slack_notify: bool = True,
    tpv_return_threshold: float = 0.008,
    drift_threshold: float = 0.08,
    max_contrib_threshold: float = 0.006,
    log_path: Optional[os.PathLike[str] | str] = None,
) -> dict[str, Any]:
    # New implementation (2026-02): delegate to v2.
    # Keep legacy parameters for backwards compatibility with existing code/tests.
    return _rebalance_portfolio_once_v2(
        env_file=env_file,
        db_path=db_path,
        settings_module=settings_module,
        module=module,
        trade=trade,
        slack_notify=slack_notify,
        log_path=log_path,
    )
    if False:  # legacy implementation (kept for reference)
        """
    resolved_db_path = Path(db_path) if db_path is not None else load_sqlite_db_path(env_file)
    initialize_db(resolved_db_path)

    log_path_resolved = _resolve_rebalance_log_path(log_path, settings_module=settings_module)

    latest_run = read_latest_portfolio_run(resolved_db_path)
    if latest_run is None:
        raise RuntimeError(f"No portfolio_runs found in db: {resolved_db_path}")

    portfolio_run_id = int(latest_run["id"])
    tpv_ref = float(latest_run.get("tpv") or 0.0)
    if tpv_ref <= 0:
        raise RuntimeError("Invalid TPV_ref (portfolio_runs.tpv). Build initial portfolio first.")

    global_running = read_global_running(env_file=env_file, db_path=resolved_db_path, default=True)
    if not bool(global_running):
        mode_label = "TRADE" if trade else "DRY-RUN"
        text = "\n".join(
            [
                f"[rebalance] skipped ({mode_label})",
                "reason=global_running=false",
                f"portfolio_run_id={portfolio_run_id} tpv_ref={tpv_ref:.2f}",
                "",
                "symbols:",
                "(skipped)",
                "",
                "actions:",
                "```(none)```",
            ]
        )

        # Persist a run record for audit.
        try:
            create_rebalance_run(
                portfolio_run_id=portfolio_run_id,
                tpv_ref=tpv_ref,
                tpv_now=None,
                tpv_return=None,
                drift=None,
                max_contrib=None,
                triggered=False,
                dry_run=not bool(trade),
                error="SKIPPED: global_running=false",
                metrics={
                    "portfolio_run_id": portfolio_run_id,
                    "tpv_ref": tpv_ref,
                    "skipped": True,
                    "global_running": False,
                    "reason": "global_running=false",
                },
                env_file=env_file,
                db_path=resolved_db_path,
            )
        except Exception:
            # ignore audit persistence failures
            pass

        _append_run_log(text=text, log_path=log_path_resolved)
        if slack_notify:
            try:
                send_slack_message(text, env_file=env_file)
            except Exception:
                pass

        return {
            "db_path": str(resolved_db_path),
            "portfolio_run_id": portfolio_run_id,
            "skipped": True,
            "global_running": False,
            "triggered": False,
            "actions": [],
        }

    plan_map = read_planned_positions_for_run(resolved_db_path, portfolio_run_id)
    snap_map = read_symbol_snapshots_for_run(resolved_db_path, portfolio_run_id)
    symbols = sorted(plan_map.keys())

    ensure_rebalance_state_rows(
        portfolio_run_id,
        symbols,
        env_file=env_file,
        db_path=resolved_db_path,
    )
    state_map, _ = read_rebalance_state_map(
        portfolio_run_id,
        env_file=env_file,
        db_path=resolved_db_path,
    )

    config = load_mt5_config(env_file)
    initialized = False
    positions: list[Any] = []
    actions: list[dict[str, Any]] = []
    metrics: dict[str, Any] = {}
    error: Optional[str] = None
    text: Optional[str] = None
    triggered: bool = False

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
            raise Mt5ConnectionError(f"MT5 initialize failed. last_error={_safe_last_error(module)}")

        raw_positions = module.positions_get()
        if raw_positions:
            positions = list(raw_positions)

        if not positions:
            mode_label = "TRADE" if trade else "DRY-RUN"
            text = "\n".join(
                [
                    f"[rebalance] skipped ({mode_label})",
                    "reason=no_positions",
                    f"portfolio_run_id={portfolio_run_id} tpv_ref={tpv_ref:.2f}",
                    "",
                    "positions:",
                    "```(none)```",
                    "",
                    "actions:",
                    "```(none)```",
                ]
            )

            # Persist a run record for audit.
            try:
                create_rebalance_run(
                    portfolio_run_id=portfolio_run_id,
                    tpv_ref=tpv_ref,
                    tpv_now=None,
                    tpv_return=None,
                    drift=None,
                    max_contrib=None,
                    triggered=False,
                    dry_run=not bool(trade),
                    error="SKIPPED: no_positions",
                    metrics={
                        "portfolio_run_id": portfolio_run_id,
                        "tpv_ref": tpv_ref,
                        "skipped": True,
                        "reason": "no_positions",
                        "positions_count": 0,
                    },
                    env_file=env_file,
                    db_path=resolved_db_path,
                )
            except Exception:
                pass

            _append_run_log(text=text, log_path=log_path_resolved)
            if slack_notify:
                try:
                    send_slack_message(text, env_file=env_file)
                except Exception:
                    pass

            return {
                "db_path": str(resolved_db_path),
                "portfolio_run_id": portfolio_run_id,
                "skipped": True,
                "reason": "no_positions",
                "triggered": False,
                "actions": [],
            }

        exposures: dict[str, SymbolExposure] = {}
        current_notional: dict[str, float] = {}
        current_prices: dict[str, float] = {}
        target_weights: dict[str, float] = {}
        current_weights: dict[str, float] = {}
        contribs: dict[str, float] = {}

        for sym in symbols:
            p = plan_map.get(sym) or {}
            target_weights[sym] = float(p.get("weight") or 0.0)

        for sym in symbols:
            p = plan_map.get(sym) or {}
            base_direction = str(p.get("direction") or "BUY").upper()
            exp = _aggregate_positions_for_symbol(
                positions,
                symbol=sym,
                base_direction=base_direction,
                module=module,
            )
            exposures[sym] = exp

            sell = _fetch_sell_price(sym, module=module)
            if sell is not None:
                current_prices[sym] = float(sell)

            snap = snap_map.get(sym)
            usd_per_lot_now = None
            if sell is not None and snap is not None:
                usd_pair_sell = None
                if snap.usd_pair_symbol:
                    usd_pair_sell = _fetch_sell_price(str(snap.usd_pair_symbol), module=module)
                usd_per_lot_now = _calc_usd_per_lot(
                    sell=sell,
                    contract_size=snap.contract_size,
                    currency_profit=snap.currency_profit,
                    usd_pair_symbol=snap.usd_pair_symbol,
                    usd_pair_sell=usd_pair_sell,
                )

            # Fallback to reference usd_per_lot if live calc fails.
            if usd_per_lot_now is None and snap is not None and snap.usd_per_lot is not None:
                usd_per_lot_now = float(snap.usd_per_lot)

            net_abs_lot = abs(float(exp.net_volume_signed))
            nominal = None
            if usd_per_lot_now is not None and math.isfinite(usd_per_lot_now):
                nominal = net_abs_lot * float(usd_per_lot_now)
            current_notional[sym] = float(nominal or 0.0)

            contribs[sym] = float(exp.profit) / float(tpv_ref)

        tpv_now = float(sum(max(v, 0.0) for v in current_notional.values()))
        tpv_return = (tpv_now - tpv_ref) / tpv_ref

        for sym, nom in current_notional.items():
            if tpv_now > 0 and nom > 0:
                current_weights[sym] = float(nom) / tpv_now
            else:
                current_weights[sym] = 0.0

        drift = _compute_drift(current_weights=current_weights, target_weights=target_weights)
        max_contrib = max(contribs.values()) if contribs else 0.0

        triggered = (
            float(tpv_return) >= float(tpv_return_threshold)
            and float(drift) >= float(drift_threshold)
        ) or (
            float(max_contrib) >= float(max_contrib_threshold)
            and float(drift) >= float(drift_threshold)
        )

        metrics = {
            "portfolio_run_id": portfolio_run_id,
            "tpv_ref": tpv_ref,
            "tpv_now": tpv_now,
            "tpv_return": tpv_return,
            "drift": drift,
            "max_contrib": max_contrib,
        }

        # Persist run (even if not triggered) for audit.
        rebalance_run_id, _ = create_rebalance_run(
            portfolio_run_id=portfolio_run_id,
            tpv_ref=tpv_ref,
            tpv_now=tpv_now,
            tpv_return=tpv_return,
            drift=drift,
            max_contrib=max_contrib,
            triggered=bool(triggered),
            dry_run=not bool(trade),
            error=None,
            metrics=metrics,
            env_file=env_file,
            db_path=resolved_db_path,
        )

        if triggered:
            hedge_plan: list[PlannedPosition] = []

            for sym in symbols:
                exp = exposures.get(sym)
                if exp is None:
                    continue

                snap = snap_map.get(sym)
                if snap is None:
                    continue

                price_now = current_prices.get(sym)
                if price_now is None:
                    continue

                atr_pct_ref = plan_map.get(sym, {}).get("atr_pct")
                if atr_pct_ref is None:
                    atr_pct_ref = snap.atr_pct

                if exp.base_avg_entry is None or exp.base_volume <= 0:
                    continue

                diff_pct = _price_diff_pct(
                    price_now=float(price_now),
                    avg_entry=float(exp.base_avg_entry),
                    base_direction=exp.base_direction,
                )
                r = _r_value(price_diff_pct=float(diff_pct or 0.0), atr_pct_ref=float(atr_pct_ref or 0.0))
                stage_reached = _tp_stage_from_r(r)

                prev_state = state_map.get(sym)
                prev_stage = int(prev_state.tp_stage) if prev_state is not None else 0
                new_stage = max(prev_stage, stage_reached)

                # Update peak (directional) for future use.
                last_peak = prev_state.last_peak_price if prev_state is not None else None
                if exp.base_direction == "BUY":
                    if last_peak is None or float(price_now) > float(last_peak):
                        last_peak = float(price_now)
                else:
                    if last_peak is None or float(price_now) < float(last_peak):
                        last_peak = float(price_now)

                desired_ratio = _target_hedge_ratio(new_stage)
                desired_hedge = float(exp.base_volume) * float(desired_ratio)
                already_hedged = float(exp.hedge_volume)
                to_hedge = max(0.0, desired_hedge - already_hedged)

                # Round to volume constraints.
                to_hedge_rounded = 0.0
                if to_hedge > 0:
                    to_hedge_rounded = _round_to_volume_constraints(
                        to_hedge,
                        volume_min=snap.volume_min,
                        volume_step=snap.volume_step,
                        volume_max=snap.volume_max,
                    )
                min_lot = float(snap.volume_min or 0.0)
                if to_hedge_rounded < min_lot:
                    to_hedge_rounded = 0.0

                if to_hedge_rounded > 0:
                    hedge_direction = _opposite_direction(exp.base_direction)
                    comment = f"rebalance_tp{new_stage}"
                    hedge_plan.append(
                        PlannedPosition(
                            symbol=sym,
                            atr_pct=float(atr_pct_ref) if atr_pct_ref is not None else None,
                            weight=None,
                            risk_usd=None,
                            risk_per_lot_at_1atr=None,
                            ideal_lot=None,
                            lot=float(to_hedge_rounded),
                            direction=hedge_direction,
                            volume_min=snap.volume_min,
                            volume_step=snap.volume_step,
                            volume_max=snap.volume_max,
                            usd_per_lot=snap.usd_per_lot,
                            usd_nominal=None,
                            reason=None,
                        )
                    )

                    actions.append(
                        {
                            "symbol": sym,
                            "action": "HEDGE",
                            "direction": hedge_direction,
                            "volume": float(to_hedge_rounded),
                            "stage": int(new_stage),
                            "r": float(r) if r is not None else None,
                            "price": float(price_now),
                            "atr_pct_ref": float(atr_pct_ref) if atr_pct_ref is not None else None,
                            "reason": "tp_stage_reached",
                            "comment": comment,
                        }
                    )

                # Persist updated state (even if no action, stage/peak may advance).
                new_state = RebalanceState(
                    portfolio_run_id=int(portfolio_run_id),
                    symbol=str(sym),
                    tp_stage=int(new_stage),
                    hedged_volume=float(already_hedged + float(to_hedge_rounded)),
                    last_peak_price=float(last_peak) if last_peak is not None else None,
                    updated_at=_now_utc_iso(),
                )
                upsert_rebalance_state(
                    new_state,
                    env_file=env_file,
                    db_path=resolved_db_path,
                )

            if actions:
                persist_rebalance_actions(
                    rebalance_run_id,
                    actions,
                    env_file=env_file,
                    db_path=resolved_db_path,
                )

            # Send hedge orders
            requests = []
            results = []
            if hedge_plan:
                requests = build_mt5_market_order_requests(
                    hedge_plan,
                    module=module,
                    comment="rebalance_hedge",
                )
                # override per-order comment if provided in actions (best-effort)
                for req in requests:
                    sym = req.get("symbol")
                    # pick the latest action comment
                    c = None
                    for a in reversed(actions):
                        if a.get("symbol") == sym:
                            c = a.get("comment")
                            break
                    if c:
                        req["comment"] = str(c)

                if trade:
                    results = execute_mt5_market_orders(requests, module=module)

        # Build text report for Slack + file log.
        held_symbols = []
        for sym in symbols:
            exp = exposures.get(sym)
            if exp is None:
                continue
            if float(exp.base_volume) > 0.0 or float(exp.hedge_volume) > 0.0:
                held_symbols.append(sym)

        atr_period = latest_run.get("atr_period")
        if atr_period is None:
            atr_period = getattr(settings_module, "ATR_PERIOD", None)
        time_frame = latest_run.get("time_frame")
        if time_frame is None:
            time_frame = getattr(settings_module, "TIME_FRAME", None)

        header = [
            "sym",
            "dir",
            "base",
            "hedge",
            "PnL$",
            "diff%",
            "ATR%ref",
            "ATR%now",
            "R(ref)",
            "stage",
            "nextR",
            "need%",
        ]
        table_rows: list[list[str]] = []

        for sym in held_symbols:
            exp = exposures.get(sym)
            if exp is None:
                continue

            snap = snap_map.get(sym)
            atr_pct_ref = plan_map.get(sym, {}).get("atr_pct")
            if atr_pct_ref is None and snap is not None:
                atr_pct_ref = snap.atr_pct

            price_now = current_prices.get(sym)
            diff_pct = None
            r_ref = None
            if price_now is not None and exp.base_avg_entry is not None:
                diff_pct = _price_diff_pct(
                    price_now=float(price_now),
                    avg_entry=float(exp.base_avg_entry),
                    base_direction=exp.base_direction,
                )
                r_ref = _r_value(
                    price_diff_pct=float(diff_pct or 0.0),
                    atr_pct_ref=float(atr_pct_ref or 0.0),
                )

            stage_reached = _tp_stage_from_r(r_ref)
            prev_state = state_map.get(sym)
            prev_stage = int(prev_state.tp_stage) if prev_state is not None else 0
            stage_now = max(prev_stage, stage_reached)

            next_r = _next_tp_threshold_r(r_ref)
            need_pct = None
            if (
                next_r is not None
                and diff_pct is not None
                and atr_pct_ref is not None
                and math.isfinite(float(atr_pct_ref))
            ):
                need_pct = (float(next_r) * float(atr_pct_ref)) - float(diff_pct)
                if need_pct < 0:
                    need_pct = 0.0

            atr_now_pct = None
            if atr_period is not None and time_frame is not None:
                try:
                    live = fetch_symbol_snapshot(
                        sym,
                        module=module,
                        atr_period=int(atr_period),
                        time_frame=time_frame,
                    )
                    atr_now_pct = live.atr_pct
                except Exception:
                    atr_now_pct = None

            table_rows.append(
                [
                    str(sym),
                    str(exp.base_direction),
                    _fmt_or_dash(exp.base_volume, digits=2),
                    _fmt_or_dash(exp.hedge_volume, digits=2),
                    _fmt_or_dash(exp.profit, digits=2),
                    _fmt_pct_or_dash(diff_pct, digits=2),
                    _fmt_pct_or_dash(atr_pct_ref, digits=2),
                    _fmt_pct_or_dash(atr_now_pct, digits=2),
                    _fmt_or_dash(r_ref, digits=2),
                    f"{prev_stage}->{stage_now}",
                    _fmt_or_dash(next_r, digits=1) if next_r is not None else "-",
                    _fmt_pct_or_dash(need_pct, digits=2),
                ]
            )

        table = _plain_table(
            header,
            table_rows,
            right_align={2, 3, 4, 5, 6, 7, 8, 10, 11},
        )

        actions_text = "(none)"
        if actions:
            actions_lines = []
            for a in actions:
                actions_lines.append(
                    f"- {a['symbol']} {a['action']} {a['direction']} vol={a['volume']:.2f} stage={a.get('stage')} r={_fmt_or_dash(a.get('r'), digits=2)}"
                )
            actions_text = "\n".join(actions_lines)

        mode_label = "TRADE" if trade else "DRY-RUN"
        status_label = "triggered" if bool(triggered) else "not_triggered"
        lines = [
            f"[rebalance] {status_label} ({mode_label})",
            f"portfolio_run_id={portfolio_run_id} tpv_ref={tpv_ref:.2f} tpv_now={tpv_now:.2f} tpv_return={tpv_return*100:.3f}%",
            f"drift={drift*100:.2f}% (th={drift_threshold*100:.2f}%) max_contrib={max_contrib*100:.3f}%",
            "",
            "symbols:",
            f"```{table}```" if table_rows else "(none)",
            "",
            "actions:",
            f"```{actions_text}```",
        ]
        text = "\n".join(lines)

        # Always write file log.
        _append_run_log(text=text, log_path=log_path_resolved)

        # Slack: keep noise low (notify only when triggered).
        if slack_notify and bool(triggered):
            send_slack_message(text, env_file=env_file)

        return {
            "db_path": str(resolved_db_path),
            "portfolio_run_id": portfolio_run_id,
            "metrics": metrics,
            "triggered": bool(triggered),
            "actions": actions,
        }
    except Exception as e:
        error = f"{type(e).__name__}: {e}"
        tb = traceback.format_exc()

        mode_label = "TRADE" if trade else "DRY-RUN"
        lines = [
            f"[rebalance] error ({mode_label})",
            f"error={error}",
            "",
            "traceback:",
            f"```{tb}```",
        ]
        text = "\n".join(lines)

        _append_run_log(text=text, log_path=log_path_resolved)
        if slack_notify:
            try:
                send_slack_message(text, env_file=env_file)
            except Exception:
                pass

        # Persist a failed run record.
        create_rebalance_run(
            portfolio_run_id=portfolio_run_id,
            tpv_ref=tpv_ref if "tpv_ref" in locals() else None,
            tpv_now=metrics.get("tpv_now") if metrics else None,
            tpv_return=metrics.get("tpv_return") if metrics else None,
            drift=metrics.get("drift") if metrics else None,
            max_contrib=metrics.get("max_contrib") if metrics else None,
            triggered=False,
            dry_run=not bool(trade),
            error=error,
            metrics=metrics,
            env_file=env_file,
            db_path=resolved_db_path,
        )
        raise
    finally:
        if initialized:
            try:
                module.shutdown()
            except Exception:
                pass

        """


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Rebalance existing portfolio on MT5")
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
    parser.add_argument(
        "--trade",
        action="store_true",
        help="Actually send orders (overrides settings default)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Force dry-run (overrides settings default)",
    )
    parser.add_argument(
        "--no-slack",
        action="store_true",
        help="Disable Slack notification",
    )
    parser.add_argument(
        "--log",
        default=None,
        help="Path to log file (default: logs/rebalance_portfolio.log)",
    )
    args = parser.parse_args(argv)

    if bool(args.trade) and bool(args.dry_run):
        parser.error("--trade and --dry-run are mutually exclusive")

    default_trade = bool(getattr(settings, "REBALANCE_TRADE_DEFAULT", False))
    if bool(args.trade):
        trade_flag = True
    elif bool(args.dry_run):
        trade_flag = False
    else:
        trade_flag = default_trade

    rebalance_portfolio_once(
        env_file=args.env,
        db_path=args.db,
        settings_module=settings,
        module=mt5,
        trade=bool(trade_flag),
        slack_notify=not bool(args.no_slack),
        log_path=args.log,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
