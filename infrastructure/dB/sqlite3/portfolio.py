from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

from infrastructure.dB.sqlite3.initialize_db import (
    connect,
    initialize_db,
    load_sqlite_db_path,
)
from infrastructure.mt5.symbols_report import SymbolSnapshot


def persist_initial_portfolio_run(
    rows: Sequence[SymbolSnapshot],
    plan: Sequence[Any],
    *,
    env_file: Optional[os.PathLike[str] | str] = None,
    db_path: Optional[os.PathLike[str] | str] = None,
    created_at: Optional[str] = None,
    time_frame: Optional[int] = None,
    atr_period: Optional[int] = None,
    tpv_leverage: Optional[float] = None,
    risk_pct: Optional[float] = None,
    equity: Optional[float] = None,
    tpv: Optional[float] = None,
    weights_title: Optional[str] = None,
    mt5_order_comment: Optional[str] = None,
) -> tuple[int, Path]:
    """Persist snapshots + planned positions as one run.

    Returns: (run_id, db_path)
    """

    resolved_db_path = Path(db_path) if db_path is not None else load_sqlite_db_path(env_file)
    initialize_db(resolved_db_path)

    created_at_value = (
        created_at
        if created_at is not None
        else datetime.now(timezone.utc).isoformat(timespec="seconds")
    )
    env_file_value = str(env_file) if env_file is not None else None

    conn = connect(resolved_db_path)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO portfolio_runs (
                created_at,
                env_file,
                time_frame,
                atr_period,
                tpv_leverage,
                risk_pct,
                equity,
                tpv,
                weights_title,
                mt5_order_comment
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                created_at_value,
                env_file_value,
                int(time_frame) if time_frame is not None else None,
                int(atr_period) if atr_period is not None else None,
                float(tpv_leverage) if tpv_leverage is not None else None,
                float(risk_pct) if risk_pct is not None else None,
                float(equity) if equity is not None else None,
                float(tpv) if tpv is not None else None,
                str(weights_title) if weights_title is not None else None,
                str(mt5_order_comment) if mt5_order_comment is not None else None,
            ),
        )
        run_id = int(cur.lastrowid)

        snapshot_rows = [
            (
                run_id,
                r.symbol,
                1 if bool(r.exists) else 0,
                r.sell,
                r.atr,
                r.atr_pct,
                r.contract_size,
                r.currency_profit,
                r.usd_pair_symbol,
                r.usd_pair_sell,
                r.usd_per_lot,
                r.volume_min,
                r.volume_step,
                r.volume_max,
                r.lot_for_base,
                r.usd_for_base_lot,
                r.usd_diff_to_base,
                r.error,
                r.usd_pair_error,
                r.atr_error,
            )
            for r in rows
        ]
        cur.executemany(
            """
            INSERT INTO symbol_snapshots (
                run_id,
                symbol,
                exists_flag,
                sell,
                atr,
                atr_pct,
                contract_size,
                currency_profit,
                usd_pair_symbol,
                usd_pair_sell,
                usd_per_lot,
                volume_min,
                volume_step,
                volume_max,
                lot_for_base,
                usd_for_base_lot,
                usd_diff_to_base,
                error,
                usd_pair_error,
                atr_error
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            snapshot_rows,
        )

        plan_rows = [
            (
                run_id,
                str(getattr(p, "symbol")),
                str(getattr(p, "direction", "")),
                float(getattr(p, "lot")),
                getattr(p, "atr_pct", None),
                getattr(p, "weight", None),
                getattr(p, "risk_usd", None),
                getattr(p, "risk_per_lot_at_1atr", None),
                getattr(p, "ideal_lot", None),
                getattr(p, "volume_min", None),
                getattr(p, "volume_step", None),
                getattr(p, "volume_max", None),
                getattr(p, "usd_per_lot", None),
                getattr(p, "usd_nominal", None),
                getattr(p, "reason", None),
            )
            for p in plan
        ]
        cur.executemany(
            """
            INSERT INTO planned_positions (
                run_id,
                symbol,
                direction,
                lot,
                atr_pct,
                weight,
                risk_usd,
                risk_per_lot_at_1atr,
                ideal_lot,
                volume_min,
                volume_step,
                volume_max,
                usd_per_lot,
                usd_nominal,
                reason
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            plan_rows,
        )

        conn.commit()
        return run_id, resolved_db_path
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def read_latest_run_id(db_path: os.PathLike[str] | str) -> Optional[int]:
    """Utility for debugging/tests."""
    conn = connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute("SELECT id FROM portfolio_runs ORDER BY id DESC LIMIT 1")
        row = cur.fetchone()
        if not row:
            return None
        return int(row[0])
    finally:
        conn.close()


def read_portfolio_run(db_path: os.PathLike[str] | str, run_id: int) -> Optional[dict[str, Any]]:
    conn = connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, created_at, time_frame, atr_period, tpv_leverage, risk_pct, equity, tpv,
                   weights_title, mt5_order_comment
            FROM portfolio_runs
            WHERE id=?
            """,
            (int(run_id),),
        )
        row = cur.fetchone()
        if not row:
            return None
        return {
            "id": int(row[0]),
            "created_at": str(row[1]),
            "time_frame": int(row[2]) if row[2] is not None else None,
            "atr_period": int(row[3]) if row[3] is not None else None,
            "tpv_leverage": float(row[4]) if row[4] is not None else None,
            "risk_pct": float(row[5]) if row[5] is not None else None,
            "equity": float(row[6]) if row[6] is not None else None,
            "tpv": float(row[7]) if row[7] is not None else None,
            "weights_title": str(row[8]) if row[8] is not None else None,
            "mt5_order_comment": str(row[9]) if row[9] is not None else None,
        }
    finally:
        conn.close()


def read_latest_portfolio_run(db_path: os.PathLike[str] | str) -> Optional[dict[str, Any]]:
    latest = read_latest_run_id(db_path)
    if latest is None:
        return None
    return read_portfolio_run(db_path, latest)


def read_planned_positions_for_run(
    db_path: os.PathLike[str] | str, run_id: int
) -> dict[str, dict[str, Any]]:
    conn = connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT
                symbol,
                direction,
                lot,
                atr_pct,
                weight,
                risk_usd,
                risk_per_lot_at_1atr,
                ideal_lot,
                volume_min,
                volume_step,
                volume_max,
                usd_per_lot,
                usd_nominal,
                reason
            FROM planned_positions
            WHERE run_id=?
            """,
            (int(run_id),),
        )
        rows = cur.fetchall()
        out: dict[str, dict[str, Any]] = {}
        for r in rows:
            sym = str(r[0])
            out[sym] = {
                "symbol": sym,
                "direction": str(r[1]),
                "lot": float(r[2]),
                "atr_pct": float(r[3]) if r[3] is not None else None,
                "weight": float(r[4]) if r[4] is not None else None,
                "risk_usd": float(r[5]) if r[5] is not None else None,
                "risk_per_lot_at_1atr": float(r[6]) if r[6] is not None else None,
                "ideal_lot": float(r[7]) if r[7] is not None else None,
                "volume_min": float(r[8]) if r[8] is not None else None,
                "volume_step": float(r[9]) if r[9] is not None else None,
                "volume_max": float(r[10]) if r[10] is not None else None,
                "usd_per_lot": float(r[11]) if r[11] is not None else None,
                "usd_nominal": float(r[12]) if r[12] is not None else None,
                "reason": str(r[13]) if r[13] is not None else None,
            }
        return out
    finally:
        conn.close()
