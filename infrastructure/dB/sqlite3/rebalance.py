from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

from infrastructure.dB.sqlite3.initialize_db import (
    connect,
    initialize_db,
    load_sqlite_db_path,
)


@dataclass(frozen=True)
class RebalanceState:
    portfolio_run_id: int
    symbol: str
    tp_stage: int
    hedged_volume: float
    last_peak_price: Optional[float]
    updated_at: str


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def ensure_rebalance_state_rows(
    portfolio_run_id: int,
    symbols: Iterable[str],
    *,
    env_file: Optional[os.PathLike[str] | str] = None,
    db_path: Optional[os.PathLike[str] | str] = None,
) -> Path:
    resolved_db_path = Path(db_path) if db_path is not None else load_sqlite_db_path(env_file)
    initialize_db(resolved_db_path)

    created_at = _now_utc_iso()
    conn = connect(resolved_db_path)
    try:
        cur = conn.cursor()
        rows = [
            (int(portfolio_run_id), str(symbol), 0, 0.0, None, created_at)
            for symbol in symbols
        ]
        cur.executemany(
            """
            INSERT OR IGNORE INTO rebalance_state (
                portfolio_run_id,
                symbol,
                tp_stage,
                hedged_volume,
                last_peak_price,
                updated_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        conn.commit()
        return resolved_db_path
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def read_rebalance_state_map(
    portfolio_run_id: int,
    *,
    env_file: Optional[os.PathLike[str] | str] = None,
    db_path: Optional[os.PathLike[str] | str] = None,
) -> tuple[dict[str, RebalanceState], Path]:
    resolved_db_path = Path(db_path) if db_path is not None else load_sqlite_db_path(env_file)
    initialize_db(resolved_db_path)

    conn = connect(resolved_db_path)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT portfolio_run_id, symbol, tp_stage, hedged_volume, last_peak_price, updated_at
            FROM rebalance_state
            WHERE portfolio_run_id=?
            """,
            (int(portfolio_run_id),),
        )
        rows = cur.fetchall()
        m: dict[str, RebalanceState] = {}
        for r in rows:
            st = RebalanceState(
                portfolio_run_id=int(r[0]),
                symbol=str(r[1]),
                tp_stage=int(r[2] or 0),
                hedged_volume=float(r[3] or 0.0),
                last_peak_price=float(r[4]) if r[4] is not None else None,
                updated_at=str(r[5]),
            )
            m[st.symbol] = st
        return m, resolved_db_path
    finally:
        conn.close()


def upsert_rebalance_state(
    state: RebalanceState,
    *,
    env_file: Optional[os.PathLike[str] | str] = None,
    db_path: Optional[os.PathLike[str] | str] = None,
) -> Path:
    resolved_db_path = Path(db_path) if db_path is not None else load_sqlite_db_path(env_file)
    initialize_db(resolved_db_path)

    conn = connect(resolved_db_path)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO rebalance_state (
                portfolio_run_id,
                symbol,
                tp_stage,
                hedged_volume,
                last_peak_price,
                updated_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(portfolio_run_id, symbol)
            DO UPDATE SET
                tp_stage=excluded.tp_stage,
                hedged_volume=excluded.hedged_volume,
                last_peak_price=excluded.last_peak_price,
                updated_at=excluded.updated_at
            """,
            (
                int(state.portfolio_run_id),
                str(state.symbol),
                int(state.tp_stage),
                float(state.hedged_volume),
                float(state.last_peak_price) if state.last_peak_price is not None else None,
                str(state.updated_at),
            ),
        )
        conn.commit()
        return resolved_db_path
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def create_rebalance_run(
    *,
    portfolio_run_id: int,
    tpv_ref: Optional[float],
    tpv_now: Optional[float],
    tpv_return: Optional[float],
    drift: Optional[float],
    max_contrib: Optional[float],
    triggered: bool,
    dry_run: bool,
    error: Optional[str] = None,
    metrics: Optional[dict[str, Any]] = None,
    created_at: Optional[str] = None,
    env_file: Optional[os.PathLike[str] | str] = None,
    db_path: Optional[os.PathLike[str] | str] = None,
) -> tuple[int, Path]:
    resolved_db_path = Path(db_path) if db_path is not None else load_sqlite_db_path(env_file)
    initialize_db(resolved_db_path)

    created_at_value = created_at if created_at is not None else _now_utc_iso()
    metrics_json = json.dumps(metrics or {}, ensure_ascii=False)

    conn = connect(resolved_db_path)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO rebalance_runs (
                created_at,
                portfolio_run_id,
                tpv_ref,
                tpv_now,
                tpv_return,
                drift,
                max_contrib,
                triggered_flag,
                dry_run_flag,
                error,
                metrics_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(created_at_value),
                int(portfolio_run_id),
                float(tpv_ref) if tpv_ref is not None else None,
                float(tpv_now) if tpv_now is not None else None,
                float(tpv_return) if tpv_return is not None else None,
                float(drift) if drift is not None else None,
                float(max_contrib) if max_contrib is not None else None,
                1 if bool(triggered) else 0,
                1 if bool(dry_run) else 0,
                str(error) if error is not None else None,
                str(metrics_json),
            ),
        )
        run_id = int(cur.lastrowid)
        conn.commit()
        return run_id, resolved_db_path
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def persist_rebalance_actions(
    rebalance_run_id: int,
    actions: Iterable[dict[str, Any]],
    *,
    created_at: Optional[str] = None,
    env_file: Optional[os.PathLike[str] | str] = None,
    db_path: Optional[os.PathLike[str] | str] = None,
) -> Path:
    resolved_db_path = Path(db_path) if db_path is not None else load_sqlite_db_path(env_file)
    initialize_db(resolved_db_path)

    created_at_value = created_at if created_at is not None else _now_utc_iso()

    conn = connect(resolved_db_path)
    try:
        cur = conn.cursor()
        rows = []
        for a in actions:
            rows.append(
                (
                    int(rebalance_run_id),
                    str(a.get("symbol")),
                    str(a.get("action")),
                    str(a.get("direction")),
                    float(a.get("volume")),
                    int(a.get("stage")) if a.get("stage") is not None else None,
                    float(a.get("r")) if a.get("r") is not None else None,
                    float(a.get("price")) if a.get("price") is not None else None,
                    float(a.get("atr_pct_ref")) if a.get("atr_pct_ref") is not None else None,
                    str(a.get("reason")) if a.get("reason") is not None else None,
                    str(a.get("comment")) if a.get("comment") is not None else None,
                    str(created_at_value),
                )
            )

        cur.executemany(
            """
            INSERT INTO rebalance_actions (
                rebalance_run_id,
                symbol,
                action,
                direction,
                volume,
                stage,
                r,
                price,
                atr_pct_ref,
                reason,
                comment,
                created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        conn.commit()
        return resolved_db_path
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
