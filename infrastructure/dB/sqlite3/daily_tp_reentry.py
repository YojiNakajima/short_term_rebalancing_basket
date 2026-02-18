from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

from infrastructure.dB.sqlite3.initialize_db import connect, initialize_db, load_sqlite_db_path


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass(frozen=True)
class DailyTpReentryState:
    ymd: str
    portfolio_run_id: int
    symbol: str
    tp_stage: int
    tp_price_1: Optional[float]
    tp_price_2: Optional[float]
    tp_price_3: Optional[float]
    reentry_done_1: bool
    reentry_done_2: bool
    reentry_done_3: bool
    reentry_price_1: Optional[float]
    reentry_price_2: Optional[float]
    reentry_price_3: Optional[float]
    updated_at: str


def ensure_daily_tp_reentry_state_rows(
    *,
    ymd: str,
    portfolio_run_id: int,
    symbols: Iterable[str],
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
            (
                str(ymd),
                int(portfolio_run_id),
                str(symbol),
                0,
                None,
                None,
                None,
                0,
                0,
                0,
                None,
                None,
                None,
                created_at,
            )
            for symbol in symbols
        ]
        cur.executemany(
            """
            INSERT OR IGNORE INTO daily_tp_reentry_state (
                ymd,
                portfolio_run_id,
                symbol,
                tp_stage,
                tp_price_1,
                tp_price_2,
                tp_price_3,
                reentry_done_1,
                reentry_done_2,
                reentry_done_3,
                reentry_price_1,
                reentry_price_2,
                reentry_price_3,
                updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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


def read_daily_tp_reentry_state_map(
    *,
    ymd: str,
    portfolio_run_id: int,
    env_file: Optional[os.PathLike[str] | str] = None,
    db_path: Optional[os.PathLike[str] | str] = None,
) -> tuple[dict[str, DailyTpReentryState], Path]:
    resolved_db_path = Path(db_path) if db_path is not None else load_sqlite_db_path(env_file)
    initialize_db(resolved_db_path)

    conn = connect(resolved_db_path)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT
                ymd,
                portfolio_run_id,
                symbol,
                tp_stage,
                tp_price_1,
                tp_price_2,
                tp_price_3,
                reentry_done_1,
                reentry_done_2,
                reentry_done_3,
                reentry_price_1,
                reentry_price_2,
                reentry_price_3,
                updated_at
            FROM daily_tp_reentry_state
            WHERE ymd=? AND portfolio_run_id=?
            """,
            (str(ymd), int(portfolio_run_id)),
        )
        rows = cur.fetchall()
        m: dict[str, DailyTpReentryState] = {}
        for r in rows:
            st = DailyTpReentryState(
                ymd=str(r[0]),
                portfolio_run_id=int(r[1]),
                symbol=str(r[2]),
                tp_stage=int(r[3] or 0),
                tp_price_1=float(r[4]) if r[4] is not None else None,
                tp_price_2=float(r[5]) if r[5] is not None else None,
                tp_price_3=float(r[6]) if r[6] is not None else None,
                reentry_done_1=bool(int(r[7] or 0)),
                reentry_done_2=bool(int(r[8] or 0)),
                reentry_done_3=bool(int(r[9] or 0)),
                reentry_price_1=float(r[10]) if r[10] is not None else None,
                reentry_price_2=float(r[11]) if r[11] is not None else None,
                reentry_price_3=float(r[12]) if r[12] is not None else None,
                updated_at=str(r[13]),
            )
            m[st.symbol] = st
        return m, resolved_db_path
    finally:
        conn.close()


def upsert_daily_tp_reentry_state(
    state: DailyTpReentryState,
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
            INSERT INTO daily_tp_reentry_state (
                ymd,
                portfolio_run_id,
                symbol,
                tp_stage,
                tp_price_1,
                tp_price_2,
                tp_price_3,
                reentry_done_1,
                reentry_done_2,
                reentry_done_3,
                reentry_price_1,
                reentry_price_2,
                reentry_price_3,
                updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(ymd, portfolio_run_id, symbol)
            DO UPDATE SET
                tp_stage=excluded.tp_stage,
                tp_price_1=excluded.tp_price_1,
                tp_price_2=excluded.tp_price_2,
                tp_price_3=excluded.tp_price_3,
                reentry_done_1=excluded.reentry_done_1,
                reentry_done_2=excluded.reentry_done_2,
                reentry_done_3=excluded.reentry_done_3,
                reentry_price_1=excluded.reentry_price_1,
                reentry_price_2=excluded.reentry_price_2,
                reentry_price_3=excluded.reentry_price_3,
                updated_at=excluded.updated_at
            """,
            (
                str(state.ymd),
                int(state.portfolio_run_id),
                str(state.symbol),
                int(state.tp_stage),
                float(state.tp_price_1) if state.tp_price_1 is not None else None,
                float(state.tp_price_2) if state.tp_price_2 is not None else None,
                float(state.tp_price_3) if state.tp_price_3 is not None else None,
                1 if bool(state.reentry_done_1) else 0,
                1 if bool(state.reentry_done_2) else 0,
                1 if bool(state.reentry_done_3) else 0,
                float(state.reentry_price_1) if state.reentry_price_1 is not None else None,
                float(state.reentry_price_2) if state.reentry_price_2 is not None else None,
                float(state.reentry_price_3) if state.reentry_price_3 is not None else None,
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
