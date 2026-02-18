from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from infrastructure.dB.sqlite3.initialize_db import connect, initialize_db, load_sqlite_db_path


def upsert_daily_pv_tpv(
    *,
    ymd: str,
    equity: float,
    pv: float,
    tpv: float,
    pv_leverage: float,
    leverage: float,
    tpv_leverage: float,
    fetched_at: str,
    env_file: Optional[os.PathLike[str] | str] = None,
    db_path: Optional[os.PathLike[str] | str] = None,
) -> Path:
    resolved_db_path = Path(db_path) if db_path is not None else load_sqlite_db_path(env_file)
    initialize_db(resolved_db_path)

    conn = connect(resolved_db_path)
    try:
        conn.execute(
            """
            INSERT INTO daily_pv_tpv (
                ymd,
                equity,
                pv,
                tpv,
                pv_leverage,
                leverage,
                tpv_leverage,
                fetched_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(ymd)
            DO UPDATE SET
                equity=excluded.equity,
                pv=excluded.pv,
                tpv=excluded.tpv,
                pv_leverage=excluded.pv_leverage,
                leverage=excluded.leverage,
                tpv_leverage=excluded.tpv_leverage,
                fetched_at=excluded.fetched_at
            """,
            (
                str(ymd),
                float(equity),
                float(pv),
                float(tpv),
                float(pv_leverage),
                float(leverage),
                float(tpv_leverage),
                str(fetched_at),
            ),
        )
        conn.commit()
        return resolved_db_path
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def upsert_daily_symbol_price(
    *,
    ymd: str,
    symbol: str,
    sell: Optional[float],
    fetched_at: str,
    error: Optional[str] = None,
    env_file: Optional[os.PathLike[str] | str] = None,
    db_path: Optional[os.PathLike[str] | str] = None,
) -> Path:
    resolved_db_path = Path(db_path) if db_path is not None else load_sqlite_db_path(env_file)
    initialize_db(resolved_db_path)

    conn = connect(resolved_db_path)
    try:
        conn.execute(
            """
            INSERT INTO daily_symbol_price (ymd, symbol, sell, fetched_at, error)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(ymd, symbol)
            DO UPDATE SET
                sell=excluded.sell,
                fetched_at=excluded.fetched_at,
                error=excluded.error
            """,
            (
                str(ymd),
                str(symbol),
                None if sell is None else float(sell),
                str(fetched_at),
                None if error is None else str(error),
            ),
        )
        conn.commit()
        return resolved_db_path
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def upsert_daily_atr(
    *,
    ymd: str,
    symbol: str,
    atr: Optional[float],
    atr_pct: Optional[float],
    time_frame: Optional[int],
    atr_period: Optional[int],
    fetched_at: str,
    error: Optional[str] = None,
    env_file: Optional[os.PathLike[str] | str] = None,
    db_path: Optional[os.PathLike[str] | str] = None,
) -> Path:
    resolved_db_path = Path(db_path) if db_path is not None else load_sqlite_db_path(env_file)
    initialize_db(resolved_db_path)

    conn = connect(resolved_db_path)
    try:
        conn.execute(
            """
            INSERT INTO daily_atr (
                ymd,
                symbol,
                atr,
                atr_pct,
                time_frame,
                atr_period,
                fetched_at,
                error
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(ymd, symbol)
            DO UPDATE SET
                atr=excluded.atr,
                atr_pct=excluded.atr_pct,
                time_frame=excluded.time_frame,
                atr_period=excluded.atr_period,
                fetched_at=excluded.fetched_at,
                error=excluded.error
            """,
            (
                str(ymd),
                str(symbol),
                None if atr is None else float(atr),
                None if atr_pct is None else float(atr_pct),
                None if time_frame is None else int(time_frame),
                None if atr_period is None else int(atr_period),
                str(fetched_at),
                None if error is None else str(error),
            ),
        )
        conn.commit()
        return resolved_db_path
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def read_daily_pv_tpv(
    *,
    ymd: str,
    env_file: Optional[os.PathLike[str] | str] = None,
    db_path: Optional[os.PathLike[str] | str] = None,
) -> Optional[dict[str, float | str | None]]:
    """Read daily PV/TPV snapshot for the given date.

    Returns None if the row does not exist.
    """

    resolved_db_path = Path(db_path) if db_path is not None else load_sqlite_db_path(env_file)
    initialize_db(resolved_db_path)

    conn = connect(resolved_db_path)
    try:
        row = conn.execute(
            """
            SELECT ymd, equity, pv, tpv, pv_leverage, leverage, tpv_leverage, fetched_at
            FROM daily_pv_tpv
            WHERE ymd=?
            """,
            (str(ymd),),
        ).fetchone()
        if not row:
            return None
        return {
            "ymd": str(row[0]),
            "equity": float(row[1]),
            "pv": float(row[2]),
            "tpv": float(row[3]),
            "pv_leverage": float(row[4]) if row[4] is not None else None,
            "leverage": float(row[5]) if row[5] is not None else None,
            "tpv_leverage": float(row[6]) if row[6] is not None else None,
            "fetched_at": str(row[7]),
        }
    finally:
        conn.close()


def read_daily_symbol_price(
    *,
    ymd: str,
    symbol: str,
    env_file: Optional[os.PathLike[str] | str] = None,
    db_path: Optional[os.PathLike[str] | str] = None,
) -> Optional[dict[str, float | str | None]]:
    """Read daily symbol sell(bid) price snapshot.

    Returns None if the row does not exist.
    """

    resolved_db_path = Path(db_path) if db_path is not None else load_sqlite_db_path(env_file)
    initialize_db(resolved_db_path)

    conn = connect(resolved_db_path)
    try:
        row = conn.execute(
            """
            SELECT ymd, symbol, sell, fetched_at, error
            FROM daily_symbol_price
            WHERE ymd=? AND symbol=?
            """,
            (str(ymd), str(symbol)),
        ).fetchone()
        if not row:
            return None
        return {
            "ymd": str(row[0]),
            "symbol": str(row[1]),
            "sell": float(row[2]) if row[2] is not None else None,
            "fetched_at": str(row[3]),
            "error": str(row[4]) if row[4] is not None else None,
        }
    finally:
        conn.close()


def read_daily_atr(
    *,
    ymd: str,
    symbol: str,
    env_file: Optional[os.PathLike[str] | str] = None,
    db_path: Optional[os.PathLike[str] | str] = None,
) -> Optional[dict[str, float | str | None]]:
    """Read daily ATR/ATR% snapshot.

    Returns None if the row does not exist.
    """

    resolved_db_path = Path(db_path) if db_path is not None else load_sqlite_db_path(env_file)
    initialize_db(resolved_db_path)

    conn = connect(resolved_db_path)
    try:
        row = conn.execute(
            """
            SELECT ymd, symbol, atr, atr_pct, time_frame, atr_period, fetched_at, error
            FROM daily_atr
            WHERE ymd=? AND symbol=?
            """,
            (str(ymd), str(symbol)),
        ).fetchone()
        if not row:
            return None
        return {
            "ymd": str(row[0]),
            "symbol": str(row[1]),
            "atr": float(row[2]) if row[2] is not None else None,
            "atr_pct": float(row[3]) if row[3] is not None else None,
            "time_frame": int(row[4]) if row[4] is not None else None,
            "atr_period": int(row[5]) if row[5] is not None else None,
            "fetched_at": str(row[6]),
            "error": str(row[7]) if row[7] is not None else None,
        }
    finally:
        conn.close()
