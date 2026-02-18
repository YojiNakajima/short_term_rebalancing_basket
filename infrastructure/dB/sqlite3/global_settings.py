from __future__ import annotations

import os
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Optional

from infrastructure.dB.sqlite3.initialize_db import (
    connect,
    initialize_db,
    load_sqlite_db_path,
)


def read_global_running(
    *,
    env_file: Optional[os.PathLike[str] | str] = None,
    db_path: Optional[os.PathLike[str] | str] = None,
    default: bool = True,
) -> bool:
    """Read singleton global_running flag.

    If the row does not exist, it will be created with the default True value.
    """

    resolved_db_path = Path(db_path) if db_path is not None else load_sqlite_db_path(env_file)
    initialize_db(resolved_db_path)

    conn = connect(resolved_db_path)
    try:
        # Ensure singleton row exists.
        conn.execute(
            "INSERT OR IGNORE INTO global_settings (id, global_running) VALUES (1, 1)"
        )
        row = conn.execute(
            "SELECT global_running FROM global_settings WHERE id=1"
        ).fetchone()
        if not row:
            return bool(default)
        try:
            return bool(int(row[0]))
        except Exception:
            return bool(default)
    finally:
        conn.close()


def set_global_running(
    running: bool,
    *,
    env_file: Optional[os.PathLike[str] | str] = None,
    db_path: Optional[os.PathLike[str] | str] = None,
) -> Path:
    resolved_db_path = Path(db_path) if db_path is not None else load_sqlite_db_path(env_file)
    initialize_db(resolved_db_path)

    conn = connect(resolved_db_path)
    try:
        # Ensure singleton row exists, then update.
        conn.execute(
            "INSERT OR IGNORE INTO global_settings (id, global_running) VALUES (1, 1)"
        )
        conn.execute(
            "UPDATE global_settings SET global_running=? WHERE id=1",
            (1 if bool(running) else 0,),
        )
        conn.commit()
        return resolved_db_path
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def upsert_daily_equity(
    *,
    ymd: str,
    equity: float,
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
            INSERT INTO daily_equities (ymd, equity, fetched_at)
            VALUES (?, ?, ?)
            ON CONFLICT(ymd)
            DO UPDATE SET
                equity=excluded.equity,
                fetched_at=excluded.fetched_at
            """,
            (str(ymd), float(equity), str(fetched_at)),
        )
        conn.commit()
        return resolved_db_path
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def today_ymd_local() -> str:
    return date.today().isoformat()


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
