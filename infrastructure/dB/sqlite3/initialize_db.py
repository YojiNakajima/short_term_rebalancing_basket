from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Optional

from dotenv import dotenv_values


def _default_env_path() -> Path:
    # project_root / .env
    return Path(__file__).resolve().parents[3] / ".env"


def default_db_path() -> Path:
    # project_root / data / portfolio.sqlite3
    return Path(__file__).resolve().parents[3] / "data" / "portfolio.sqlite3"


def load_sqlite_db_path(env_file: Optional[os.PathLike[str] | str] = None) -> Path:
    env_path = Path(env_file) if env_file is not None else _default_env_path()

    env_values: dict[str, Optional[str]] = {}
    if env_path.exists():
        # interpolate=False to avoid unexpected substitutions
        env_values = dotenv_values(str(env_path), interpolate=False)

    def get_value(key: str) -> Optional[str]:
        v = env_values.get(key)
        if v is not None:
            return v
        return os.getenv(key)

    raw = get_value("SQLITE_DB_PATH")
    if raw:
        return Path(raw)
    return default_db_path()


def connect(db_path: os.PathLike[str] | str) -> sqlite3.Connection:
    p = Path(db_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(p))
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def initialize_db(db_path: os.PathLike[str] | str) -> None:
    """Create required tables for portfolio build/rebalance runs.

    This function is idempotent.
    """

    conn = connect(db_path)
    try:
        conn.executescript(
            """
            -- --- Global settings (singleton row) ---
            CREATE TABLE IF NOT EXISTS global_settings (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                global_running INTEGER NOT NULL DEFAULT 1
            );

            -- --- Daily equities snapshot ---
            CREATE TABLE IF NOT EXISTS daily_equities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ymd TEXT NOT NULL,
                equity REAL NOT NULL,
                fetched_at TEXT NOT NULL,
                UNIQUE (ymd)
            );

            -- --- Daily PV/TPV snapshot ---
            CREATE TABLE IF NOT EXISTS daily_pv_tpv (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ymd TEXT NOT NULL,
                equity REAL NOT NULL,
                pv REAL NOT NULL,
                tpv REAL NOT NULL,
                pv_leverage REAL,
                leverage REAL,
                tpv_leverage REAL,
                fetched_at TEXT NOT NULL,
                UNIQUE (ymd)
            );

            -- --- Daily symbol prices (bid/sell) snapshot ---
            CREATE TABLE IF NOT EXISTS daily_symbol_price (
                ymd TEXT NOT NULL,
                symbol TEXT NOT NULL,
                sell REAL,
                fetched_at TEXT NOT NULL,
                error TEXT,
                PRIMARY KEY (ymd, symbol)
            );

            -- --- Daily ATR% snapshot ---
            CREATE TABLE IF NOT EXISTS daily_atr (
                ymd TEXT NOT NULL,
                symbol TEXT NOT NULL,
                atr REAL,
                atr_pct REAL,
                time_frame INTEGER,
                atr_period INTEGER,
                fetched_at TEXT NOT NULL,
                error TEXT,
                PRIMARY KEY (ymd, symbol)
            );

            CREATE TABLE IF NOT EXISTS portfolio_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                env_file TEXT,
                time_frame INTEGER,
                atr_period INTEGER,
                tpv_leverage REAL,
                risk_pct REAL,
                equity REAL,
                tpv REAL,
                weights_title TEXT,
                mt5_order_comment TEXT
            );

            CREATE TABLE IF NOT EXISTS symbol_snapshots (
                run_id INTEGER NOT NULL,
                symbol TEXT NOT NULL,
                exists_flag INTEGER NOT NULL,
                sell REAL,
                atr REAL,
                atr_pct REAL,
                contract_size REAL,
                currency_profit TEXT,
                usd_pair_symbol TEXT,
                usd_pair_sell REAL,
                usd_per_lot REAL,
                volume_min REAL,
                volume_step REAL,
                volume_max REAL,
                lot_for_base REAL,
                usd_for_base_lot REAL,
                usd_diff_to_base REAL,
                error TEXT,
                usd_pair_error TEXT,
                atr_error TEXT,
                PRIMARY KEY (run_id, symbol),
                FOREIGN KEY (run_id) REFERENCES portfolio_runs(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS planned_positions (
                run_id INTEGER NOT NULL,
                symbol TEXT NOT NULL,
                direction TEXT NOT NULL,
                lot REAL NOT NULL,
                atr_pct REAL,
                weight REAL,
                risk_usd REAL,
                risk_per_lot_at_1atr REAL,
                ideal_lot REAL,
                volume_min REAL,
                volume_step REAL,
                volume_max REAL,
                usd_per_lot REAL,
                usd_nominal REAL,
                reason TEXT,
                PRIMARY KEY (run_id, symbol),
                FOREIGN KEY (run_id) REFERENCES portfolio_runs(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_portfolio_runs_created_at
                ON portfolio_runs(created_at);
            CREATE INDEX IF NOT EXISTS idx_symbol_snapshots_symbol
                ON symbol_snapshots(symbol);
            CREATE INDEX IF NOT EXISTS idx_planned_positions_symbol
                ON planned_positions(symbol);

            CREATE INDEX IF NOT EXISTS idx_daily_equities_ymd
                ON daily_equities(ymd);

            CREATE INDEX IF NOT EXISTS idx_daily_pv_tpv_ymd
                ON daily_pv_tpv(ymd);
            CREATE INDEX IF NOT EXISTS idx_daily_symbol_price_ymd
                ON daily_symbol_price(ymd);
            CREATE INDEX IF NOT EXISTS idx_daily_symbol_price_symbol
                ON daily_symbol_price(symbol);
            CREATE INDEX IF NOT EXISTS idx_daily_atr_ymd
                ON daily_atr(ymd);
            CREATE INDEX IF NOT EXISTS idx_daily_atr_symbol
                ON daily_atr(symbol);

            -- --- Rebalance (state + history) ---
            CREATE TABLE IF NOT EXISTS rebalance_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                portfolio_run_id INTEGER NOT NULL,
                tpv_ref REAL,
                tpv_now REAL,
                tpv_return REAL,
                drift REAL,
                max_contrib REAL,
                triggered_flag INTEGER NOT NULL,
                dry_run_flag INTEGER NOT NULL,
                error TEXT,
                metrics_json TEXT,
                FOREIGN KEY (portfolio_run_id) REFERENCES portfolio_runs(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS rebalance_state (
                portfolio_run_id INTEGER NOT NULL,
                symbol TEXT NOT NULL,
                tp_stage INTEGER NOT NULL DEFAULT 0,
                hedged_volume REAL NOT NULL DEFAULT 0,
                last_peak_price REAL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (portfolio_run_id, symbol),
                FOREIGN KEY (portfolio_run_id) REFERENCES portfolio_runs(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS rebalance_actions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                rebalance_run_id INTEGER NOT NULL,
                symbol TEXT NOT NULL,
                action TEXT NOT NULL,
                direction TEXT NOT NULL,
                volume REAL NOT NULL,
                stage INTEGER,
                r REAL,
                price REAL,
                atr_pct_ref REAL,
                reason TEXT,
                comment TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (rebalance_run_id) REFERENCES rebalance_runs(id) ON DELETE CASCADE
            );

            -- --- Daily TP (partial take profit) & Re-entry state ---
            -- ymd-scoped state so that multiple rebalance runs in the same day can share TP/Re-entry progress.
            CREATE TABLE IF NOT EXISTS daily_tp_reentry_state (
                ymd TEXT NOT NULL,
                portfolio_run_id INTEGER NOT NULL,
                symbol TEXT NOT NULL,
                tp_stage INTEGER NOT NULL DEFAULT 0,
                tp_price_1 REAL,
                tp_price_2 REAL,
                tp_price_3 REAL,
                reentry_done_1 INTEGER NOT NULL DEFAULT 0,
                reentry_done_2 INTEGER NOT NULL DEFAULT 0,
                reentry_done_3 INTEGER NOT NULL DEFAULT 0,
                reentry_price_1 REAL,
                reentry_price_2 REAL,
                reentry_price_3 REAL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (ymd, portfolio_run_id, symbol),
                FOREIGN KEY (portfolio_run_id) REFERENCES portfolio_runs(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_rebalance_runs_created_at
                ON rebalance_runs(created_at);
            CREATE INDEX IF NOT EXISTS idx_rebalance_state_symbol
                ON rebalance_state(symbol);
            CREATE INDEX IF NOT EXISTS idx_rebalance_actions_symbol
                ON rebalance_actions(symbol);

            CREATE INDEX IF NOT EXISTS idx_daily_tp_reentry_state_ymd
                ON daily_tp_reentry_state(ymd);
            CREATE INDEX IF NOT EXISTS idx_daily_tp_reentry_state_symbol
                ON daily_tp_reentry_state(symbol);
            """
        )

        # Ensure singleton settings row exists with default global_running=True (1).
        conn.execute(
            "INSERT OR IGNORE INTO global_settings (id, global_running) VALUES (1, 1)"
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def main() -> None:
    db_path = load_sqlite_db_path()
    print(f"[INFO] sqlite db path: {db_path}")

    initialize_db(db_path)
    print("[OK] initialize_db completed")

    conn = connect(db_path)
    try:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        tables = [r[0] for r in rows]
        print(f"[INFO] tables ({len(tables)}): {', '.join(tables)}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
