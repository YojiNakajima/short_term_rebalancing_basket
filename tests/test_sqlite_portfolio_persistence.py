import sqlite3
import tempfile
import unittest
from pathlib import Path

from infrastructure.dB.sqlite3.initialize_db import initialize_db
from infrastructure.dB.sqlite3.portfolio import persist_initial_portfolio_run
from infrastructure.mt5.build_initial_portfolio import PlannedPosition
from infrastructure.mt5.symbols_report import SymbolSnapshot


class TestSqlitePortfolioPersistence(unittest.TestCase):
    def test_initialize_and_persist_run(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            db_path = Path(d) / "portfolio.sqlite3"

            initialize_db(db_path)

            conn = sqlite3.connect(str(db_path))
            try:
                cur = conn.cursor()
                cur.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name IN ('global_settings', 'daily_equities') ORDER BY name"
                )
                self.assertEqual(cur.fetchall(), [("daily_equities",), ("global_settings",)])

                cur.execute("SELECT id, global_running FROM global_settings")
                self.assertEqual(cur.fetchall(), [(1, 1)])
            finally:
                conn.close()

            rows = [
                SymbolSnapshot(
                    symbol="XAUUSD",
                    exists=True,
                    sell=2000.0,
                    atr=20.0,
                    atr_pct=1.0,
                    usd_per_lot=2000.0,
                    volume_min=0.01,
                    volume_step=0.01,
                    volume_max=10.0,
                    lot_for_base=0.01,
                    usd_for_base_lot=2000.0,
                    usd_diff_to_base=0.0,
                ),
                SymbolSnapshot(
                    symbol="USOIL",
                    exists=True,
                    sell=75.0,
                    atr=1.5,
                    atr_pct=2.0,
                    usd_per_lot=75000.0,
                    volume_min=0.01,
                    volume_step=0.01,
                    volume_max=100.0,
                    lot_for_base=0.01,
                    usd_for_base_lot=750.0,
                    usd_diff_to_base=-1250.0,
                ),
            ]

            plan = [
                PlannedPosition(
                    symbol="XAUUSD",
                    atr_pct=1.0,
                    weight=0.6,
                    risk_usd=600.0,
                    risk_per_lot_at_1atr=20.0,
                    ideal_lot=0.3,
                    lot=0.3,
                    direction="BUY",
                    volume_min=0.01,
                    volume_step=0.01,
                    volume_max=10.0,
                    usd_per_lot=2000.0,
                    usd_nominal=600.0,
                    reason=None,
                ),
                PlannedPosition(
                    symbol="USOIL",
                    atr_pct=2.0,
                    weight=0.4,
                    risk_usd=400.0,
                    risk_per_lot_at_1atr=1500.0,
                    ideal_lot=0.05,
                    lot=0.05,
                    direction="BUY",
                    volume_min=0.01,
                    volume_step=0.01,
                    volume_max=100.0,
                    usd_per_lot=75000.0,
                    usd_nominal=3750.0,
                    reason=None,
                ),
            ]

            run_id, saved_path = persist_initial_portfolio_run(
                rows,
                plan,
                db_path=db_path,
                created_at="2026-02-16T00:00:00+00:00",
                time_frame=4,
                atr_period=14,
                tpv_leverage=5.0,
                risk_pct=0.01,
                equity=10000.0,
                tpv=50000.0,
                weights_title="Inverse-Vol Weights",
                mt5_order_comment="initial_portfolio",
            )

            self.assertEqual(saved_path, db_path)
            self.assertGreater(run_id, 0)

            conn = sqlite3.connect(str(db_path))
            try:
                cur = conn.cursor()
                cur.execute("SELECT COUNT(*) FROM portfolio_runs")
                self.assertEqual(cur.fetchone()[0], 1)
                cur.execute("SELECT COUNT(*) FROM symbol_snapshots")
                self.assertEqual(cur.fetchone()[0], 2)
                cur.execute("SELECT COUNT(*) FROM planned_positions")
                self.assertEqual(cur.fetchone()[0], 2)

                cur.execute(
                    "SELECT symbol, lot FROM planned_positions WHERE run_id=? ORDER BY symbol",
                    (run_id,),
                )
                self.assertEqual(cur.fetchall(), [("USOIL", 0.05), ("XAUUSD", 0.3)])
            finally:
                conn.close()


if __name__ == "__main__":
    unittest.main()
