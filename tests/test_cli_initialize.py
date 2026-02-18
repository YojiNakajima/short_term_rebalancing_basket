import sqlite3
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


class TestCliInitialize(unittest.TestCase):
    def test_initialize_sets_global_running_and_persists_daily_equity(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            db_path = Path(d) / "portfolio.sqlite3"

            fake_settings = SimpleNamespace(
                PV_LEVERAGE=5,
                TPV_LEVERAGE=1.3,
                LEVERAGE=1.0,
                ATR_PERIOD=14,
                TIME_FRAME=4,
                target_symbols={},
            )

            class FakeMt5:
                def __init__(self):
                    self.shutdown_called = False

                def initialize(self, **kwargs):
                    return True

                def shutdown(self):
                    self.shutdown_called = True

                def account_info(self):
                    return SimpleNamespace(equity=12345.67)

            fake = FakeMt5()

            with patch(
                "interfaces.cli.initialize.load_mt5_config",
                return_value=SimpleNamespace(
                    path="p",
                    login=1,
                    password="x",
                    server="s",
                    portable=False,
                ),
            ):
                from interfaces.cli.initialize import initialize_once

                out = initialize_once(
                    env_file="dummy.env",
                    db_path=db_path,
                    module=fake,
                    settings_module=fake_settings,
                    slack_sender=None,
                )

            self.assertTrue(fake.shutdown_called)
            self.assertEqual(Path(out["db_path"]), db_path)
            self.assertAlmostEqual(out["equity"], 12345.67, places=2)
            self.assertIn("fetched_at", out)

            conn = sqlite3.connect(str(db_path))
            try:
                cur = conn.cursor()
                cur.execute("SELECT global_running FROM global_settings WHERE id=1")
                self.assertEqual(cur.fetchone()[0], 1)

                cur.execute(
                    "SELECT ymd, equity, fetched_at FROM daily_equities WHERE ymd=?",
                    (out["ymd"],),
                )
                row = cur.fetchone()
                self.assertIsNotNone(row)
                self.assertEqual(row[0], out["ymd"])
                self.assertAlmostEqual(float(row[1]), 12345.67, places=2)
                self.assertEqual(row[2], out["fetched_at"])

                cur.execute(
                    "SELECT ymd, equity, pv, tpv, fetched_at FROM daily_pv_tpv WHERE ymd=?",
                    (out["ymd"],),
                )
                row = cur.fetchone()
                self.assertIsNotNone(row)
                self.assertEqual(row[0], out["ymd"])
                self.assertAlmostEqual(float(row[1]), 12345.67, places=2)
                self.assertAlmostEqual(float(row[2]), 12345.67 * 5.0, places=2)
                self.assertAlmostEqual(float(row[3]), 12345.67 * 5.0 * 1.0 * 1.3, places=2)
                self.assertEqual(row[4], out["fetched_at"])
            finally:
                conn.close()


if __name__ == "__main__":
    unittest.main()
