import sqlite3
import tempfile
import unittest
from pathlib import Path


class TestCliDeinitialize(unittest.TestCase):
    def test_deinitialize_sets_global_running_false(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            db_path = Path(d) / "portfolio.sqlite3"

            from interfaces.cli.deinitialize import deinitialize_once

            out = deinitialize_once(db_path=db_path)

            self.assertEqual(Path(out["db_path"]), db_path)
            self.assertFalse(out["global_running"])

            conn = sqlite3.connect(str(db_path))
            try:
                cur = conn.cursor()
                cur.execute("SELECT global_running FROM global_settings WHERE id=1")
                row = cur.fetchone()
                self.assertEqual(row, (0,))
            finally:
                conn.close()


if __name__ == "__main__":
    unittest.main()