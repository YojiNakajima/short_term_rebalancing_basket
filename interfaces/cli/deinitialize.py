from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Optional

from infrastructure.dB.sqlite3.global_settings import set_global_running
from infrastructure.dB.sqlite3.initialize_db import initialize_db, load_sqlite_db_path


def deinitialize_once(
    *,
    env_file: Optional[os.PathLike[str] | str] = None,
    db_path: Optional[os.PathLike[str] | str] = None,
    module: Any = None,
) -> dict[str, Any]:
    """Deactivate global running flag.

    `module` is reserved for future extensions / symmetry with initialize.py.
    """

    resolved_db_path = Path(db_path) if db_path is not None else load_sqlite_db_path(env_file)
    initialize_db(resolved_db_path)

    set_global_running(False, env_file=env_file, db_path=resolved_db_path)

    return {
        "db_path": str(resolved_db_path),
        "global_running": False,
    }


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Deinitialize system state (global_running=false)")
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
    args = parser.parse_args(argv)

    out = deinitialize_once(env_file=args.env, db_path=args.db)
    print("[OK] deinitialize completed")
    print(f"db_path={out['db_path']}")
    print("global_running=False")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())