from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from dotenv import dotenv_values

try:
    import MetaTrader5 as mt5
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "MetaTrader5 package is required. Install dependencies from requirements.txt"
    ) from e


class Mt5ConnectionError(RuntimeError):
    pass


@dataclass(frozen=True)
class Mt5Config:
    path: str
    login: int
    password: str
    server: str
    portable: bool = False


def _parse_bool(value: Optional[str], default: bool = False) -> bool:
    if value is None:
        return default

    v = value.strip().lower()
    if v in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if v in {"0", "false", "f", "no", "n", "off"}:
        return False
    return default


def _default_env_path() -> Path:
    # project_root / .env
    return Path(__file__).resolve().parents[2] / ".env"


def load_mt5_config(env_file: Optional[os.PathLike[str] | str] = None) -> Mt5Config:
    env_path = Path(env_file) if env_file is not None else _default_env_path()

    env_values: dict[str, Optional[str]] = {}
    if env_path.exists():
        # NOTE: interpolate=False is important because MT5_PASSWORD may start with '$'
        # and dotenv interpolation could drop/alter the value.
        env_values = dotenv_values(str(env_path), interpolate=False)

    def get_value(key: str) -> Optional[str]:
        v = env_values.get(key)
        if v is not None:
            return v
        return os.getenv(key)

    path = get_value("MT5_PATH")
    login = get_value("MT5_LOGIN")
    password = get_value("MT5_PASSWORD")
    server = get_value("MT5_SERVER")
    portable = _parse_bool(get_value("MT5_PORTABLE"), default=False)

    missing = [
        k
        for k, v in {
            "MT5_PATH": path,
            "MT5_LOGIN": login,
            "MT5_PASSWORD": password,
            "MT5_SERVER": server,
        }.items()
        if not v
    ]
    if missing:
        raise ValueError(f"Missing required MT5 config values: {', '.join(missing)}")

    try:
        login_int = int(str(login))
    except ValueError as e:
        raise ValueError("MT5_LOGIN must be an integer") from e

    return Mt5Config(
        path=str(path),
        login=login_int,
        password=str(password),
        server=str(server),
        portable=portable,
    )


def connect(config: Mt5Config, *, module: Any = mt5) -> Any:
    initialized = module.initialize(
        path=config.path,
        login=config.login,
        password=config.password,
        server=config.server,
        portable=config.portable,
    )
    if not initialized:
        err = None
        try:
            err = module.last_error()
        except Exception:
            err = None
        raise Mt5ConnectionError(f"MT5 initialize failed. last_error={err}")

    info = module.account_info()
    if info is None:
        err = None
        try:
            err = module.last_error()
        except Exception:
            err = None
        raise Mt5ConnectionError(f"MT5 account_info() returned None. last_error={err}")

    return info


def connect_and_print_account_info(
    env_file: Optional[os.PathLike[str] | str] = None, *, module: Any = mt5
) -> None:
    config = load_mt5_config(env_file)
    initialized = False
    try:
        initialized = module.initialize(
            path=config.path,
            login=config.login,
            password=config.password,
            server=config.server,
            portable=config.portable,
        )
        if not initialized:
            raise Mt5ConnectionError(f"MT5 initialize failed. last_error={module.last_error()}")

        info = module.account_info()
        if info is None:
            raise Mt5ConnectionError(
                f"MT5 account_info() returned None. last_error={module.last_error()}"
            )

        print("MT5 account_info:")
        print(info)
    finally:
        if initialized:
            module.shutdown()


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Connect to MT5 and print account info")
    parser.add_argument(
        "--env",
        default=None,
        help="Path to .env file (default: project root .env)",
    )
    args = parser.parse_args(argv)

    connect_and_print_account_info(args.env)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
