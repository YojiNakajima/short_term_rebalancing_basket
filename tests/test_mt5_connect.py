import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import Mock

from infrastructure.mt5.connect import (
    Mt5Config,
    Mt5ConnectionError,
    connect,
    connect_and_print_account_info,
    load_mt5_config,
)


class TestMt5Connect(unittest.TestCase):
    def test_load_mt5_config_does_not_interpolate_dollar_password(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            env_path = Path(d) / ".env"
            env_path.write_text(
                "\n".join(
                    [
                        "MT5_PATH=C:/mt5/terminal64.exe",
                        "MT5_LOGIN=123",
                        "MT5_PASSWORD=$U480atakkin",
                        "MT5_SERVER=Some-Server",
                        "MT5_PORTABLE=false",
                    ]
                ),
                encoding="utf-8",
            )

            cfg = load_mt5_config(env_path)
            self.assertEqual(cfg.password, "$U480atakkin")

    def test_connect_calls_initialize_with_expected_args(self) -> None:
        module = Mock()
        module.initialize.return_value = True
        module.account_info.return_value = {"login": 123}

        cfg = Mt5Config(
            path="C:/mt5/terminal64.exe",
            login=123,
            password="pass",
            server="Some-Server",
            portable=False,
        )

        info = connect(cfg, module=module)
        self.assertEqual(info, {"login": 123})
        module.initialize.assert_called_once_with(
            path=cfg.path,
            login=cfg.login,
            password=cfg.password,
            server=cfg.server,
            portable=cfg.portable,
        )
        module.account_info.assert_called_once_with()

    def test_connect_raises_on_initialize_failure(self) -> None:
        module = Mock()
        module.initialize.return_value = False
        module.last_error.return_value = (1, "init failed")

        cfg = Mt5Config(
            path="C:/mt5/terminal64.exe",
            login=123,
            password="pass",
            server="Some-Server",
            portable=False,
        )

        with self.assertRaises(Mt5ConnectionError) as ctx:
            connect(cfg, module=module)

        self.assertIn("init failed", str(ctx.exception))

    def test_connect_and_print_account_info_shuts_down(self) -> None:
        module = Mock()
        module.initialize.return_value = True
        module.account_info.return_value = {"login": 123}
        module.shutdown.return_value = True

        with tempfile.TemporaryDirectory() as d:
            env_path = Path(d) / ".env"
            env_path.write_text(
                "\n".join(
                    [
                        "MT5_PATH=C:/mt5/terminal64.exe",
                        "MT5_LOGIN=123",
                        "MT5_PASSWORD=pass",
                        "MT5_SERVER=Some-Server",
                        "MT5_PORTABLE=false",
                    ]
                ),
                encoding="utf-8",
            )

            buf = io.StringIO()
            with redirect_stdout(buf):
                connect_and_print_account_info(env_path, module=module)

        out = buf.getvalue()
        self.assertIn("MT5 account_info:", out)
        module.shutdown.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
