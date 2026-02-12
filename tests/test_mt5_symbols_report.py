import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from infrastructure.mt5.symbols_report import (
    collect_target_symbol_snapshots,
    fetch_symbol_snapshot,
    get_enabled_target_symbols,
)


class TestMt5SymbolsReport(unittest.TestCase):
    def test_get_enabled_target_symbols_uses_only_explicit_true(self) -> None:
        settings_module = SimpleNamespace(
            target_symbols={"AAA": True, "BBB": False, "CCC": 1, "DDD": "true"}
        )
        self.assertEqual(get_enabled_target_symbols(settings_module=settings_module), ["AAA"])

    def test_fetch_symbol_snapshot_marks_missing_symbol(self) -> None:
        module = Mock()
        module.symbol_info.return_value = None
        module.last_error.return_value = (1, "not found")

        snap = fetch_symbol_snapshot("MISSING", module=module)
        self.assertFalse(snap.exists)
        self.assertIn("not found", snap.error or "")

    def test_fetch_symbol_snapshot_selects_when_not_visible(self) -> None:
        module = Mock()

        info1 = SimpleNamespace(
            visible=False,
            currency_profit="USD",
            volume_min=0.01,
            volume_step=0.01,
            trade_contract_size=100.0,
        )
        info2 = SimpleNamespace(
            visible=True,
            currency_profit="USD",
            volume_min=0.01,
            volume_step=0.01,
            trade_contract_size=100.0,
        )
        module.symbol_info.side_effect = [info1, info2]
        module.symbol_select.return_value = True
        tick = SimpleNamespace(bid=1.0)
        module.symbol_info_tick.return_value = tick

        snap = fetch_symbol_snapshot("XAUUSD", module=module)
        self.assertTrue(snap.exists)
        self.assertEqual(snap.currency_profit, "USD")
        self.assertEqual(snap.volume_min, 0.01)
        self.assertEqual(snap.volume_step, 0.01)
        self.assertEqual(snap.sell, 1.0)
        self.assertEqual(snap.contract_size, 100.0)
        self.assertEqual(snap.usd_per_lot, 100.0)
        self.assertIsNone(snap.usd_pair_symbol)
        self.assertIsNone(snap.usd_pair_sell)
        module.symbol_select.assert_called_once_with("XAUUSD", True)

    def test_fetch_symbol_snapshot_fetches_usd_pair_for_non_usd_profit_currency(self) -> None:
        module = Mock()

        module.symbol_info.side_effect = lambda sym: {
            "JP225": SimpleNamespace(
                visible=True,
                currency_profit="JPY",
                volume_min=0.1,
                volume_step=0.1,
                trade_contract_size=10.0,
            ),
            "USDJPY": SimpleNamespace(visible=True),
        }.get(sym)

        module.symbol_info_tick.side_effect = lambda sym: {
            "JP225": SimpleNamespace(bid=100.0),
            "USDJPY": SimpleNamespace(bid=150.0),
        }.get(sym)

        snap = fetch_symbol_snapshot("JP225", module=module)
        self.assertTrue(snap.exists)
        self.assertEqual(snap.currency_profit, "JPY")
        self.assertEqual(snap.sell, 100.0)
        self.assertEqual(snap.contract_size, 10.0)
        self.assertEqual(snap.usd_pair_symbol, "USDJPY")
        self.assertEqual(snap.usd_pair_sell, 150.0)
        self.assertIsNone(snap.usd_pair_error)
        self.assertAlmostEqual(snap.usd_per_lot or 0.0, (100.0 * 10.0) / 150.0)

    def test_fetch_symbol_snapshot_converts_non_usd_profit_currency_with_ccyusd_pair(self) -> None:
        module = Mock()

        module.symbol_info.side_effect = lambda sym: {
            "EURO": SimpleNamespace(
                visible=True,
                currency_profit="EUR",
                volume_min=1.0,
                volume_step=1.0,
                trade_contract_size=10.0,
            ),
            "EURUSD": SimpleNamespace(visible=True),
        }.get(sym)

        module.symbol_info_tick.side_effect = lambda sym: {
            "EURO": SimpleNamespace(bid=2.0),
            "EURUSD": SimpleNamespace(bid=1.1),
        }.get(sym)

        snap = fetch_symbol_snapshot("EURO", module=module)
        self.assertTrue(snap.exists)
        self.assertEqual(snap.currency_profit, "EUR")
        self.assertEqual(snap.usd_pair_symbol, "EURUSD")
        self.assertEqual(snap.usd_pair_sell, 1.1)
        self.assertAlmostEqual(snap.usd_per_lot or 0.0, 2.0 * 10.0 * 1.1)

    def test_collect_target_symbol_snapshots_shuts_down(self) -> None:
        module = Mock()
        module.initialize.return_value = True
        module.shutdown.return_value = True
        module.symbol_info.return_value = SimpleNamespace(
            visible=True,
            currency_profit="USD",
            volume_min=0.01,
            volume_step=0.01,
            trade_contract_size=1.0,
        )
        module.symbol_info_tick.return_value = SimpleNamespace(bid=1.0)

        settings_module = SimpleNamespace(target_symbols={"AAA": True})

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

            rows = collect_target_symbol_snapshots(
                env_path, settings_module=settings_module, module=module
            )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].symbol, "AAA")
        self.assertEqual(rows[0].contract_size, 1.0)
        module.shutdown.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
