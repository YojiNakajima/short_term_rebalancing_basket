import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import infrastructure.mt5.build_initial_portfolio as bip
from infrastructure.mt5.build_initial_portfolio import (
    PlannedPosition,
    build_initial_portfolio,
    build_initial_portfolio_from_snapshots,
    build_mt5_market_order_requests,
    execute_mt5_market_orders,
    inverse_volatility_weights,
    trade_planned_positions_on_mt5,
    _format_portfolio_table,
)
from infrastructure.mt5.symbols_report import SymbolSnapshot


class TestMt5RebalancingPortfolio(unittest.TestCase):
    def test_inverse_volatility_weights_normalizes(self) -> None:
        rows = [
            SymbolSnapshot(symbol="AAA", exists=True, atr_pct=2.0),
            SymbolSnapshot(symbol="BBB", exists=True, atr_pct=4.0),
        ]

        w = inverse_volatility_weights(rows)

        self.assertAlmostEqual(w["AAA"], 2.0 / 3.0)
        self.assertAlmostEqual(w["BBB"], 1.0 / 3.0)
        self.assertAlmostEqual(sum(w.values()), 1.0)

    def test_build_initial_portfolio_from_snapshots_computes_lots(self) -> None:
        rows = [
            SymbolSnapshot(
                symbol="AAA",
                exists=True,
                atr_pct=2.0,
                lot_for_base=1.0,
                usd_for_base_lot=1000.0,
                usd_per_lot=1000.0,
                volume_min=0.01,
                volume_step=0.01,
            ),
            SymbolSnapshot(
                symbol="BBB",
                exists=True,
                atr_pct=4.0,
                lot_for_base=1.0,
                usd_for_base_lot=1000.0,
                usd_per_lot=1000.0,
                volume_min=0.01,
                volume_step=0.01,
            ),
        ]

        # TPV=200,000 / RISK=1% => 2,000 USD risk budget
        plan = build_initial_portfolio_from_snapshots(rows, tpv=200_000.0, risk_pct=0.01)
        p_aaa = next(p for p in plan if p.symbol == "AAA")
        p_bbb = next(p for p in plan if p.symbol == "BBB")

        # weights: AAA=2/3, BBB=1/3
        self.assertAlmostEqual(p_aaa.weight or 0.0, 2.0 / 3.0)
        self.assertAlmostEqual(p_bbb.weight or 0.0, 1.0 / 3.0)

        # AAA risk_per_unit = 1000 * 0.02 = 20 => units=1333.33/20=66.66 => lot=66.67
        self.assertAlmostEqual(p_aaa.risk_usd or 0.0, 2000.0 * (2.0 / 3.0))
        self.assertAlmostEqual(p_aaa.lot, 66.67, places=2)

        # BBB risk_per_unit = 1000 * 0.04 = 40 => units=666.66/40=16.66 => lot=16.67
        self.assertAlmostEqual(p_bbb.risk_usd or 0.0, 2000.0 * (1.0 / 3.0))
        self.assertAlmostEqual(p_bbb.lot, 16.67, places=2)

    def test_build_initial_portfolio_from_snapshots_fallbacks_to_min_lot(self) -> None:
        rows = [
            SymbolSnapshot(
                symbol="OK",
                exists=True,
                atr_pct=2.0,
                lot_for_base=1.0,
                usd_for_base_lot=1000.0,
                volume_min=0.01,
                volume_step=0.01,
            ),
            SymbolSnapshot(
                symbol="MISSING_ATR",
                exists=True,
                atr_pct=None,
                volume_min=0.1,
                volume_step=0.1,
            ),
        ]

        plan = build_initial_portfolio_from_snapshots(rows, tpv=200_000.0, risk_pct=0.01)
        p = next(p for p in plan if p.symbol == "MISSING_ATR")

        self.assertEqual(p.lot, 0.1)
        self.assertEqual(p.reason, "fallback_min_lot")
        self.assertIsNone(p.weight)

    def test_build_initial_portfolio_uses_collector_and_settings(self) -> None:
        rows = [
            SymbolSnapshot(
                symbol="AAA",
                exists=True,
                atr_pct=2.0,
                lot_for_base=1.0,
                usd_for_base_lot=1000.0,
                volume_min=0.01,
                volume_step=0.01,
            )
        ]

        calls = []

        def fake_collector(env_file, *, settings_module, module):
            calls.append((env_file, settings_module, module))
            return rows, 10_000.0

        settings_module = SimpleNamespace(TPV_LEVERAGE=20, RISK_PCT=0.01)
        module = object()

        plan, equity, tpv, risk_pct = build_initial_portfolio(
            "dummy.env",
            settings_module=settings_module,
            module=module,
            collector=fake_collector,
        )

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0], "dummy.env")
        self.assertIs(calls[0][1], settings_module)
        self.assertIs(calls[0][2], module)

        self.assertEqual(equity, 10_000.0)
        self.assertEqual(tpv, 200_000.0)
        self.assertEqual(risk_pct, 0.01)
        self.assertEqual([p.symbol for p in plan], ["AAA"])

    def test_format_portfolio_table_includes_direction(self) -> None:
        plan = [
            PlannedPosition(
                symbol="AAA",
                atr_pct=1.0,
                weight=None,
                risk_usd=None,
                risk_per_lot_at_1atr=None,
                ideal_lot=None,
                lot=0.1,
            )
        ]

        table = _format_portfolio_table(plan)

        self.assertIn("direction", table.splitlines()[0])
        self.assertIn("BUY", table)

    def test_build_mt5_market_order_requests_uses_ask_for_buy(self) -> None:
        class FakeMt5:
            ORDER_TYPE_BUY = 0
            ORDER_TYPE_SELL = 1
            TRADE_ACTION_DEAL = 2
            ORDER_TIME_GTC = 3
            ORDER_FILLING_IOC = 4

            def symbol_select(self, symbol, enable):
                return True

            def symbol_info_tick(self, symbol):
                return SimpleNamespace(ask=1.2345, bid=1.2340)

            def symbol_info(self, symbol):
                return SimpleNamespace(filling_mode=self.ORDER_FILLING_IOC)

        plan = [
            PlannedPosition(
                symbol="AAA",
                atr_pct=None,
                weight=None,
                risk_usd=None,
                risk_per_lot_at_1atr=None,
                ideal_lot=None,
                lot=0.12,
                direction="BUY",
            )
        ]

        reqs = build_mt5_market_order_requests(plan, module=FakeMt5(), deviation=10, comment="c")
        self.assertEqual(len(reqs), 1)
        req = reqs[0]

        self.assertEqual(req["symbol"], "AAA")
        self.assertAlmostEqual(req["volume"], 0.12)
        self.assertEqual(req["type"], FakeMt5.ORDER_TYPE_BUY)
        self.assertAlmostEqual(req["price"], 1.2345)
        self.assertEqual(req["deviation"], 10)
        self.assertEqual(req["comment"], "c")

    def test_trade_planned_positions_on_mt5_dry_run_does_not_send_orders(self) -> None:
        class FakeMt5:
            ORDER_TYPE_BUY = 0
            ORDER_TYPE_SELL = 1
            TRADE_ACTION_DEAL = 2
            ORDER_TIME_GTC = 3
            ORDER_FILLING_IOC = 4
            TRADE_RETCODE_DONE = 10009

            def __init__(self):
                self.sent = []
                self.shutdown_called = False

            def initialize(self, **kwargs):
                return True

            def shutdown(self):
                self.shutdown_called = True

            def symbol_select(self, symbol, enable):
                return True

            def symbol_info_tick(self, symbol):
                return SimpleNamespace(ask=1.1000, bid=1.0999)

            def symbol_info(self, symbol):
                return SimpleNamespace(filling_mode=self.ORDER_FILLING_IOC)

            def order_send(self, req):
                self.sent.append(req)
                return SimpleNamespace(retcode=self.TRADE_RETCODE_DONE)

        fake = FakeMt5()

        plan = [
            PlannedPosition(
                symbol="AAA",
                atr_pct=None,
                weight=None,
                risk_usd=None,
                risk_per_lot_at_1atr=None,
                ideal_lot=None,
                lot=0.10,
                direction="BUY",
            )
        ]

        with patch(
            "infrastructure.mt5.build_initial_portfolio.load_mt5_config",
            return_value=SimpleNamespace(
                path="p",
                login=1,
                password="x",
                server="s",
                portable=False,
            ),
        ):
            reqs, results = trade_planned_positions_on_mt5(plan, env_file="dummy.env", module=fake, dry_run=True)

        self.assertEqual(len(reqs), 1)
        self.assertEqual(results, [])
        self.assertEqual(fake.sent, [])
        self.assertTrue(fake.shutdown_called)

    def test_trade_planned_positions_on_mt5_trade_sends_orders(self) -> None:
        class FakeMt5:
            ORDER_TYPE_BUY = 0
            ORDER_TYPE_SELL = 1
            TRADE_ACTION_DEAL = 2
            ORDER_TIME_GTC = 3
            ORDER_FILLING_IOC = 4
            TRADE_RETCODE_DONE = 10009

            def __init__(self):
                self.sent = []
                self.shutdown_called = False

            def initialize(self, **kwargs):
                return True

            def shutdown(self):
                self.shutdown_called = True

            def symbol_select(self, symbol, enable):
                return True

            def symbol_info_tick(self, symbol):
                return SimpleNamespace(ask=1.2000, bid=1.1999)

            def symbol_info(self, symbol):
                return SimpleNamespace(filling_mode=self.ORDER_FILLING_IOC)

            def order_send(self, req):
                self.sent.append(req)
                return SimpleNamespace(retcode=self.TRADE_RETCODE_DONE)

        fake = FakeMt5()

        plan = [
            PlannedPosition(
                symbol="AAA",
                atr_pct=None,
                weight=None,
                risk_usd=None,
                risk_per_lot_at_1atr=None,
                ideal_lot=None,
                lot=0.10,
                direction="BUY",
            )
        ]

        with patch(
            "infrastructure.mt5.build_initial_portfolio.load_mt5_config",
            return_value=SimpleNamespace(
                path="p",
                login=1,
                password="x",
                server="s",
                portable=False,
            ),
        ):
            reqs, results = trade_planned_positions_on_mt5(plan, env_file="dummy.env", module=fake, dry_run=False)

        self.assertEqual(len(reqs), 1)
        self.assertEqual(len(results), 1)
        self.assertEqual(len(fake.sent), 1)
        self.assertTrue(fake.shutdown_called)

    def test_execute_mt5_market_orders_retries_on_unsupported_filling_mode(self) -> None:
        class FakeMt5:
            TRADE_RETCODE_DONE = 10009
            TRADE_RETCODE_PLACED = 10008
            TRADE_RETCODE_INVALID_FILL = 10030

            ORDER_FILLING_FOK = 0
            ORDER_FILLING_IOC = 1
            ORDER_FILLING_RETURN = 2

            def __init__(self):
                self.calls = []

            def order_send(self, req):
                self.calls.append(dict(req))
                filling = req.get("type_filling")
                if filling == self.ORDER_FILLING_IOC:
                    return SimpleNamespace(
                        retcode=self.TRADE_RETCODE_INVALID_FILL,
                        comment="Unsupported filling mode",
                    )
                if filling in {self.ORDER_FILLING_RETURN, None}:
                    return SimpleNamespace(retcode=self.TRADE_RETCODE_DONE, comment="ok")
                return SimpleNamespace(retcode=99999, comment="other")

        fake = FakeMt5()
        requests = [{"symbol": "XAUUSD", "type_filling": fake.ORDER_FILLING_IOC}]

        results = execute_mt5_market_orders(requests, module=fake)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].retcode, fake.TRADE_RETCODE_DONE)
        # First try IOC (fails), then RETURN (succeeds).
        self.assertEqual(len(fake.calls), 2)
        self.assertEqual(fake.calls[0].get("type_filling"), fake.ORDER_FILLING_IOC)
        self.assertEqual(fake.calls[1].get("type_filling"), fake.ORDER_FILLING_RETURN)
        # Request is updated to the recovered filling mode.
        self.assertEqual(requests[0].get("type_filling"), fake.ORDER_FILLING_RETURN)

    def test_execute_mt5_market_orders_tries_all_filling_candidates_including_omit(self) -> None:
        """Regression: do not stop early when retcode/comment changes during retries.

        Some brokers return different retcodes/comments for unsupported filling modes.
        We must still try the remaining candidates, including omitting type_filling.
        """

        class FakeMt5:
            TRADE_RETCODE_DONE = 10009
            TRADE_RETCODE_PLACED = 10008
            TRADE_RETCODE_INVALID_FILL = 10030

            ORDER_FILLING_FOK = 0
            ORDER_FILLING_IOC = 1
            ORDER_FILLING_RETURN = 2

            def __init__(self):
                self.calls = []

            def order_send(self, req):
                self.calls.append(dict(req))
                filling = req.get("type_filling") if "type_filling" in req else None

                # First: IOC => explicit unsupported filling error.
                if filling == self.ORDER_FILLING_IOC:
                    return SimpleNamespace(
                        retcode=self.TRADE_RETCODE_INVALID_FILL,
                        comment="Unsupported filling mode",
                    )

                # Next candidates fail with a different retcode/comment (simulates broker variance).
                if filling in {self.ORDER_FILLING_RETURN, self.ORDER_FILLING_FOK}:
                    return SimpleNamespace(retcode=10011, comment="initial_portfolio")

                # Finally: omitting type_filling succeeds.
                if filling is None:
                    return SimpleNamespace(retcode=self.TRADE_RETCODE_DONE, comment="ok")

                return SimpleNamespace(retcode=99999, comment="other")

        fake = FakeMt5()
        requests = [{"symbol": "FR40", "type_filling": fake.ORDER_FILLING_IOC}]

        results = execute_mt5_market_orders(requests, module=fake)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].retcode, fake.TRADE_RETCODE_DONE)

        # Must have tried: IOC (orig) -> RETURN -> FOK -> omit.
        self.assertGreaterEqual(len(fake.calls), 4)
        self.assertEqual(fake.calls[0].get("type_filling"), fake.ORDER_FILLING_IOC)

        # Request is updated to the actually used filling mode (omitted).
        self.assertNotIn("type_filling", requests[0])

    def test_main_interactive_yes_triggers_trade_without_second_confirmation(self) -> None:
        plan = [
            PlannedPosition(
                symbol="AAA",
                atr_pct=None,
                weight=None,
                risk_usd=None,
                risk_per_lot_at_1atr=None,
                ideal_lot=None,
                lot=0.10,
                direction="BUY",
            )
        ]

        with patch.dict(os.environ, {"PYCHARM_HOSTED": "1"}, clear=False), patch(
            "infrastructure.mt5.build_initial_portfolio.build_initial_portfolio_with_diagnostics",
            return_value=(plan, 10_000.0, 200_000.0, 0.01, None, "Weights"),
        ), patch(
            "infrastructure.mt5.build_initial_portfolio.trade_planned_positions_on_mt5",
            return_value=([{"symbol": "AAA"}], [SimpleNamespace(retcode=0)]),
        ) as trade_mock, patch(
            "infrastructure.mt5.build_initial_portfolio.sys.stdin.isatty",
            return_value=False,
        ), patch(
            "builtins.input",
            return_value="YES",
        ) as input_mock:
            rc = bip.main(["--no-plot"])

        self.assertEqual(rc, 0)
        trade_mock.assert_called_once()
        # One interactive choice prompt only; no second confirmation prompt.
        self.assertEqual(input_mock.call_count, 1)

    def test_main_interactive_dry_triggers_dry_run(self) -> None:
        plan = [
            PlannedPosition(
                symbol="AAA",
                atr_pct=None,
                weight=None,
                risk_usd=None,
                risk_per_lot_at_1atr=None,
                ideal_lot=None,
                lot=0.10,
                direction="BUY",
            )
        ]

        with patch.dict(os.environ, {"PYCHARM_HOSTED": "1"}, clear=False), patch(
            "infrastructure.mt5.build_initial_portfolio.build_initial_portfolio_with_diagnostics",
            return_value=(plan, 10_000.0, 200_000.0, 0.01, None, "Weights"),
        ), patch(
            "infrastructure.mt5.build_initial_portfolio.trade_planned_positions_on_mt5",
            return_value=([{"symbol": "AAA"}], []),
        ) as trade_mock, patch(
            "infrastructure.mt5.build_initial_portfolio.sys.stdin.isatty",
            return_value=False,
        ), patch(
            "builtins.input",
            return_value="DRY",
        ):
            rc = bip.main(["--no-plot"])

        self.assertEqual(rc, 0)
        self.assertEqual(trade_mock.call_count, 1)
        self.assertTrue(trade_mock.call_args.kwargs.get("dry_run"))

    def test_main_trade_flag_with_no_confirmation_aborts(self) -> None:
        plan = [
            PlannedPosition(
                symbol="AAA",
                atr_pct=None,
                weight=None,
                risk_usd=None,
                risk_per_lot_at_1atr=None,
                ideal_lot=None,
                lot=0.10,
                direction="BUY",
            )
        ]

        with patch(
            "infrastructure.mt5.build_initial_portfolio.build_initial_portfolio_with_diagnostics",
            return_value=(plan, 10_000.0, 200_000.0, 0.01, None, "Weights"),
        ), patch(
            "infrastructure.mt5.build_initial_portfolio.trade_planned_positions_on_mt5",
        ) as trade_mock, patch(
            "builtins.input",
            return_value="NO",
        ):
            rc = bip.main(["--no-plot", "--trade"])

        self.assertEqual(rc, 0)
        trade_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
