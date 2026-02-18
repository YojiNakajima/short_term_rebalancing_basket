import sqlite3
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


from infrastructure.dB.sqlite3.initialize_db import initialize_db
from infrastructure.dB.sqlite3.global_settings import set_global_running
from infrastructure.dB.sqlite3.daily_metrics import (
    upsert_daily_atr,
    upsert_daily_pv_tpv,
    upsert_daily_symbol_price,
)
from infrastructure.dB.sqlite3.portfolio import persist_initial_portfolio_run
from infrastructure.mt5.build_initial_portfolio import PlannedPosition
from infrastructure.mt5.symbols_report import SymbolSnapshot


TEST_YMD = "2026-02-18"
FETCHED_AT = "2026-02-18T00:00:00+00:00"


class TestMt5RebalancePortfolio(unittest.TestCase):
    def test_main_uses_settings_trade_default_when_no_flag(self) -> None:
        import infrastructure.mt5.rebalance_portfolio as rp

        called = {}

        def fake_once(*, trade: bool, **kwargs):
            called["trade"] = bool(trade)
            return {}

        with patch.object(
            rp,
            "settings",
            SimpleNamespace(REBALANCE_TRADE_DEFAULT=True),
        ), patch.object(rp, "rebalance_portfolio_once", side_effect=fake_once):
            rc = rp.main(["--no-slack"])

        self.assertEqual(rc, 0)
        self.assertTrue(called.get("trade"))

    def test_main_dry_run_flag_overrides_settings_trade_default(self) -> None:
        import infrastructure.mt5.rebalance_portfolio as rp

        called = {}

        def fake_once(*, trade: bool, **kwargs):
            called["trade"] = bool(trade)
            return {}

        with patch.object(
            rp,
            "settings",
            SimpleNamespace(REBALANCE_TRADE_DEFAULT=True),
        ), patch.object(rp, "rebalance_portfolio_once", side_effect=fake_once):
            rc = rp.main(["--no-slack", "--dry-run"])

        self.assertEqual(rc, 0)
        self.assertFalse(called.get("trade"))

    def test_rebalance_hedges_when_pv_exceeds_daily_tpv_and_r_reaches_stage1(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            db_path = Path(d) / "portfolio.sqlite3"
            initialize_db(db_path)

            # Seed daily metrics (daily initialize outcome)
            upsert_daily_pv_tpv(
                ymd=TEST_YMD,
                equity=10_000.0,
                pv=60_000.0,
                tpv=45_000.0,
                pv_leverage=5.0,
                leverage=1.0,
                tpv_leverage=1.3,
                fetched_at=FETCHED_AT,
                db_path=db_path,
            )
            for sym, sell in [("AAA", 100.0), ("BBB", 70.0)]:
                upsert_daily_symbol_price(
                    ymd=TEST_YMD,
                    symbol=sym,
                    sell=sell,
                    fetched_at=FETCHED_AT,
                    db_path=db_path,
                )
                upsert_daily_atr(
                    ymd=TEST_YMD,
                    symbol=sym,
                    atr=2.0,
                    atr_pct=2.0,
                    time_frame=4,
                    atr_period=14,
                    fetched_at=FETCHED_AT,
                    db_path=db_path,
                )

            rows = [
                SymbolSnapshot(symbol="AAA", exists=True, sell=100.0, atr=2.0, atr_pct=2.0, volume_min=0.01, volume_step=0.01, volume_max=100.0),
                SymbolSnapshot(symbol="BBB", exists=True, sell=70.0, atr=1.4, atr_pct=2.0, volume_min=0.01, volume_step=0.01, volume_max=100.0),
            ]
            plan = [
                PlannedPosition(symbol="AAA", lot=1.0, direction="BUY", atr_pct=2.0, weight=0.5, risk_usd=0.0, risk_per_lot_at_1atr=None, ideal_lot=None, volume_min=0.01, volume_step=0.01, volume_max=100.0, usd_per_lot=100_000.0, usd_nominal=100_000.0, reason=None),
                PlannedPosition(symbol="BBB", lot=1.0, direction="BUY", atr_pct=2.0, weight=0.5, risk_usd=0.0, risk_per_lot_at_1atr=None, ideal_lot=None, volume_min=0.01, volume_step=0.01, volume_max=100.0, usd_per_lot=70_000.0, usd_nominal=70_000.0, reason=None),
            ]
            run_id, _ = persist_initial_portfolio_run(
                rows,
                plan,
                db_path=db_path,
                created_at="2026-02-16T00:00:00+00:00",
                time_frame=4,
                atr_period=14,
                tpv_leverage=5.0,
                risk_pct=0.01,
                equity=10_000.0,
                tpv=160_000.0,
                weights_title="w",
                mt5_order_comment="initial_portfolio",
            )
            self.assertGreater(run_id, 0)

            fake_settings = SimpleNamespace(
                PV_LEVERAGE=5.0,
                CIRCUIT_BREAKER_THRESHOLD=0.5,
                TP_ATR_1ST=1.0,
                TP_ATR_2ND=1.5,
                TP_ATR_3RD=1.9,
                TP_AMOUNT_1ST=0.5,
                TP_AMOUNT_2ND=0.3,
                TP_AMOUNT_3RD=0.2,
            )

            class FakeMt5:
                POSITION_TYPE_BUY = 0
                POSITION_TYPE_SELL = 1
                ORDER_TYPE_BUY = 0
                ORDER_TYPE_SELL = 1
                TRADE_ACTION_DEAL = 2
                ORDER_TIME_GTC = 3
                ORDER_FILLING_IOC = 4

                def __init__(self):
                    self.shutdown_called = False

                def initialize(self, **kwargs):
                    return True

                def shutdown(self):
                    self.shutdown_called = True

                def account_info(self):
                    return SimpleNamespace(equity=10_000.0)

                def positions_get(self):
                    return [
                        SimpleNamespace(symbol="AAA", type=self.POSITION_TYPE_BUY, volume=1.0),
                        SimpleNamespace(symbol="BBB", type=self.POSITION_TYPE_BUY, volume=1.0),
                    ]

                def symbol_select(self, symbol, enable):
                    return True

                def symbol_info_tick(self, symbol):
                    if symbol == "AAA":
                        return SimpleNamespace(bid=102.0, ask=102.01)
                    if symbol == "BBB":
                        return SimpleNamespace(bid=70.0, ask=70.01)
                    return SimpleNamespace(bid=1.0, ask=1.01)

                def symbol_info(self, symbol):
                    return SimpleNamespace(filling_mode=self.ORDER_FILLING_IOC)

            fake = FakeMt5()

            with patch(
                "infrastructure.mt5.rebalance_portfolio.load_mt5_config",
                return_value=SimpleNamespace(path="p", login=1, password="x", server="s", portable=False),
            ), patch(
                "infrastructure.mt5.rebalance_portfolio.send_slack_message",
                return_value="ok",
            ) as slack_mock, patch(
                "infrastructure.mt5.rebalance_portfolio.today_ymd_local",
                return_value=TEST_YMD,
            ):
                from infrastructure.mt5.rebalance_portfolio import rebalance_portfolio_once

                out = rebalance_portfolio_once(
                    env_file="dummy.env",
                    db_path=db_path,
                    settings_module=fake_settings,
                    module=fake,
                    trade=False,
                    slack_notify=True,
                )

            self.assertTrue(out["triggered"])
            self.assertEqual(len(out["actions"]), 1)
            a = out["actions"][0]
            self.assertEqual(a["symbol"], "AAA")
            self.assertEqual(a["action"], "HEDGE")
            self.assertEqual(a["direction"], "SELL")
            self.assertAlmostEqual(a["volume"], 0.50, places=2)
            self.assertEqual(a["stage"], 1)
            self.assertTrue(fake.shutdown_called)
            self.assertEqual(slack_mock.call_count, 1)

            text = slack_mock.call_args.args[0]
            self.assertIn("[rebalance] tp_check", text)
            self.assertIn("AAA", text)
            self.assertIn("tp_px", text)
            self.assertIn("actions:", text)

            conn = sqlite3.connect(str(db_path))
            try:
                cur = conn.cursor()
                cur.execute("SELECT COUNT(*) FROM rebalance_runs")
                self.assertEqual(cur.fetchone()[0], 1)
                cur.execute("SELECT COUNT(*) FROM rebalance_actions")
                self.assertEqual(cur.fetchone()[0], 1)
                cur.execute(
                    "SELECT tp_stage FROM rebalance_state WHERE portfolio_run_id=? AND symbol=?",
                    (run_id, "AAA"),
                )
                self.assertEqual(cur.fetchone()[0], 1)

                cur.execute(
                    "SELECT tp_stage, tp_price_1 FROM daily_tp_reentry_state WHERE ymd=? AND portfolio_run_id=? AND symbol=?",
                    (TEST_YMD, run_id, "AAA"),
                )
                row = cur.fetchone()
                self.assertIsNotNone(row)
                self.assertEqual(int(row[0]), 1)
                self.assertAlmostEqual(float(row[1]), 102.0, places=6)
            finally:
                conn.close()


    def test_rebalance_does_not_duplicate_hedge_when_mt5_already_has_rebalance_positions(self) -> None:
        """If MT5 already has hedge positions (comment starts with rebalance_),
        rebalance should not place duplicated opposite trades even if DB state is not updated.
        """

        with tempfile.TemporaryDirectory() as d:
            db_path = Path(d) / "portfolio.sqlite3"
            initialize_db(db_path)

            upsert_daily_pv_tpv(
                ymd=TEST_YMD,
                equity=10_000.0,
                pv=60_000.0,
                tpv=45_000.0,
                pv_leverage=5.0,
                leverage=1.0,
                tpv_leverage=1.3,
                fetched_at=FETCHED_AT,
                db_path=db_path,
            )
            upsert_daily_symbol_price(
                ymd=TEST_YMD,
                symbol="AAA",
                sell=100.0,
                fetched_at=FETCHED_AT,
                db_path=db_path,
            )
            upsert_daily_atr(
                ymd=TEST_YMD,
                symbol="AAA",
                atr=2.0,
                atr_pct=2.0,
                time_frame=4,
                atr_period=14,
                fetched_at=FETCHED_AT,
                db_path=db_path,
            )

            rows = [
                SymbolSnapshot(
                    symbol="AAA",
                    exists=True,
                    sell=100.0,
                    atr=2.0,
                    atr_pct=2.0,
                    volume_min=0.01,
                    volume_step=0.01,
                    volume_max=100.0,
                )
            ]
            plan = [
                PlannedPosition(
                    symbol="AAA",
                    lot=1.0,
                    direction="BUY",
                    atr_pct=2.0,
                    weight=1.0,
                    risk_usd=0.0,
                    risk_per_lot_at_1atr=None,
                    ideal_lot=None,
                    volume_min=0.01,
                    volume_step=0.01,
                    volume_max=100.0,
                    usd_per_lot=100_000.0,
                    usd_nominal=100_000.0,
                    reason=None,
                )
            ]
            run_id, _ = persist_initial_portfolio_run(
                rows,
                plan,
                db_path=db_path,
                created_at="2026-02-16T00:00:00+00:00",
                time_frame=4,
                atr_period=14,
                tpv_leverage=5.0,
                risk_pct=0.01,
                equity=10_000.0,
                tpv=160_000.0,
                weights_title="w",
                mt5_order_comment="initial_portfolio",
            )

            fake_settings = SimpleNamespace(
                PV_LEVERAGE=5.0,
                CIRCUIT_BREAKER_THRESHOLD=0.5,
                TP_ATR_1ST=1.0,
                TP_ATR_2ND=1.5,
                TP_ATR_3RD=1.9,
                TP_AMOUNT_1ST=0.5,
                TP_AMOUNT_2ND=0.3,
                TP_AMOUNT_3RD=0.2,
            )

            class FakeMt5:
                POSITION_TYPE_BUY = 0
                POSITION_TYPE_SELL = 1
                ORDER_TYPE_BUY = 0
                ORDER_TYPE_SELL = 1
                TRADE_ACTION_DEAL = 2
                ORDER_TIME_GTC = 3
                ORDER_FILLING_IOC = 4

                def __init__(self):
                    self.shutdown_called = False

                def initialize(self, **kwargs):
                    return True

                def shutdown(self):
                    self.shutdown_called = True

                def account_info(self):
                    return SimpleNamespace(equity=10_000.0)

                def positions_get(self):
                    return [
                        # base position (not rebalance_ comment)
                        SimpleNamespace(
                            symbol="AAA",
                            type=self.POSITION_TYPE_BUY,
                            volume=1.0,
                            price_open=100.0,
                            comment="initial_portfolio",
                        ),
                        # already-hedged position (rebalance_ comment)
                        SimpleNamespace(
                            symbol="AAA",
                            type=self.POSITION_TYPE_SELL,
                            volume=0.50,
                            price_open=102.0,
                            comment="rebalance_tp_hedge",
                        ),
                    ]

                def symbol_select(self, symbol, enable):
                    return True

                def symbol_info_tick(self, symbol):
                    return SimpleNamespace(bid=102.0, ask=102.01)

                def symbol_info(self, symbol):
                    return SimpleNamespace(filling_mode=self.ORDER_FILLING_IOC)

            fake = FakeMt5()

            with patch(
                "infrastructure.mt5.rebalance_portfolio.load_mt5_config",
                return_value=SimpleNamespace(path="p", login=1, password="x", server="s", portable=False),
            ), patch(
                "infrastructure.mt5.rebalance_portfolio.today_ymd_local",
                return_value=TEST_YMD,
            ):
                from infrastructure.mt5.rebalance_portfolio import rebalance_portfolio_once

                out = rebalance_portfolio_once(
                    env_file="dummy.env",
                    db_path=db_path,
                    settings_module=fake_settings,
                    module=fake,
                    trade=False,
                    slack_notify=False,
                )

            self.assertFalse(out["triggered"])
            self.assertEqual(out["actions"], [])
            self.assertTrue(fake.shutdown_called)

            conn = sqlite3.connect(str(db_path))
            try:
                cur = conn.cursor()
                cur.execute("SELECT COUNT(*) FROM rebalance_runs")
                self.assertEqual(cur.fetchone()[0], 1)
                cur.execute("SELECT COUNT(*) FROM rebalance_actions")
                self.assertEqual(cur.fetchone()[0], 0)
                cur.execute(
                    "SELECT tp_stage, hedged_volume FROM rebalance_state WHERE portfolio_run_id=? AND symbol=?",
                    (run_id, "AAA"),
                )
                row = cur.fetchone()
                self.assertIsNotNone(row)
                self.assertEqual(int(row[0]), 1)
                self.assertAlmostEqual(float(row[1]), 0.50, places=6)
            finally:
                conn.close()


    def test_rebalance_no_op_when_pv_not_exceeds_daily_tpv(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            db_path = Path(d) / "portfolio.sqlite3"
            initialize_db(db_path)

            upsert_daily_pv_tpv(
                ymd=TEST_YMD,
                equity=10_000.0,
                pv=50_000.0,
                tpv=70_000.0,
                pv_leverage=5.0,
                leverage=1.0,
                tpv_leverage=1.3,
                fetched_at=FETCHED_AT,
                db_path=db_path,
            )

            rows = [SymbolSnapshot(symbol="AAA", exists=True, sell=100.0, atr=2.0, atr_pct=2.0, volume_min=0.01, volume_step=0.01, volume_max=100.0)]
            plan = [PlannedPosition(symbol="AAA", lot=1.0, direction="BUY", atr_pct=2.0, weight=1.0, risk_usd=0.0, risk_per_lot_at_1atr=None, ideal_lot=None, volume_min=0.01, volume_step=0.01, volume_max=100.0, usd_per_lot=100_000.0, usd_nominal=100_000.0, reason=None)]
            _run_id, _ = persist_initial_portfolio_run(
                rows,
                plan,
                db_path=db_path,
                created_at="2026-02-16T00:00:00+00:00",
                time_frame=4,
                atr_period=14,
                tpv_leverage=5.0,
                risk_pct=0.01,
                equity=10_000.0,
                tpv=160_000.0,
                weights_title="w",
                mt5_order_comment="initial_portfolio",
            )

            fake_settings = SimpleNamespace(
                PV_LEVERAGE=5.0,
                CIRCUIT_BREAKER_THRESHOLD=0.5,
                TP_ATR_1ST=1.0,
                TP_ATR_2ND=1.5,
                TP_ATR_3RD=1.9,
                TP_AMOUNT_1ST=0.5,
                TP_AMOUNT_2ND=0.3,
                TP_AMOUNT_3RD=0.2,
            )

            class FakeMt5:
                def __init__(self):
                    self.shutdown_called = False

                def initialize(self, **kwargs):
                    return True

                def shutdown(self):
                    self.shutdown_called = True

                def account_info(self):
                    return SimpleNamespace(equity=10_000.0)

                def positions_get(self):
                    return [SimpleNamespace(symbol="AAA", type=0, volume=1.0)]

            fake = FakeMt5()

            with patch(
                "infrastructure.mt5.rebalance_portfolio.load_mt5_config",
                return_value=SimpleNamespace(path="p", login=1, password="x", server="s", portable=False),
            ), patch(
                "infrastructure.mt5.rebalance_portfolio.send_slack_message",
                return_value="ok",
            ) as slack_mock, patch(
                "infrastructure.mt5.rebalance_portfolio.today_ymd_local",
                return_value=TEST_YMD,
            ):
                from infrastructure.mt5.rebalance_portfolio import rebalance_portfolio_once

                out = rebalance_portfolio_once(
                    env_file="dummy.env",
                    db_path=db_path,
                    settings_module=fake_settings,
                    module=fake,
                    trade=False,
                    slack_notify=True,
                )

            self.assertFalse(out["triggered"])
            self.assertEqual(out["actions"], [])
            self.assertTrue(fake.shutdown_called)
            self.assertEqual(slack_mock.call_count, 1)


    def test_rebalance_circuit_breaker_closes_all_and_turns_global_running_off(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            db_path = Path(d) / "portfolio.sqlite3"
            initialize_db(db_path)

            upsert_daily_pv_tpv(
                ymd=TEST_YMD,
                equity=10_000.0,
                pv=120_000.0,
                tpv=130_000.0,
                pv_leverage=5.0,
                leverage=1.0,
                tpv_leverage=1.3,
                fetched_at=FETCHED_AT,
                db_path=db_path,
            )

            rows = [SymbolSnapshot(symbol="AAA", exists=True, sell=100.0, atr=2.0, atr_pct=2.0, volume_min=0.01, volume_step=0.01, volume_max=100.0)]
            plan = [PlannedPosition(symbol="AAA", lot=1.0, direction="BUY", atr_pct=2.0, weight=1.0, risk_usd=0.0, risk_per_lot_at_1atr=None, ideal_lot=None, volume_min=0.01, volume_step=0.01, volume_max=100.0, usd_per_lot=100_000.0, usd_nominal=100_000.0, reason=None)]
            _run_id, _ = persist_initial_portfolio_run(
                rows,
                plan,
                db_path=db_path,
                created_at="2026-02-16T00:00:00+00:00",
                time_frame=4,
                atr_period=14,
                tpv_leverage=5.0,
                risk_pct=0.01,
                equity=10_000.0,
                tpv=160_000.0,
                weights_title="w",
                mt5_order_comment="initial_portfolio",
            )
            set_global_running(True, db_path=db_path)

            fake_settings = SimpleNamespace(
                PV_LEVERAGE=5.0,
                CIRCUIT_BREAKER_THRESHOLD=0.5,
                TP_ATR_1ST=1.0,
                TP_ATR_2ND=1.5,
                TP_ATR_3RD=1.9,
                TP_AMOUNT_1ST=0.5,
                TP_AMOUNT_2ND=0.3,
                TP_AMOUNT_3RD=0.2,
            )

            class FakeMt5:
                POSITION_TYPE_BUY = 0
                POSITION_TYPE_SELL = 1
                ORDER_TYPE_BUY = 0
                ORDER_TYPE_SELL = 1
                TRADE_ACTION_DEAL = 2
                ORDER_TIME_GTC = 3
                ORDER_FILLING_IOC = 4
                TRADE_RETCODE_DONE = 10009

                def __init__(self):
                    self.shutdown_called = False
                    self.sent = []

                def initialize(self, **kwargs):
                    return True

                def shutdown(self):
                    self.shutdown_called = True

                def account_info(self):
                    return SimpleNamespace(equity=10_000.0)

                def positions_get(self):
                    return [
                        SimpleNamespace(symbol="AAA", type=self.POSITION_TYPE_BUY, volume=1.0, ticket=101),
                        SimpleNamespace(symbol="AAA", type=self.POSITION_TYPE_BUY, volume=2.0, ticket=102),
                    ]

                def symbol_select(self, symbol, enable):
                    return True

                def symbol_info_tick(self, symbol):
                    return SimpleNamespace(bid=99.9, ask=100.1)

                def symbol_info(self, symbol):
                    return SimpleNamespace(filling_mode=self.ORDER_FILLING_IOC)

                def order_send(self, req):
                    self.sent.append(req)
                    return SimpleNamespace(retcode=self.TRADE_RETCODE_DONE)

            fake = FakeMt5()

            with patch(
                "infrastructure.mt5.rebalance_portfolio.load_mt5_config",
                return_value=SimpleNamespace(path="p", login=1, password="x", server="s", portable=False),
            ), patch(
                "infrastructure.mt5.rebalance_portfolio.send_slack_message",
                return_value="ok",
            ) as slack_mock, patch(
                "infrastructure.mt5.rebalance_portfolio.today_ymd_local",
                return_value=TEST_YMD,
            ):
                from infrastructure.mt5.rebalance_portfolio import rebalance_portfolio_once

                out = rebalance_portfolio_once(
                    env_file="dummy.env",
                    db_path=db_path,
                    settings_module=fake_settings,
                    module=fake,
                    trade=True,
                    slack_notify=True,
                )

            self.assertTrue(out["triggered"])
            self.assertTrue(fake.shutdown_called)
            self.assertEqual(len(fake.sent), 2)
            self.assertEqual(slack_mock.call_count, 1)
            self.assertIn("circuit_breaker", slack_mock.call_args.args[0])

            conn = sqlite3.connect(str(db_path))
            try:
                cur = conn.cursor()
                cur.execute("SELECT global_running FROM global_settings WHERE id=1")
                self.assertEqual(int(cur.fetchone()[0]), 0)
            finally:
                conn.close()


    def test_rebalance_skips_when_global_running_false_and_writes_log(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            db_path = Path(d) / "portfolio.sqlite3"
            log_path = Path(d) / "rebalance.log"
            initialize_db(db_path)

            upsert_daily_pv_tpv(
                ymd=TEST_YMD,
                equity=10_000.0,
                pv=50_000.0,
                tpv=60_000.0,
                pv_leverage=5.0,
                leverage=1.0,
                tpv_leverage=1.3,
                fetched_at=FETCHED_AT,
                db_path=db_path,
            )

            rows = [
                SymbolSnapshot(
                    symbol="AAA",
                    exists=True,
                    sell=100.0,
                    atr=2.0,
                    atr_pct=2.0,
                    contract_size=1000.0,
                    currency_profit="USD",
                    usd_per_lot=100_000.0,
                    volume_min=0.01,
                    volume_step=0.01,
                    volume_max=100.0,
                    lot_for_base=1.0,
                    usd_for_base_lot=100_000.0,
                    usd_diff_to_base=0.0,
                )
            ]

            plan = [
                PlannedPosition(
                    symbol="AAA",
                    atr_pct=2.0,
                    weight=1.0,
                    risk_usd=0.0,
                    risk_per_lot_at_1atr=None,
                    ideal_lot=None,
                    lot=1.0,
                    direction="BUY",
                    volume_min=0.01,
                    volume_step=0.01,
                    volume_max=100.0,
                    usd_per_lot=100_000.0,
                    usd_nominal=100_000.0,
                    reason=None,
                )
            ]

            run_id, _ = persist_initial_portfolio_run(
                rows,
                plan,
                db_path=db_path,
                created_at="2026-02-16T00:00:00+00:00",
                time_frame=4,
                atr_period=14,
                tpv_leverage=5.0,
                risk_pct=0.01,
                equity=10_000.0,
                tpv=100_000.0,
                weights_title="w",
                mt5_order_comment="initial_portfolio",
            )
            self.assertGreater(run_id, 0)

            set_global_running(False, db_path=db_path)

            class FakeMt5:
                def initialize(self, **kwargs):
                    raise AssertionError("MT5 should not be initialized when global_running=false")

                def shutdown(self):
                    raise AssertionError("shutdown should not be called when initialize was skipped")

            from infrastructure.mt5.rebalance_portfolio import rebalance_portfolio_once

            with patch(
                "infrastructure.mt5.rebalance_portfolio.today_ymd_local",
                return_value=TEST_YMD,
            ):
                out = rebalance_portfolio_once(
                    env_file=None,
                    db_path=db_path,
                    module=FakeMt5(),
                    trade=False,
                    slack_notify=False,
                    log_path=log_path,
                )

            self.assertTrue(out.get("skipped"))
            self.assertFalse(out.get("global_running"))
            self.assertFalse(out.get("triggered"))
            self.assertEqual(out.get("actions"), [])

            self.assertTrue(log_path.exists())
            log_text = log_path.read_text(encoding="utf-8")
            self.assertIn("[rebalance] skipped", log_text)
            self.assertIn("global_running=false", log_text)

            conn = sqlite3.connect(str(db_path))
            try:
                cur = conn.cursor()
                cur.execute("SELECT COUNT(*) FROM rebalance_runs")
                self.assertEqual(cur.fetchone()[0], 1)
            finally:
                conn.close()


    def test_rebalance_skips_and_notifies_when_no_positions(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            db_path = Path(d) / "portfolio.sqlite3"
            log_path = Path(d) / "rebalance.log"
            initialize_db(db_path)

            upsert_daily_pv_tpv(
                ymd=TEST_YMD,
                equity=10_000.0,
                pv=50_000.0,
                tpv=60_000.0,
                pv_leverage=5.0,
                leverage=1.0,
                tpv_leverage=1.3,
                fetched_at=FETCHED_AT,
                db_path=db_path,
            )

            rows = [
                SymbolSnapshot(
                    symbol="AAA",
                    exists=True,
                    sell=100.0,
                    atr=2.0,
                    atr_pct=2.0,
                    contract_size=1000.0,
                    currency_profit="USD",
                    usd_per_lot=100_000.0,
                    volume_min=0.01,
                    volume_step=0.01,
                    volume_max=100.0,
                    lot_for_base=1.0,
                    usd_for_base_lot=100_000.0,
                    usd_diff_to_base=0.0,
                )
            ]

            plan = [
                PlannedPosition(
                    symbol="AAA",
                    atr_pct=2.0,
                    weight=1.0,
                    risk_usd=0.0,
                    risk_per_lot_at_1atr=None,
                    ideal_lot=None,
                    lot=1.0,
                    direction="BUY",
                    volume_min=0.01,
                    volume_step=0.01,
                    volume_max=100.0,
                    usd_per_lot=100_000.0,
                    usd_nominal=100_000.0,
                    reason=None,
                )
            ]

            run_id, _ = persist_initial_portfolio_run(
                rows,
                plan,
                db_path=db_path,
                created_at="2026-02-16T00:00:00+00:00",
                time_frame=4,
                atr_period=14,
                tpv_leverage=5.0,
                risk_pct=0.01,
                equity=10_000.0,
                tpv=100_000.0,
                weights_title="w",
                mt5_order_comment="initial_portfolio",
            )
            self.assertGreater(run_id, 0)

            set_global_running(True, db_path=db_path)

            class FakeMt5:
                def __init__(self):
                    self.shutdown_called = False

                def initialize(self, **kwargs):
                    return True

                def account_info(self):
                    return SimpleNamespace(equity=10_000.0)

                def positions_get(self):
                    return []

                def shutdown(self):
                    self.shutdown_called = True

            fake = FakeMt5()

            with patch(
                "infrastructure.mt5.rebalance_portfolio.load_mt5_config",
                return_value=SimpleNamespace(
                    path="p",
                    login=1,
                    password="x",
                    server="s",
                    portable=False,
                ),
            ), patch(
                "infrastructure.mt5.rebalance_portfolio.send_slack_message",
                return_value="ok",
            ) as slack_mock, patch(
                "infrastructure.mt5.rebalance_portfolio.today_ymd_local",
                return_value=TEST_YMD,
            ):
                from infrastructure.mt5.rebalance_portfolio import rebalance_portfolio_once

                out = rebalance_portfolio_once(
                    env_file="dummy.env",
                    db_path=db_path,
                    module=fake,
                    trade=False,
                    slack_notify=True,
                    log_path=log_path,
                )

            self.assertFalse(out.get("triggered"))
            self.assertEqual(out.get("actions"), [])
            self.assertTrue(fake.shutdown_called)

            self.assertEqual(slack_mock.call_count, 1)
            slack_text = slack_mock.call_args.args[0]
            self.assertIn("reason=no_positions", slack_text)

            self.assertTrue(log_path.exists())
            log_text = log_path.read_text(encoding="utf-8")
            self.assertIn("INFO", log_text)
            self.assertIn("reason=no_positions", log_text)


if __name__ == "__main__":
    unittest.main()
