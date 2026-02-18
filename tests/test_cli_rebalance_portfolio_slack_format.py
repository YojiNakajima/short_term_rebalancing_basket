import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


from infrastructure.dB.sqlite3.daily_metrics import (
    upsert_daily_atr,
    upsert_daily_pv_tpv,
    upsert_daily_symbol_price,
)
from infrastructure.dB.sqlite3.global_settings import set_global_running
from infrastructure.dB.sqlite3.initialize_db import initialize_db
from infrastructure.dB.sqlite3.portfolio import persist_initial_portfolio_run
from infrastructure.mt5.build_initial_portfolio import PlannedPosition
from infrastructure.mt5.symbols_report import SymbolSnapshot


TEST_YMD = "2026-02-18"
FETCHED_AT = "2026-02-18T00:00:00+00:00"


class TestCliRebalancePortfolioSlackFormat(unittest.TestCase):
    def test_slack_text_ref_now_are_2dp(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            db_path = Path(d) / "portfolio.sqlite3"
            initialize_db(db_path)
            set_global_running(True, db_path=db_path)

            upsert_daily_pv_tpv(
                ymd=TEST_YMD,
                equity=10_000.0,
                pv=10_000.0,
                tpv=9_000.0,
                pv_leverage=1.0,
                leverage=1.0,
                tpv_leverage=1.0,
                fetched_at=FETCHED_AT,
                db_path=db_path,
            )

            upsert_daily_symbol_price(
                ymd=TEST_YMD,
                symbol="AAA",
                sell=100.1234,
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
            _run_id, _ = persist_initial_portfolio_run(
                rows,
                plan,
                db_path=db_path,
                created_at="2026-02-16T00:00:00+00:00",
                time_frame=4,
                atr_period=14,
                tpv_leverage=1.0,
                risk_pct=0.01,
                equity=10_000.0,
                tpv=100_000.0,
                weights_title="w",
                mt5_order_comment="initial_portfolio",
            )

            fake_settings = SimpleNamespace(
                PV_LEVERAGE=1.0,
                CIRCUIT_BREAKER_THRESHOLD=0.8,
                # Keep thresholds high to avoid triggering hedge actions in this formatting test.
                TP_ATR_1ST=100.0,
                TP_ATR_2ND=150.0,
                TP_ATR_3RD=190.0,
                TP_AMOUNT_1ST=0.5,
                TP_AMOUNT_2ND=0.3,
                TP_AMOUNT_3RD=0.2,
            )

            class FakeMt5:
                POSITION_TYPE_BUY = 0
                POSITION_TYPE_SELL = 1

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
                        SimpleNamespace(
                            symbol="AAA",
                            type=self.POSITION_TYPE_BUY,
                            volume=1.0,
                            price_open=100.0,
                            comment=None,
                            profit=0.0,
                        )
                    ]

                def symbol_select(self, symbol, enable):
                    return True

                def symbol_info_tick(self, symbol):
                    return SimpleNamespace(bid=123.4567, ask=123.5567)

            fake = FakeMt5()

            with patch(
                "interfaces.cli.rebalance_portfolio.load_mt5_config",
                return_value=SimpleNamespace(path="p", login=1, password="x", server="s", portable=False),
            ), patch(
                "interfaces.cli.rebalance_portfolio.today_ymd_local",
                return_value=TEST_YMD,
            ), patch(
                "interfaces.cli.rebalance_portfolio.send_slack_message",
                return_value="ok",
            ) as slack_mock:
                from interfaces.cli.rebalance_portfolio import rebalance_portfolio_once

                out = rebalance_portfolio_once(
                    env_file="dummy.env",
                    db_path=db_path,
                    settings_module=fake_settings,
                    module=fake,
                    trade=False,
                    slack_notify=True,
                    log_path=Path(d) / "rebalance_portfolio.log",
                )

            self.assertIn("ymd", out["metrics"])
            self.assertTrue(fake.shutdown_called)
            self.assertEqual(slack_mock.call_count, 1)

            text = slack_mock.call_args.args[0]

            # ref/now should be rounded to 2 decimal places in the Slack table.
            self.assertIn("100.12", text)
            self.assertIn("123.46", text)
            self.assertNotIn("100.12340", text)
            self.assertNotIn("123.45670", text)
