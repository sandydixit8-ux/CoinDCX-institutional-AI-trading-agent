"""Regression tests for live-trading robustness fixes:

- signed GET requests must transmit the JSON body (401 otherwise)
- LiveBroker sizing equity comes from the real USDT wallet balance
- a failed broker close keeps the position instead of forgetting it
- live mode sizes from the real balance, and zero balance blocks sizing
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pytest

from ta_agent.bot import TradingBot
from ta_agent.brokers import LiveBroker
from ta_agent.coindcx_client import CoinDCXClient, CoinDCXError
from ta_agent.settings import Settings
from ta_agent.strategy import PositionState


@pytest.fixture
def settings():
    s = Settings.load("config.json")
    s.mode = "paper"
    return s


class FakeResponse:
    def __init__(self, payload, status_code=200, text=""):
        self._payload = payload
        self.status_code = status_code
        self.text = text or repr(payload)

    def json(self):
        return self._payload


class TestSignedGetBody:
    def test_get_signed_transmits_signed_body(self, settings):
        """CoinDCX validates the signature against the received body; a GET
        without the body yields 401."""
        client = CoinDCXClient(api_key="k", api_secret="s")
        captured = {}

        def fake_get(url, **kwargs):
            captured["url"] = url
            captured["kwargs"] = kwargs
            return FakeResponse([{"currency_short_name": "USDT", "balance": "1.0"}])

        client.session.get = fake_get
        data = client._get_signed("/exchange/v1/derivatives/futures/wallets")
        assert data[0]["currency_short_name"] == "USDT"
        assert captured["kwargs"].get("data"), "signed GET must send the JSON body"
        assert "X-AUTH-SIGNATURE" in captured["kwargs"].get("headers", {})


class TestLiveBrokerBalances:
    def test_equity_from_usdt_wallet(self, settings):
        client = type("C", (), {
            "get_futures_wallets": lambda self: [
                {"currency_short_name": "INR", "balance": "37.0"},
                {"currency_short_name": "USDT", "balance": "250.5"},
            ]})()
        b = LiveBroker(client, settings)
        bal = b.get_balances()
        assert bal["USDT"] == 250.5
        assert bal["equity"] == 250.5

    def test_equity_zero_when_no_usdt_wallet(self, settings):
        client = type("C", (), {
            "get_futures_wallets": lambda self: [
                {"currency_short_name": "INR", "balance": "37.0"},
            ]})()
        b = LiveBroker(client, settings)
        assert b.get_balances()["equity"] == 0.0

    def test_balances_error_returns_empty(self, settings):
        client = type("C", (), {
            "get_futures_wallets": lambda self: (_ for _ in ()).throw(
                CoinDCXError("boom"))})()
        b = LiveBroker(client, settings)
        assert b.get_balances() == {}


class TestCloseFailureKeepsPosition:
    def test_none_close_keeps_position_and_does_not_record_exit(self, settings, tmp_path):
        bot = TradingBot(settings, store_path=tmp_path / "j.db", synthetic=True)
        # Open a position in the bot's tracking but NOT in the paper broker,
        # so close_position() cannot fill and returns None.
        bot.open_positions["BTC"] = PositionState(
            coin="BTC", pair="B-BTC_USDT", side="long", entry=100.0, qty=0.01,
            notional=1.0, stop_loss=95.0, take_profit=115.0, entry_time_ms=1,
            peak_price=100.0, reason="BTC-1", confidence=0.9,
        )
        bot._close_position("BTC", bot.open_positions["BTC"], 101.0, "test")
        assert "BTC" in bot.open_positions, "failed close must keep the position"
        assert bot.store.closed_trades() == [], "failed close must not record an exit"


class TestLiveEquitySync:
    def test_syncs_real_balance(self, settings, tmp_path):
        bot = TradingBot(settings, store_path=tmp_path / "j.db", synthetic=True)
        bot.s.mode = "live"
        bot.broker = type("B", (), {"get_balances": lambda self: {"USDT": 250.5, "equity": 250.5}})()
        bot._sync_equity()
        assert bot.risk.state.equity == 250.5

    def test_zero_balance_blocks_sizing(self, settings, tmp_path):
        bot = TradingBot(settings, store_path=tmp_path / "j.db", synthetic=True)
        bot.s.mode = "live"
        bot.broker = type("B", (), {"get_balances": lambda self: {"equity": 0.0}})()
        bot._sync_equity()
        assert bot.risk.state.equity == 0.0

    def test_first_sync_resets_peak_no_spurious_drawdown(self, settings, tmp_path):
        """Zero real balance must not produce a -100% drawdown vs the config
        default peak (which caused a false CRITICAL monitor alert)."""
        bot = TradingBot(settings, store_path=tmp_path / "j.db", synthetic=True)
        bot.s.mode = "live"
        bot.broker = type("B", (), {"get_balances": lambda self: {"equity": 0.0}})()
        bot._sync_equity()
        assert bot.risk.state.peak_equity == 0.0
        peak = max(bot.risk.state.peak_equity, bot.risk.state.equity)
        dd = (bot.risk.state.equity - peak) / peak if peak > 0 else 0.0
        assert dd == 0.0

    def test_paper_mode_never_syncs(self, settings, tmp_path):
        bot = TradingBot(settings, store_path=tmp_path / "j.db", synthetic=True)
        bot.broker = type("B", (), {"get_balances": lambda self: {"USDT": 999.0, "equity": 999.0}})()
        bot._sync_equity()
        assert bot.risk.state.equity == 10_000.0


class TestZeroEquityDrawdown:
    def test_no_drawdown_alert_when_equity_zero(self, settings, tmp_path):
        from ta_agent.monitor import FailureMonitor
        bot = TradingBot(settings, store_path=tmp_path / "j.db", synthetic=True)
        bot.s.mode = "live"
        bot.broker = type("B", (), {"get_balances": lambda self: {"equity": 0.0}})()
        bot._sync_equity()
        mon = FailureMonitor(settings)
        alerts = mon.check_drawdown(bot.risk.state)
        assert alerts == [], "unfunded wallet (equity 0) must not trigger drawdown CRITICAL"

    def test_drawdown_alert_fires_when_funded(self, settings, tmp_path):
        from ta_agent.monitor import FailureMonitor
        bot = TradingBot(settings, store_path=tmp_path / "j.db", synthetic=True)
        bot.risk.state.peak_equity = 10_000.0
        bot.risk.state.equity = 8_000.0
        mon = FailureMonitor(settings)
        alerts = mon.check_drawdown(bot.risk.state)
        assert any(a["rule"] == "drawdown_limit" and a["severity"] == "critical"
                   for a in alerts)
