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


class TestKillSwitch:
    """HALT/PAUSE marker files + flatten-on-critical hardening for live mode."""

    def _bot_with_open_position(self, settings, tmp_path):
        bot = TradingBot(settings, store_path=tmp_path / "j.db", synthetic=True)
        bot.broker.mark_prices = {"B-BTC_USDT": 101.0}
        bot.broker.place_order("B-BTC_USDT", "buy", 0.01, order_type="market_order")
        bot.open_positions["BTC"] = PositionState(
            coin="BTC", pair="B-BTC_USDT", side="long", entry=100.0, qty=0.01,
            notional=1.0, stop_loss=95.0, take_profit=115.0, entry_time_ms=1,
            peak_price=100.0, reason="BTC-1", confidence=0.9,
        )
        bot.risk.state.open_positions = 1
        return bot

    def test_markers_status(self, settings, tmp_path):
        bot = TradingBot(settings, store_path=tmp_path / "j.db", synthetic=True)
        bot.s.data_dir = str(tmp_path)
        assert bot._check_markers() == "run"
        (tmp_path / "PAUSE").write_text("x", encoding="utf-8")
        assert bot._check_markers() == "pause"
        (tmp_path / "HALT").write_text("x", encoding="utf-8")
        assert bot._check_markers() == "halt"

    def test_flatten_all_closes_open_positions(self, settings, tmp_path):
        bot = self._bot_with_open_position(settings, tmp_path)
        bot._flatten_all("test")
        assert "BTC" not in bot.open_positions, "flatten must remove the position"
        assert bot.risk.state.open_positions == 0
        assert bot.broker.get_positions() == [], "flatten must close at the broker"

    def test_halt_marker_blocks_entries_and_keeps_loop_alive(self, settings, tmp_path, monkeypatch, caplog):
        bot = TradingBot(settings, store_path=tmp_path / "j.db", synthetic=True)
        bot.s.data_dir = str(tmp_path)
        (tmp_path / "HALT").write_text("x", encoding="utf-8")
        scanned = {"n": 0}
        bot._scan_entries = lambda *a, **k: scanned.__setitem__("n", scanned["n"] + 1)
        bot.run(cycles=3, interval_seconds=0, dry_cycles=0)
        assert scanned["n"] == 0, "HALT must prevent entry scans"
        assert bot.open_positions == {}

    def test_pause_manages_exits_but_no_entries(self, settings, tmp_path, monkeypatch):
        bot = self._bot_with_open_position(settings, tmp_path)
        bot.s.data_dir = str(tmp_path)
        (tmp_path / "PAUSE").write_text("x", encoding="utf-8")
        managed = {"n": 0}
        scanned = {"n": 0}
        bot._manage_position = lambda *a, **k: managed.__setitem__("n", managed["n"] + 1)
        bot._scan_entries = lambda *a, **k: scanned.__setitem__("n", scanned["n"] + 1)
        bot.run(cycles=2, interval_seconds=0, dry_cycles=0)
        assert managed["n"] > 0, "PAUSE must still manage open positions"
        assert scanned["n"] == 0, "PAUSE must not scan for entries"

    def test_critical_alert_flattens_and_writes_halt(self, settings, tmp_path, monkeypatch):
        bot = self._bot_with_open_position(settings, tmp_path)
        bot.s.mode = "live"
        bot.s.data_dir = str(tmp_path)
        bot.monitor = type("M", (), {
            "check": lambda *a, **k: [{
                "severity": "critical", "rule": "drawdown_limit",
                "detail": "dd beyond limit", "ts": 1, "meta": {}}]})()
        bot.run(cycles=1, interval_seconds=0, dry_cycles=0)
        assert "BTC" not in bot.open_positions, "critical alert must flatten"
        assert (tmp_path / "HALT").exists(), "critical alert must write HALT marker"

    def test_paper_mode_does_not_halt_on_critical(self, settings, tmp_path):
        bot = self._bot_with_open_position(settings, tmp_path)
        bot.s.data_dir = str(tmp_path)
        bot.monitor = type("M", (), {
            "check": lambda *a, **k: [{
                "severity": "critical", "rule": "drawdown_limit",
                "detail": "dd beyond limit", "ts": 1, "meta": {}}]})()
        bot.run(cycles=1, interval_seconds=0, dry_cycles=0)
        assert not (tmp_path / "HALT").exists(), "paper mode must not self-halt"


class TestAvailableCoinsFallback:
    """A transient network failure in available_coins() (a raw CoinDCX HTTP GET
    at the start of every bot loop) must fall back to the watchlist, not crash
    the daemon into a restart loop."""

    def test_failure_falls_back_to_watchlist(self, settings, tmp_path, monkeypatch):
        from ta_agent.coindcx_client import CoinDCXError
        bot = TradingBot(settings, store_path=tmp_path / "j.db", synthetic=True)

        def boom(self):
            raise CoinDCXError("simulated network failure in available_coins")

        monkeypatch.setattr(type(bot.feed), "available_coins", boom)
        bot.run(cycles=1, interval_seconds=0, dry_cycles=1)  # must not raise

    def test_success_uses_available_coins(self, settings, tmp_path, monkeypatch, caplog):
        bot = TradingBot(settings, store_path=tmp_path / "j.db", synthetic=True)
        called = {"n": 0}

        def counting(self):
            called["n"] += 1
            return ["BTC"]

        monkeypatch.setattr(type(bot.feed), "available_coins", counting)
        bot.run(cycles=1, interval_seconds=0, dry_cycles=1)
        assert called["n"] == 1, "available_coins() should be consulted on success"
