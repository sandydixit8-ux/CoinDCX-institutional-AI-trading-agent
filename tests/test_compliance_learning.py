"""Phase 3: self-learning verification + consolidated spec-compliance report."""
import json
from types import SimpleNamespace

import pytest

from ta_agent.compliance import spec_compliance_report
from ta_agent.features import FEATURE_COLUMNS
from ta_agent.self_learning import LearningJournal, learning_report
from ta_agent.settings import Settings
from ta_agent.store import TradeStore


def plan(coin="BTC", side="long", conf=0.95):
    return SimpleNamespace(coin=coin, pair=f"B-{coin}_USDT", side=side, entry=100.0,
                           stop_loss=95.0, take_profit=115.0, confidence=conf,
                           probability=0.62, rr=3.0, trigger="breakout",
                           timeframe="1h", reason="test", ai_signals={},
                           technical_signals={}, as_report=lambda: {})


FEATURES = [0.5] * len(FEATURE_COLUMNS)


def seed_outcomes(st, n_wins=40, n_loss=20):
    for i in range(n_wins):
        st.record_trade_entry(plan(), f"w{i}", 1.0, 100.0, 1000 + i,
                              features=[float(i % 10) / 10.0] * len(FEATURE_COLUMNS))
        st.record_trade_exit(f"w{i}", 110.0, "take_profit", 10.0, 0.0)
    for i in range(n_loss):
        st.record_trade_entry(plan(), f"l{i}", 1.0, 100.0, 2000 + i,
                              features=[float(i % 10) / 10.0] * len(FEATURE_COLUMNS))
        st.record_trade_exit(f"l{i}", 95.0, "stop_loss", -5.0, 0.0)


class FakeModel:
    def __init__(self):
        self._fitted = True
        self.fit_calls = 0

    def fit(self, X, y):
        self.fit_calls += 1
        self._fitted = True


class TestLearningReport:
    def test_edge_calibration_adaptation(self, tmp_path):
        st = TradeStore(str(tmp_path / "l.db"))
        seed_outcomes(st)
        j = LearningJournal(st, buffer_size=500)
        for r in st.closed_trades():
            j.observe(r["outcome"], r["confidence"], 0.62, coin="BTC",
                      side="long", trigger="breakout",
                      feature_vector=json.loads(r["meta"])["features"])
        rep = learning_report(store=st)
        assert rep["decided"] == 60
        assert rep["overall_win_rate"] == pytest.approx(40 / 60, abs=1e-3)
        assert "BTC:long:breakout" in rep["edge_by_setup"]
        assert rep["edge_by_setup"]["BTC:long:breakout"]["n"] == 60
        assert rep["calibration_curve"] and rep["calibration_curve"][0]["bucket"] == "[0.60,0.70)"
        assert rep["adaptation"]["samples"] == 60
        assert "refit_history" in rep

    def test_insufficient_data_is_graceful(self, tmp_path):
        st = TradeStore(str(tmp_path / "e.db"))
        rep = learning_report(store=st)
        assert rep["decided"] == 0
        assert rep["overall_win_rate"] == 0.0

    def test_refit_records_history(self, tmp_path):
        st = TradeStore(str(tmp_path / "r.db"))
        seed_outcomes(st)
        j = LearningJournal(st, buffer_size=500)
        assert j.refit_ml(FakeModel()) is True
        refits = st.get_param("learning.refits", [])
        assert len(refits) == 1
        assert refits[0]["samples"] == 60
        assert refits[0]["fitted"] is True
        assert refits[0]["win_rate"] == pytest.approx(40 / 60, abs=1e-3)

    def test_refit_requires_samples(self, tmp_path):
        st = TradeStore(str(tmp_path / "s.db"))
        j = LearningJournal(st)
        assert j.refit_ml(FakeModel()) is False
        assert st.get_param("learning.refits", []) == []


MATRIX = {
    "scenarios": {
        "bull": {"30d": {"strict": {"n_trades": 0, "metrics": {},
                                    "compliance": {"passed": True, "hard_failed": False,
                                                   "violations": [], "checks": []}},
                         "edge": {"n_trades": 5, "metrics": {},
                                  "compliance": {"passed": True, "hard_failed": False,
                                                 "violations": [], "checks": []}}}},
        "bear": {"30d": {"strict": {"n_trades": 0, "metrics": {},
                                    "compliance": {"passed": True, "hard_failed": False,
                                                   "violations": [], "checks": []}},
                         "edge": {"n_trades": 5, "metrics": {},
                                  "compliance": {"passed": True, "hard_failed": False,
                                                 "violations": [], "checks": []}}}},
    },
    "summary": {"profile_pass": {"strict": True, "edge": True}, "overall": "PASS"},
}


class TestSpecCompliance:
    def test_matrix_only_report(self, tmp_path):
        m = tmp_path / "matrix.json"
        m.write_text(json.dumps(MATRIX), encoding="utf-8")
        s = Settings()
        s.risk = {"per_trade": 0.01, "daily": 0.03, "weekly": 0.06, "monthly": 0.10,
                  "max_positions": 5, "max_coin_weight": 0.25,
                  "max_correlated_weight": 0.50, "kelly_fraction": 0.25,
                  "atr_mult_sl": 1.5, "min_rr": 3.0, "min_confidence": 0.90}
        rep = spec_compliance_report(s, matrix_path=m, out_dir=tmp_path / "out")
        assert rep["validation_matrix"]["passed_cells"] == 4
        assert rep["verdict"]["overall"] == "PASS"
        assert (tmp_path / "out" / "spec_compliance_report.json").exists()
        assert (tmp_path / "out" / "spec_compliance_report.md").exists()

    def test_risk_mismatch_fails(self, tmp_path):
        s = Settings()
        s.risk = {"per_trade": 0.05}  # wrong on purpose
        rep = spec_compliance_report(s)
        item = next(i for i in rep["verdict"]["items"] if i["item"] == "risk_rules")
        assert item["status"] == "FAIL"
        assert rep["verdict"]["overall"] == "FAIL"

    def test_full_journal_pipeline(self, tmp_path):
        st = TradeStore(str(tmp_path / "j.db"))
        seed_outcomes(st)
        s = Settings()
        s.risk = {"per_trade": 0.01, "daily": 0.03, "weekly": 0.06, "monthly": 0.10,
                  "max_positions": 5, "max_coin_weight": 0.25,
                  "max_correlated_weight": 0.50, "kelly_fraction": 0.25,
                  "atr_mult_sl": 1.5, "min_rr": 3.0, "min_confidence": 0.90,
                  "circuit_breaker": {"consecutive": 3}}
        s.backtest = {"initial_capital": 10000.0}
        rep = spec_compliance_report(s, matrix_path=None, store_path=str(tmp_path / "j.db"))
        pt = rep["per_trade_compliance"]["journal"]
        assert pt["trades"] == 60
        assert pt["passed"] is True
        assert rep["period_loss_limits"]["available"] is True
        assert rep["self_learning"]["decided"] == 60
        assert rep["verdict"]["overall"] == "PASS"

    def test_journal_violations_detected(self, tmp_path):
        st = TradeStore(str(tmp_path / "v.db"))
        p = plan(conf=0.50)
        st.record_trade_entry(p, "v1", 1.0, 100.0, 1000)
        st.record_trade_exit("v1", 95.0, "stop_loss", -5.0, 0.0)
        s = Settings()
        s.risk = {"per_trade": 0.01, "daily": 0.03, "weekly": 0.06, "monthly": 0.10,
                  "max_positions": 5, "max_coin_weight": 0.25,
                  "max_correlated_weight": 0.50, "kelly_fraction": 0.25,
                  "atr_mult_sl": 1.5, "min_rr": 3.0, "min_confidence": 0.90}
        rep = spec_compliance_report(s, store_path=str(tmp_path / "v.db"))
        pt = rep["per_trade_compliance"]["journal"]
        assert pt["passed"] is False
        assert pt["violations"]["confidence_below"] == 1
