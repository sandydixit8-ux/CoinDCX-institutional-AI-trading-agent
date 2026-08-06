"""Self-learning: post-trade review, rolling edge estimation, calibration and
ML model refitting from the trade journal.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .features import FEATURE_COLUMNS
from .store import TradeStore

log = logging.getLogger("ta_agent.learning")


class LearningJournal:
    def __init__(self, store: Optional[TradeStore], buffer_size: int = 300):
        self.store = store
        self.buffer_size = buffer_size
        self._buffer: List[dict] = []
        self.edge_estimates: Dict[str, dict] = {}

    # ------------------------------------------------------------------
    def observe(self, outcome: str, confidence: float, probability: float,
                coin: str = "", side: str = "", trigger: str = "",
                feature_vector: Optional[list] = None) -> None:
        rec = {
            "outcome": outcome,
            "confidence": confidence,
            "probability": probability,
            "coin": coin,
            "side": side,
            "trigger": trigger,
            "features": feature_vector,
        }
        self._buffer.append(rec)
        if len(self._buffer) > self.buffer_size:
            self._buffer = self._buffer[-self.buffer_size:]
        self._update_edge(coin, side, trigger, outcome)
        if self.store:
            params = self.store.get_param("learning", {})
            params.setdefault("outcomes", []).append({
                "outcome": outcome, "confidence": confidence,
                "probability": probability, "coin": coin, "side": side, "trigger": trigger,
            })
            self.store.set_param("learning", params)

    def _update_edge(self, coin: str, side: str, trigger: str, outcome: str) -> None:
        win = 1.0 if outcome == "win" else 0.0
        key = f"{coin}:{side}:{trigger}"
        prev = self.edge_estimates.get(key, {"n": 0, "wins": 0, "win_rate": 0.5})
        prev["n"] += 1
        prev["wins"] += win
        prev["win_rate"] = prev["wins"] / prev["n"]
        prev["ewma"] = prev["wins"] if prev["n"] == 1 else \
            0.1 * win + 0.9 * prev.get("ewma", 0.5)
        self.edge_estimates[key] = prev

    def calibration_offset(self, min_samples: int = 20) -> float:
        """Return (observed_win_rate - confidence_implied) adjustment."""
        wins = [r for r in self._buffer if r["outcome"] == "win"]
        if len(self._buffer) < min_samples:
            return 0.0
        return (len(wins) / len(self._buffer)) - 0.5

    def worst_setups(self, n: int = 5) -> List[dict]:
        ranked = sorted(self.edge_estimates.items(),
                        key=lambda kv: kv[1]["win_rate"])
        return [{"setup": k, **v} for k, v in ranked[:n] if v["n"] >= 3]

    # ------------------------------------------------------------------
    def refit_ml(self, model) -> bool:
        """Rebuild the MLScorer from stored feature vectors + outcomes.

        Records a refit event in ``learning.refits`` so adaptation can be
        audited over time.
        """
        if not self.store or model is None:
            return False
        rows = self.store.closed_trades()
        X, y = [], []
        wins = losses = 0
        for r in rows:
            try:
                meta = json.loads(r["meta"] or "{}")
            except Exception:
                meta = {}
            feats = meta.get("features")
            if not feats or len(feats) != len(FEATURE_COLUMNS):
                continue
            outcome = r["outcome"]
            if outcome not in ("win", "loss"):
                continue
            if outcome == "win":
                wins += 1
            else:
                losses += 1
            X.append(feats)
            y.append(1.0 if outcome == "win" else 0.0)
        if len(X) < 30:
            return False
        model.fit(np.asarray(X, dtype=float), np.asarray(y, dtype=float))
        fitted = bool(getattr(model, "_fitted", True))
        wr = wins / (wins + losses) if (wins + losses) else 0.0
        refits = self.store.get_param("learning.refits", [])
        refits.append({
            "ts": int(time.time() * 1000),
            "samples": len(X),
            "win_rate": round(wr, 4),
            "calibration_offset": round(self.calibration_offset(), 4),
            "fitted": fitted,
        })
        self.store.set_param("learning.refits", refits)
        return fitted


# ---------------------------------------------------------------------------
# Learning verification report
# ---------------------------------------------------------------------------
def _bucket_label(lo: float, hi: float) -> str:
    return f"[{lo:.2f},{hi:.2f})"


def learning_report(store: Optional[TradeStore] = None,
                    journal: Optional[LearningJournal] = None,
                    min_samples: int = 3,
                    out_dir: Optional[Path] = None) -> dict:
    """Audit the self-learning loop: edge per setup, probability calibration,
    adaptation over time, and ML refit history.

    Uses the full persisted outcome history from the store when available
    (falling back to the journal's rolling buffer).
    """
    outcomes: List[dict] = []
    if store is not None:
        outcomes = list(store.get_param("learning", {}).get("outcomes", []))
        if not outcomes:
            # Backtester/runners write trades directly to the journal; derive
            # the outcome stream from the trade rows when no observe() records
            # have been persisted.
            for r in store.closed_trades():
                if r["outcome"] not in ("win", "loss"):
                    continue
                outcomes.append({
                    "outcome": r["outcome"],
                    "confidence": float(r["confidence"] or 0.0),
                    "probability": float(r["probability"] or 0.5),
                    "coin": r["coin"], "side": r["side"],
                    "trigger": r["trigger"] or "",
                })
    if not outcomes and journal is not None:
        outcomes = list(journal._buffer)

    wins = [o for o in outcomes if o["outcome"] == "win"]
    losses = [o for o in outcomes if o["outcome"] == "loss"]
    decided = len(wins) + len(losses)
    win_rate = len(wins) / decided if decided else 0.0

    # edge per setup (coin:side:trigger)
    edge: Dict[str, dict] = {}
    for o in outcomes:
        if o["outcome"] not in ("win", "loss"):
            continue
        key = f"{o.get('coin', '?')}:{o.get('side', '?')}:{o.get('trigger', '?')}"
        e = edge.setdefault(key, {"n": 0, "wins": 0, "win_rate": 0.0,
                                  "avg_confidence": 0.0})
        e["n"] += 1
        e["wins"] += 1 if o["outcome"] == "win" else 0
        e["avg_confidence"] = (e["avg_confidence"] * (e["n"] - 1)
                               + float(o.get("confidence", 0.0))) / e["n"]
    edge_used = {k: v for k, v in edge.items() if v["n"] >= min_samples}
    for e in edge_used.values():
        e["win_rate"] = e["wins"] / e["n"]
    ranked = sorted(edge_used.items(), key=lambda kv: kv[1]["win_rate"])

    # calibration: realized win rate vs predicted probability buckets
    buckets: Dict[str, dict] = {}
    for o in outcomes:
        if o["outcome"] not in ("win", "loss"):
            continue
        p = max(0.0, min(1.0, float(o.get("probability", 0.5))))
        lo = int(p * 10) / 10
        hi = min(lo + 0.1, 1.0)
        b = buckets.setdefault(_bucket_label(lo, hi), {"n": 0, "wins": 0})
        b["n"] += 1
        b["wins"] += 1 if o["outcome"] == "win" else 0
    calibration = []
    err_terms, err_n = 0.0, 0
    for label, b in sorted(buckets.items()):
        if b["n"] < min_samples:
            continue
        mid = (float(label[1:-1].split(",")[0]) + float(label[1:-1].split(",")[1])) / 2
        wr = b["wins"] / b["n"]
        calibration.append({"bucket": label, "predicted": round(mid, 2),
                            "realized": round(wr, 4), "n": b["n"]})
        err_terms += abs(wr - mid) * b["n"]
        err_n += b["n"]
    calibration_error = err_terms / err_n if err_n else 0.0

    # adaptation: first half vs second half of the outcome stream
    seq = [o for o in outcomes if o["outcome"] in ("win", "loss")]
    adaptation = {"samples": len(seq), "first_half_win_rate": 0.0,
                  "second_half_win_rate": 0.0, "edge_delta": 0.0,
                  "rolling_win_rate": []}
    if len(seq) >= 2 * min_samples:
        half = len(seq) // 2
        fh = seq[:half]
        sh = seq[half:]
        fwr = sum(1 for o in fh if o["outcome"] == "win") / len(fh)
        swr = sum(1 for o in sh if o["outcome"] == "win") / len(sh)
        adaptation.update(first_half_win_rate=round(fwr, 4),
                          second_half_win_rate=round(swr, 4),
                          edge_delta=round(swr - fwr, 4))
        chunk = max(5, len(seq) // 10)
        for i in range(0, len(seq), chunk):
            part = seq[i:i + chunk]
            adaptation["rolling_win_rate"].append(round(
                sum(1 for o in part if o["outcome"] == "win") / len(part), 4))

    refits = store.get_param("learning.refits", []) if store else []

    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "observations": len(outcomes),
        "decided": decided,
        "overall_win_rate": round(win_rate, 4),
        "calibration_offset": round(win_rate - 0.5, 4),
        "calibration_error": round(calibration_error, 4),
        "edge_by_setup": {k: {"n": v["n"], "wins": v["wins"],
                              "win_rate": round(v["win_rate"], 4),
                              "avg_confidence": round(v["avg_confidence"], 4)}
                          for k, v in edge_used.items()},
        "best_setups": [{"setup": k, **v} for k, v in ranked[-3:]],
        "worst_setups": [{"setup": k, **v} for k, v in ranked[:3]],
        "calibration_curve": calibration,
        "adaptation": adaptation,
        "refit_history": refits,
    }
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "learning_report.json").write_text(
            json.dumps(report, indent=2, default=str), encoding="utf-8")
        (out_dir / "learning_report.md").write_text(
            _learning_markdown(report), encoding="utf-8")
    return report


def _learning_markdown(r: dict) -> str:
    lines = [f"# Self-learning report — {r['generated_at']}", "",
             f"Observations: **{r['observations']}** (decided: {r['decided']})",
             f"Overall win rate: **{r['overall_win_rate']:.1%}**",
             f"Calibration offset: {r['calibration_offset']:+.1%}  "
             f"(realized vs 50/50 baseline)",
             f"Calibration error: {r['calibration_error']:.1%}  "
             f"(mean |realized - predicted| across buckets)", ""]
    if r["edge_by_setup"]:
        lines += ["## Edge by setup", "",
                  "| Setup | Trades | Wins | Win rate | Avg conf |",
                  "|---|---|---|---|---|"]
        for k, v in sorted(r["edge_by_setup"].items(), key=lambda kv: -kv[1]["win_rate"]):
            lines.append(f"| {k} | {v['n']} | {v['wins']} | {v['win_rate']:.0%} | "
                         f"{v['avg_confidence']:.0%} |")
        lines.append("")
    if r["calibration_curve"]:
        lines += ["## Probability calibration", "",
                  "| Bucket | Predicted | Realized | N |", "|---|---|---|---|"]
        for c in r["calibration_curve"]:
            lines.append(f"| {c['bucket']} | {c['predicted']:.2f} | "
                         f"{c['realized']:.0%} | {c['n']} |")
        lines.append("")
    ad = r["adaptation"]
    lines += ["## Adaptation over time", "",
              f"Samples: {ad['samples']}",
              f"First-half win rate: **{ad['first_half_win_rate']:.1%}**",
              f"Second-half win rate: **{ad['second_half_win_rate']:.1%}**",
              f"Edge delta: {ad['edge_delta']:+.1%}",
              f"Rolling (per chunk): {', '.join(f'{x:.0%}' for x in ad['rolling_win_rate'])}",
              ""]
    if r["refit_history"]:
        lines += ["## ML refit history", "",
                  "| When | Samples | Win rate at refit | Cal offset | Fitted |",
                  "|---|---|---|---|---|"]
        for f in r["refit_history"]:
            when = time.strftime("%Y-%m-%d %H:%M", time.gmtime(f["ts"] / 1000))
            lines.append(f"| {when} | {f['samples']} | {f['win_rate']:.1%} | "
                         f"{f['calibration_offset']:+.1%} | {f['fitted']} |")
    return "\n".join(lines)
