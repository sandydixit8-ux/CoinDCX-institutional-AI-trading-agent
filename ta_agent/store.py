"""SQLite persistence: trade journal, equity curve, learned parameters."""
from __future__ import annotations

import json
import sqlite3
import time
from typing import Dict, List, Optional

import numpy as np


class TradeStore:
    def __init__(self, path: str):
        self.path = path
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trade_key TEXT UNIQUE,
                coin TEXT, pair TEXT, side TEXT,
                entry REAL, stop_loss REAL, take_profit REAL,
                quantity REAL, notional REAL,
                confidence REAL, probability REAL, rr REAL,
                trigger TEXT, timeframe TEXT, reason TEXT,
                entry_time INTEGER,
                exit_time INTEGER, exit_price REAL, exit_reason TEXT,
                pnl REAL, pnl_pct REAL, fees REAL, risk_pct REAL,
                outcome TEXT,
                meta TEXT
            );
            CREATE TABLE IF NOT EXISTS equity (
                ts INTEGER, equity REAL, peak REAL, drawdown REAL, trade_id INTEGER
            );
            CREATE TABLE IF NOT EXISTS params (
                key TEXT PRIMARY KEY, value TEXT, updated_at INTEGER
            );
            CREATE TABLE IF NOT EXISTS monitor_alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts INTEGER, severity TEXT, rule TEXT, detail TEXT, meta TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_trades_time ON trades(entry_time);
            CREATE INDEX IF NOT EXISTS idx_equity_ts ON equity(ts);
            CREATE INDEX IF NOT EXISTS idx_alerts_ts ON monitor_alerts(ts);
            """
        )
        self._migrate()
        self.conn.commit()

    def _migrate(self) -> None:
        """Add columns added after the initial schema (idempotent)."""
        cols = {r["name"] for r in self.conn.execute("PRAGMA table_info(trades)").fetchall()}
        additions = {
            "regime": "TEXT",
            "context": "TEXT",
            "lessons": "TEXT",
        }
        for name, decl in additions.items():
            if name not in cols:
                self.conn.execute(f"ALTER TABLE trades ADD COLUMN {name} {decl}")
        self.conn.commit()

    # ------------------------------------------------------------------
    def record_trade_entry(self, plan, trade_key: str, qty: float, notional: float,
                           entry_time_ms: int, features: Optional[list] = None,
                           context: Optional[dict] = None,
                           regime: Optional[str] = None) -> int:
        meta = {"ai": getattr(plan, "ai_signals", {}), "tech": getattr(plan, "technical_signals", {}),
                "report": plan.as_report()}
        if features is not None:
            meta["features"] = features
        entry_time_ms = int(entry_time_ms)
        cur = self.conn.execute(
            """
            INSERT INTO trades (trade_key, coin, pair, side, entry, stop_loss, take_profit,
                                quantity, notional, confidence, probability, rr, trigger,
                                timeframe, reason, entry_time, meta, regime, context)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (trade_key, plan.coin, plan.pair, plan.side, plan.entry, plan.stop_loss,
             plan.take_profit, qty, notional, plan.confidence, plan.probability, plan.rr,
             plan.trigger, plan.timeframe, plan.reason, entry_time_ms,
             json.dumps(meta), regime, json.dumps(context or {})),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def record_trade_exit(self, trade_key: str, exit_price: float, exit_reason: str,
                          pnl: float, fees: float) -> None:
        risk_pct = self.get_risk_pct(trade_key)
        pnl_pct = pnl / (self.get_notional(trade_key) + 1e-12) if risk_pct is None else None
        if pnl_pct is None:
            notional = self.get_notional(trade_key)
            pnl_pct = pnl / notional if notional else 0.0
        outcome = "win" if pnl > 0 else ("loss" if pnl < 0 else "breakeven")
        self.conn.execute(
            """
            UPDATE trades SET exit_time=?, exit_price=?, exit_reason=?, pnl=?, pnl_pct=?,
                              fees=?, outcome=? WHERE trade_key=?
            """,
            (int(time.time() * 1000), exit_price, exit_reason, pnl, pnl_pct, fees,
             outcome, trade_key),
        )
        self.conn.commit()

    def get_risk_pct(self, trade_key: str) -> Optional[float]:
        row = self.conn.execute("SELECT risk_pct FROM trades WHERE trade_key=?", (trade_key,)).fetchone()
        return float(row["risk_pct"]) if row and row["risk_pct"] is not None else None

    def get_notional(self, trade_key: str) -> float:
        row = self.conn.execute("SELECT notional FROM trades WHERE trade_key=?", (trade_key,)).fetchone()
        return float(row["notional"]) if row and row["notional"] is not None else 0.0

    def set_risk_pct(self, trade_key: str, risk_pct: float) -> None:
        self.conn.execute("UPDATE trades SET risk_pct=? WHERE trade_key=?", (risk_pct, trade_key))
        self.conn.commit()

    def record_trade_lessons(self, trade_key: str, lessons: dict) -> None:
        self.conn.execute("UPDATE trades SET lessons=? WHERE trade_key=?",
                          (json.dumps(lessons), trade_key))
        self.conn.commit()

    def append_alert(self, severity: str, rule: str, detail: str,
                     meta: Optional[dict] = None) -> None:
        self.conn.execute(
            "INSERT INTO monitor_alerts (ts, severity, rule, detail, meta) VALUES (?,?,?,?,?)",
            (int(time.time() * 1000), severity, rule, detail, json.dumps(meta or {})),
        )
        self.conn.commit()

    def recent_alerts(self, limit: int = 200, severity: Optional[str] = None) -> List[sqlite3.Row]:
        if severity:
            return self.conn.execute(
                "SELECT * FROM monitor_alerts WHERE severity=? ORDER BY id DESC LIMIT ?",
                (severity, limit)).fetchall()
        return self.conn.execute(
            "SELECT * FROM monitor_alerts ORDER BY id DESC LIMIT ?", (limit,)).fetchall()

    def append_equity(self, ts: int, equity: float, peak: float, drawdown: float,
                      trade_id: Optional[int] = None) -> None:
        self.conn.execute(
            "INSERT INTO equity (ts, equity, peak, drawdown, trade_id) VALUES (?,?,?,?,?)",
            (int(ts), float(equity), float(peak), float(drawdown), trade_id),
        )
        self.conn.commit()

    # ------------------------------------------------------------------
    def closed_trades(self) -> List[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM trades WHERE exit_time IS NOT NULL ORDER BY exit_time"
        ).fetchall()

    def open_trades(self) -> List[sqlite3.Row]:
        return self.conn.execute("SELECT * FROM trades WHERE exit_time IS NULL").fetchall()

    def recent_trades(self, limit: int = 100) -> List[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM trades ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()

    def equity_curve(self) -> List[Dict]:
        rows = self.conn.execute("SELECT ts, equity, peak, drawdown FROM equity ORDER BY ts").fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    def set_param(self, key: str, value) -> None:
        self.conn.execute(
            "INSERT INTO params (key, value, updated_at) VALUES (?,?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
            (key, json.dumps(value), int(time.time() * 1000)),
        )
        self.conn.commit()

    def get_param(self, key: str, default=None):
        row = self.conn.execute("SELECT value FROM params WHERE key=?", (key,)).fetchone()
        if not row:
            return default
        try:
            return json.loads(row["value"])
        except json.JSONDecodeError:
            return row["value"]

    def all_params(self) -> Dict[str, object]:
        rows = self.conn.execute("SELECT key, value FROM params").fetchall()
        out = {}
        for r in rows:
            try:
                out[r["key"]] = json.loads(r["value"])
            except json.JSONDecodeError:
                out[r["key"]] = r["value"]
        return out

    def close(self) -> None:
        self.conn.close()
