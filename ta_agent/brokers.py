"""Broker abstraction: identical interface for paper (simulated) and live
(CoinDCX) execution.
"""
from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .coindcx_client import CoinDCXClient, CoinDCXError
from .settings import Settings

log = logging.getLogger("ta_agent.broker")

FEES = {
    "market": 0.0005,   # taker (futures)
    "limit": 0.0005,
}


@dataclass
class BrokerOrder:
    id: str
    pair: str
    side: str
    order_type: str
    quantity: float
    price: Optional[float]
    status: str = "open"
    filled_quantity: float = 0.0
    avg_price: Optional[float] = None
    created_ms: int = 0
    meta: dict = field(default_factory=dict)


@dataclass
class BrokerPosition:
    pair: str
    side: str
    quantity: float
    entry_price: float
    notional: float
    unrealized_pnl: float = 0.0
    position_id: Optional[str] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    meta: dict = field(default_factory=dict)


class Broker:
    def get_balances(self) -> Dict[str, float]: ...
    def place_order(self, pair, side, quantity, order_type="market_order", price=None, **kw) -> BrokerOrder: ...
    def get_positions(self) -> List[BrokerPosition]: ...
    def get_position(self, pair: str) -> Optional[BrokerPosition]: ...
    def close_position(self, pair: str, quantity: Optional[float] = None) -> Optional[BrokerOrder]: ...
    def set_tpsl(self, pair: str, stop_loss: float, take_profit: float, quantity: float) -> dict: ...
    def cancel_order(self, order_id: str) -> bool: ...


# ----------------------------------------------------------------------
# Paper broker
# ----------------------------------------------------------------------
class PaperBroker(Broker):
    def __init__(self, settings: Settings, initial_cash: float = 10_000.0):
        self.s = settings
        self.cash = initial_cash
        self.positions: Dict[str, BrokerPosition] = {}
        self.orders: Dict[str, BrokerOrder] = {}
        self.taker_fee = settings.taker_fee()
        self.slippage = settings.slippage()

    def get_balances(self) -> Dict[str, float]:
        equity = self.cash + sum(p.notional for p in self.positions.values())
        return {"USDT": self.cash, "equity": equity}

    def place_order(self, pair, side, quantity, order_type="market_order", price=None, **kw) -> BrokerOrder:
        oid = uuid.uuid4().hex
        order = BrokerOrder(id=oid, pair=pair, side=side, order_type=order_type,
                            quantity=quantity, price=price, created_ms=int(time.time() * 1000))
        if order_type == "market_order":
            fill_price = self._mark_price(pair)
            if fill_price is None:
                order.status = "rejected"
                self.orders[oid] = order
                return order
            order.avg_price = fill_price * (1 + self.slippage) if side == "buy" else fill_price * (1 - self.slippage)
            order.filled_quantity = quantity
            order.status = "filled"
            self._apply_fill(order, fill_price)
        else:
            order.status = "open"
        self.orders[oid] = order
        return order

    def get_positions(self) -> List[BrokerPosition]:
        return list(self.positions.values())

    def get_position(self, pair: str) -> Optional[BrokerPosition]:
        return self.positions.get(pair)

    def close_position(self, pair: str, quantity: Optional[float] = None) -> Optional[BrokerOrder]:
        pos = self.positions.get(pair)
        if not pos:
            return None
        qty = quantity or pos.quantity
        fill = self._mark_price(pair)
        if fill is None:
            return None
        sell_side = "sell" if pos.side == "long" else "buy"
        px = fill * (1 - self.slippage) if sell_side == "sell" else fill * (1 + self.slippage)
        pnl = (px - pos.entry_price) * qty if pos.side == "long" else (pos.entry_price - px) * qty
        fees = px * qty * self.taker_fee
        pnl -= fees
        self.cash += px * qty if sell_side == "sell" else self.cash  # shorts reduce cash by cost
        self.cash -= fees
        remaining = pos.quantity - qty
        if remaining <= 1e-12:
            del self.positions[pair]
        else:
            pos.quantity = remaining
            pos.notional = remaining * pos.entry_price
        return BrokerOrder(id=uuid.uuid4().hex, pair=pair, side=sell_side, order_type="market_order",
                           quantity=qty, price=px, status="filled", filled_quantity=qty,
                           avg_price=px, meta={"pnl": pnl})

    def set_tpsl(self, pair: str, stop_loss: float, take_profit: float, quantity: float) -> dict:
        return {"ok": True, "pair": pair, "stop_loss": stop_loss, "take_profit": take_profit}

    def cancel_order(self, order_id: str) -> bool:
        o = self.orders.get(order_id)
        if o and o.status == "open":
            o.status = "cancelled"
            return True
        return False

    def _mark_price(self, pair: str) -> Optional[float]:
        """Price feed hook. Override in live/paper loop by setting self.mark_prices."""
        if getattr(self, "mark_prices", None) is not None:
            return self.mark_prices.get(pair)
        pos = self.positions.get(pair)
        return pos.entry_price if pos else None

    def _apply_fill(self, order: BrokerOrder, ref_price: float) -> None:
        pair = order.pair
        px = order.avg_price or ref_price
        fees = px * order.quantity * self.taker_fee
        if order.side == "buy":
            self.cash -= px * order.quantity + fees
            existing = self.positions.get(pair)
            if existing:
                tot_qty = existing.quantity + order.quantity
                existing.entry_price = (existing.entry_price * existing.quantity + px * order.quantity) / tot_qty
                existing.quantity = tot_qty
                existing.notional = existing.quantity * existing.entry_price
            else:
                self.positions[pair] = BrokerPosition(pair=pair, side="long", quantity=order.quantity,
                                                      entry_price=px, notional=px * order.quantity)
        else:
            self.cash += px * order.quantity - fees
            existing = self.positions.get(pair)
            if existing:
                tot_qty = existing.quantity + order.quantity
                existing.entry_price = (existing.entry_price * existing.quantity + px * order.quantity) / tot_qty
                existing.quantity = tot_qty
                existing.notional = existing.quantity * existing.entry_price
            else:
                self.positions[pair] = BrokerPosition(pair=pair, side="short", quantity=order.quantity,
                                                      entry_price=px, notional=px * order.quantity)


# ----------------------------------------------------------------------
# Live broker (CoinDCX futures)
# ----------------------------------------------------------------------
class LiveBroker(Broker):
    def __init__(self, client: CoinDCXClient, settings: Settings):
        self.c = client
        self.s = settings
        self.margin_currency = "USDT"
        self._orders: Dict[str, BrokerOrder] = {}
        self._last_positions: Dict[str, BrokerPosition] = {}

    def get_balances(self) -> Dict[str, float]:
        try:
            wallets = self.c.get_futures_wallets()
            out: Dict[str, float] = {}
            for w in wallets:
                currency = w.get("currency_short_name") or w.get("currency")
                out[currency] = float(w.get("balance", 0.0))
            # Sizing equity for USDT-margined futures = free USDT wallet balance.
            out["equity"] = out.get("USDT", 0.0)
            return out
        except CoinDCXError as exc:
            log.error("get_balances failed: %s", exc)
            return {}

    def place_order(self, pair, side, quantity, order_type="market_order", price=None,
                    stop_loss=None, take_profit=None, leverage=None, **kw) -> BrokerOrder:
        oid = f"local-{uuid.uuid4().hex}"
        order = BrokerOrder(id=oid, pair=pair, side=side, order_type=order_type,
                            quantity=quantity, price=price, created_ms=int(time.time() * 1000))
        try:
            res = self.c.create_futures_order(
                pair=pair, side=side, quantity=quantity, order_type=order_type,
                price=price, leverage=leverage or self.s.leverage,
                take_profit_price=take_profit, stop_loss_price=stop_loss,
                margin_currency=self.margin_currency,
            )
        except CoinDCXError as exc:
            order.status = "rejected"
            order.meta["error"] = str(exc)
            self._orders[oid] = order
            return order
        broker_id = res.get("id")
        order.id = broker_id or oid
        order.meta["broker_id"] = broker_id
        order.status = "filled" if res.get("status") in ("filled", "init", "open") else "open"
        order.avg_price = float(res["avg_price"]) if res.get("avg_price") else price
        order.meta["raw"] = res
        self._orders[order.id] = order
        return order

    def get_positions(self) -> List[BrokerPosition]:
        try:
            rows = self.c.get_futures_positions(self.margin_currency)
        except CoinDCXError as exc:
            log.error("get_positions failed: %s", exc)
            return list(self._last_positions.values())
        out: List[BrokerPosition] = []
        for r in rows:
            active = float(r.get("active_pos", 0.0))
            if abs(active) < 1e-12:
                continue
            pair = r.get("pair")
            side = "long" if active > 0 else "short"
            qty = abs(active)
            entry = float(r.get("avg_price", 0.0))
            out.append(BrokerPosition(
                pair=pair, side=side, quantity=qty, entry_price=entry,
                notional=qty * entry, position_id=r.get("id"),
                stop_loss=r.get("stop_loss_trigger"),
                take_profit=r.get("take_profit_trigger"),
                meta={"raw": r},
            ))
        self._last_positions = {p.pair: p for p in out}
        return out

    def get_position(self, pair: str) -> Optional[BrokerPosition]:
        for p in self.get_positions():
            if p.pair == pair:
                return p
        return None

    def close_position(self, pair: str, quantity: Optional[float] = None) -> Optional[BrokerOrder]:
        pos = self.get_position(pair)
        if not pos:
            return None
        qty = quantity or pos.quantity
        try:
            res = self.c.exit_futures_position(pos.position_id, pair, qty, self.margin_currency)
        except CoinDCXError as exc:
            log.error("exit %s failed: %s", pair, exc)
            return None
        return BrokerOrder(id=uuid.uuid4().hex, pair=pair, side="exit", order_type="market_order",
                           quantity=qty, status="filled", meta={"raw": res})

    def set_tpsl(self, pair: str, stop_loss: float, take_profit: float, quantity: float) -> dict:
        pos = self.get_position(pair)
        try:
            res = self.c.create_futures_tpsl(
                pair=pair, side=("sell" if pos and pos.side == "long" else "buy"),
                take_profit_price=take_profit, stop_loss_price=stop_loss,
                quantity=quantity, margin_currency=self.margin_currency,
                position_id=pos.position_id if pos else None,
            )
            return {"ok": True, "raw": res}
        except CoinDCXError as exc:
            log.error("set_tpsl %s failed: %s", pair, exc)
            return {"ok": False, "error": str(exc)}

    def cancel_order(self, order_id: str) -> bool:
        try:
            self.c.cancel_futures_order(order_id, self.margin_currency)
            return True
        except CoinDCXError as exc:
            log.error("cancel %s failed: %s", order_id, exc)
            return False
