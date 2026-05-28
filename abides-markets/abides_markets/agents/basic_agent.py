"""
basic_agent.py
==============

Trend-following / breakout trading agent for ABIDES.

Originally this file defined a very simple BasicAgent that woke up once at
noon and sent a single market order. This version keeps the same ABIDES
TradingAgent lifecycle (__init__, kernel_starting, wakeup, receive_message,
get_wake_frequency, kernel_stopping) but replaces the one-shot logic with
a recurring trend-following / breakout strategy.

Strategy in one paragraph
-------------------------
The agent watches the mid-price of one symbol. On every wakeup it appends
the current mid to a rolling history, then computes a recent high, recent
low, a short moving average and a long moving average. It BUYS when price
breaks above the recent high or the short MA crosses above the long MA,
and SELLS in the symmetric case. Trades are throttled by a cooldown, capped
by max_position, and (optionally) flattened near the close.
"""

from typing import Deque, List, Optional
from collections import deque

from abides_core import Message, NanosecondTime
from abides_core.utils import str_to_ns

from abides_markets.orders import Side
from abides_markets.agents.trading_agent import TradingAgent


class BasicAgent(TradingAgent):
    """
    Trend-following / breakout agent.

    Kept the class name `BasicAgent` so existing ABIDES configs that import
    it keep working. The behaviour, however, is no longer "basic".
    """

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        id: int,
        name: str = "BasicAgent",
        type: str = "BasicAgent",
        symbol: str = "ABM",
        starting_cash: int = 10_000_000,
        order_size: int = 100,
        lookback_window: int = 50,
        short_window: int = 20,
        long_window: int = 50,
        max_position: int = 1000,
        wake_frequency: str = "1min",
        cooldown_time: str = "5min",
        flatten_before_close: bool = True,
        log_orders: bool = True,
        random_state=None,
    ) -> None:
        super().__init__(
            id=id,
            name=name,
            type=type,
            starting_cash=starting_cash,
            log_orders=log_orders,
            random_state=random_state,
        )

        # --- strategy parameters ---
        self.symbol: str = symbol
        self.order_size: int = order_size
        self.lookback_window: int = lookback_window
        self.short_window: int = short_window
        self.long_window: int = long_window
        self.max_position: int = max_position

        # Convert human-readable durations to nanoseconds (ABIDES native unit)
        self.wake_frequency_ns: NanosecondTime = str_to_ns(wake_frequency)
        self.cooldown_ns: NanosecondTime = str_to_ns(cooldown_time)

        self.flatten_before_close: bool = flatten_before_close

        # --- internal state ---
        # Rolling price history sized to the largest window we need.
        max_window = max(self.lookback_window, self.long_window) + 5
        self.price_history: Deque[float] = deque(maxlen=max_window)

        # Last trade timestamp (in ns) for cooldown enforcement.
        self.last_trade_time: Optional[NanosecondTime] = None

        # Track previous SMA relationship to detect crossovers.
        self.prev_short_ma: Optional[float] = None
        self.prev_long_ma: Optional[float] = None

        # Convenience: stop trading flag (used near close).
        self.trading: bool = True

    # ------------------------------------------------------------------ #
    # Kernel lifecycle
    # ------------------------------------------------------------------ #
    def kernel_starting(self, start_time: NanosecondTime) -> None:
        # Register an exchange before calling super (ABIDES convention).
        self.exchange_id: int = self.kernel.find_agents_by_type(
            type(self).__bases__[0]  # not used; placeholder to mirror BasicAgent
        )[0] if False else 0  # most ABIDES configs set this via TradingAgent

        super().kernel_starting(start_time)

    def kernel_stopping(self) -> None:
        # Optional: log final position / PnL for the judges.
        try:
            holdings = self.holdings.get(self.symbol, 0)
            print(
                f"[{self.name}] Final holdings in {self.symbol}: {holdings}, "
                f"cash={self.holdings.get('CASH', 0)}"
            )
        except Exception:
            pass
        super().kernel_stopping()

    # ------------------------------------------------------------------ #
    # Wakeup loop
    # ------------------------------------------------------------------ #
    def wakeup(self, current_time: NanosecondTime) -> None:
        super().wakeup(current_time)

        if not self.mkt_open or not self.mkt_close:
            # Exchange hasn't told us its hours yet; reschedule.
            self.set_wakeup(current_time + self.wake_frequency_ns)
            return

        if current_time < self.mkt_open:
            self.set_wakeup(self.mkt_open)
            return

        if current_time >= self.mkt_close:
            return  # day over

        # Optionally flatten position in the last few minutes.
        if (
            self.flatten_before_close
            and current_time >= self.mkt_close - str_to_ns("5min")
        ):
            self._flatten_position(current_time)
            self.set_wakeup(current_time + self.wake_frequency_ns)
            return

        # Standard path: ask the exchange for the top of book.
        # The response arrives in receive_message; we act there.
        self.get_current_spread(self.symbol)

        # Always reschedule so we keep checking the market.
        self.set_wakeup(current_time + self.wake_frequency_ns)

    # ------------------------------------------------------------------ #
    # Message handling
    # ------------------------------------------------------------------ #
    def receive_message(
        self, current_time: NanosecondTime, sender_id: int, message: Message
    ) -> None:
        super().receive_message(current_time, sender_id, message)

        # We only act after a spread (quote) update.
        if not self.mkt_open or current_time >= self.mkt_close:
            return

        mid = self.get_mid_price()
        if mid is None:
            return

        self.update_price_history(mid)
        signal = self.compute_signal()

        if signal == 0:
            return

        if not self.can_trade_now(current_time):
            return

        self.place_trend_order(signal, current_time)

    # ------------------------------------------------------------------ #
    # Strategy helpers
    # ------------------------------------------------------------------ #
    def get_mid_price(self) -> Optional[float]:
        """Return the current mid-price, or None if quotes are unavailable."""
        # Check if symbol has been initialized in known_bids
        if self.symbol not in self.known_bids or self.symbol not in self.known_asks:
            return None
        
        bid, _, ask, _ = self.get_known_bid_ask(self.symbol)
        if bid is None or ask is None or bid <= 0 or ask <= 0:
            return None
        return (bid + ask) / 2.0

    def update_price_history(self, mid: float) -> None:
        self.price_history.append(mid)

    def _sma(self, prices: List[float], window: int) -> Optional[float]:
        if len(prices) < window:
            return None
        return sum(prices[-window:]) / window

    def compute_signal(self) -> int:
        """
        Returns:
            +1 -> buy signal
            -1 -> sell signal
             0 -> do nothing
        """
        n_needed = max(self.lookback_window, self.long_window) + 1
        if len(self.price_history) < n_needed:
            return 0

        prices = list(self.price_history)
        current = prices[-1]

        # Breakout levels computed on the window EXCLUDING the current bar,
        # so "breaks above recent_high" is meaningful.
        window = prices[-(self.lookback_window + 1):-1]
        recent_high = max(window)
        recent_low = min(window)

        short_ma = self._sma(prices, self.short_window)
        long_ma = self._sma(prices, self.long_window)

        signal = 0

        # --- Rule 1: breakout ---
        if current > recent_high:
            signal = +1
        elif current < recent_low:
            signal = -1

        # --- Rule 2: moving-average crossover ---
        if (
            signal == 0
            and short_ma is not None
            and long_ma is not None
            and self.prev_short_ma is not None
            and self.prev_long_ma is not None
        ):
            crossed_up = self.prev_short_ma <= self.prev_long_ma and short_ma > long_ma
            crossed_dn = self.prev_short_ma >= self.prev_long_ma and short_ma < long_ma
            if crossed_up:
                signal = +1
            elif crossed_dn:
                signal = -1

        self.prev_short_ma = short_ma
        self.prev_long_ma = long_ma
        return signal

    def can_trade_now(self, current_time: NanosecondTime) -> bool:
        if self.last_trade_time is None:
            return True
        return (current_time - self.last_trade_time) >= self.cooldown_ns

    def _current_position(self) -> int:
        return self.holdings.get(self.symbol, 0)

    def place_trend_order(self, signal: int, current_time: NanosecondTime) -> None:
        position = self._current_position()

        if signal > 0:
            # BUY -> we want to lift the ask, so we send a BID order.
            if position >= self.max_position:
                return
            size = min(self.order_size, self.max_position - position)
            self.place_market_order(self.symbol, size, Side.BID)
        elif signal < 0:
            # SELL -> we want to hit the bid, so we send an ASK order.
            if position <= -self.max_position:
                return
            size = min(self.order_size, self.max_position + position)
            self.place_market_order(self.symbol, size, Side.ASK)
        else:
            return

        self.last_trade_time = current_time

    def _flatten_position(self, current_time: NanosecondTime) -> None:
        position = self._current_position()
        if position == 0:
            return
        if position > 0:
            self.place_market_order(self.symbol, abs(position), Side.ASK)
        else:
            self.place_market_order(self.symbol, abs(position), Side.BID)
        self.last_trade_time = current_time

    # ------------------------------------------------------------------ #
    # Wake scheduling
    # ------------------------------------------------------------------ #
    def get_wake_frequency(self) -> NanosecondTime:
        return self.wake_frequency_ns
