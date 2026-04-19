from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any

from config import GameConfig


@dataclass(slots=True)
class Coin:
    coin_id: str
    creator_id: int
    price: float
    total_supply: float
    circulating_supply: float
    trading_volume: float = 0.0
    created_turn: int = 0
    initial_liquidity: float = 0.0
    creator_locked_supply: float = 0.0
    last_traded_turn: int = 0
    intrinsic_value: float = 1.0
    passive_liquidity: float = 0.0
    bull_score: float = 0.0
    panic_score: float = 0.0
    pending_buy_pressure: float = 0.0
    pending_sell_pressure: float = 0.0
    consecutive_sell_pressure: int = 0
    bid_price: float = 1.0
    ask_price: float = 1.0
    exchange_price: float = 1.0
    price_history: list[float] = field(default_factory=list)
    volume_history: list[float] = field(default_factory=list)
    # Lifecycle tracking
    turns_below_min_price: int = 0
    turns_low_activity: int = 0
    lifecycle_stage: str = "birth"
    peak_price: float = 0.0
    death_turn: int = -1
    attention_score: float = 0.5
    # Economic tracking
    buy_volume: float = 0.0  # Total buy volume this turn
    sell_volume: float = 0.0  # Total sell volume this turn
    liquidity_pool: float = 0.0  # Actual liquidity backing the coin
    market_cap: float = 0.0  # price * circulating_supply
    holder_count: int = 0  # Number of unique holders

    def snapshot(self) -> dict[str, Any]:
        return {
            "coin_id": self.coin_id,
            "creator_id": self.creator_id,
            "price": round(self.price, 4),
            "total_supply": round(self.total_supply, 4),
            "circulating_supply": round(self.circulating_supply, 4),
            "trading_volume": round(self.trading_volume, 4),
            "created_turn": self.created_turn,
            "initial_liquidity": round(self.initial_liquidity, 4),
            "creator_locked_supply": round(self.creator_locked_supply, 4),
            "intrinsic_value": round(self.intrinsic_value, 4),
            "passive_liquidity": round(self.passive_liquidity, 4),
            "bull_score": round(self.bull_score, 4),
            "bid_price": round(self.bid_price, 4),
            "ask_price": round(self.ask_price, 4),
            "exchange_price": round(self.exchange_price, 4),
            "pending_buy_pressure": round(self.pending_buy_pressure, 4),
            "pending_sell_pressure": round(self.pending_sell_pressure, 4),
            "lifecycle_stage": self.lifecycle_stage,
            "peak_price": round(self.peak_price, 4),
            "turns_below_min_price": self.turns_below_min_price,
            "turns_low_activity": self.turns_low_activity,
            "attention_score": round(self.attention_score, 4),
        }


class Market:
    def __init__(self, config: GameConfig) -> None:
        self.config = config
        self.random = random.Random(config.seed + 404)
        self.coins: dict[str, Coin] = {}
        self.coin_counter = 0

    def reset(self) -> None:
        self.coins.clear()
        self.coin_counter = 0
        self.random = random.Random(self.config.seed + 404)

    def create_coin(self, creator_id: int, current_turn: int) -> tuple[Coin, float]:
        self.coin_counter += 1
        coin_id = f"C{self.coin_counter:03d}"
        total_supply = min(self.config.market.initial_coin_supply, self.config.market.max_supply)
        creator_supply = total_supply * self.config.market.creator_allocation
        initial_liquidity = total_supply * self.config.market.initial_liquidity_fraction
        creator_supply = min(creator_supply, total_supply - initial_liquidity)
        coin = Coin(
            coin_id=coin_id,
            creator_id=creator_id,
            price=self.config.market.base_coin_price,
            total_supply=total_supply,
            circulating_supply=initial_liquidity,
            trading_volume=0.0,
            created_turn=current_turn,
            initial_liquidity=initial_liquidity,
            creator_locked_supply=creator_supply,
            last_traded_turn=current_turn,
            intrinsic_value=self.config.market.base_coin_price * self.config.market.intrinsic_value_weight,
            passive_liquidity=self.config.market.passive_liquidity_base,
            bull_score=0.0,
            pending_buy_pressure=0.0,
            pending_sell_pressure=0.0,
            bid_price=self.config.market.base_coin_price * (1.0 - self.config.market.min_spread_bps * 0.5),
            ask_price=self.config.market.base_coin_price * (1.0 + self.config.market.min_spread_bps * 0.5),
            exchange_price=self.config.market.base_coin_price,
            price_history=[self.config.market.base_coin_price],
            volume_history=[0.0],
        )
        self.coins[coin_id] = coin
        return coin, creator_supply

    def _soft_floor_support(self, coin: Coin) -> float:
        if coin.intrinsic_value <= 0:
            return 0.0
        deviation = max(0.0, (coin.intrinsic_value - coin.price) / coin.intrinsic_value)
        return min(0.45, deviation * self.config.market.intrinsic_rebound_strength * (1.0 + coin.passive_liquidity / max(self.config.market.passive_liquidity_base, 1.0) * 0.25))

    def _overextension_pressure(self, coin: Coin) -> float:
        intrinsic_ratio = coin.price / max(coin.intrinsic_value, self.config.market.min_price)
        cap_ratio = coin.price / max(self.config.market.max_price, self.config.market.min_price)
        intrinsic_excess = max(0.0, intrinsic_ratio - 1.85)
        cap_excess = max(0.0, cap_ratio - 0.84)
        pressure = intrinsic_excess * 0.16 + cap_excess * 0.55
        if len(coin.price_history) > 1 and coin.price_history[-1] > coin.price_history[-2]:
            pressure += min(0.08, (coin.price_history[-1] - coin.price_history[-2]) / max(coin.price_history[-2], self.config.market.min_price) * 0.18)
        return min(0.42, pressure)

    def _uptrend_strength(self, coin: Coin) -> float:
        if len(coin.price_history) < 4:
            return 0.0
        recent_moves: list[float] = []
        for index in range(max(1, len(coin.price_history) - 4), len(coin.price_history)):
            previous_price = max(coin.price_history[index - 1], self.config.market.min_price)
            recent_moves.append((coin.price_history[index] - previous_price) / previous_price)
        positive_moves = [move for move in recent_moves if move > 0]
        if not positive_moves:
            return 0.0
        return min(0.18, sum(positive_moves) / max(len(recent_moves), 1))

    def _correction_bias(self, coin: Coin) -> float:
        if len(coin.price_history) < 2:
            return 0.0
        previous_price = coin.price_history[-1]
        earlier_price = coin.price_history[-2]
        if earlier_price <= 0:
            return 0.0
        previous_move = (previous_price - earlier_price) / max(earlier_price, self.config.market.min_price)
        # Reduced multiplier for smoother correction
        return -max(-0.08, min(0.08, previous_move * 0.35))

    def _update_quotes(self, coin: Coin) -> None:
        spread = max(
            self.config.market.min_spread_bps,
            self.config.market.min_spread_bps + abs(self._correction_bias(coin)) * self.config.market.spread_volatility_scale + coin.panic_score * 0.012,
        )
        coin.bid_price = max(self.config.market.min_price, coin.price * (1.0 - spread * 0.5))
        coin.ask_price = min(self.config.market.max_price, coin.price * (1.0 + spread * 0.5))
        mismatch = self.random.uniform(-self.config.market.exchange_mismatch_scale, self.config.market.exchange_mismatch_scale)
        coin.exchange_price = min(self.config.market.max_price, max(self.config.market.min_price, coin.price * (1.0 + mismatch)))

    def creator_sellable_fraction(self, coin_id: str, current_turn: int) -> float:
        coin = self.coins[coin_id]
        age = max(0, current_turn - coin.created_turn)
        if age < self.config.market.creator_lock_turns:
            return 0.0
        unlocked = (age - self.config.market.creator_lock_turns + 1) * self.config.market.creator_unlock_per_turn
        return max(0.0, min(1.0, unlocked))

    def buy(self, coin_id: str, cash_spent: float, current_turn: int) -> tuple[float, float, dict[str, Any]]:
        """
        Buy coins with supply/demand pricing.
        Price increases based on buy pressure relative to liquidity.
        """
        coin = self.coins[coin_id]
        price_before = coin.price

        # Cap spending to max transaction
        capped_spend = min(max(cash_spent, 0.0), self.config.market.max_transaction)

        # Calculate effective liquidity (higher = less price impact)
        # More realistic: liquidity scales with market cap and trading volume
        market_cap = price_before * coin.circulating_supply
        effective_liquidity = (
            coin.liquidity_pool +
            market_cap * 0.15 +  # 15% of market cap provides liquidity
            self.config.market.liquidity_depth
        )

        # Price impact: reduced for more realistic movements
        # Real crypto: large trades move price 1-5%, not 50-90%
        buy_pressure = capped_spend / max(effective_liquidity, 1.0)
        price_impact = math.tanh(buy_pressure * self.config.market.price_impact_buy * 0.3)  # Reduced by 70%

        # Add mean reversion - only for extreme deviations (>50% from intrinsic)
        if coin.intrinsic_value > 0:
            deviation = (coin.price - coin.intrinsic_value) / max(coin.intrinsic_value, 1.0)
            if abs(deviation) > 0.5:  # Only apply when price is >50% away from intrinsic
                mean_reversion = -deviation * 0.002  # Very weak 0.2% reversion
                price_impact += mean_reversion

        # Update price with realistic bounds (max 20% move per turn for growth potential)
        price_impact = max(-0.20, min(0.20, price_impact))
        coin.price = min(
            self.config.market.max_price,
            max(coin.price * 0.80, price_before * (1.0 + price_impact))  # Min 80% of previous
        )

        # Calculate units bought at average price
        avg_price = (price_before + coin.price) / 2.0
        units_bought = capped_spend / max(avg_price, self.config.market.min_price)

        # Update coin metrics
        coin.buy_volume += capped_spend
        coin.trading_volume += capped_spend
        coin.last_traded_turn = current_turn
        coin.market_cap = coin.price * coin.circulating_supply

        # Update liquidity pool (trading adds liquidity)
        coin.liquidity_pool = min(
            coin.liquidity_pool * 1.5,
            coin.liquidity_pool + capped_spend * 0.03  # Reduced liquidity addition
        )

        event = {
            "type": "buy",
            "coin_id": coin_id,
            "cash_spent": round(capped_spend, 4),
            "units_bought": round(units_bought, 4),
            "price_before": round(price_before, 4),
            "price_after": round(coin.price, 4),
            "buy_pressure": round(buy_pressure, 4),
            "pressure": round(buy_pressure, 4),  # Alias for backwards compatibility
            "liquidity": round(effective_liquidity, 4),
        }
        return units_bought, coin.price, event

    def sell(self, coin_id: str, units_sold: float, creator_share: float, current_turn: int) -> tuple[float, float, dict[str, Any]]:
        """
        Sell coins with liquidity-aware pricing.
        Large sales significantly impact price and return less cash.
        """
        coin = self.coins[coin_id]
        price_before = coin.price

        # Cap units to max transaction value
        max_units = self.config.market.max_transaction / max(price_before, self.config.market.min_price)
        capped_units = min(units_sold, max_units)

        # Calculate effective liquidity (same as buy for symmetry)
        market_cap = price_before * coin.circulating_supply
        effective_liquidity = (
            coin.liquidity_pool +
            market_cap * 0.15 +  # 15% of market cap provides liquidity
            self.config.market.liquidity_depth
        )

        # Market share being sold (higher = more impact)
        market_share = capped_units / max(coin.circulating_supply, 1.0)

        # Sell pressure based on units and liquidity
        sell_value = capped_units * price_before
        sell_pressure = sell_value / max(effective_liquidity, 1.0)

        # Price impact: reduced for realistic movements (same as buy)
        # Crash multiplier reduced for less extreme dumps
        impact_multiplier = 1.0 + market_share * (self.config.market.crash_multiplier * 0.4)  # Reduced by 60%
        price_impact = math.tanh(sell_pressure * self.config.market.price_impact_sell * 0.3 * impact_multiplier)  # Reduced by 70%

        # Add mean reversion - only for extreme deviations (>50% from intrinsic)
        if coin.intrinsic_value > 0:
            deviation = (coin.price - coin.intrinsic_value) / max(coin.intrinsic_value, 1.0)
            if abs(deviation) > 0.5:  # Only apply when price is >50% away from intrinsic
                mean_reversion = -deviation * 0.002  # Very weak 0.2% reversion
                price_impact += mean_reversion

        # Update price with realistic bounds (max 20% move per turn for growth potential)
        price_impact = max(-0.20, min(0.20, price_impact))
        coin.price = max(
            self.config.market.min_price,
            min(price_before * 1.20, price_before * (1.0 - price_impact))  # Max 120% of previous
        )

        # Calculate realized cash at AVERAGE price (not current price!)
        # This ensures large sales return less than price * units
        avg_price = (price_before + coin.price) / 2.0
        realized_cash = capped_units * avg_price

        # Update coin metrics
        coin.sell_volume += sell_value
        coin.trading_volume += sell_value
        coin.last_traded_turn = current_turn
        coin.market_cap = coin.price * coin.circulating_supply

        # Reduce liquidity pool (selling drains liquidity, but less severely)
        coin.liquidity_pool = max(
            self.config.market.failure_liquidity_threshold * 0.5,
            coin.liquidity_pool - sell_value * 0.02  # Reduced drain
        )

        event = {
            "type": "sell",
            "coin_id": coin_id,
            "units_sold": round(capped_units, 4),
            "gross_value": round(sell_value, 4),
            "realized_cash": round(realized_cash, 4),
            "price_before": round(price_before, 4),
            "price_after": round(coin.price, 4),
            "market_share": round(market_share, 4),
            "sell_pressure": round(sell_pressure, 4),
            "liquidity": round(effective_liquidity, 4),
            "creator_dump": creator_share >= self.config.market.rug_pull_fraction,
            "suspicious_dump": market_share >= self.config.market.suspicious_dump_fraction or sell_pressure >= 0.35,
        }
        return realized_cash, coin.price, event

    def apply_turn_effects(self, current_turn: int, usage_context: dict[str, dict[str, float]] | None = None) -> list[str]:
        """Apply end-of-turn effects to all coins. Returns list of dead coin IDs."""
        dead_coins: list[str] = []

        for coin in self.coins.values():
            usage = (usage_context or {}).get(coin.coin_id, {})
            holders = usage.get("holders", 0.0)
            miners = usage.get("miners", 0.0)
            held_units = usage.get("held_units", 0.0)
            trades = usage.get("trades", 0.0)
            buy_pressure = usage.get("buy_pressure", 0.0)
            sell_pressure = usage.get("sell_pressure", 0.0)
            buy_pressure += coin.pending_buy_pressure
            sell_pressure += coin.pending_sell_pressure
            hold_ratio = held_units / max(coin.total_supply, 1.0)
            mining_support = min(0.35, miners * 0.04)
            usage_support = min(0.3, trades * 0.015)
            holder_support = min(0.35, holders * 0.025 + hold_ratio * 0.18)

            # Update intrinsic value
            coin.intrinsic_value = max(
                self.config.market.min_price * 1.5,
                min(
                    self.config.market.max_price * 0.6,
                    self.config.market.base_coin_price * self.config.market.intrinsic_value_weight * (1.0 + mining_support + usage_support + holder_support),
                ),
            )

            # Track inactive turns
            inactive_turns = current_turn - coin.last_traded_turn
            recent_volume = coin.volume_history[-1] - coin.volume_history[-2] if len(coin.volume_history) > 1 else coin.trading_volume

            # Update bull score
            drift_scale = self.config.economy.stochastic_price_drift * (1.0 + min(recent_volume / max(self.config.market.max_transaction, 1.0), 2.0) * 0.25)
            net_flow = (buy_pressure - sell_pressure) / max(buy_pressure + sell_pressure, 1.0)
            coin.bull_score = max(-1.0, min(1.0, coin.bull_score * 0.84 + net_flow * 0.72 + usage_support * 0.08 - coin.panic_score * 0.3))

            # ============= PRICE STABILITY CONTROLS =============
            # Reduced multipliers for smoother price movement
            cycle_bias = coin.bull_score * 0.025  # Reduced from 0.045
            drift = self.random.uniform(-drift_scale, drift_scale) * 0.5  # Reduced drift by 50%
            correction_bias = self._correction_bias(coin) * 0.6  # Reduced correction
            rebound_bias = self._soft_floor_support(coin)
            overextension_pressure = self._overextension_pressure(coin)
            uptrend_strength = self._uptrend_strength(coin) * 0.7  # Reduced uptrend tracking
            market_maker_push = rebound_bias * (1.0 + coin.panic_score * self.config.market.passive_liquidity_panic_boost * 0.5)  # Reduced
            overextension_drag = overextension_pressure * self.random.uniform(0.22, 0.85)
            if uptrend_strength > 0.035:
                overextension_drag += min(0.05, uptrend_strength * 0.35)  # Reduced from 0.08, 0.55

            # Calculate raw price change
            price_before = coin.price
            coin.price = coin.price * (1.0 + drift + cycle_bias + correction_bias + market_maker_push - overextension_drag)

            # Passive price bleed for inactive coins
            if self.config.market.passive_price_bleed_enabled and inactive_turns >= self.config.market.inactive_decay_turns:
                bleed = self.config.market.inactive_price_decay * max(0.25, 1.0 - rebound_bias)
                # Cap bleed to prevent sudden drops
                bleed = min(bleed, self.config.market.max_turn_price_move * 0.5)
                coin.price = max(
                    self.config.market.min_price,
                    coin.price * (1.0 - bleed),
                )
                coin.trading_volume *= 0.98
            
            # Liquidity dynamics
            if coin.panic_score < 0.28:
                coin.passive_liquidity = min(
                    self.config.market.passive_liquidity_base * 2.5,
                    coin.passive_liquidity + self.config.market.passive_liquidity_base * 0.035,
                )
            else:
                coin.passive_liquidity = min(
                    self.config.market.passive_liquidity_base * 3.0,
                    coin.passive_liquidity + self.config.market.passive_liquidity_base * 0.02 * (1.0 + coin.panic_score * self.config.market.passive_liquidity_panic_boost),
                )
            
            # ============= LIFECYCLE TRACKING =============
            # Update peak price
            if coin.price > coin.peak_price:
                coin.peak_price = coin.price
            
            # Determine lifecycle stage
            age = len(coin.price_history)
            price_ratio = coin.price / max(coin.peak_price, self.config.market.min_price) if coin.peak_price > 0 else 1.0
            
            if age < 5:
                coin.lifecycle_stage = "birth"
            elif coin.price > coin.peak_price * 0.95 and coin.bull_score > 0.2:
                coin.lifecycle_stage = "growth"
            elif price_ratio >= 0.85:
                coin.lifecycle_stage = "peak"
            elif price_ratio < 0.85 and price_ratio >= 0.5:
                coin.lifecycle_stage = "decline"
            else:
                coin.lifecycle_stage = "death"
            
            # ============= COIN DEATH MECHANICS =============
            if self.config.market.coin_death_enabled:
                # Track turns below minimum price
                if coin.price <= self.config.market.min_price * 1.5:
                    coin.turns_below_min_price += 1
                else:
                    coin.turns_below_min_price = max(0, coin.turns_below_min_price - 1)
                
                # Track low activity turns
                if recent_volume < self.config.market.min_trading_activity_per_turn:
                    coin.turns_low_activity += 1
                else:
                    coin.turns_low_activity = max(0, coin.turns_low_activity - 1)
                
                # Check death conditions
                should_die = False
                death_reason = ""
                
                # Condition 1: Price too low for too long
                if coin.turns_below_min_price >= self.config.market.coin_death_price_turns:
                    should_die = True
                    death_reason = "price_below_minimum"
                
                # Condition 2: No trading activity for too long
                elif coin.turns_low_activity >= self.config.market.coin_death_activity_turns:
                    should_die = True
                    death_reason = "no_activity"
                
                # Condition 3: Liquidity too low
                elif coin.passive_liquidity <= self.config.market.coin_death_liquidity_threshold:
                    should_die = True
                    death_reason = "low_liquidity"
                
                if should_die:
                    coin.death_turn = current_turn
                    coin.lifecycle_stage = "death"
                    dead_coins.append(coin.coin_id)
            
            # Mean reversion for non-dying coins
            failure_state = (
                coin.consecutive_sell_pressure >= self.config.market.sustained_failure_turns
                and miners <= 0
                and holders <= 0
                and coin.passive_liquidity <= self.config.market.failure_liquidity_threshold
            )
            
            if not failure_state and coin.price < coin.intrinsic_value * 0.75:
                coin.price = min(
                    self.config.market.max_price,
                    coin.price + (coin.intrinsic_value - coin.price) * min(0.22, 0.08 + rebound_bias),
                )
            elif not failure_state and overextension_pressure > 0.06 and uptrend_strength > 0.02:
                crash_bias = max(0.0, sell_pressure - buy_pressure) / max(buy_pressure + sell_pressure, 1.0)
                crash_risk = min(
                    0.28,
                    overextension_pressure * 0.65
                    + max(0.0, coin.bull_score) * 0.12
                    + coin.panic_score * 0.16
                    + crash_bias * 0.22,
                )
                if sell_pressure > buy_pressure * 1.05 or coin.panic_score > 0.16 or self.random.random() < crash_risk:
                    coin.price = max(
                        coin.intrinsic_value * 0.88,
                        coin.price * (1.0 - min(0.24, crash_risk * self.random.uniform(0.5, 1.0) + uptrend_strength * 0.35)),
                    )
                    coin.panic_score = min(1.0, coin.panic_score + min(0.22, crash_risk * 0.7))
            elif failure_state:
                coin.price = max(
                    self.config.market.min_price * 0.6,
                    coin.price * (1.0 - self.config.market.inactive_price_decay * 1.6),
                )
            
            # Clamp values
            coin.price = min(max(coin.price, self.config.market.min_price), self.config.market.max_price)
            coin.total_supply = min(coin.total_supply, self.config.market.max_supply)
            coin.circulating_supply = min(coin.circulating_supply, coin.total_supply)
            
            # Decay scores
            coin.panic_score = max(0.0, coin.panic_score * 0.9)
            coin.pending_buy_pressure *= self.config.market.pending_pressure_decay
            coin.pending_sell_pressure *= self.config.market.pending_pressure_decay

            # Update attention score based on activity
            activity_score = min(1.0, (trades + holders + miners) / 10.0)
            coin.attention_score = coin.attention_score * (1.0 - self.config.market.attention_decay_rate) + activity_score * self.config.market.attention_decay_rate

            self._update_quotes(coin)

            # Record history
            coin.price_history.append(coin.price)
            coin.volume_history.append(coin.trading_volume)
        
        # Remove dead coins
        for coin_id in dead_coins:
            del self.coins[coin_id]
        
        return dead_coins

    def get_attention_coins(self, agent_id: int, holdings: dict[str, float], current_turn: int) -> list[str]:
        """
        Get the subset of coins an agent is paying attention to.
        Agents prefer: coins they hold, coins with recent activity, coins with high attention scores.
        """
        if not self.coins:
            return []
        
        max_slots = self.config.market.agent_attention_slots
        scored_coins: list[tuple[str, float]] = []
        
        for coin_id, coin in self.coins.items():
            score = coin.attention_score
            
            # Boost for held coins
            if coin_id in holdings and holdings[coin_id] > 0:
                score += self.config.market.attention_hold_bias
            
            # Boost for recently active coins
            if current_turn - coin.last_traded_turn < 5:
                score += self.config.market.attention_activity_bias
            
            # Boost for high volume coins
            if coin.trading_volume > self.config.market.max_transaction:
                score += 0.2
            
            scored_coins.append((coin_id, score))
        
        # Sort by score descending
        scored_coins.sort(key=lambda x: x[1], reverse=True)
        
        # Return top coins up to attention limit
        return [coin_id for coin_id, _ in scored_coins[:max_slots]]

    def update_dynamic_coin_limits(self, num_agents: int) -> None:
        """
        Update max_active_coins based on agent population.
        Target: ~1 coin per 2-4 agents.
        """
        target_coins = max(3, int(num_agents * self.config.market.target_coin_per_agent_ratio))
        max_coins = max(target_coins, self.config.market.max_coins_per_agent * num_agents)
        self.config.market.max_active_coins = min(max_coins, 25)  # Hard cap at 25

    def should_allow_coin_creation(self) -> bool:
        """Check if a new coin can be created based on population limits."""
        return len(self.coins) < self.config.market.max_active_coins

    def get_coin_creation_bias(self, num_agents: int) -> float:
        """
        Return a bias factor for coin creation based on current population.
        Returns 1.0 when under target, 0.0 when at/over limit.
        """
        target_coins = max(3, int(num_agents * self.config.market.target_coin_per_agent_ratio))
        current_coins = len(self.coins)
        
        if current_coins >= target_coins:
            return 0.0
        elif current_coins >= self.config.market.max_active_coins:
            return 0.0
        else:
            # Linear interpolation: more bias when far from target
            return min(1.0, (target_coins - current_coins) / max(target_coins, 1))

    def coin_stats(self) -> list[dict[str, Any]]:
        stats: list[dict[str, Any]] = []
        for coin in self.coins.values():
            previous_price = coin.price_history[-2] if len(coin.price_history) > 1 else coin.price
            trend = (coin.price - previous_price) / max(previous_price, 1e-6)
            stats.append(
                {
                    "coin_id": coin.coin_id,
                    "creator_id": coin.creator_id,
                    "price": coin.price,
                    "volume": coin.trading_volume,
                    "trend": trend,
                    "age": len(coin.price_history) - 1,
                    "intrinsic_value": coin.intrinsic_value,
                    "passive_liquidity": coin.passive_liquidity,
                    "bull_score": coin.bull_score,
                    "bid_price": coin.bid_price,
                    "ask_price": coin.ask_price,
                    "exchange_price": coin.exchange_price,
                    "lifecycle_stage": coin.lifecycle_stage,
                    "attention_score": round(coin.attention_score, 3),
                }
            )
        return sorted(stats, key=lambda item: item["volume"], reverse=True)
