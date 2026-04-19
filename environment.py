from __future__ import annotations

import csv
import json
import math
import random
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator

from agent import Action, ActionType, EvolutionAgent, StrategyGenome
from config import GameConfig
from market import Market
from event_detector import EventDetector


@dataclass(slots=True)
class AgentState:
    agent_id: int
    cash: float
    mining_power: float
    mining_allocation: str | None = None
    holdings: dict[str, float] = field(default_factory=dict)
    cost_basis: dict[str, float] = field(default_factory=dict)
    leverage_positions: dict[str, dict[str, float]] = field(default_factory=dict)
    coins_created: list[str] = field(default_factory=list)
    mining_profit_history: deque[float] = field(default_factory=lambda: deque(maxlen=8))
    work_streak: int = 0
    reaction_speed: float = 0.45
    risk_trigger_threshold: float = 0.08
    sentiment_memory: float = 0.0
    volatility_memory: float = 0.0
    rally_memory: float = 0.0
    drop_memory: float = 0.0
    cooldown: int = 0
    creator_effectiveness: float = 1.0
    failed_coins: list[str] = field(default_factory=list)


class CryptoTradingEnvironment:
    def __init__(self, config: GameConfig, agents: list[EvolutionAgent] | None = None) -> None:
        self.config = config
        self.market = Market(config)
        self.random = random.Random(config.seed)
        self.agents = agents or [
            EvolutionAgent(i, StrategyGenome.random(random.Random(config.seed + i + 1)), config.seed + i + 1)
            for i in range(config.num_agents)
        ]
        self.agent_states: dict[int, AgentState] = {}
        self.turn = 0
        self.episode = 0
        self.event_log: list[dict[str, Any]] = []
        self.replay: list[dict[str, Any]] = []
        self.metrics: dict[str, Any] = {}
        self.current_total_hashrate = 0.0
        self.current_difficulty = config.mining.base_difficulty
        self.current_electricity_price = config.mining.base_electricity_price
        self.log_listener: Callable[[str], None] | None = None
        self.event_detector = EventDetector()
        self._reset_metrics()

    def set_agents(self, agents: list[EvolutionAgent]) -> None:
        self.agents = agents

    def _reset_metrics(self) -> None:
        self.metrics = {
            "episode_rewards": defaultdict(list),
            "net_worth_history": defaultdict(list),
            "agent_coin_ownership_history": defaultdict(list),  # Track coins per agent per turn
            "rug_pulls": Counter(),
            "action_distribution": Counter(),
            "agent_action_distribution": defaultdict(Counter),
            "coins_created_history": [],
            "wealth_snapshots": [],
            "suspicious_events": [],
            "average_reward": [],
            "network_hashrate_history": [],
            "electricity_price_history": [],
            "difficulty_history": [],
            "mining_profit_history": defaultdict(list),
            "mining_participation_history": defaultdict(list),
            "average_mining_profit_history": [],
        }

    def reset(self) -> None:
        self.market.reset()
        self.turn = 0
        self.event_log = []
        self.replay = []
        self.event_detector.reset()

        # Initialize dynamic coin population limits based on agent count
        self.market.update_dynamic_coin_limits(len(self.agents))
        
        self.agent_states = {
            agent.agent_id: AgentState(
                agent_id=agent.agent_id,
                cash=self.config.starting_money,
                mining_power=self.random.uniform(
                    self.config.mining.min_starting_mining_power,
                    self.config.mining.max_starting_mining_power,
                ),
                reaction_speed=agent.genome.reaction_speed,
                risk_trigger_threshold=agent.genome.risk_trigger_threshold,
            )
            for agent in self.agents
        }
        self.current_total_hashrate = 0.0
        self.current_difficulty = self.config.mining.base_difficulty
        self.current_electricity_price = self.config.mining.base_electricity_price
        for agent in self.agents:
            agent.reset_episode()

    def _clamp(self, value: float, lower: float, upper: float) -> float:
        if not math.isfinite(value):
            return upper if value > 0 else lower
        return max(lower, min(upper, value))

    def _normalize_wealth_delta(self, worth_before: float, worth_after: float) -> float:
        safe_before = max(0.0, worth_before)
        safe_after = max(0.0, worth_after)
        baseline = max(math.log1p(self.config.starting_money), 1e-6)
        delta = (math.log1p(safe_after) - math.log1p(safe_before)) / baseline
        return self._clamp(delta, -self.config.evolution.max_turn_reward, self.config.evolution.max_turn_reward)

    def _inflation_multiplier(self) -> float:
        return max(0.25, math.exp(-self.turn * self.config.economy.inflation_rate_per_turn))

    def _current_work_income(self, agent_id: int) -> float:
        state = self.agent_states[agent_id]
        streak_factor = max(
            self.config.economy.min_work_income_factor,
            1.0 - state.work_streak * self.config.economy.work_decay_per_streak,
        )
        return self.config.mining.work_income * self._inflation_multiplier() * streak_factor

    def _apply_transaction_fee(self, gross_amount: float) -> float:
        return gross_amount * self.config.economy.transaction_fee_rate

    def _update_cost_basis_buy(self, state: AgentState, coin_id: str, added_units: float, total_spent: float) -> None:
        if added_units <= 0:
            return
        previous_units = max(0.0, state.holdings.get(coin_id, 0.0) - added_units)
        previous_basis = state.cost_basis.get(coin_id, self.market.coins[coin_id].price)
        blended_cost = ((previous_units * previous_basis) + total_spent) / max(previous_units + added_units, 1e-6)
        state.cost_basis[coin_id] = blended_cost

    def _update_cost_basis_sell(self, state: AgentState, coin_id: str) -> None:
        if state.holdings.get(coin_id, 0.0) <= 0.01:
            state.cost_basis.pop(coin_id, None)

    def _max_unrealized_gain(self, agent_id: int) -> float:
        state = self.agent_states[agent_id]
        best_gain = 0.0
        for coin_id, units in state.holdings.items():
            if units <= 0.01 or coin_id not in self.market.coins:
                continue
            basis = max(state.cost_basis.get(coin_id, self.market.coins[coin_id].price), self.config.market.min_price)
            gain = (self.market.coins[coin_id].price - basis) / basis
            best_gain = max(best_gain, gain)
        return best_gain

    def _max_recent_drop(self) -> float:
        max_drop = 0.0
        for coin in self.market.coins.values():
            if len(coin.price_history) < 2:
                continue
            previous = max(coin.price_history[-2], self.config.market.min_price)
            drop = (coin.price_history[-1] - previous) / previous
            max_drop = min(max_drop, drop)
        return abs(min(0.0, max_drop))

    def _max_recent_rally(self) -> float:
        max_rally = 0.0
        for coin in self.market.coins.values():
            if len(coin.price_history) < 2:
                continue
            previous = max(coin.price_history[-2], self.config.market.min_price)
            rally = (coin.price_history[-1] - previous) / previous
            max_rally = max(max_rally, rally)
        return max_rally

    def _finalize_reward(self, reward: float, event: dict[str, Any], worth_before: float, worth_after: float) -> float:
        reward += self._normalize_wealth_delta(worth_before, worth_after)
        wealth_jump = (worth_after - worth_before) / max(self.config.starting_money, 1.0)
        if abs(wealth_jump) > self.config.evolution.spike_penalty_threshold:
            reward -= (abs(wealth_jump) - self.config.evolution.spike_penalty_threshold) * self.config.evolution.spike_penalty_scale
            event["spike_penalty"] = round(
                (abs(wealth_jump) - self.config.evolution.spike_penalty_threshold) * self.config.evolution.spike_penalty_scale,
                4,
            )
            self._record_suspicious(event, "economic_spike")
        return self._clamp(reward, -self.config.evolution.max_turn_reward, self.config.evolution.max_turn_reward)

    def _creator_sellable_units(self, agent_id: int, coin_id: str, current_units: float) -> float:
        if coin_id not in self.market.coins:
            return 0.0
        coin = self.market.coins[coin_id]
        if coin.creator_id != agent_id:
            return current_units
        unlocked_fraction = self.market.creator_sellable_fraction(coin_id, self.turn)
        return current_units * unlocked_fraction

    def _log_capped_event(self, event: dict[str, Any]) -> None:
        if event.get("capped_transaction"):
            self._record_suspicious(event, "transaction_cap_hit")
        if event.get("capped_price"):
            self._record_suspicious(event, "price_cap_hit")

    def _is_action_disabled(self, action_type: ActionType) -> bool:
        """Check if an action type is disabled in the config."""
        # Normalize to uppercase for comparison since GUI stores uppercase
        return action_type.value.upper() in self.config.disabled_actions

    def available_actions(self, agent_id: int) -> list[Action]:
        state = self.agent_states[agent_id]
        
        # Base actions (WORK, MINE, HOLD) - only add if not disabled
        actions = []
        if not self._is_action_disabled(ActionType.WORK):
            actions.append(Action(ActionType.WORK))
        if not self._is_action_disabled(ActionType.MINE):
            actions.append(Action(ActionType.MINE))
        if not self._is_action_disabled(ActionType.HOLD):
            actions.append(Action(ActionType.HOLD))

        if state.cooldown > 0:
            if not self._is_action_disabled(ActionType.SELL):
                for coin_id, units in state.holdings.items():
                    sellable_units = self._creator_sellable_units(agent_id, coin_id, units)
                    if sellable_units > 0.5:
                        actions.append(Action(ActionType.SELL, coin_id=coin_id, fraction=0.25))
                        actions.append(Action(ActionType.SELL, coin_id=coin_id, fraction=0.5))
            return actions

        # Use attention system to limit which coins the agent can see
        attention_coins = set(self.market.get_attention_coins(agent_id, state.holdings, self.turn))

        # Coin creation - check dynamic limits
        create_total_cost = self.config.market.create_coin_cost + self.config.market.create_liquidity_cost
        if (
            state.cash >= create_total_cost
            and self.market.should_allow_coin_creation()
            and len(state.coins_created) < self.config.market.max_coins_per_agent
            and not self._is_action_disabled(ActionType.CREATE)
        ):
            actions.append(Action(ActionType.CREATE))

        # Only show actions for coins in agent's attention set
        if attention_coins:
            for coin_id in attention_coins:
                if coin_id not in self.market.coins:
                    continue
                
                if not self._is_action_disabled(ActionType.MINE):
                    actions.append(Action(ActionType.MINE, coin_id=coin_id))
                if not self._is_action_disabled(ActionType.TREND):
                    actions.append(Action(ActionType.TREND, coin_id=coin_id, fraction=0.25))
                if not self._is_action_disabled(ActionType.SPECULATE):
                    actions.append(Action(ActionType.SPECULATE, coin_id=coin_id, fraction=0.25))
                if not self._is_action_disabled(ActionType.ARBITRAGE):
                    actions.append(Action(ActionType.ARBITRAGE, coin_id=coin_id, fraction=0.2))
                if not self._is_action_disabled(ActionType.MARKET_MAKE):
                    actions.append(Action(ActionType.MARKET_MAKE, coin_id=coin_id, fraction=0.2))
                if state.cash >= 4.0 and not self._is_action_disabled(ActionType.LEVERAGE):
                    actions.append(Action(ActionType.LEVERAGE, coin_id=coin_id, fraction=0.18))

            if not self._is_action_disabled(ActionType.BUY):
                for coin_id in attention_coins:
                    if coin_id not in self.market.coins:
                        continue
                    for fraction in self.config.market.buy_fractions:
                        spend = min(state.cash * fraction, self.config.market.max_transaction)
                        if spend >= 1.0:
                            actions.append(Action(ActionType.BUY, coin_id=coin_id, fraction=fraction))

            if not self._is_action_disabled(ActionType.SELL):
                for coin_id, units in state.holdings.items():
                    sellable_units = self._creator_sellable_units(agent_id, coin_id, units)
                    if sellable_units <= 0:
                        continue
                    for fraction in self.config.market.sell_fractions:
                        if sellable_units * fraction >= 0.5:
                            actions.append(Action(ActionType.SELL, coin_id=coin_id, fraction=fraction))

        return actions

    def _perceived_coin_stats(self, agent_id: int) -> list[dict[str, Any]]:
        perceived: list[dict[str, Any]] = []
        # Only show coins the agent is paying attention to
        attention_coins = set(self.market.get_attention_coins(agent_id, self.agent_states[agent_id].holdings, self.turn))
        
        for coin_id, coin in self.market.coins.items():
            if coin_id not in attention_coins:
                continue
            price_index = max(0, len(coin.price_history) - 2)
            observed_price = coin.price_history[price_index]
            previous_index = max(0, price_index - 1)
            previous_price = coin.price_history[previous_index]
            trend = (observed_price - previous_price) / max(previous_price, 1e-6)
            noise_rng = random.Random(self.config.seed * 1000 + agent_id * 97 + self.turn * 17 + sum(ord(ch) for ch in coin.coin_id))
            noisy_price = max(self.config.market.min_price, observed_price * (1.0 + noise_rng.uniform(-0.025, 0.025)))
            noisy_trend = trend + noise_rng.uniform(-0.015, 0.015)
            perceived.append(
                {
                    "coin_id": coin.coin_id,
                    "creator_id": coin.creator_id,
                    "price": noisy_price,
                    "volume": coin.trading_volume,
                    "trend": noisy_trend,
                    "age": len(coin.price_history) - 1,
                    "bid_price": coin.bid_price,
                    "ask_price": coin.ask_price,
                    "exchange_price": coin.exchange_price,
                }
            )
        return sorted(perceived, key=lambda item: item["volume"], reverse=True)[:5]

    def observe(self, agent_id: int) -> dict[str, Any]:
        state = self.agent_states[agent_id]
        holdings_value = self.holdings_value(agent_id)
        visible_coins = self._perceived_coin_stats(agent_id)[:5]
        rising_coins = sum(1 for coin in visible_coins if coin["trend"] > 0.04)
        market_sentiment = sum(coin["trend"] for coin in visible_coins) / max(len(visible_coins), 1)
        market_volatility = (
            sum(abs(coin["trend"]) for coin in visible_coins) / max(len(visible_coins), 1) if visible_coins else 0.0
        )
        smoothing = state.reaction_speed
        state.sentiment_memory = state.sentiment_memory * (1.0 - smoothing) + market_sentiment * smoothing
        state.volatility_memory = state.volatility_memory * (1.0 - smoothing) + market_volatility * smoothing
        rally_now = self._max_recent_rally()
        drop_now = self._max_recent_drop()
        state.rally_memory = state.rally_memory * (1.0 - smoothing) + rally_now * smoothing
        state.drop_memory = state.drop_memory * (1.0 - smoothing) + drop_now * smoothing
        richest_known_target = 0.0
        highest_known_mining_power = 0.0
        for target_id, other_state in self.agent_states.items():
            if target_id == agent_id:
                continue
            richest_known_target = max(richest_known_target, other_state.cash)
            highest_known_mining_power = max(highest_known_mining_power, other_state.mining_power)
        return {
            "cash": state.cash,
            "mining_power": state.mining_power,
            "holdings_value": holdings_value,
            "coins_owned": len([units for units in state.holdings.values() if units > 0]),
            "coins_created": len(state.coins_created),
            "visible_coins": visible_coins,
            "rising_coins": rising_coins,
            "market_sentiment": state.sentiment_memory,
            "market_volatility": state.volatility_memory,
            "richest_known_target": richest_known_target,
            "highest_known_mining_power": highest_known_mining_power,
            "electricity_price": self.current_electricity_price,
            "network_hashrate": self.current_total_hashrate,
            "difficulty": self.current_difficulty,
            "inflation_multiplier": self._inflation_multiplier(),
            "current_work_income": self._current_work_income(agent_id),
            "max_unrealized_gain": self._max_unrealized_gain(agent_id),
            "recent_drop": state.drop_memory,
            "recent_rally": state.rally_memory,
            "leverage_exposure": sum(position.get("collateral", 0.0) for position in state.leverage_positions.values()),
            "cooldown": state.cooldown,
            "estimated_mining_margin": self.config.mining.block_reward_value / max(self.current_difficulty, 1e-6)
            - (state.mining_power * self.current_electricity_price),
            "last_mining_profit": state.mining_profit_history[-1] if state.mining_profit_history else 0.0,
            "turn": self.turn,
        }

    def holdings_value(self, agent_id: int) -> float:
        state = self.agent_states[agent_id]
        value = 0.0
        for coin_id, units in state.holdings.items():
            if coin_id in self.market.coins:
                coin = self.market.coins[coin_id]
                effective_units = units
                if coin.creator_id == agent_id:
                    unlocked_fraction = self.market.creator_sellable_fraction(coin_id, self.turn)
                    unlocked_units = units * unlocked_fraction
                    locked_units = max(0.0, units - unlocked_units)
                    effective_units = unlocked_units + locked_units * 0.12
                value += effective_units * coin.price
        return self._clamp(value, 0.0, self.config.market.max_supply * self.config.market.max_price * max(len(self.market.coins), 1))

    def net_worth(self, agent_id: int) -> float:
        state = self.agent_states[agent_id]
        return state.cash + self.holdings_value(agent_id)

    def _log(self, message: str) -> None:
        if self.log_listener is not None:
            self.log_listener(message)
        if self.config.debug:
            print(message)

    def _record_suspicious(self, turn_event: dict[str, Any], reason: str) -> None:
        entry = {"episode": self.episode, "turn": self.turn, "reason": reason, **turn_event}
        self.metrics["suspicious_events"].append(entry)
        self._log(f"[suspicious] episode={self.episode} turn={self.turn} reason={reason} event={turn_event}")

    def apply_action(self, agent_id: int, action: Action) -> tuple[float, dict[str, Any]]:
        actor = self.agent_states[agent_id]
        reward = 0.0
        event: dict[str, Any] = {"agent_id": agent_id, "action": action.key}
        actor_cash_before = actor.cash
        applied_cooldown = 0

        if action.action_type == ActionType.WORK:
            income = self._current_work_income(agent_id)
            actor.cash += income
            actor.work_streak += 1
            reward += 0.05
            event["income"] = round(income, 4)
            event["work_streak"] = actor.work_streak

        elif action.action_type == ActionType.CREATE:
            # Use dynamic coin population limits
            if not self.market.should_allow_coin_creation() or len(actor.coins_created) >= self.config.market.max_coins_per_agent:
                event["blocked"] = "coin_limit"
                reward -= 0.1
                return reward, event
            actor.work_streak = 0
            actor.cash -= self.config.market.create_coin_cost + self.config.market.create_liquidity_cost
            coin, creator_supply = self.market.create_coin(agent_id, self.turn)
            actor.holdings[coin.coin_id] = actor.holdings.get(coin.coin_id, 0.0) + creator_supply
            actor.coins_created.append(coin.coin_id)
            reward += self.config.evolution.create_reward_bonus
            event["coin_id"] = coin.coin_id
            event["creator_supply"] = round(creator_supply, 4)
            event["liquidity_seeded"] = round(coin.initial_liquidity, 4)

        elif action.action_type == ActionType.BUY and action.coin_id and action.fraction:
            actor.work_streak = 0
            total_spend = min(actor.cash * action.fraction, self.config.market.max_transaction)
            
            # "Buy the dip" incentive - reduced fees when price is below intrinsic value
            coin = self.market.coins[action.coin_id]
            price_ratio = coin.price / max(coin.intrinsic_value, coin.price)
            if price_ratio < 0.7:  # Price is significantly below intrinsic value
                fee_rate = self.config.economy.transaction_fee_rate * 0.3  # 70% fee reduction
            elif price_ratio < 0.9:
                fee_rate = self.config.economy.transaction_fee_rate * 0.6  # 40% fee reduction
            else:
                fee_rate = self.config.economy.transaction_fee_rate
            
            fee = total_spend * fee_rate
            spend = max(0.0, total_spend - fee)
            actor.cash -= total_spend
            units, new_price, market_event = self.market.buy(action.coin_id, spend, self.turn)
            actor.holdings[action.coin_id] = actor.holdings.get(action.coin_id, 0.0) + units
            self._update_cost_basis_buy(actor, action.coin_id, units, total_spend)
            reward += self._clamp(market_event["pressure"], 0.0, 0.2)
            
            # Extra reward for buying undervalued coins
            if price_ratio < 0.7:
                reward += 0.15  # Bonus for buying the dip
            event.update(market_event)
            event["new_price"] = round(new_price, 4)
            event["transaction_fee"] = round(fee, 4)
            if action.fraction >= 0.35:
                applied_cooldown = 1
            self._log_capped_event(event)

        elif action.action_type == ActionType.SELL and action.coin_id and action.fraction:
            actor.work_streak = 0
            current_units = actor.holdings.get(action.coin_id, 0.0)
            basis = actor.cost_basis.get(action.coin_id, self.market.coins[action.coin_id].price)
            gain_ratio = (self.market.coins[action.coin_id].price - basis) / max(basis, self.config.market.min_price)
            sellable_units = self._creator_sellable_units(agent_id, action.coin_id, current_units)
            units_to_sell = sellable_units * action.fraction
            actor_share = current_units / max(self.market.coins[action.coin_id].circulating_supply, 1.0)
            price_before = self.market.coins[action.coin_id].price
            realized_cash, new_price, market_event = self.market.sell(action.coin_id, units_to_sell, actor_share, self.turn)
            
            # Check for rug pull
            fraction_sold = market_event["units_sold"] / max(current_units, 1.0)
            self.event_detector.check_rug_pull(
                agent_id, action.coin_id, fraction_sold,
                price_before, new_price, self
            )
            fee = self._apply_transaction_fee(realized_cash)
            net_cash = max(0.0, realized_cash - fee)
            actor.cash += net_cash
            actor.holdings[action.coin_id] = max(0.0, current_units - market_event["units_sold"])
            self._update_cost_basis_sell(actor, action.coin_id)
            reward += self._clamp(net_cash / max(self.config.starting_money, 1.0), 0.0, 0.35)
            reward += self._clamp(gain_ratio * 0.22, 0.0, 0.22)
            if market_event["creator_dump"] and self.market.coins[action.coin_id].creator_id == agent_id:
                self.metrics["rug_pulls"][agent_id] += 1
                self._record_suspicious(event | market_event, "rug_pull")
            if market_event["suspicious_dump"]:
                reward += self.config.evolution.dump_penalty
                self._record_suspicious(event | market_event, "large_dump")
            event.update(market_event)
            event["new_price"] = round(new_price, 4)
            event["transaction_fee"] = round(fee, 4)
            event["realized_cash_net"] = round(net_cash, 4)
            event["max_gain_triggered"] = gain_ratio >= self.config.economy.profit_take_threshold_low
            if gain_ratio >= self.config.economy.profit_take_threshold_mid or action.fraction >= 0.5:
                applied_cooldown = 1
            if self.market.coins[action.coin_id].creator_id == agent_id and sellable_units < current_units:
                event["creator_locked_units"] = round(current_units - sellable_units, 4)
            self._log_capped_event(event)

        elif action.action_type == ActionType.MINE:
            actor.work_streak = 0
            raise ValueError("Mining actions are resolved in a turn-level batch")

        elif action.action_type == ActionType.HOLD:
            actor.work_streak = 0
            reward -= 0.01
            event["status"] = "holding"

        elif action.action_type == ActionType.TREND and action.coin_id and action.fraction:
            actor.work_streak = 0
            coin_stats = next((coin for coin in self.market.coin_stats() if coin["coin_id"] == action.coin_id), None)
            trend = coin_stats["trend"] if coin_stats is not None else 0.0
            if trend > 0.01:
                total_spend = min(actor.cash * action.fraction, self.config.market.max_transaction)
                fee = self._apply_transaction_fee(total_spend)
                spend = max(0.0, total_spend - fee)
                actor.cash -= total_spend
                units, new_price, market_event = self.market.buy(action.coin_id, spend, self.turn)
                actor.holdings[action.coin_id] = actor.holdings.get(action.coin_id, 0.0) + units
                self._update_cost_basis_buy(actor, action.coin_id, units, total_spend)
                reward += 0.08 + self._clamp(trend * 3.0, 0.0, 0.35)
                event.update(market_event)
                event["new_price"] = round(new_price, 4)
                event["strategy"] = "trend_follow_buy"
                event["transaction_fee"] = round(fee, 4)
                applied_cooldown = 1
            else:
                basis = actor.cost_basis.get(action.coin_id, self.market.coins[action.coin_id].price)
                gain_ratio = (self.market.coins[action.coin_id].price - basis) / max(basis, self.config.market.min_price)
                sellable_units = self._creator_sellable_units(agent_id, action.coin_id, actor.holdings.get(action.coin_id, 0.0))
                units_to_sell = sellable_units * action.fraction
                realized_cash, new_price, market_event = self.market.sell(action.coin_id, units_to_sell, 0.0, self.turn)
                fee = self._apply_transaction_fee(realized_cash)
                net_cash = max(0.0, realized_cash - fee)
                actor.cash += net_cash
                actor.holdings[action.coin_id] = max(0.0, actor.holdings.get(action.coin_id, 0.0) - market_event["units_sold"])
                self._update_cost_basis_sell(actor, action.coin_id)
                reward += self._clamp(net_cash / max(self.config.starting_money, 1.0), 0.0, 0.18)
                reward += self._clamp(gain_ratio * 0.18, 0.0, 0.16)
                event.update(market_event)
                event["new_price"] = round(new_price, 4)
                event["strategy"] = "trend_exit"
                event["transaction_fee"] = round(fee, 4)
                event["realized_cash_net"] = round(net_cash, 4)
                event["max_gain_triggered"] = gain_ratio >= self.config.economy.profit_take_threshold_low
                if gain_ratio >= self.config.economy.profit_take_threshold_mid:
                    applied_cooldown = 1

        elif action.action_type == ActionType.SPECULATE and action.coin_id and action.fraction:
            actor.work_streak = 0
            coin = self.market.coins[action.coin_id]
            stake = min(actor.cash * max(action.fraction, 0.1), self.config.market.max_transaction * 0.85)
            actor.cash -= stake
            fee = self._apply_transaction_fee(stake)
            direction = 1 if self.random.random() < 0.5 else -1
            shock = self.random.uniform(-0.22, 0.24) + coin.price_history[-1] / max(coin.price_history[0], 1.0) * 0.01 - 0.01
            pnl_ratio = self._clamp(direction * shock, -0.85, 1.1)
            pnl = stake * pnl_ratio - fee
            actor.cash += max(0.0, stake + pnl)
            reward += math.tanh(pnl / max(self.config.starting_money * 0.4, 1.0))
            event.update(
                {
                    "coin_id": action.coin_id,
                    "stake": round(stake, 4),
                    "transaction_fee": round(fee, 4),
                    "direction": "long" if direction > 0 else "short",
                    "pnl": round(pnl, 4),
                    "return_ratio": round(pnl_ratio, 4),
                }
            )
            applied_cooldown = 1

        elif action.action_type == ActionType.ARBITRAGE and action.coin_id and action.fraction:
            actor.work_streak = 0
            coin = self.market.coins[action.coin_id]
            mispricing = abs(coin.exchange_price - coin.price) / max(coin.price, self.config.market.min_price)
            net_edge = mispricing - self.config.market.arbitrage_cost_threshold
            stake = min(actor.cash * max(action.fraction, 0.1), self.config.market.max_transaction * 0.4)
            profit = stake * self._clamp(net_edge * 0.75, 0.0, 0.05)
            actor.cash += profit
            reward += math.tanh(profit / max(self.config.starting_money * 0.25, 1.0))
            event.update(
                {
                    "coin_id": action.coin_id,
                    "stake": round(stake, 4),
                    "profit": round(profit, 4),
                    "mispricing": round(mispricing, 4),
                    "net_edge": round(net_edge, 4),
                }
            )
            applied_cooldown = 2

        elif action.action_type == ActionType.LEVERAGE and action.coin_id and action.fraction:
            actor.work_streak = 0
            coin = self.market.coins[action.coin_id]
            trend = (coin.price_history[-1] - coin.price_history[-2]) / max(coin.price_history[-2], self.config.market.min_price) if len(coin.price_history) > 1 else 0.0
            leverage_multiple = self._clamp(2.0 + (self.net_worth(agent_id) / max(self.config.starting_money, 1.0)) * 0.45, 2.0, 5.5)
            stake = min(actor.cash * max(action.fraction, 0.1), self.config.market.max_transaction * 0.5)
            trading_fee = stake * (0.012 + leverage_multiple * 0.005)
            total_upfront = stake + trading_fee
            if total_upfront > actor.cash:
                stake = actor.cash / max(1.0 + 0.012 + leverage_multiple * 0.005, 1e-6)
                trading_fee = stake * (0.012 + leverage_multiple * 0.005)
                total_upfront = stake + trading_fee
            actor.cash -= total_upfront
            direction = 1 if trend >= -0.015 else -1
            actor.leverage_positions[action.coin_id] = {
                "coin_id": action.coin_id,
                "entry_price": coin.price,
                "collateral": stake,
                "leverage": leverage_multiple,
                "direction": float(direction),
                "opened_turn": float(self.turn),
            }
            reward += 0.02
            event.update(
                {
                    "coin_id": action.coin_id,
                    "stake": round(stake, 4),
                    "leverage": round(leverage_multiple, 2),
                    "trading_fee": round(trading_fee, 4),
                    "position_opened": True,
                    "direction": "long" if direction > 0 else "short",
                }
            )
            applied_cooldown = 2

        elif action.action_type == ActionType.MARKET_MAKE and action.coin_id and action.fraction:
            actor.work_streak = 0
            coin = self.market.coins[action.coin_id]
            recent_volume = coin.volume_history[-1] - coin.volume_history[-2] if len(coin.volume_history) > 1 else coin.trading_volume
            spread = max(self.config.market.min_spread_bps, (coin.ask_price - coin.bid_price) / max(coin.price, self.config.market.min_price))
            completed_cycles = max(0.0, recent_volume / max(self.config.market.max_transaction * 0.75, 1.0))
            volatility_cost = abs(coin.price_history[-1] - coin.price_history[-2]) / max(coin.price_history[-2], self.config.market.min_price) if len(coin.price_history) > 1 else 0.0
            stake = min(actor.cash * max(action.fraction, 0.1), self.config.market.max_transaction * 0.5)
            gross_spread_capture = spread * min(completed_cycles, 2.0)
            profit = stake * self._clamp(gross_spread_capture - volatility_cost * 0.04, 0.0, 0.07)
            actor.cash += profit
            reward += math.tanh(profit / max(self.config.starting_money * 0.25, 1.0))
            event.update(
                {
                    "coin_id": action.coin_id,
                    "stake": round(stake, 4),
                    "profit": round(profit, 4),
                    "spread": round(spread, 4),
                    "completed_cycles": round(completed_cycles, 4),
                }
            )
            if completed_cycles > 0.8:
                applied_cooldown = 1

        else:
            raise ValueError(f"Unsupported action: {action}")

        actor.cash = max(0.0, actor.cash)
        actor.cooldown = max(actor.cooldown, applied_cooldown)
        event["cash_delta"] = round(actor.cash - actor_cash_before, 4)
        if self.net_worth(agent_id) > self.config.starting_money * 3.5:
            self._record_suspicious(event, "extreme_wealth_gain")

        self.metrics["action_distribution"][action.action_type.value] += 1
        self.metrics["agent_action_distribution"][agent_id][action.action_type.value] += 1
        return reward, event

    def _settle_leverage_positions(self) -> list[tuple[int, float, dict[str, Any], float, float]]:
        settlements: list[tuple[int, float, dict[str, Any], float, float]] = []
        for agent_id, state in self.agent_states.items():
            if not state.leverage_positions:
                continue
            for coin_id, position in list(state.leverage_positions.items()):
                if position.get("opened_turn", -1.0) >= self.turn:
                    continue
                if coin_id not in self.market.coins:
                    del state.leverage_positions[coin_id]
                    continue
                worth_before = self.net_worth(agent_id)
                coin = self.market.coins[coin_id]
                entry_price = max(position["entry_price"], self.config.market.min_price)
                price_move = (coin.price - entry_price) / entry_price
                direction = 1.0 if position["direction"] >= 0 else -1.0
                leverage_multiple = position["leverage"]
                collateral = position["collateral"]
                pnl_ratio = direction * price_move * leverage_multiple
                liquidation_threshold = -(
                    1.0
                    - self.config.economy.leverage_margin_buffer
                    - min(0.22, self._max_recent_drop() * 0.9 + coin.panic_score * 0.24)
                )
                liquidated = pnl_ratio <= liquidation_threshold
                event: dict[str, Any] = {
                    "agent_id": agent_id,
                    "action": f"{ActionType.LEVERAGE.value}_settle|{coin_id}",
                    "coin_id": coin_id,
                    "entry_price": round(entry_price, 4),
                    "price_now": round(coin.price, 4),
                    "price_move": round(price_move, 4),
                    "leverage": round(leverage_multiple, 2),
                    "direction": "long" if direction > 0 else "short",
                    "collateral": round(collateral, 4),
                    "liquidated": liquidated,
                }
                if liquidated:
                    if direction > 0:
                        synthetic_units = min(
                            self.config.market.max_transaction / max(coin.bid_price, self.config.market.min_price),
                            collateral * leverage_multiple * 0.18 / max(coin.bid_price, self.config.market.min_price),
                        )
                        forced_cash, _, forced_event = self.market.sell(coin_id, synthetic_units, 0.0, self.turn)
                        event["type"] = "sell"
                        event["gross_value"] = round(forced_event["gross_value"], 4)
                        event["forced_sale_cash"] = round(forced_cash, 4)
                        event["forced_units"] = round(forced_event["units_sold"], 4)
                        event["price_after"] = forced_event["price_after"]
                    else:
                        synthetic_spend = min(self.config.market.max_transaction, collateral * leverage_multiple * 0.16)
                        forced_units, _, forced_event = self.market.buy(coin_id, synthetic_spend, self.turn)
                        event["type"] = "buy"
                        event["forced_cover_spend"] = round(synthetic_spend, 4)
                        event["forced_units"] = round(forced_units, 4)
                        event["price_after"] = forced_event["price_after"]
                    reward = -0.25 - math.tanh(collateral / max(self.config.starting_money * 0.35, 1.0))
                    state.cooldown = max(state.cooldown, 2)
                    self._record_suspicious(event, "liquidation")
                else:
                    capped_ratio = self._clamp(pnl_ratio, -0.7, 1.15)
                    settlement_value = collateral * max(0.0, 1.0 + capped_ratio)
                    state.cash += settlement_value
                    reward = math.tanh((settlement_value - collateral) / max(self.config.starting_money * 0.28, 1.0))
                    event["pnl"] = round(settlement_value - collateral, 4)
                    event["settlement_value"] = round(settlement_value, 4)
                del state.leverage_positions[coin_id]
                worth_after = self.net_worth(agent_id)
                settlements.append((agent_id, reward, event, worth_before, worth_after))
        return settlements

    def _resolve_mining_actions(self, mining_actions: list[tuple[int, Action]]) -> dict[int, tuple[float, dict[str, Any]]]:
        """
        Mining as a costly production process.
        coins_mined = base_rate / (1 + total_miners_on_coin)
        profit = (coins_mined × price) - mining_cost
        """
        if not mining_actions:
            self.current_total_hashrate = 0.0
            self.current_difficulty = self.config.mining.base_difficulty
            self.current_electricity_price = self.config.mining.base_electricity_price
            self.metrics["average_mining_profit_history"].append(0.0)
            return {}

        # Group miners by coin
        per_coin_miners = {}
        for agent_id, action in mining_actions:
            coin_id = action.coin_id or "general"
            if coin_id not in per_coin_miners:
                per_coin_miners[coin_id] = []
            per_coin_miners[coin_id].append(agent_id)

        effective_hashrates: dict[int, float] = {}
        total_hashrate = 0.0
        
        for agent_id, action in mining_actions:
            state = self.agent_states[agent_id]
            effective_hashrate = state.mining_power
            
            # Competition penalty: more miners = less reward per miner
            coin_id = action.coin_id or "general"
            num_miners = len(per_coin_miners.get(coin_id, []))
            if coin_id and coin_id in self.market.coins:
                # Resource-constrained: rewards split among miners
                competition_penalty = 1.0 / (1.0 + (num_miners - 1) * 0.3)
                effective_hashrate = state.mining_power * competition_penalty
            
            effective_hashrates[agent_id] = max(0.05, effective_hashrate)
            total_hashrate += effective_hashrates[agent_id]
        
        # Difficulty scales with total hashrate
        difficulty = self.config.mining.base_difficulty + total_hashrate * self.config.mining.difficulty_scale
        electricity_price = self.config.mining.base_electricity_price + total_hashrate * self.config.mining.electricity_price_scale * 0.2
        
        # Reward pool based on difficulty (higher difficulty = lower rewards)
        base_reward_pool = self.config.mining.block_reward_value / max(difficulty, 0.1)

        self.current_total_hashrate = total_hashrate
        self.current_difficulty = difficulty
        self.current_electricity_price = electricity_price

        results: dict[int, tuple[float, dict[str, Any]]] = {}
        profits: list[float] = []
        
        for agent_id, action in mining_actions:
            state = self.agent_states[agent_id]
            allocation = action.coin_id
            state.mining_allocation = allocation or "general"
            state.work_streak = 0
            hashrate = effective_hashrates[agent_id]
            
            # MINING COST (energy cost × effort)
            mining_cost = state.mining_power * electricity_price
            
            # COINS MINED (base rate / competition)
            coin_id = action.coin_id or "general"
            num_miners = len(per_coin_miners.get(coin_id, []))
            base_mining_rate = self.config.mining.baseline_positive_income / (1.0 + total_hashrate * 0.05)
            coins_mined = base_mining_rate / (1.0 + (num_miners - 1) * 0.3)
            
            # VALUE FROM MINING
            if allocation and allocation in self.market.coins:
                coin = self.market.coins[allocation]
                # Miner receives newly minted coins
                reward_units = coins_mined
                state.holdings[allocation] = state.holdings.get(allocation, 0.0) + reward_units
                
                # Increase supply (dilution!)
                coin.total_supply = min(self.config.market.max_supply, coin.total_supply + reward_units)
                coin.circulating_supply = min(coin.total_supply, coin.circulating_supply + reward_units)
                
                # Value = coins × current price
                reward_value = reward_units * coin.price
            else:
                # General mining pays in cash (but less profitable)
                reward_value = coins_mined * 0.5  # Discount for not mining specific coin
                state.cash += reward_value
            
            # PROFIT CALCULATION
            profit = reward_value - mining_cost
            state.cash = max(0.0, state.cash - mining_cost)
            profits.append(profit)

            # Reward signal
            normalized_profit = math.tanh(profit / max(self.config.starting_money * 0.1, 1.0))
            reward = self.config.evolution.mine_reward_bonus + normalized_profit
            if profit < 0:
                reward += self.config.evolution.mine_loss_penalty * 0.3
            
            event = {
                "agent_id": agent_id,
                "action": ActionType.MINE.value if allocation is None else f"{ActionType.MINE.value}|{allocation}",
                "coin_id": allocation,
                "hashrate": round(hashrate, 4),
                "raw_mining_power": round(state.mining_power, 4),
                "total_network_hashrate": round(total_hashrate, 4),
                "difficulty": round(difficulty, 4),
                "electricity_price": round(electricity_price, 4),
                "mining_cost": round(mining_cost, 4),
                "coins_mined": round(coins_mined, 4),
                "reward_units": round(reward_units if allocation else 0, 4),
                "reward_value": round(reward_value, 4),
                "profit": round(profit, 4),
                "num_competitors": num_miners,
            }
            results[agent_id] = (reward, event)
        
        if profits:
            self.metrics["average_mining_profit_history"].append(sum(profits) / len(profits))
        return results

    def _apply_end_of_turn_decay(self) -> None:
        for agent_id in self.agent_states:
            if len(self.metrics["mining_participation_history"][agent_id]) < len(self.metrics["coins_created_history"]) + 1:
                self.metrics["mining_participation_history"][agent_id].append(0)
            state = self.agent_states[agent_id]
            inflation_drag = state.cash * self.config.economy.cash_drag_rate * (1.0 + self.turn / max(self.config.turns_per_episode, 1))
            state.cash = max(0.0, state.cash - inflation_drag)
            holdings_drag = self.holdings_value(agent_id) * self.config.economy.holding_cost_rate
            for coin_id in list(state.holdings.keys()):
                if coin_id in self.market.coins:
                    coin = self.market.coins[coin_id]
                    # Intrinsic value grows over time based on holder activity
                    holders = sum(1 for s in self.agent_states.values() if coin_id in s.holdings and s.holdings[coin_id] > 0)
                    miners = sum(1 for s in self.agent_states.values() if s.mining_allocation == coin_id)
                    growth_factor = 1.0 + (holders + miners) * 0.002  # 0.2% growth per holder/miner
                    coin.intrinsic_value = min(coin.intrinsic_value * growth_factor, self.config.market.max_price * 0.8)
            state.cash = max(0.0, state.cash - holdings_drag)
            state.cooldown = max(0, state.cooldown - 1)

    def _market_usage_context(self) -> dict[str, dict[str, float]]:
        usage: dict[str, dict[str, float]] = {
            coin_id: {"holders": 0.0, "miners": 0.0, "held_units": 0.0, "trades": 0.0, "buy_pressure": 0.0, "sell_pressure": 0.0}
            for coin_id in self.market.coins
        }
        for state in self.agent_states.values():
            if state.mining_allocation in usage:
                usage[state.mining_allocation]["miners"] += 1.0
            for coin_id, units in state.holdings.items():
                if coin_id in usage and units > 0.01:
                    usage[coin_id]["holders"] += 1.0
                    usage[coin_id]["held_units"] += units
        for event in self.event_log[-max(20, len(self.agent_states) * 3):]:
            coin_id = event.get("coin_id")
            if coin_id not in usage:
                continue
            if event.get("type") == "buy":
                usage[coin_id]["trades"] += 1.0
                usage[coin_id]["buy_pressure"] += float(event.get("cash_spent", 0.0))
            elif event.get("type") == "sell":
                usage[coin_id]["trades"] += 1.0
                usage[coin_id]["sell_pressure"] += float(event.get("gross_value", event.get("forced_sale_cash", 0.0)))
        return usage

    def _snapshot_turn(self) -> None:
        dead_coins = self.market.apply_turn_effects(self.turn, self._market_usage_context())
        
        # Update event detector
        self.event_detector.update(self)
        
        # Record coin deaths
        for coin_id in dead_coins:
            coin_history = self.event_detector.price_history.get(coin_id, [])
            lifespan = len(coin_history)
            self.event_detector.check_coin_death(coin_id, lifespan, "death_conditions_met")
            self.event_log.append({
                "type": "coin_death",
                "turn": self.turn,
                "coin_id": coin_id,
                "reason": "death_conditions_met",
            })
        
        wealth_snapshot = {agent_id: round(self.net_worth(agent_id), 4) for agent_id in self.agent_states}
        self.metrics["wealth_snapshots"].append(wealth_snapshot)
        self.metrics["coins_created_history"].append(len(self.market.coins))
        self.metrics["network_hashrate_history"].append(round(self.current_total_hashrate, 4))
        self.metrics["electricity_price_history"].append(round(self.current_electricity_price, 4))
        self.metrics["difficulty_history"].append(round(self.current_difficulty, 4))
        for agent_id, value in wealth_snapshot.items():
            self.metrics["net_worth_history"][agent_id].append(value)
            # Track coin ownership per agent per turn (TOTAL UNITS, not just coin types)
            total_units = sum(units for cid, units in self.agent_states[agent_id].holdings.items() 
                           if units > 0 and cid in self.market.coins)
            self.metrics["agent_coin_ownership_history"][agent_id].append(total_units)
        if self.config.output.save_replay:
            self.replay.append(
                {
                    "episode": self.episode,
                    "turn": self.turn,
                    "wealth": wealth_snapshot,
                    "coins": [coin.snapshot() for coin in self.market.coins.values()],
                    "dead_coins": dead_coins,
                }
            )

    def market_snapshot(self) -> list[dict[str, Any]]:
        snapshot: list[dict[str, Any]] = []
        for coin in self.market.coin_stats():
            snapshot.append(
                {
                    "coin_id": coin["coin_id"],
                    "creator_id": coin["creator_id"],
                    "price": round(coin["price"], 4),
                    "trend": round(coin["trend"], 4),
                    "volume": round(coin["volume"], 4),
                    "intrinsic_value": round(coin["intrinsic_value"], 4),
                    "passive_liquidity": round(coin["passive_liquidity"], 4),
                    "age": coin["age"],
                    "lifecycle_stage": coin.get("lifecycle_stage", "unknown"),
                    "attention_score": coin.get("attention_score", 0.0),
                }
            )
        return snapshot

    def network_snapshot(self) -> dict[str, Any]:
        return {
            "total_hashrate": round(self.current_total_hashrate, 4),
            "difficulty": round(self.current_difficulty, 4),
            "electricity_price": round(self.current_electricity_price, 4),
        }

    def agent_snapshot(self) -> list[dict[str, Any]]:
        snapshot: list[dict[str, Any]] = []
        for agent_id, state in sorted(self.agent_states.items()):
            holdings = {coin_id: round(units, 2) for coin_id, units in state.holdings.items() if units > 0.01}
            snapshot.append(
                {
                    "agent_id": agent_id,
                    "cash": round(state.cash, 4),
                    "mining_power": round(state.mining_power, 4),
                    "mining_allocation": state.mining_allocation,
                    "last_mining_profit": round(state.mining_profit_history[-1], 4) if state.mining_profit_history else 0.0,
                    "net_worth": round(self.net_worth(agent_id), 4),
                    "creator_effectiveness": round(state.creator_effectiveness, 3),
                    "failed_coins": state.failed_coins,
                    "holdings": holdings,
                    "coins_created": list(state.coins_created),
                }
            )
        return snapshot

    def describe_event(self, event: dict[str, Any]) -> str:
        # Handle coin death events (no action key)
        if event.get("type") == "coin_death":
            return f"💀 Coin {event.get('coin_id', '?')} died after turn {event.get('turn', 0)} - {event.get('reason', 'unknown')}"
        
        action = str(event.get("action", ""))
        agent_id = event.get("agent_id", "?")
        
        if not action:
            return f"Event: {event}"
        
        if action.startswith(ActionType.TREND.value):
            strategy = event.get("strategy", "trend")
            return f"Agent {agent_id} ran trend strategy on {event.get('coin_id')} via {strategy}"
        if action.startswith(ActionType.SPECULATE.value):
            return f"Agent {agent_id} speculated {event.get('direction')} on {event.get('coin_id')} for pnl ${event.get('pnl', 0):.2f}"
        if action.startswith(ActionType.ARBITRAGE.value):
            return f"Agent {agent_id} arbitraged {event.get('coin_id')} for ${event.get('profit', 0):.2f}"
        if action.startswith(ActionType.LEVERAGE.value):
            if "settle|" in action:
                if event.get("liquidated"):
                    return f"Agent {agent_id} was liquidated on {event.get('coin_id')} after a {event.get('price_move', 0):.1%} move"
                return f"Agent {agent_id} settled leveraged {event.get('coin_id')} for pnl ${event.get('pnl', 0):.2f}"
            if event.get("liquidated"):
                return f"Agent {agent_id} was liquidated on {event.get('coin_id')} at {event.get('leverage', 0):.1f}x"
            if event.get("position_opened"):
                return f"Agent {agent_id} opened a {event.get('leverage', 0):.1f}x leveraged {event.get('direction')} on {event.get('coin_id')}"
            return f"Agent {agent_id} traded {event.get('coin_id')} with {event.get('leverage', 0):.1f}x leverage for pnl ${event.get('pnl', 0):.2f}"
        if action.startswith(ActionType.MARKET_MAKE.value):
            return f"Agent {agent_id} market-made {event.get('coin_id')} for ${event.get('profit', 0):.2f}"
        if action.startswith(ActionType.BUY.value):
            return (
                f"Agent {agent_id} bought {event.get('units_bought', 0):.2f} of {event.get('coin_id')} "
                f"for ${event.get('cash_spent', 0):.2f} at ${event.get('price_after', 0):.3f}"
            )
        if action.startswith(ActionType.SELL.value):
            return (
                f"Agent {agent_id} sold {event.get('units_sold', 0):.2f} of {event.get('coin_id')} "
                f"for ${event.get('realized_cash', 0):.2f}; price now ${event.get('price_after', 0):.3f}"
            )
        if action.startswith(ActionType.MINE.value):
            if event.get("coin_id"):
                return (
                    f"Agent {agent_id} mined {event.get('reward_units', 0):.2f} of {event.get('coin_id')} "
                    f"with profit ${event.get('profit', 0):.2f}"
                )
            if event.get("jackpot"):
                return f"Agent {agent_id} hit the mining jackpot for profit ${event.get('profit', 0):.2f}"
            return f"Agent {agent_id} mined the general network for profit ${event.get('profit', 0):.2f}"
        if action == ActionType.CREATE.value:
            return (
                f"Agent {agent_id} created {event.get('coin_id')} with {event.get('creator_supply', 0):.0f} locked tokens "
                f"and {event.get('liquidity_seeded', 0):.0f} liquidity"
            )
        if action == ActionType.WORK.value:
            return f"Agent {agent_id} worked and earned ${event.get('income', 0):.2f}"
        if action == ActionType.HOLD.value:
            return f"Agent {agent_id} held positions and waited"
        return json.dumps(event, sort_keys=True)

    def event_highlights(self, event: dict[str, Any]) -> list[str]:
        highlights: list[str] = []
        if event.get("suspicious_dump"):
            highlights.append("Large sell-off detected")
        if event.get("creator_dump"):
            highlights.append("Potential rug pull by coin creator")
        if event.get("capped_transaction"):
            highlights.append("Trade capped for stability")
        if event.get("capped_price"):
            highlights.append("Price limit reached")
        if event.get("supply_capped"):
            highlights.append("Coin supply cap reached")
        if event.get("profit", 0) < 0:
            highlights.append("Mining is running at a loss")
        if event.get("jackpot"):
            highlights.append("Mining jackpot awarded")
        if event.get("liquidated"):
            highlights.append("Leverage liquidation")
        if event.get("max_gain_triggered"):
            highlights.append("Profit-taking pressure")
        if event.get("electricity_price", 0) > self.config.mining.base_electricity_price * 1.8:
            highlights.append("Electricity spike")
        if event.get("total_network_hashrate", 0) > self.config.num_agents * self.config.mining.starting_mining_power * 0.7:
            highlights.append("Mining competition intensified")
        if abs(event.get("price_after", event.get("price_before", 0)) - event.get("price_before", 0)) >= 0.12:
            highlights.append("Rapid price change")
        if event.get("type") == "sell" and event.get("market_share", 0) >= 0.35:
            highlights.append("High market-share dump")
        return highlights

    def live_snapshot(self, event: dict[str, Any] | None = None) -> dict[str, Any]:
        recent_events = self.event_log[-8:]
        suspicious_events = self.metrics["suspicious_events"][-6:]
        return {
            "episode": self.episode,
            "turn": self.turn,
            "market": self.market_snapshot(),
            "network": self.network_snapshot(),
            "agents": self.agent_snapshot(),
            "recent_events": recent_events,
            "suspicious_events": suspicious_events,
            "current_event": event,
            "highlights": self.event_highlights(event) if event else [],
        }

    def _finalize_episode(self) -> dict[str, Any]:
        for agent in self.agents:
            self.metrics["episode_rewards"][agent.agent_id].append(round(agent.total_reward, 4))
        avg_reward = sum(agent.total_reward for agent in self.agents) / max(len(self.agents), 1)
        self.metrics["average_reward"].append(round(avg_reward, 4))
        
        # Store event summaries in metrics
        self.metrics["event_summaries"] = self.event_detector.get_summary(top_n=15)
        self.metrics["major_events"] = [e.summary() for e in self.event_detector.get_all_events() if e.severity >= 4]
        
        return self.episode_summary()

    def run_episode_iter(self, training: bool = True) -> Iterator[dict[str, Any]]:
        self.reset()
        self.episode += 1
        self._log(f"=== Episode {self.episode} ===")
        for turn in range(self.config.turns_per_episode):
            self.turn = turn
            self._log(f"[turn {turn:03d}] active_coins={len(self.market.coins)}")
            yield {"kind": "turn_start", "turn": turn, "snapshot": self.live_snapshot()}
            planned_actions: dict[int, tuple[dict[str, Any], list[Action], list[Action], float]] = {}
            mining_queue: list[tuple[int, Action]] = []
            for agent in self.agents:
                agent_id = agent.agent_id
                state = self.observe(agent_id)
                actions = self.available_actions(agent_id)
                chosen_actions = agent.select_actions(state, actions, training=training)
                planned_actions[agent_id] = (state, actions, chosen_actions, self.net_worth(agent_id))
                for action in chosen_actions:
                    if action.action_type == ActionType.MINE:
                        mining_queue.append((agent_id, action))
                    self._log(f"agent={agent_id} action={action.key}")

            mining_results = self._resolve_mining_actions(mining_queue)
            for agent in self.agents:
                agent_id = agent.agent_id
                state, actions, chosen_actions, worth_before = planned_actions[agent_id]
                running_worth_before = worth_before
                for action in chosen_actions:
                    if action.action_type == ActionType.MINE:
                        reward, event = mining_results[agent_id]
                    else:
                        reward, event = self.apply_action(agent_id, action)
                    worth_after = self.net_worth(agent_id)
                    reward = self._finalize_reward(reward, event, running_worth_before, worth_after)
                    running_worth_before = worth_after
                    agent.record_reward(reward)
                    self._log(f"market_event={json.dumps(event, sort_keys=True)} reward={reward:.4f}")
                    full_event = {"episode": self.episode, "turn": turn, "reward": reward, **event}
                    self.event_log.append(full_event)
                    yield {
                        "kind": "action",
                        "turn": turn,
                        "agent_id": agent_id,
                        "reward": reward,
                        "event": full_event,
                        "description": self.describe_event(full_event),
                        "snapshot": self.live_snapshot(full_event),
                    }
            for agent_id, reward, event, worth_before, worth_after in self._settle_leverage_positions():
                reward = self._finalize_reward(reward, event, worth_before, worth_after)
                agent = next(agent for agent in self.agents if agent.agent_id == agent_id)
                agent.record_reward(reward)
                self._log(f"market_event={json.dumps(event, sort_keys=True)} reward={reward:.4f}")
                full_event = {"episode": self.episode, "turn": turn, "reward": reward, **event}
                self.event_log.append(full_event)
                yield {
                    "kind": "action",
                    "turn": turn,
                    "agent_id": agent_id,
                    "reward": reward,
                    "event": full_event,
                    "description": self.describe_event(full_event),
                    "snapshot": self.live_snapshot(full_event),
                }
            self._apply_end_of_turn_decay()
            self._snapshot_turn()
            yield {"kind": "turn_end", "turn": turn, "snapshot": self.live_snapshot()}

        summary = self._finalize_episode()
        yield {"kind": "episode_complete", "summary": summary, "snapshot": self.live_snapshot()}

    def run_episode(self, training: bool = True) -> dict[str, Any]:
        summary: dict[str, Any] | None = None
        for payload in self.run_episode_iter(training=training):
            if payload["kind"] == "episode_complete":
                summary = payload["summary"]
        if summary is None:
            raise RuntimeError("Episode finished without a summary")
        return summary

    def episode_summary(self) -> dict[str, Any]:
        return {
            "episode": self.episode,
            "coins_created": len(self.market.coins),
            "avg_reward": self.metrics["average_reward"][-1] if self.metrics["average_reward"] else 0.0,
            "final_wealth": {agent_id: round(self.net_worth(agent_id), 4) for agent_id in self.agent_states},
        }

    def train(self) -> list[dict[str, Any]]:
        summaries = []
        for _ in range(self.config.episodes):
            summaries.append(self.run_episode(training=True))
        return summaries

    def save_models(self) -> None:
        if not self.config.output.save_models:
            return
        for agent in self.agents:
            model_path = self.config.output.model_dir / f"agent_{agent.agent_id}.pkl"
            agent.save(model_path)

    def load_models(self) -> None:
        for agent in self.agents:
            model_path = self.config.output.model_dir / f"agent_{agent.agent_id}.pkl"
            if model_path.exists():
                agent.load(model_path)

    def export_csv(self, output_dir: Path) -> None:
        if not self.config.output.export_csv:
            return
        event_path = output_dir / "event_log.csv"
        with event_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=sorted({key for row in self.event_log for key in row}))
            writer.writeheader()
            for row in self.event_log:
                writer.writerow(row)

        reward_path = output_dir / "episode_rewards.csv"
        with reward_path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["agent_id", "episode_index", "reward"])
            for agent_id, rewards in self.metrics["episode_rewards"].items():
                for index, reward in enumerate(rewards, start=1):
                    writer.writerow([agent_id, index, reward])

        generation_path = output_dir / "generation_fitness.csv"
        with generation_path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["generation", "best_fitness", "average_fitness", "dominant_behavior"])
            best = self.metrics.get("generation_best_fitness", [])
            avg = self.metrics.get("generation_average_fitness", [])
            dominant = self.metrics.get("generation_dominant_behaviors", [])
            for index in range(max(len(best), len(avg), len(dominant))):
                writer.writerow(
                    [
                        index + 1,
                        best[index] if index < len(best) else "",
                        avg[index] if index < len(avg) else "",
                        dominant[index] if index < len(dominant) else "",
                    ]
                )

    def export_replay(self, output_dir: Path) -> None:
        if not self.config.output.save_replay:
            return
        replay_path = output_dir / "replay.json"
        with replay_path.open("w") as handle:
            json.dump(self.replay, handle, indent=2)
