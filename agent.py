from __future__ import annotations

import math
import pickle
import random
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any


class ActionType(str, Enum):
    CREATE = "create"
    BUY = "buy"
    SELL = "sell"
    WORK = "work"
    MINE = "mine"
    TREND = "trend"
    SPECULATE = "speculate"
    ARBITRAGE = "arbitrage"
    LEVERAGE = "leverage"
    HOLD = "hold"
    MARKET_MAKE = "market_make"


@dataclass(frozen=True, slots=True)
class Action:
    action_type: ActionType
    coin_id: str | None = None
    target_id: int | None = None
    fraction: float | None = None

    @property
    def key(self) -> str:
        parts = [self.action_type.value]
        if self.coin_id is not None:
            parts.append(self.coin_id)
        if self.target_id is not None:
            parts.append(f"target:{self.target_id}")
        if self.fraction is not None:
            parts.append(f"fraction:{self.fraction:.2f}")
        return "|".join(parts)


@dataclass(slots=True)
class StrategyGenome:
    buy_weight: float
    sell_weight: float
    mine_weight: float
    work_weight: float
    create_coin_weight: float
    trend_trade_weight: float
    speculation_weight: float
    arbitrage_weight: float
    leverage_weight: float
    hold_weight: float
    market_making_weight: float
    risk_tolerance: float
    trend_sensitivity: float
    mining_aggressiveness: float
    profit_threshold: float
    exploration_randomness: float
    coin_creation_bias: float
    wealth_preservation: float
    volume_sensitivity: float
    mining_switch_bias: float
    fomo_sensitivity: float
    panic_sensitivity: float
    overconfidence_bias: float
    reaction_speed: float
    risk_trigger_threshold: float

    @classmethod
    def random(cls, rng: random.Random) -> StrategyGenome:
        return cls(
            buy_weight=rng.uniform(-1.5, 1.5),
            sell_weight=rng.uniform(-1.5, 1.5),
            mine_weight=rng.uniform(-1.5, 1.5),
            work_weight=rng.uniform(-1.0, 1.0),
            create_coin_weight=rng.uniform(-1.0, 1.2),
            trend_trade_weight=rng.uniform(-1.2, 1.8),
            speculation_weight=rng.uniform(-1.0, 2.0),
            arbitrage_weight=rng.uniform(-1.0, 1.6),
            leverage_weight=rng.uniform(-1.3, 2.2),
            hold_weight=rng.uniform(-1.0, 1.5),
            market_making_weight=rng.uniform(-1.0, 1.8),
            risk_tolerance=rng.uniform(0.0, 2.0),
            trend_sensitivity=rng.uniform(-2.0, 2.5),
            mining_aggressiveness=rng.uniform(-1.5, 2.5),
            profit_threshold=rng.uniform(-0.4, 1.2),
            exploration_randomness=rng.uniform(0.01, 0.5),
            coin_creation_bias=rng.uniform(-1.0, 1.5),
            wealth_preservation=rng.uniform(0.0, 2.0),
            volume_sensitivity=rng.uniform(-1.0, 1.5),
            mining_switch_bias=rng.uniform(-1.0, 1.5),
            fomo_sensitivity=rng.uniform(0.0, 2.2),
            panic_sensitivity=rng.uniform(0.0, 2.2),
            overconfidence_bias=rng.uniform(-0.4, 1.8),
            reaction_speed=rng.uniform(0.18, 0.72),
            risk_trigger_threshold=rng.uniform(0.02, 0.18),
        )


class EvolutionAgent:
    def __init__(self, agent_id: int, genome: StrategyGenome, seed: int) -> None:
        self.agent_id = agent_id
        self.genome = genome
        self.random = random.Random(seed)
        self.total_reward = 0.0
        self.fitness = 0.0

    def reset_episode(self) -> None:
        self.total_reward = 0.0
        self.fitness = 0.0

    def record_reward(self, reward: float) -> None:
        self.total_reward += reward

    def select_actions(
        self,
        observation: dict[str, Any],
        available_actions: list[Action],
        training: bool = True,
    ) -> list[Action]:
        if not available_actions:
            raise ValueError("No available actions for agent")

        if training and self.random.random() < self.genome.exploration_randomness:
            chosen = [self.random.choice(available_actions)]
        else:
            scored_actions: list[tuple[float, Action]] = []
            for action in available_actions:
                scored_actions.append((self.score_action(observation, action), action))
            scored_actions.sort(key=lambda item: item[0], reverse=True)
            top_slice = scored_actions[: min(5, len(scored_actions))]
            max_score = top_slice[0][0]
            temperature = max(0.18, 0.72 - self.genome.reaction_speed * 0.45)
            weights = [math.exp((score - max_score) / temperature) for score, _ in top_slice]
            chosen = [self.random.choices([action for _, action in top_slice], weights=weights, k=1)[0]]

        wealth = observation["cash"] + observation["holdings_value"]
        can_take_second = wealth > 1.5 * 120.0 or self.genome.risk_tolerance > 1.25 or self.genome.overconfidence_bias > 1.0
        if can_take_second:
            scored_actions = sorted(
                ((self.score_action(observation, action), action) for action in available_actions if action not in chosen),
                key=lambda item: item[0],
                reverse=True,
            )
            for _, candidate in scored_actions:
                if candidate.action_type == ActionType.HOLD:
                    continue
                if candidate.action_type == chosen[0].action_type and candidate.coin_id == chosen[0].coin_id:
                    continue
                if candidate.action_type == ActionType.MINE and any(existing.action_type == ActionType.MINE for existing in chosen):
                    continue
                if {candidate.action_type, chosen[0].action_type} == {ActionType.BUY, ActionType.SELL}:
                    continue
                chosen.append(candidate)
                break
        return chosen

    def score_action(self, observation: dict[str, Any], action: Action) -> float:
        cash = observation["cash"]
        holdings_value = observation["holdings_value"]
        top_coin = observation["visible_coins"][0] if observation["visible_coins"] else None
        top_trend = top_coin["trend"] if top_coin else 0.0
        top_volume = top_coin["volume"] if top_coin else 0.0
        market_sentiment = observation.get("market_sentiment", 0.0)
        market_volatility = observation.get("market_volatility", 0.0)
        wealth_ratio = cash / max(cash + holdings_value, 1.0)
        mining_margin = observation["estimated_mining_margin"]
        electricity_pressure = observation["electricity_price"] / max(observation["difficulty"], 1e-6)
        inflation_pressure = 1.0 - observation.get("inflation_multiplier", 1.0)
        current_work_income = observation.get("current_work_income", 0.0)
        max_unrealized_gain = observation.get("max_unrealized_gain", 0.0)
        recent_drop = observation.get("recent_drop", 0.0)
        recent_rally = observation.get("recent_rally", 0.0)
        leverage_exposure = observation.get("leverage_exposure", 0.0)
        capital_advantage = max(0.0, (cash + holdings_value) / 120.0 - 1.0)
        overconfidence = capital_advantage * self.genome.overconfidence_bias
        fomo = max(0.0, market_sentiment) * self.genome.fomo_sensitivity
        panic = max(0.0, -market_sentiment) * self.genome.panic_sensitivity
        profit_take_urge = 0.0
        if max_unrealized_gain >= 0.2:
            profit_take_urge += 0.22
        if max_unrealized_gain >= 0.5:
            profit_take_urge += 0.32
        if max_unrealized_gain >= 1.0:
            profit_take_urge += 0.4
        panic_wave = max(0.0, recent_drop - 0.02) * self.genome.panic_sensitivity * 3.5
        fomo_wave = max(0.0, recent_rally - 0.02) * self.genome.fomo_sensitivity * 3.0
        reaction_gate = max(0.0, market_volatility - self.genome.risk_trigger_threshold)
        synchronized_penalty = observation.get("cooldown", 0) * 0.25

        score = self.random.uniform(-0.03, 0.03)
        if action.action_type == ActionType.WORK:
            score += self.genome.work_weight
            score += self.genome.wealth_preservation * (1.0 - wealth_ratio)
            score += max(0.0, electricity_pressure - 1.0) * 0.6
            score += current_work_income / 8.0
            score -= inflation_pressure * (0.8 + self.genome.risk_tolerance * 0.2)
            score -= capital_advantage * 0.45
            score -= fomo * 0.2
            score -= fomo_wave * 0.3
            score += synchronized_penalty * 0.12

        elif action.action_type == ActionType.CREATE:
            score += self.genome.create_coin_weight
            score += self.genome.coin_creation_bias
            score += self.genome.risk_tolerance * wealth_ratio
            score -= self.genome.wealth_preservation * (1.0 - wealth_ratio)
            score += capital_advantage * self.genome.risk_tolerance * 0.2
            score += overconfidence * 0.15

        elif action.action_type == ActionType.BUY:
            score += self.genome.buy_weight
            score += self.genome.trend_sensitivity * top_trend
            score += self.genome.volume_sensitivity * min(top_volume / 100.0, 2.0)
            score += self.genome.risk_tolerance * wealth_ratio
            score -= self.genome.wealth_preservation * (1.0 - wealth_ratio)
            score += capital_advantage * (0.18 + self.genome.risk_tolerance * 0.12)
            score += fomo * 0.35 + overconfidence * 0.1
            score += fomo_wave * (0.25 + reaction_gate * 2.1)
            score -= panic_wave * (0.18 + reaction_gate * 1.35)
            score -= profit_take_urge * 0.25
            score -= synchronized_penalty * 0.28
            if action.fraction is not None:
                score += self.genome.risk_tolerance * action.fraction
                score -= self.genome.wealth_preservation * action.fraction * 0.5

        elif action.action_type == ActionType.SELL:
            score += self.genome.sell_weight
            score += self.genome.wealth_preservation * (1.0 - wealth_ratio)
            score -= self.genome.trend_sensitivity * top_trend * 0.6
            score += capital_advantage * 0.08
            score += panic * 0.4
            score += profit_take_urge * (0.62 + reaction_gate * 1.75)
            score += panic_wave * (0.35 + reaction_gate * 1.85)
            if action.fraction is not None:
                score += action.fraction * max(0.0, self.genome.profit_threshold)
                score += action.fraction * profit_take_urge * 0.7

        elif action.action_type == ActionType.MINE:
            score += self.genome.mine_weight
            score += self.genome.mining_aggressiveness * mining_margin
            score -= electricity_pressure * (1.2 - self.genome.risk_tolerance * 0.25)
            score += self.genome.mining_switch_bias if action.coin_id else self.genome.mining_aggressiveness * 0.25
            score += capital_advantage * self.genome.risk_tolerance * 0.08
            if action.coin_id and top_coin and action.coin_id == top_coin["coin_id"]:
                score += self.genome.trend_sensitivity * top_trend * 0.5

        elif action.action_type == ActionType.TREND:
            score += self.genome.trend_trade_weight
            score += self.genome.trend_sensitivity * top_trend * 1.25
            score += fomo * 0.45
            score -= panic * 0.15
            score += fomo_wave * (0.28 + reaction_gate * 2.4)
            score -= panic_wave * (0.22 + reaction_gate * 1.9)
            score -= profit_take_urge * 0.38
            score -= synchronized_penalty * 0.2
            if action.fraction is not None:
                score += action.fraction * (0.25 + self.genome.risk_tolerance * 0.25)

        elif action.action_type == ActionType.SPECULATE:
            score += self.genome.speculation_weight
            score += market_volatility * (0.8 + self.genome.risk_tolerance * 0.3)
            score += overconfidence * 0.35
            score -= self.genome.wealth_preservation * 0.25
            score += fomo_wave * (0.12 + reaction_gate * 1.1)
            score += panic_wave * (0.08 + reaction_gate * 0.85)

        elif action.action_type == ActionType.ARBITRAGE:
            score += self.genome.arbitrage_weight
            score += market_volatility * 0.2
            score += self.genome.wealth_preservation * 0.2
            score -= overconfidence * 0.1

        elif action.action_type == ActionType.LEVERAGE:
            score += self.genome.leverage_weight
            score += market_volatility * 0.35
            score += overconfidence * 0.35
            score += fomo * 0.18
            score -= self.genome.wealth_preservation * 0.55
            score -= panic * 0.12
            score += fomo_wave * (0.08 + reaction_gate * 1.45)
            score -= panic_wave * (0.18 + reaction_gate * 1.7)
            score -= min(leverage_exposure / max(cash + holdings_value, 1.0), 1.0) * 0.7
            score -= profit_take_urge * 0.22
            if recent_rally > 0.05:
                score += 0.28 + recent_rally * 3.2
            if recent_drop > 0.05:
                score -= 0.18 + recent_drop * 1.8
            score -= synchronized_penalty * 0.35

        elif action.action_type == ActionType.HOLD:
            score += self.genome.hold_weight
            score += self.genome.wealth_preservation * 0.55
            score -= inflation_pressure * 0.25
            score -= fomo * 0.25
            score -= panic * 0.1
            score -= profit_take_urge * 0.8
            score -= panic_wave * 0.35
            score += synchronized_penalty * 0.08

        elif action.action_type == ActionType.MARKET_MAKE:
            score += self.genome.market_making_weight
            score += min(top_volume / 60.0, 1.5) * 0.6
            score += market_volatility * 0.15
            score += self.genome.wealth_preservation * 0.2
            score += recent_drop * 0.2

        return score

    def genome_dict(self) -> dict[str, float]:
        return asdict(self.genome)

    def save(self, path: Path) -> None:
        payload = {"agent_id": self.agent_id, "genome": self.genome_dict()}
        with path.open("wb") as handle:
            pickle.dump(payload, handle)

    def load(self, path: Path) -> None:
        with path.open("rb") as handle:
            payload = pickle.load(handle)
        self.genome = StrategyGenome(**payload["genome"])
