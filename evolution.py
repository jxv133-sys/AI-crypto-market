from __future__ import annotations

import json
import math
import random
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from statistics import mean
from typing import Any, Callable

from agent import ActionType, EvolutionAgent, StrategyGenome
from config import GameConfig
from environment import CryptoTradingEnvironment


class EvolutionTrainer:
    def __init__(self, config: GameConfig) -> None:
        self.config = config
        self.random = random.Random(config.seed)
        self.population = self._create_population()
        self.last_environment: CryptoTradingEnvironment | None = None
        self.best_agent: EvolutionAgent | None = None
        self.history: dict[str, Any] = {
            "best_fitness": [],
            "average_fitness": [],
            "best_genomes": [],
            "strategy_distribution": [],
            "dominant_behaviors": [],
            "strategy_diversity": [],
            "behavior_counts": [],
        }

    def _create_population(self) -> list[EvolutionAgent]:
        population: list[EvolutionAgent] = []
        population_size = max(2, self.config.evolution.population_size)
        self.config.num_agents = population_size
        for agent_id in range(population_size):
            genome_rng = random.Random(self.config.seed + agent_id + 101)
            population.append(EvolutionAgent(agent_id, StrategyGenome.random(genome_rng), self.config.seed + agent_id + 1))
        return population

    def _clone_agent(self, source: EvolutionAgent, agent_id: int) -> EvolutionAgent:
        genome = StrategyGenome(**asdict(source.genome))
        clone = EvolutionAgent(agent_id, genome, self.config.seed + 5000 + agent_id + self.random.randint(0, 10_000))
        # Preserve fitness details for analysis
        if hasattr(source, '_fitness_details'):
            clone._fitness_details = source._fitness_details.copy()
        return clone

    def _mutate_genome(self, genome: StrategyGenome) -> StrategyGenome:
        payload = asdict(genome)
        for key, value in payload.items():
            if self.random.random() <= self.config.evolution.mutation_rate:
                payload[key] = value + self.random.gauss(0.0, self.config.evolution.mutation_strength)
        payload["exploration_randomness"] = min(0.75, max(0.0, payload["exploration_randomness"]))
        payload["profit_threshold"] = min(2.0, max(-1.0, payload["profit_threshold"]))
        return StrategyGenome(**payload)

    def _crossover(self, parent_a: StrategyGenome, parent_b: StrategyGenome) -> StrategyGenome:
        child: dict[str, float] = {}
        for key, value in asdict(parent_a).items():
            if self.random.random() < 0.5:
                child[key] = value
            else:
                child[key] = getattr(parent_b, key)
            if self.random.random() < 0.25:
                child[key] = (value + getattr(parent_b, key)) / 2.0
        return StrategyGenome(**child)

    def _fitness(self, env: CryptoTradingEnvironment, agent: EvolutionAgent) -> float:
        """
        Calculate agent fitness based on selected objective.

        Objectives:
        - cash_maximizer: Rewards cash only, no penalty for market holdings
        - balanced: Rewards both cash and market holdings equally
        - net_worth: Rewards total wealth (default, legacy behavior)
        """
        agent_id = agent.agent_id
        state = env.agent_states[agent_id]
        objective = self.config.fitness_objective

        # ============= BASE COMPONENTS =============
        final_cash = state.cash
        final_holdings_value = env.holdings_value(agent_id)
        final_wealth = final_cash + final_holdings_value

        # ============= OBJECTIVE-SPECIFIC CALCULATION =============
        if objective == "cash_maximizer":
            # Only reward cash - market holdings don't hurt fitness
            cash_fitness = math.log1p(max(final_cash, 0.0)) / max(math.log1p(self.config.starting_money), 1e-6)

            # Small bonus for having some market presence (encourages trading)
            market_bonus = 0.1 if final_holdings_value > 0 else 0.0

            # Realized profit bonus
            realized_profit_bonus = self._calculate_realized_profit_bonus(env, agent_id)

            # No liquidity penalty - agents can go all-in on coins
            raw_fitness = cash_fitness * 0.7 + realized_profit_bonus * 0.2 + market_bonus * 0.1
            stability_factor = 1.0  # No stability scaling
            holdings_discount = 0.0
            asset_fitness = 0.0
            liquidity_ratio = 1.0

        elif objective == "balanced":
            # Reward both cash and holdings
            cash_fitness = math.log1p(max(final_cash, 0.0)) / max(math.log1p(self.config.starting_money), 1e-6)

            # Holdings are discounted but still valuable
            holdings_discount = self._calculate_holdings_discount(env, agent_id, state)
            discounted_holdings_value = final_holdings_value * (1.0 - holdings_discount)
            asset_fitness = math.log1p(max(discounted_holdings_value, 0.0)) / max(math.log1p(self.config.starting_money), 1e-6)

            # Realized profit bonus
            realized_profit_bonus = self._calculate_realized_profit_bonus(env, agent_id)

            # Moderate liquidity preference
            liquidity_ratio = final_cash / max(final_wealth, 1.0) if final_wealth > 0 else 0.0
            stability_factor = self._calculate_stability_factor(liquidity_ratio)

            raw_fitness = cash_fitness * 0.40 + asset_fitness * 0.35 + realized_profit_bonus * 0.25

        else:  # net_worth (default/legacy)
            # Legacy behavior - reward total wealth
            cash_fitness = math.log1p(max(final_cash, 0.0)) / max(math.log1p(self.config.starting_money), 1e-6)

            holdings_discount = self._calculate_holdings_discount(env, agent_id, state)
            discounted_holdings_value = final_holdings_value * (1.0 - holdings_discount)
            asset_fitness = math.log1p(max(discounted_holdings_value, 0.0)) / max(math.log1p(self.config.starting_money), 1e-6)

            realized_profit_bonus = self._calculate_realized_profit_bonus(env, agent_id)

            liquidity_ratio = final_cash / max(final_wealth, 1.0) if final_wealth > 0 else 0.0
            stability_factor = self._calculate_stability_factor(liquidity_ratio)

            raw_fitness = cash_fitness * 0.50 + asset_fitness * 0.25 + realized_profit_bonus * 0.25

        # Apply stability factor (except for cash_maximizer)
        fitness = raw_fitness * stability_factor

        # Store details for analysis
        agent._fitness_details = {
            "objective": objective,
            "cash_fitness": cash_fitness,
            "asset_fitness": asset_fitness,
            "holdings_discount": holdings_discount,
            "realized_profit_bonus": realized_profit_bonus,
            "liquidity_ratio": liquidity_ratio,
            "stability_factor": stability_factor,
            "raw_fitness": raw_fitness,
            "final_cash": final_cash,
            "final_holdings_value": final_holdings_value,
            "final_wealth": final_wealth,
        }

        return fitness
    
    def _calculate_stability_factor(self, liquidity_ratio: float) -> float:
        """
        Calculate stability scaling factor based on liquidity ratio.
        
        Liquidity Ratio | Stability Factor | Effect
        ----------------|------------------|--------
        >= 30%          | 1.0              | Full fitness
        20%             | 0.85             | Slight reduction
        10%             | 0.50             | Half fitness
        0%              | 0.25             | Severe reduction
        
        This ensures illiquid agents (all holdings, no cash) 
        receive significantly reduced fitness.
        """
        if liquidity_ratio >= 0.30:
            return 1.0
        elif liquidity_ratio >= 0.20:
            # Linear interpolation: 0.85 to 1.0
            return 0.85 + (liquidity_ratio - 0.20) / 0.10 * 0.15
        elif liquidity_ratio >= 0.10:
            # Linear interpolation: 0.50 to 0.85
            return 0.50 + (liquidity_ratio - 0.10) / 0.10 * 0.35
        else:
            # Linear interpolation: 0.25 to 0.50
            return 0.25 + liquidity_ratio / 0.10 * 0.25
    
    def _calculate_holdings_discount(self, env: CryptoTradingEnvironment, agent_id: int, state) -> float:
        """
        Calculate discount factor for unrealized holdings based on market liquidity.
        
        Holdings are worth less than cash because they must be sold to realize value.
        The discount reflects how difficult it would be to convert to cash.
        
        Factors:
        - Base illiquidity: 30% (holdings aren't cash)
        - Low trading volume: +more discount
        - Price volatility: +more discount
        - Lifecycle stage: risky stages get more discount
        """
        if not state.holdings:
            return 0.0
        
        total_discount = 0.0
        total_weight = 0.0
        
        for coin_id, units in state.holdings.items():
            if units <= 0 or coin_id not in env.market.coins:
                continue
            
            coin = env.market.coins[coin_id]
            holdings_value = units * coin.price
            
            # Base discount: holdings aren't cash (30%)
            base_discount = 0.30
            
            # Volume adjustment: can we sell without crashing price?
            if coin.trading_volume > 0:
                # If we could sell our entire position in one turn
                sellable_fraction = min(1.0, coin.trading_volume / (holdings_value + 1e-6))
                base_discount *= (1.0 - sellable_fraction * 0.4)  # Reduce discount by up to 40%
            
            # Price stability: stable = less discount
            if len(coin.price_history) >= 5:
                recent_prices = coin.price_history[-5:]
                price_volatility = max(recent_prices) / max(min(recent_prices), 1e-6) - 1
                volatility_penalty = min(0.25, price_volatility * 0.4)
                base_discount += volatility_penalty
            
            # Lifecycle stage adjustment
            lifecycle_discounts = {
                "birth": 0.10,      # New coins are uncertain
                "growth": 0.0,      # Growing coins are good
                "peak": 0.05,       # At peak, could go either way
                "decline": 0.20,    # Declining coins are risky
                "death": 0.95,      # Dead coins worth nothing
            }
            base_discount += lifecycle_discounts.get(coin.lifecycle_stage, 0.10)
            
            # Clamp discount between 0.05 and 0.95
            coin_discount = max(0.05, min(0.95, base_discount))
            
            total_discount += coin_discount * holdings_value
            total_weight += holdings_value
        
        if total_weight <= 0:
            return 0.0
        
        return total_discount / total_weight
    
    def _calculate_realized_profit_bonus(self, env: CryptoTradingEnvironment, agent_id: int) -> float:
        """
        Reward agents who successfully sold assets for profit.
        Analyzes sell events to find profitable exits.
        """
        realized_gains = 0.0
        
        for event in env.event_log:
            if event.get("agent_id") != agent_id:
                continue
            if event.get("type") not in ("sell", "trend", "speculate"):
                continue
            
            # Get realized cash from the sale
            realized_cash = event.get("realized_cash", 0.0)
            pnl = event.get("pnl", 0.0)
            
            if realized_cash > 0:
                realized_gains += realized_cash
            elif pnl > 0:
                realized_gains += pnl
        
        # Normalize by episode length and starting money
        # Target: realize at least 50% of starting money per episode
        avg_gain_per_turn = realized_gains / max(self.config.turns_per_episode, 1)
        normalized_bonus = avg_gain_per_turn / max(self.config.starting_money * 0.5, 1.0)
        
        # Cap bonus at 2.0
        return min(2.0, normalized_bonus)

    def _agent_behavior_label(self, env: CryptoTradingEnvironment, agent: EvolutionAgent) -> str:
        return self._dominant_observed_behavior(env, agent.agent_id, agent.genome)

    def _apply_diversity_adjustment(self, env: CryptoTradingEnvironment) -> dict[int, str]:
        labels_by_agent: dict[int, str] = {}
        for agent in self.population:
            labels_by_agent[agent.agent_id] = self._agent_behavior_label(env, agent)
        behavior_counts = Counter(labels_by_agent.values())
        population_size = max(len(self.population), 1)
        for agent in self.population:
            label = labels_by_agent[agent.agent_id]
            share = behavior_counts[label] / population_size
            agent.fitness -= share * self.config.evolution.overcrowding_penalty
            agent.fitness += (1.0 - share) * self.config.evolution.rarity_bonus
        return labels_by_agent

    def _dominant_behavior(self, genome: StrategyGenome) -> str:
        weights = {
            "buy": genome.buy_weight,
            "sell": genome.sell_weight,
            "mine": genome.mine_weight + genome.mining_aggressiveness,
            "work": genome.work_weight + genome.wealth_preservation,
            "create": genome.create_coin_weight + genome.coin_creation_bias,
            "trend": genome.trend_trade_weight + genome.trend_sensitivity,
            "speculate": genome.speculation_weight + genome.risk_tolerance,
            "arbitrage": genome.arbitrage_weight + genome.wealth_preservation,
            "leverage": genome.leverage_weight + genome.overconfidence_bias,
            "hold": genome.hold_weight + genome.wealth_preservation,
            "market_make": genome.market_making_weight + genome.volume_sensitivity,
        }
        return max(weights, key=weights.get)

    def _dominant_observed_behavior(self, env: CryptoTradingEnvironment, agent_id: int, genome: StrategyGenome) -> str:
        action_counts = env.metrics["agent_action_distribution"][agent_id]
        ordered_actions = [action.value for action in ActionType]
        if any(action_counts.values()):
            return max(ordered_actions, key=lambda action: (action_counts.get(action, 0), action))
        return self._dominant_behavior(genome)

    def _record_generation(
        self,
        generation: int,
        env: CryptoTradingEnvironment,
        ranked: list[EvolutionAgent],
        labels_by_agent: dict[int, str],
    ) -> dict[str, Any]:
        best = ranked[0]
        avg_fitness = mean(agent.fitness for agent in ranked)
        dominant = labels_by_agent[best.agent_id]
        best_action_mix = {action.value: env.metrics["agent_action_distribution"][best.agent_id].get(action.value, 0) for action in ActionType}
        behavior_counts = Counter(labels_by_agent.values())
        distribution = {
            "buy_weight": round(mean(agent.genome.buy_weight for agent in ranked), 4),
            "sell_weight": round(mean(agent.genome.sell_weight for agent in ranked), 4),
            "mine_weight": round(mean(agent.genome.mine_weight for agent in ranked), 4),
            "work_weight": round(mean(agent.genome.work_weight for agent in ranked), 4),
            "create_weight": round(mean(agent.genome.create_coin_weight for agent in ranked), 4),
            "trend_weight": round(mean(agent.genome.trend_trade_weight for agent in ranked), 4),
            "speculate_weight": round(mean(agent.genome.speculation_weight for agent in ranked), 4),
            "arbitrage_weight": round(mean(agent.genome.arbitrage_weight for agent in ranked), 4),
            "leverage_weight": round(mean(agent.genome.leverage_weight for agent in ranked), 4),
            "hold_weight": round(mean(agent.genome.hold_weight for agent in ranked), 4),
            "market_make_weight": round(mean(agent.genome.market_making_weight for agent in ranked), 4),
        }
        summary = {
            "generation": generation,
            "best_fitness": round(best.fitness, 4),
            "average_fitness": round(avg_fitness, 4),
            "best_agent_id": best.agent_id,
            "dominant_behavior": dominant,
            "best_agent_action_mix": best_action_mix,
            "behavior_counts": dict(behavior_counts),
            "strategy_diversity": len(behavior_counts),
            "best_genome": best.genome_dict(),
            "final_wealth": {agent_id: round(env.net_worth(agent_id), 4) for agent_id in env.agent_states},
        }
        self.history["best_fitness"].append(summary["best_fitness"])
        self.history["average_fitness"].append(summary["average_fitness"])
        self.history["best_genomes"].append(best.genome_dict())
        self.history["strategy_distribution"].append(distribution)
        self.history["dominant_behaviors"].append(dominant)
        self.history["strategy_diversity"].append(summary["strategy_diversity"])
        self.history["behavior_counts"].append(dict(behavior_counts))
        return summary

    def _next_generation(self, ranked: list[EvolutionAgent], labels_by_agent: dict[int, str]) -> list[EvolutionAgent]:
        population_size = len(ranked)
        elite_count = min(self.config.evolution.elite_count, population_size)
        survivor_count = max(elite_count + 1, int(population_size * self.config.evolution.selection_ratio))
        niche_preservation_count = max(0, self.config.evolution.niche_preservation_count)
        selected_ids: set[int] = set(agent.agent_id for agent in ranked[:elite_count])
        niche_groups: dict[str, list[EvolutionAgent]] = {}
        for agent in ranked:
            niche_groups.setdefault(labels_by_agent[agent.agent_id], []).append(agent)
        for niche_agents in niche_groups.values():
            for niche_agent in niche_agents[:niche_preservation_count]:
                selected_ids.add(niche_agent.agent_id)
        survivors: list[EvolutionAgent] = []
        for agent in ranked:
            if agent.agent_id in selected_ids:
                survivors.append(agent)
        for agent in ranked:
            if len(survivors) >= survivor_count:
                break
            if agent.agent_id not in selected_ids:
                survivors.append(agent)
                selected_ids.add(agent.agent_id)
        immigrant_count = max(1, int(population_size * self.config.evolution.random_agent_fraction))
        next_population: list[EvolutionAgent] = [
            self._clone_agent(agent, agent_id=index) for index, agent in enumerate(survivors[:elite_count])
        ]

        target_before_immigrants = max(elite_count, population_size - immigrant_count)
        while len(next_population) < target_before_immigrants:
            parent_a = self.random.choice(survivors)
            parent_b = self.random.choice(survivors)
            if self.random.random() < self.config.evolution.crossover_rate:
                genome = self._crossover(parent_a.genome, parent_b.genome)
            else:
                genome = StrategyGenome(**asdict(parent_a.genome))
            genome = self._mutate_genome(genome)
            child = EvolutionAgent(len(next_population), genome, self.config.seed + 9000 + len(next_population) + self.random.randint(0, 10000))
            next_population.append(child)
        while len(next_population) < population_size:
            agent_id = len(next_population)
            genome = StrategyGenome.random(random.Random(self.config.seed + 20000 + agent_id + self.random.randint(0, 10000)))
            next_population.append(EvolutionAgent(agent_id, genome, self.config.seed + 30000 + agent_id + self.random.randint(0, 10000)))
        return next_population

    def run(
        self,
        progress_callback: Callable[[dict[str, Any], CryptoTradingEnvironment], None] | None = None,
    ) -> tuple[list[dict[str, Any]], CryptoTradingEnvironment]:
        generation_summaries: list[dict[str, Any]] = []
        best_environment: CryptoTradingEnvironment | None = None

        for generation in range(1, self.config.generations + 1):
            env = CryptoTradingEnvironment(self.config, agents=self.population)
            env.run_episode(training=False)
            for agent in self.population:
                agent.fitness = self._fitness(env, agent)
            labels_by_agent = self._apply_diversity_adjustment(env)
            ranked = sorted(self.population, key=lambda agent: agent.fitness, reverse=True)
            summary = self._record_generation(generation, env, ranked, labels_by_agent)
            generation_summaries.append(summary)
            if self.best_agent is None or ranked[0].fitness > self.best_agent.fitness:
                self.best_agent = self._clone_agent(ranked[0], ranked[0].agent_id)
                best_environment = env
            env.metrics["generation_best_fitness"] = list(self.history["best_fitness"])
            env.metrics["generation_average_fitness"] = list(self.history["average_fitness"])
            env.metrics["generation_best_genomes"] = list(self.history["best_genomes"])
            env.metrics["generation_strategy_distribution"] = list(self.history["strategy_distribution"])
            env.metrics["generation_dominant_behaviors"] = list(self.history["dominant_behaviors"])
            env.metrics["generation_strategy_diversity"] = list(self.history["strategy_diversity"])
            env.metrics["generation_behavior_counts"] = list(self.history["behavior_counts"])
            if progress_callback is not None:
                progress_callback(summary, env)
            if self.config.debug:
                print(
                    f"[generation {generation:03d}] best_fitness={summary['best_fitness']:.4f} "
                    f"avg_fitness={summary['average_fitness']:.4f} dominant={summary['dominant_behavior']} "
                    f"diversity={summary['strategy_diversity']} "
                    f"genome={json.dumps(summary['best_genome'], sort_keys=True)}"
                )
            self.population = self._next_generation(ranked, labels_by_agent)

        if best_environment is None:
            raise RuntimeError("Evolution run finished without an environment")
        best_environment.metrics["generation_best_fitness"] = list(self.history["best_fitness"])
        best_environment.metrics["generation_average_fitness"] = list(self.history["average_fitness"])
        best_environment.metrics["generation_best_genomes"] = list(self.history["best_genomes"])
        best_environment.metrics["generation_strategy_distribution"] = list(self.history["strategy_distribution"])
        best_environment.metrics["generation_dominant_behaviors"] = list(self.history["dominant_behaviors"])
        best_environment.metrics["generation_strategy_diversity"] = list(self.history["strategy_diversity"])
        best_environment.metrics["generation_behavior_counts"] = list(self.history["behavior_counts"])
        self.last_environment = best_environment
        return generation_summaries, best_environment

    def save_population(self, model_dir: Path) -> None:
        model_dir.mkdir(parents=True, exist_ok=True)
        for agent in self.population:
            agent.save(model_dir / f"agent_{agent.agent_id}.pkl")
        if self.best_agent is not None:
            self.best_agent.save(model_dir / "best_agent.pkl")

    def load_population(self, model_dir: Path) -> None:
        loaded: list[EvolutionAgent] = []
        for agent_id in range(self.config.evolution.population_size):
            path = model_dir / f"agent_{agent_id}.pkl"
            if path.exists():
                genome = StrategyGenome.random(random.Random(self.config.seed + agent_id + 101))
                agent = EvolutionAgent(agent_id, genome, self.config.seed + agent_id + 1)
                agent.load(path)
                loaded.append(agent)
        if loaded:
            self.population = loaded
            self.config.num_agents = len(loaded)
