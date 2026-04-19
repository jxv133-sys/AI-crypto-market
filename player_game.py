"""
Player Game Mode - Turn-based competition against AI agents
Uses the same economic model as the main simulation
"""

from __future__ import annotations

import math
from typing import Any

from agent import ActionType, EvolutionAgent, StrategyGenome
from config import GameConfig
from environment import CryptoTradingEnvironment


class PlayerGame:
    """
    Turn-based game where player competes against AI agents.
    All mechanics match the main simulation exactly.
    """
    
    def __init__(
        self,
        num_ai_agents: int = 11,  # Total will be num_ai_agents + 1 (player)
        turns_per_episode: int = 80,
        starting_money: float = 120.0,
        seed: int | None = None,
    ) -> None:
        """
        Initialize player game.
        
        Args:
            num_ai_agents: Number of AI opponents (default 11, total 12 agents)
            turns_per_episode: Turns per episode
            starting_money: Starting cash for all agents
            seed: Random seed for reproducibility
        """
        self.num_ai_agents = num_ai_agents
        self.total_agents = num_ai_agents + 1  # AI agents + player
        self.player_id = 0  # Player is always agent 0
        self.turns_per_episode = turns_per_episode
        self.starting_money = starting_money
        
        # Create config with player as agent 0
        config = GameConfig()
        config.num_agents = self.total_agents
        config.turns_per_episode = turns_per_episode
        config.starting_money = starting_money
        config.seed = seed if seed is not None else 7
        config.debug = False
        
        self.config = config
        self.env = CryptoTradingEnvironment(config)
        
        # Replace player agent with a placeholder (player chooses actions manually)
        import random
        player_genome = StrategyGenome.random(random.Random(config.seed + 999))
        self.player_agent = EvolutionAgent(self.player_id, player_genome, config.seed + 999)
        self.env.agents[self.player_id] = self.player_agent
        
        # Game state
        self.current_episode = 0
        self.current_turn = 0
        self.game_over = False
        self.waiting_for_player = False
        self.player_action: ActionType | None = None
        self.player_action_coin: str | None = None
        self.player_action_fraction: float | None = None
        
        # Statistics tracking
        self.player_turn_history: list[dict[str, Any]] = []
        self.episode_summaries: list[dict[str, Any]] = []
        
    def reset(self) -> None:
        """Reset the game to initial state."""
        self.env.reset()
        self.current_episode = 0
        self.current_turn = 0
        self.game_over = False
        self.waiting_for_player = False
        self.player_turn_history.clear()
        self.episode_summaries.clear()
        
    def get_game_state(self) -> dict[str, Any]:
        """
        Get comprehensive game state for UI display.
        Returns all information player needs to make decisions.
        """
        snapshot = self.env.live_snapshot()
        
        # Player's portfolio
        player_state = snapshot["agents"][self.player_id]
        player_holdings_value = sum(
            units * self.env.market.coins[coin_id].price
            for coin_id, units in player_state["holdings"].items()
            if coin_id in self.env.market.coins and units > 0
        )
        player_net_worth = player_state["cash"] + player_holdings_value
        
        # Market data with trends
        market_data = []
        for coin in snapshot["market"]:
            coin_id = coin["coin_id"]
            if coin_id in self.env.market.coins:
                coin_obj = self.env.market.coins[coin_id]
                # Calculate price trend
                trend = "stable"
                if len(coin_obj.price_history) >= 3:
                    recent = coin_obj.price_history[-3:]
                    if recent[-1] > recent[0] * 1.05:
                        trend = "rising"
                    elif recent[-1] < recent[0] * 0.95:
                        trend = "falling"
                
                # Calculate volatility
                volatility = "low"
                if len(coin_obj.price_history) >= 5:
                    prices = coin_obj.price_history[-5:]
                    price_range = (max(prices) - min(prices)) / max(min(prices), 0.01)
                    if price_range > 0.20:
                        volatility = "high"
                    elif price_range > 0.10:
                        volatility = "medium"
                
                market_data.append({
                    **coin,
                    "trend": trend,
                    "volatility": volatility,
                    "liquidity": coin_obj.liquidity_pool,
                    "market_cap": coin_obj.market_cap,
                })
        
        # Leaderboard with performance metrics
        leaderboard = []
        for agent in snapshot["agents"]:
            agent_id = agent["agent_id"]
            holdings_value = sum(
                units * self.env.market.coins[coin_id].price
                for coin_id, units in agent["holdings"].items()
                if coin_id in self.env.market.coins and units > 0
            )
            net_worth = agent["cash"] + holdings_value
            
            # Calculate recent performance (last 10 turns)
            recent_performance = 0.0
            if agent_id in self.env.metrics.get("net_worth_history", {}):
                history = self.env.metrics["net_worth_history"][agent_id]
                if len(history) >= 2:
                    recent_performance = (history[-1] - history[-2]) / max(history[-2], 1.0) * 100
            
            leaderboard.append({
                "agent_id": agent_id,
                "is_player": agent_id == self.player_id,
                "cash": agent["cash"],
                "holdings_value": holdings_value,
                "net_worth": net_worth,
                "recent_performance": recent_performance,
                "rank": 0,  # Will be calculated after sorting
            })
        
        # Sort and assign ranks
        leaderboard.sort(key=lambda x: x["net_worth"], reverse=True)
        for rank, entry in enumerate(leaderboard, 1):
            entry["rank"] = rank
        
        # Format recent events for display (include agent actions)
        formatted_events = []
        for event in snapshot.get("recent_events", [])[-10:]:
            if isinstance(event, dict):
                agent_id = event.get("agent_id", "?")
                action = event.get("action", "unknown")
                action_str = str(action)
                
                if "profit" in event and event["profit"] != 0:
                    profit_str = f"profit=${event['profit']:.2f}"
                elif "income" in event:
                    profit_str = f"earned=${event['income']:.2f}"
                else:
                    profit_str = ""
                
                formatted_events.append(f"Agent {agent_id}: {action_str} {profit_str}".strip())
            else:
                formatted_events.append(str(event))
        
        # Available actions for player
        available_actions = self._get_available_actions()
        
        return {
            "episode": self.current_episode,
            "turn": self.current_turn,
            "game_over": self.game_over,
            "waiting_for_player": self.waiting_for_player,
            "player": {
                "agent_id": self.player_id,
                "cash": player_state["cash"],
                "holdings": player_state["holdings"],
                "holdings_value": player_holdings_value,
                "net_worth": player_net_worth,
                "mining_power": player_state["mining_power"],
                "mining_allocation": player_state.get("mining_allocation"),
            },
            "market": market_data,
            "leaderboard": leaderboard,
            "available_actions": available_actions,
            "recent_events": formatted_events,
            "highlights": snapshot.get("highlights", []),
        }
    
    def _get_available_actions(self) -> list[dict[str, Any]]:
        """Get list of actions available to player."""
        actions = []
        player_state = self.env.agent_states[self.player_id]
        
        # Basic actions
        actions.append({
            "type": ActionType.WORK.value,
            "label": "Work",
            "description": f"Earn ${self.env._current_work_income(self.player_id):.2f}",
            "enabled": True,
        })
        
        actions.append({
            "type": ActionType.MINE.value,
            "label": "Mine (General)",
            "description": "Mine network coins",
            "enabled": True,
        })
        
        actions.append({
            "type": ActionType.HOLD.value,
            "label": "Hold",
            "description": "Wait and observe",
            "enabled": True,
        })
        
        # Coin-specific actions
        for coin_id in self.env.market.coins:
            coin = self.env.market.coins[coin_id]
            
            # Mining specific coin
            actions.append({
                "type": ActionType.MINE.value,
                "coin_id": coin_id,
                "label": f"Mining {coin_id}",
                "description": f"Mining {coin_id} coins",
                "enabled": True,
            })
            
            # Buy
            if player_state.cash >= 1.0:
                for fraction in self.config.market.buy_fractions:
                    spend = min(player_state.cash * fraction, self.config.market.max_transaction)
                    if spend >= 1.0:
                        actions.append({
                            "type": ActionType.BUY.value,
                            "coin_id": coin_id,
                            "fraction": fraction,
                            "label": f"Buy {coin_id} ({fraction*100:.0f}%)",
                            "description": f"Spend up to ${spend:.2f}",
                            "enabled": True,
                        })
            
            # Sell
            holdings = player_state.holdings.get(coin_id, 0.0)
            if holdings > 0.5:
                sellable = self.env._creator_sellable_units(self.player_id, coin_id, holdings)
                if sellable > 0.5:
                    for fraction in self.config.market.sell_fractions:
                        if holdings * fraction >= 0.5:
                            actions.append({
                                "type": ActionType.SELL.value,
                                "coin_id": coin_id,
                                "fraction": fraction,
                                "label": f"Sell {coin_id} ({fraction*100:.0f}%)",
                                "description": f"Sell {fraction*100:.0f}% of holdings",
                                "enabled": True,
                            })
            
            # Trend trading
            actions.append({
                "type": ActionType.TREND.value,
                "coin_id": coin_id,
                "label": f"Trend Trade {coin_id}",
                "description": "Trade based on price trend",
                "enabled": True,
            })
            
            # Speculate
            actions.append({
                "type": ActionType.SPECULATE.value,
                "coin_id": coin_id,
                "label": f"Speculate {coin_id}",
                "description": "Bet on price direction",
                "enabled": True,
            })
            
            # Arbitrage
            actions.append({
                "type": ActionType.ARBITRAGE.value,
                "coin_id": coin_id,
                "label": f"Arbitrage {coin_id}",
                "description": "Exploit price differences",
                "enabled": player_state.cash >= 4.0,
            })
            
            # Market making
            actions.append({
                "type": ActionType.MARKET_MAKE.value,
                "coin_id": coin_id,
                "label": f"Market Make {coin_id}",
                "description": "Profit from bid-ask spread",
                "enabled": player_state.cash >= 4.0,
            })
            
            # Leverage
            if player_state.cash >= 4.0:
                actions.append({
                    "type": ActionType.LEVERAGE.value,
                    "coin_id": coin_id,
                    "label": f"Leverage Trade {coin_id}",
                    "description": "Trade with leverage (high risk)",
                    "enabled": True,
                })
        
        # Create coin
        create_cost = self.config.market.create_coin_cost + self.config.market.create_liquidity_cost
        can_create = (
            player_state.cash >= create_cost
            and len(self.env.market.coins) < self.config.market.max_active_coins
            and len(player_state.coins_created) < self.config.market.max_coins_per_agent
        )
        actions.append({
            "type": ActionType.CREATE.value,
            "label": "Create Coin",
            "description": f"Launch new coin (cost: ${create_cost:.2f})",
            "enabled": can_create,
        })
        
        return actions
    
    def submit_player_action(
        self,
        action_type: str,
        coin_id: str | None = None,
        fraction: float | None = None,
    ) -> bool:
        """
        Submit player's chosen action.
        
        Args:
            action_type: Type of action (e.g., "buy", "sell", "mine")
            coin_id: Coin ID if action requires it
            fraction: Fraction for buy/sell actions
            
        Returns:
            True if action was valid and submitted
        """
        try:
            action_type_enum = ActionType(action_type)
        except ValueError:
            return False
        
        self.player_action = action_type_enum
        self.player_action_coin = coin_id
        self.player_action_fraction = fraction
        self.waiting_for_player = False
        
        return True
    
    def step(self) -> dict[str, Any]:
        """
        Advance game by one step.
        If waiting for player, returns immediately.
        Otherwise processes AI agents and player action.
        
        Returns:
            Game state after step
        """
        if self.game_over:
            return self.get_game_state()
        
        if self.waiting_for_player:
            # Waiting for player input
            return self.get_game_state()
        
        # Process turn
        self._process_turn()
        
        return self.get_game_state()
    
    def _process_turn(self) -> None:
        """Process one turn of the game."""
        # Get player's available actions
        available_actions = self.env.available_actions(self.player_id)
        
        if not available_actions:
            # No actions available, skip player
            self.waiting_for_player = False
        else:
            # Wait for player to choose
            self.waiting_for_player = True
            return
        
        # If we get here, process AI agents (player action handled separately)
        # This is called after player submits action
        pass
    
    def process_full_turn(self) -> dict[str, Any]:
        """
        Process a complete turn including player action and AI agents.
        Call this after player submits action.

        Returns:
            Game state after turn
        """
        if not self.waiting_for_player and self.player_action is not None:
            # Player has chosen action, execute it
            self._execute_player_action()
            self.player_action = None
            self.player_action_coin = None
            self.player_action_fraction = None
        
        # Process AI agent actions
        self._process_ai_actions()

        return self.get_game_state()
    
    def _process_ai_actions(self) -> None:
        """Process actions for all AI agents."""
        for agent in self.env.agents:
            if agent.agent_id == self.player_id:
                continue  # Skip player
            
            # Get AI's available actions
            ai_actions = self.env.available_actions(agent.agent_id)
            if not ai_actions:
                continue
            
            # Let agent choose action using its genome
            state = self.env.observe(agent.agent_id)
            chosen_actions = agent.select_actions(state, ai_actions, training=False)
            
            if not chosen_actions:
                continue
            
            # Execute first chosen action
            action = chosen_actions[0]
            
            # Handle mining separately (batched)
            if action.action_type == ActionType.MINE:
                mining_actions = [(agent.agent_id, action)]
                results = self.env._resolve_mining_actions(mining_actions)
                if agent.agent_id in results:
                    reward, event = results[agent.agent_id]
                    agent.record_reward(reward)
                    # Add to event log for display - store action string
                    self.env.event_log.append({
                        "agent_id": agent.agent_id,
                        "action": action.key,  # Store string
                        "profit": event.get("profit", 0),
                        "type": "mine",
                    })
            else:
                # Execute other actions
                reward, event = self.env.apply_action(agent.agent_id, action)
                agent.record_reward(reward)
                # Add to event log for display - store action string not object
                event["agent_id"] = agent.agent_id
                event["action"] = action.key  # Store string, not Action object
                self.env.event_log.append(event)
    
    def _execute_player_action(self) -> None:
        """Execute player's chosen action within the environment."""
        if self.player_action is None:
            return
        
        from agent import Action
        
        # Create action object
        action = Action(
            action_type=self.player_action,
            coin_id=self.player_action_coin,
            fraction=self.player_action_fraction,
        )
        
        # Mining is batched - just execute with environment's batch resolver
        if self.player_action == ActionType.MINE:
            # Create a single-item mining batch with just the player
            mining_actions = [(self.player_id, action)]
            results = self.env._resolve_mining_actions(mining_actions)
            
            if self.player_id in results:
                reward, event = results[self.player_id]
                self.player_agent.record_reward(reward)
                self.player_turn_history.append({
                    "turn": self.current_turn,
                    "action": str(action),
                    "reward": reward,
                    "event": event,
                })
            return
        
        # Execute other actions through environment (same as AI agents)
        reward, event = self.env.apply_action(self.player_id, action)
        self.player_agent.record_reward(reward)
        
        # Record for statistics
        self.player_turn_history.append({
            "turn": self.current_turn,
            "action": str(action),
            "reward": reward,
            "event": event,
        })
    
    def next_turn(self) -> dict[str, Any]:
        """
        Advance to next turn after player action is processed.
        
        Returns:
            Game state at start of new turn
        """
        # Run environment for remaining agents
        # (Player action already executed via process_full_turn)
        
        # Update turn counter
        self.current_turn += 1
        
        # Check for episode end
        if self.current_turn >= self.config.turns_per_episode:
            self._end_episode()
        
        return self.get_game_state()
    
    def _end_episode(self) -> None:
        """Handle end of episode."""
        summary = self.env._finalize_episode()
        self.episode_summaries.append(summary)
        
        self.current_episode += 1
        self.current_turn = 0
        
        # Reset for next episode
        self.env.reset()
        
        if self.current_episode >= self.config.episodes:
            self.game_over = True
    
    def get_player_statistics(self) -> dict[str, Any]:
        """Get player performance statistics."""
        if not self.player_turn_history:
            return {"error": "No data yet"}
        
        total_reward = sum(h["reward"] for h in self.player_turn_history)
        profitable_turns = sum(1 for h in self.player_turn_history if h["reward"] > 0)
        
        # Action distribution
        action_counts: dict[str, int] = {}
        for h in self.player_turn_history:
            action = h["action"].split("|")[0]
            action_counts[action] = action_counts.get(action, 0) + 1
        
        return {
            "total_turns": len(self.player_turn_history),
            "total_reward": total_reward,
            "avg_reward": total_reward / len(self.player_turn_history),
            "profitable_turns": profitable_turns,
            "win_rate": profitable_turns / len(self.player_turn_history) * 100,
            "action_distribution": action_counts,
            "episodes_completed": self.current_episode,
        }


# Need to import Action class
from agent import Action
