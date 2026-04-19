"""
Event Detection System - Identifies and reports major market events
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class MarketEvent:
    """Represents a significant market event."""
    event_type: str  # crash, surge, rug_pull, extreme_profit, extreme_loss, mining_shift, coin_death
    severity: int  # 1-10 scale
    turn: int
    coin_id: str | None = None
    agent_id: int | None = None
    description: str = ""
    details: dict[str, Any] = field(default_factory=dict)
    
    def summary(self) -> str:
        """Generate human-readable summary."""
        if self.event_type == "crash":
            return (
                f"💥 {self.coin_id} crashed {self.details.get('price_drop', 0):.0f}% "
                f"{'(Agent ' + str(self.agent_id) + ' sold ' + str(self.details.get('fraction_sold', 0) * 100, 0) + '% of supply) ' if self.agent_id is not None else ''}"
                f"in turn {self.turn}"
            )
        elif self.event_type == "surge":
            return (
                f"🚀 {self.coin_id} surged {self.details.get('price_gain', 0):.0f}% "
                f"during buying frenzy in turn {self.turn}"
            )
        elif self.event_type == "rug_pull":
            profit = self.details.get('seller_profit', 0)
            losses = self.details.get('other_losses', 0)
            return (
                f"🧨 Agent {self.agent_id} dumped {self.details.get('fraction_sold', 0):.0f}% of {self.coin_id}, "
                f"profiting ${profit:.0f} while others lost ${losses:.0f}"
            )
        elif self.event_type == "extreme_profit":
            source = self.details.get('source', 'trading')
            return (
                f"💰 Agent {self.agent_id} made {self.details.get('return_multiple', 0):.1f}x return "
                f"(${self.details.get('start_value', 0):.0f} → ${self.details.get('end_value', 0):.0f}) via {source}"
            )
        elif self.event_type == "extreme_loss":
            cause = self.details.get('cause', 'price collapse')
            return (
                f"💸 Agent {self.agent_id} lost {self.details.get('loss_percent', 0):.0f}% "
                f"(${self.details.get('start_value', 0):.0f} → ${self.details.get('end_value', 0):.0f}) due to {cause}"
            )
        elif self.event_type == "mining_shift":
            return (
                f"⛏️ Mining {self.details.get('direction', 'shifted')} for {self.coin_id} "
                f"(profitability {self.details.get('change', 0):+.0f}%)"
            )
        elif self.event_type == "coin_death":
            return (
                f"💀 {self.coin_id} died after {self.details.get('lifespan', 0)} turns "
                f"({self.details.get('cause', 'unknown')})"
            )
        else:
            return f"{self.event_type}: {self.description}"


class EventDetector:
    """
    Detects and tracks significant market events during simulation.
    """
    
    # Detection thresholds
    CRASH_THRESHOLD = 0.50  # 50% price drop
    SURGE_THRESHOLD = 0.80  # 80% price gain
    RUG_PULL_THRESHOLD = 0.50  # 50% of supply sold
    EXTREME_PROFIT_THRESHOLD = 5.0  # 5x return
    EXTREME_LOSS_THRESHOLD = 0.90  # 90% loss
    MINING_SHIFT_THRESHOLD = 0.50  # 50% profitability change
    
    def __init__(self) -> None:
        self.events: list[MarketEvent] = []
        self.price_history: dict[str, list[float]] = {}
        self.agent_values: dict[int, list[float]] = {}
        self.mining_profitability: dict[str, list[float]] = {}
        self.current_turn = 0
        
    def reset(self) -> None:
        """Reset detector for new episode."""
        self.events.clear()
        self.price_history.clear()
        self.agent_values.clear()
        self.mining_profitability.clear()
        self.current_turn = 0
        
    def update(self, env: Any) -> None:
        """
        Update detector with current environment state.
        Checks for significant events.
        """
        self.current_turn = env.turn
        
        # Track coin prices
        for coin_id, coin in env.market.coins.items():
            if coin_id not in self.price_history:
                self.price_history[coin_id] = []
            self.price_history[coin_id].append(coin.price)
            
            # Check for crashes and surges
            self._check_price_events(coin_id, coin)
            
            # Track mining profitability
            if coin_id not in self.mining_profitability:
                self.mining_profitability[coin_id] = []
            # Simplified profitability estimate
            profitability = coin.price / max(coin.total_supply, 1.0)
            self.mining_profitability[coin_id].append(profitability)
            self._check_mining_shift(coin_id)
        
        # Track agent net worth
        for agent_id, state in env.agent_states.items():
            if agent_id not in self.agent_values:
                self.agent_values[agent_id] = []
            
            holdings_value = sum(
                units * env.market.coins[coin_id].price
                for coin_id, units in state.holdings.items()
                if coin_id in env.market.coins
            )
            net_worth = state.cash + holdings_value
            self.agent_values[agent_id].append(net_worth)
            
            # Check for extreme profits/losses
            self._check_agent_events(agent_id, net_worth, state, env)
    
    def _check_price_events(self, coin_id: str, coin: Any) -> None:
        """Check for crashes and surges."""
        history = self.price_history.get(coin_id, [])
        
        if len(history) < 5:
            return
        
        # Check recent price changes (last 5 turns)
        recent = history[-5:]
        if len(recent) < 2:
            return
        
        peak = max(recent[:-1])
        trough = min(recent)
        current = recent[-1]
        
        # Check for crash
        if peak > 0:
            drop = (peak - current) / peak
            if drop > self.CRASH_THRESHOLD:
                # Check if we already logged this crash
                if not any(e.event_type == "crash" and e.coin_id == coin_id and 
                          abs(e.turn - self.current_turn) < 5 for e in self.events):
                    self.events.append(MarketEvent(
                        event_type="crash",
                        severity=min(10, int(drop * 15)),
                        turn=self.current_turn,
                        coin_id=coin_id,
                        description=f"{coin_id} crashed {drop*100:.0f}%",
                        details={
                            'price_drop': drop * 100,
                            'peak': peak,
                            'current': current,
                        }
                    ))
        
        # Check for surge
        if len(recent) >= 3 and recent[0] > 0:
            gain = (current - recent[0]) / recent[0]
            if gain > self.SURGE_THRESHOLD:
                if not any(e.event_type == "surge" and e.coin_id == coin_id and 
                          abs(e.turn - self.current_turn) < 5 for e in self.events):
                    self.events.append(MarketEvent(
                        event_type="surge",
                        severity=min(10, int(gain * 10)),
                        turn=self.current_turn,
                        coin_id=coin_id,
                        description=f"{coin_id} surged {gain*100:.0f}%",
                        details={
                            'price_gain': gain * 100,
                            'start': recent[0],
                            'current': current,
                        }
                    ))
    
    def _check_agent_events(self, agent_id: int, net_worth: float, state: Any, env: Any) -> None:
        """Check for extreme profits and losses."""
        history = self.agent_values.get(agent_id, [])
        
        if len(history) < 10:
            return
        
        # Check starting vs current value
        start_value = history[0]
        if start_value > 0:
            current_multiple = net_worth / start_value
            
            # Extreme profit
            if current_multiple > self.EXTREME_PROFIT_THRESHOLD:
                if not any(e.event_type == "extreme_profit" and e.agent_id == agent_id and 
                          abs(e.turn - self.current_turn) < 10 for e in self.events):
                    # Determine source
                    source = 'trading'
                    if len(state.coins_created) > 0:
                        source = 'coin creation'
                    elif state.mining_power > 2.0:
                        source = 'mining'
                    
                    self.events.append(MarketEvent(
                        event_type="extreme_profit",
                        severity=min(10, int(current_multiple)),
                        turn=self.current_turn,
                        agent_id=agent_id,
                        description=f"Agent {agent_id} made {current_multiple:.1f}x return",
                        details={
                            'return_multiple': current_multiple,
                            'start_value': start_value,
                            'end_value': net_worth,
                            'source': source,
                        }
                    ))
            
            # Extreme loss
            loss_percent = 1.0 - (net_worth / start_value)
            if loss_percent > self.EXTREME_LOSS_THRESHOLD:
                if not any(e.event_type == "extreme_loss" and e.agent_id == agent_id and 
                          abs(e.turn - self.current_turn) < 10 for e in self.events):
                    cause = 'price collapse'
                    if any(coin_id in state.holdings for coin_id in env.market.coins):
                        for coin_id, units in state.holdings.items():
                            if coin_id in env.market.coins:
                                coin = env.market.coins[coin_id]
                                if len(coin.price_history) > 10:
                                    if coin.price < coin.price_history[-10] * 0.5:
                                        cause = f'{coin_id} collapse'
                    
                    self.events.append(MarketEvent(
                        event_type="extreme_loss",
                        severity=min(10, int(loss_percent * 10)),
                        turn=self.current_turn,
                        agent_id=agent_id,
                        description=f"Agent {agent_id} lost {loss_percent*100:.0f}%",
                        details={
                            'loss_percent': loss_percent * 100,
                            'start_value': start_value,
                            'end_value': net_worth,
                            'cause': cause,
                        }
                    ))
    
    def _check_mining_shift(self, coin_id: str) -> None:
        """Check for mining profitability shifts."""
        history = self.mining_profitability.get(coin_id, [])
        
        if len(history) < 5:
            return
        
        recent_avg = sum(history[-3:]) / 3
        earlier_avg = sum(history[:-3]) / max(len(history) - 3, 1)
        
        if earlier_avg > 0:
            change = (recent_avg - earlier_avg) / earlier_avg
            
            if abs(change) > self.MINING_SHIFT_THRESHOLD:
                if not any(e.event_type == "mining_shift" and e.coin_id == coin_id and 
                          abs(e.turn - self.current_turn) < 5 for e in self.events):
                    direction = "profitable" if change > 0 else "unprofitable"
                    self.events.append(MarketEvent(
                        event_type="mining_shift",
                        severity=min(10, int(abs(change) * 8)),
                        turn=self.current_turn,
                        coin_id=coin_id,
                        description=f"Mining {direction} for {coin_id}",
                        details={
                            'direction': direction,
                            'change': change * 100,
                        }
                    ))
    
    def check_rug_pull(self, agent_id: int, coin_id: str, fraction_sold: float, 
                       price_before: float, price_after: float, env: Any) -> None:
        """Check for rug pull events with profit/loss tracking."""
        if fraction_sold > self.RUG_PULL_THRESHOLD:
            price_impact = (price_before - price_after) / max(price_before, 1e-6)
            
            # Calculate seller's profit
            agent_state = env.agent_states.get(agent_id)
            if agent_state:
                # Estimate profit based on cash change and holdings sold
                units_sold = agent_state.holdings.get(coin_id, 0) * fraction_sold
                avg_sell_price = (price_before + price_after) / 2.0
                sale_proceeds = units_sold * avg_sell_price
                
                # Estimate other agents' losses
                other_losses = 0.0
                for other_id, other_state in env.agent_states.items():
                    if other_id != agent_id and coin_id in other_state.holdings:
                        holdings = other_state.holdings[coin_id]
                        loss = holdings * price_impact * price_before
                        other_losses += loss
                
                self.events.append(MarketEvent(
                    event_type="rug_pull",
                    severity=min(10, int(fraction_sold * 15 + price_impact * 10)),
                    turn=self.current_turn,
                    agent_id=agent_id,
                    coin_id=coin_id,
                    description=f"Agent {agent_id} dumped {fraction_sold*100:.0f}% of {coin_id}",
                    details={
                        'fraction_sold': fraction_sold * 100,
                        'price_impact': price_impact * 100,
                        'seller_profit': sale_proceeds,
                        'other_losses': other_losses,
                    }
                ))
    
    def check_coin_death(self, coin_id: str, lifespan: int, cause: str) -> None:
        """Record coin death event."""
        self.events.append(MarketEvent(
            event_type="coin_death",
            severity=7,
            turn=self.current_turn,
            coin_id=coin_id,
            description=f"{coin_id} died after {lifespan} turns",
            details={
                'lifespan': lifespan,
                'cause': cause,
            }
        ))
    
    def get_summary(self, top_n: int = 15) -> list[str]:
        """
        Get human-readable summary of top events.
        
        Args:
            top_n: Number of top events to include (increased from 5 to 15)
            
        Returns:
            List of event summaries sorted by severity
        """
        # Sort by severity
        sorted_events = sorted(self.events, key=lambda e: e.severity, reverse=True)
        
        # Return top N summaries
        return [event.summary() for event in sorted_events[:top_n]]
    
    def get_all_events(self) -> list[MarketEvent]:
        """Get all detected events."""
        return self.events.copy()
    
    def get_events_by_type(self, event_type: str) -> list[MarketEvent]:
        """Get events of a specific type."""
        return [e for e in self.events if e.event_type == event_type]
