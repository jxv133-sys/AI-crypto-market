from __future__ import annotations

import os
import tempfile
from pathlib import Path

cache_root = Path(tempfile.gettempdir()) / "coingame_mpl_cache"
cache_root.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(cache_root))
os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))

from matplotlib.figure import Figure

from agent import ActionType
from environment import CryptoTradingEnvironment


def _create_figure(figsize: tuple[float, float]) -> Figure:
    fig = Figure(figsize=figsize)
    fig.subplots_adjust(left=0.12, right=0.97, top=0.92, bottom=0.14, hspace=0.38, wspace=0.28)
    return fig


def _style_axes(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)


def create_coin_price_figure(env: CryptoTradingEnvironment) -> Figure:
    fig = _create_figure((6, 4))
    ax = fig.add_subplot(111)
    if env.market.coins:
        # Sort coins by creation turn so newly created ones are plotted last (on top)
        sorted_coins = sorted(env.market.coins.values(), key=lambda c: c.created_turn)
        
        # First, draw vertical lines at coin creation points
        creation_turns = sorted(set(coin.created_turn for coin in env.market.coins.values()))
        for turn in creation_turns:
            if turn > 0:  # Don't mark turn 0
                ax.axvline(x=turn, color='gray', linestyle=':', alpha=0.3, linewidth=0.5, zorder=0)
        
        # Then plot coin prices on top
        for coin in sorted_coins:
            # Use different line styles based on coin age
            age = len(coin.price_history)
            if age < 5:
                # Newly created coin: dashed line, brighter color
                ax.plot(coin.price_history, label=f"{coin.coin_id} (new)", linestyle='--', linewidth=2, zorder=5)
            else:
                # Established coin: solid line
                ax.plot(coin.price_history, label=coin.coin_id, zorder=5)
        
        # Legend removed for cleaner display
        ax.grid(True, alpha=0.25)
    else:
        ax.text(0.5, 0.5, "No coins created yet", ha="center", va="center")
    _style_axes(ax, "Coin Prices", "Turn", "Price")
    return fig


def create_net_worth_figure(env: CryptoTradingEnvironment) -> Figure:
    fig = _create_figure((6, 4))
    ax = fig.add_subplot(111)
    if env.metrics["net_worth_history"]:
        for agent_id, history in env.metrics["net_worth_history"].items():
            ax.plot(history, alpha=0.7, linewidth=1)
        # Legend removed for cleaner display
    else:
        ax.text(0.5, 0.5, "No net worth data yet", ha="center", va="center")
    _style_axes(ax, "Agent Net Worth", "Turn", "Net Worth")
    return fig


def create_wealth_distribution_figure(env: CryptoTradingEnvironment) -> Figure:
    fig = _create_figure((6, 4))
    ax = fig.add_subplot(111)
    latest_wealth = env.metrics["wealth_snapshots"][-1] if env.metrics["wealth_snapshots"] else {}
    if latest_wealth:
        ax.bar([str(agent_id) for agent_id in latest_wealth], latest_wealth.values(), color="#e76f51")
    else:
        ax.text(0.5, 0.5, "No wealth snapshot yet", ha="center", va="center")
    _style_axes(ax, "Final Wealth Distribution", "Agent", "Net Worth")
    return fig


def create_behavior_mix_figure(env: CryptoTradingEnvironment) -> Figure:
    """Create a simple bar chart showing current behavior distribution."""
    fig = _create_figure((6, 4))
    ax = fig.add_subplot(111)
    
    # Try NEAT history first
    behavior_counts = env.metrics.get("generation_behavior_counts", [])
    
    # Fallback to action_distribution for NEAT
    if not behavior_counts:
        action_dist = env.metrics.get("action_distribution", {})
        if action_dist:
            behavior_counts = [action_dist]
    
    if behavior_counts:
        # Use latest generation data
        latest = behavior_counts[-1] if behavior_counts else {}
        action_order = ["buy", "sell", "mine", "work", "trend", "speculate", "arbitrage", "leverage", "market_make"]

        # Simple bar chart
        x = range(len(action_order))
        heights = [latest.get(action, 0) for action in action_order]
        ax.bar(x, heights, color=['#2a9d8f', '#e9c46a', '#f4a261', '#264653',
                                   '#e76f51', '#e9c46a', '#2a9d8f', '#f4a261', '#264653'])

        ax.set_xticks(x)
        ax.set_xticklabels([a.capitalize() for a in action_order], rotation=45, ha='right', fontsize=7)
        ax.set_ylabel('Agents Using Action')
        ax.set_title('Current Behavior Distribution')
        ax.grid(True, alpha=0.25, axis='y')
    else:
        ax.text(0.5, 0.5, "No behavior data yet", ha="center", va="center")

    fig.tight_layout()
    return fig


def create_behavior_trends_figure(env: CryptoTradingEnvironment) -> Figure:
    """Create a line chart showing behavior trends over generations."""
    fig = _create_figure((6, 4))
    ax = fig.add_subplot(111)
    
    # Try NEAT history first
    behavior_counts = env.metrics.get("generation_behavior_counts", [])
    
    # Fallback to action_distribution history for NEAT
    if not behavior_counts:
        action_history = env.metrics.get("action_distribution", [])
        if isinstance(action_history, list) and action_history:
            behavior_counts = action_history

    if behavior_counts:
        x_values = list(range(1, len(behavior_counts) + 1))
        # Focus on key behaviors
        focus_actions = ["buy", "sell", "mine", "work", "speculate", "leverage"]
        colors = ['#2a9d8f', '#e9c46a', '#f4a261', '#264653', '#e76f51', '#8d99ae']

        for action, color in zip(focus_actions, colors):
            ax.plot(x_values, [row.get(action, 0) for row in behavior_counts],
                   label=action.capitalize(), color=color, linewidth=2, alpha=0.8)

        ax.set_xlabel('Generation')
        ax.set_ylabel('Agents Using Action')
        ax.set_title('Behavior Trends Over Generations')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.25)
    else:
        ax.text(0.5, 0.5, "No behavior trend data yet", ha="center", va="center")

    fig.tight_layout()
    return fig


def create_training_stats_figure(env: CryptoTradingEnvironment) -> Figure:
    fig = _create_figure((8.6, 6.6))
    axes = fig.subplots(2, 2)
    reward_ax, network_ax, mining_ax, action_ax = axes.flatten()

    generation_best = env.metrics.get("generation_best_fitness", [])
    generation_avg = env.metrics.get("generation_average_fitness", [])
    if generation_best:
        reward_ax.plot(generation_best, label="Best Fitness", color="#1d4ed8")
        reward_ax.plot(generation_avg, label="Average Fitness", color="#111827", linestyle="--")
        reward_ax.legend(loc="best", fontsize=8)
    else:
        reward_ax.text(0.5, 0.5, "No generation fitness yet", ha="center", va="center")
    _style_axes(reward_ax, "Evolution Fitness", "Generation", "Fitness")

    if env.metrics["network_hashrate_history"]:
        network_ax.plot(env.metrics["network_hashrate_history"], color="#457b9d", label="Hashrate")
        network_ax2 = network_ax.twinx()
        network_ax2.plot(env.metrics["electricity_price_history"], color="#e76f51", linestyle="--", label="Electricity")
        network_ax2.set_ylabel("Electricity Price")
    else:
        network_ax.text(0.5, 0.5, "No mining network data", ha="center", va="center")
    _style_axes(network_ax, "Hashrate And Power Cost", "Turn", "Total Network Hashrate")

    strategy_distribution = env.metrics.get("generation_strategy_distribution", [])
    if strategy_distribution:
        mining_ax.plot([row["mine_weight"] for row in strategy_distribution], label="Mine", color="#059669")
        mining_ax.plot([row["buy_weight"] for row in strategy_distribution], label="Buy", color="#2563eb")
        mining_ax.plot([row["sell_weight"] for row in strategy_distribution], label="Sell", color="#dc2626")
        mining_ax.plot([row["work_weight"] for row in strategy_distribution], label="Work", color="#d97706")
        mining_ax.plot([row["trend_weight"] for row in strategy_distribution], label="Trend", color="#7c3aed")
        mining_ax.plot([row["leverage_weight"] for row in strategy_distribution], label="Leverage", color="#db2777")
        mining_ax.legend(loc="best", fontsize=8)
    else:
        mining_ax.text(0.5, 0.5, "No strategy distribution yet", ha="center", va="center")
    _style_axes(mining_ax, "Genome Weight Trends", "Generation", "Average Weight")

    action_labels = [action.value for action in ActionType]
    action_values = [env.metrics["action_distribution"].get(label, 0) for label in action_labels]
    if any(action_values):
        action_ax.bar(action_labels, action_values, color="#f4a261")
        action_ax.tick_params(axis="x", rotation=35)
    else:
        action_ax.bar(action_labels, action_values, color="#cbd5e1")
        action_ax.tick_params(axis="x", rotation=35)
    _style_axes(action_ax, "Action Distribution", "Action", "Count")
    return fig


def create_mining_system_figure(env: CryptoTradingEnvironment) -> Figure:
    fig = _create_figure((8.8, 6.8))
    axes = fig.subplots(2, 2)
    hashrate_ax, price_ax, participation_ax, profit_ax = axes.flatten()

    if env.metrics["network_hashrate_history"]:
        hashrate_ax.plot(env.metrics["network_hashrate_history"], color="#2a9d8f")
    else:
        hashrate_ax.text(0.5, 0.5, "No hashrate data yet", ha="center", va="center")
    _style_axes(hashrate_ax, "Total Network Hashrate", "Turn", "Hashrate")

    if env.metrics["electricity_price_history"]:
        price_ax.plot(env.metrics["electricity_price_history"], color="#e76f51")
    else:
        price_ax.text(0.5, 0.5, "No electricity data yet", ha="center", va="center")
    _style_axes(price_ax, "Electricity Price", "Turn", "Price")

    if env.metrics["mining_participation_history"]:
        for agent_id, history in env.metrics["mining_participation_history"].items():
            participation_ax.plot(history, label=f"A{agent_id}")
        participation_ax.legend(loc="best", fontsize=8)
    else:
        participation_ax.text(0.5, 0.5, "No participation data yet", ha="center", va="center")
    _style_axes(participation_ax, "Mining Participation Per Agent", "Turn", "Participation")

    if env.metrics["mining_profit_history"]:
        for agent_id, history in env.metrics["mining_profit_history"].items():
            profit_ax.plot(history, label=f"A{agent_id}")
        profit_ax.legend(loc="best", fontsize=8)
    else:
        profit_ax.text(0.5, 0.5, "No mining profit data yet", ha="center", va="center")
    _style_axes(profit_ax, "Mining Profitability Trends", "Turn", "Profit")
    return fig


def create_evolution_overview_figure(env: CryptoTradingEnvironment) -> Figure:
    fig = _create_figure((8.8, 6.8))
    axes = fig.subplots(2, 2)
    best_ax, behavior_ax, genome_ax, reward_ax = axes.flatten()

    best = env.metrics.get("generation_best_fitness", [])
    avg = env.metrics.get("generation_average_fitness", [])
    if best:
        best_ax.plot(best, label="Best", color="#1d4ed8")
        best_ax.plot(avg, label="Average", color="#111827", linestyle="--")
        best_ax.legend(loc="best", fontsize=8)
    else:
        best_ax.text(0.5, 0.5, "No evolution data yet", ha="center", va="center")
    _style_axes(best_ax, "Best Vs Average Fitness", "Generation", "Fitness")

    dominant = env.metrics.get("generation_dominant_behaviors", [])
    action_order = [action.value for action in ActionType]
    if dominant:
        behavior_counts: dict[str, int] = {action: 0 for action in action_order}
        for label in dominant:
            behavior_counts[label] = behavior_counts.get(label, 0) + 1
        behavior_ax.bar(behavior_counts.keys(), behavior_counts.values(), color="#8b5cf6")
        behavior_ax.tick_params(axis="x", rotation=25)
    else:
        behavior_ax.bar(action_order, [0 for _ in action_order], color="#cbd5e1")
        behavior_ax.tick_params(axis="x", rotation=25)
    _style_axes(behavior_ax, "Dominant Behavior Frequency", "Behavior", "Count")

    best_genomes = env.metrics.get("generation_best_genomes", [])
    if best_genomes:
        genome_ax.plot([row["risk_tolerance"] for row in best_genomes], label="Risk")
        genome_ax.plot([row["trend_sensitivity"] for row in best_genomes], label="Trend")
        genome_ax.plot([row["mining_aggressiveness"] for row in best_genomes], label="Mining")
        genome_ax.legend(loc="best", fontsize=8)
    else:
        genome_ax.text(0.5, 0.5, "No genome trend data", ha="center", va="center")
    _style_axes(genome_ax, "Top Genome Traits", "Generation", "Trait Value")

    if env.metrics["average_reward"]:
        reward_ax.plot(env.metrics["average_reward"], color="#0f766e")
        diversity = env.metrics.get("generation_strategy_diversity", [])
        if diversity:
            reward_ax2 = reward_ax.twinx()
            reward_ax2.plot(diversity, color="#7c3aed", linestyle="--")
            reward_ax2.set_ylabel("Strategy Diversity")
    else:
        reward_ax.text(0.5, 0.5, "No average reward data", ha="center", va="center")
    _style_axes(reward_ax, "Average Episode Reward", "Generation", "Reward")
    return fig


def create_coins_created_figure(env: CryptoTradingEnvironment) -> Figure:
    fig = _create_figure((6, 4))
    ax = fig.add_subplot(111)
    if env.metrics["coins_created_history"]:
        ax.plot(env.metrics["coins_created_history"], color="#2a9d8f")
    else:
        ax.text(0.5, 0.5, "No coin history yet", ha="center", va="center")
    _style_axes(ax, "Coins Created Over Time", "Turn", "Coins")
    return fig


def create_coin_ownership_figure(env: CryptoTradingEnvironment) -> Figure:
    """Create a line chart showing each agent's total coin units over turns."""
    fig = _create_figure((6, 4))
    ax = fig.add_subplot(111)
    
    # Check if we have turn-by-turn ownership history
    if env.metrics.get("agent_coin_ownership_history"):
        ownership_history = env.metrics["agent_coin_ownership_history"]
        
        # Plot each agent as a separate line (showing total UNITS owned)
        for agent_id in sorted(ownership_history.keys()):
            agent_data = ownership_history[agent_id]
            ax.plot(agent_data, alpha=0.7, linewidth=1)
        
        ax.set_xlabel('Turn')
        ax.set_ylabel('Total Coin Units')
        ax.set_title('Agent Coin Holdings Over Time')
        # Legend removed to reduce clutter - lines represent individual agents
        ax.grid(True, alpha=0.25)
        ax.set_ylim(bottom=0)
    else:
        # Fallback: show current snapshot if no history
        agent_ids = sorted(env.agent_states.keys())
        
        if agent_ids:
            created_counts = [len(env.agent_states[aid].coins_created) for aid in agent_ids]
            owned_counts = [
                sum(1 for cid, units in env.agent_states[aid].holdings.items() 
                    if units > 0 and cid in env.market.coins)
                for aid in agent_ids
            ]
            
            ax.plot(agent_ids, created_counts, 'o-', label='Coins Created', color='#2a9d8f', 
                    markersize=4, linewidth=1.5, alpha=0.8)
            ax.plot(agent_ids, owned_counts, 's-', label='Coins Owned', color='#e9c46a', 
                    markersize=4, linewidth=1.5, alpha=0.8)
            
            ax.set_xlabel('Agent ID')
            ax.set_ylabel('Number of Coins')
            ax.set_title('Coin Ownership by Agent (Current)')
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.25)
            ax.set_ylim(bottom=0)
        else:
            ax.text(0.5, 0.5, "No agent data yet", ha="center", va="center")
    
    fig.tight_layout()
    return fig


def plot_simulation_metrics(env: CryptoTradingEnvironment, output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    figures = {
        "coin_prices.png": create_coin_price_figure(env),
        "agent_net_worth.png": create_net_worth_figure(env),
        "coin_ownership.png": create_coin_ownership_figure(env),
        "behavior_mix.png": create_behavior_mix_figure(env),
        "behavior_trends.png": create_behavior_trends_figure(env),
        "mining_system.png": create_mining_system_figure(env),
        "evolution_overview.png": create_evolution_overview_figure(env),
    }
    paths: list[Path] = []
    for filename, figure in figures.items():
        path = output_dir / filename
        figure.savefig(path)
        paths.append(path)
    return paths
