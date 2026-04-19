from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class MarketConfig:
    # Coin creation costs (lowered to enable more creation)
    create_coin_cost: float = 35.0  # Reduced from 45.0
    create_liquidity_cost: float = 10.0  # Reduced from 15.0
    base_coin_price: float = 1.0  # Realistic starting price
    initial_coin_supply: float = 30.0  # Very low supply for high prices
    max_supply: float = 150.0  # Very limited
    creator_allocation: float = 0.7
    initial_liquidity_fraction: float = 0.15  # Good liquidity for trading
    creator_lock_turns: int = 5
    creator_unlock_per_turn: float = 0.14

    # Price impact and market depth (MORE VOLATILE FOR $50+ PRICES)
    price_impact_buy: float = 0.45  # Higher for more price movement
    price_impact_sell: float = 0.55  # Higher for more movement
    crash_multiplier: float = 1.5  # More amplification
    liquidity_depth: float = 80.0  # Lower = more volatile
    volatility_cap_per_trade: float = 0.15  # Allow more movement
    max_turn_price_move: float = 0.20  # Allow 20% moves per turn
    immediate_impact_fraction: float = 0.60  # More immediate impact

    # Price bounds
    min_price: float = 0.05
    max_price: float = 150.0
    max_transaction: float = 250.0  # Increased from 25 to allow large accumulation/rug pulls

    # Trading fractions
    buy_fractions: tuple[float, ...] = (0.15, 0.35, 0.65)
    sell_fractions: tuple[float, ...] = (0.25, 0.5, 1.0)

    # Rug pull detection
    rug_pull_fraction: float = 0.45
    suspicious_dump_fraction: float = 0.35

    # Coin population balance
    max_active_coins: int = 10
    max_coins_per_agent: int = 3  # Increased from 2 to allow more creation
    target_coin_per_agent_ratio: float = 0.40  # Increased for more coins

    # Coin death mechanics (more aggressive)
    passive_price_bleed_enabled: bool = True
    inactive_decay_turns: int = 8  # Reduced from 12
    inactive_price_decay: float = 0.02  # Increased from 0.015
    coin_death_price_turns: int = 12  # Reduced from 20
    coin_death_activity_turns: int = 10  # Reduced from 15
    coin_death_liquidity_threshold: float = 15.0  # Reduced from 20
    min_trading_activity_per_turn: float = 8.0  # Increased from 5.0
    coin_death_enabled: bool = True

    # Attention system (increased for rug pull potential)
    agent_attention_slots: int = 12  # Increased from 5 to allow owning more coin types
    attention_hold_bias: float = 0.6
    attention_activity_bias: float = 0.3
    attention_decay_rate: float = 0.08

    # Intrinsic value and mean reversion
    intrinsic_value_weight: float = 0.72
    intrinsic_rebound_strength: float = 0.22

    # Liquidity dynamics
    passive_liquidity_base: float = 120.0  # Increased from 100
    passive_liquidity_panic_boost: float = 0.50  # Reduced from 0.85
    pending_pressure_decay: float = 0.50  # Reduced from 0.64 (faster decay)
    sustained_failure_turns: int = 8
    failure_liquidity_threshold: float = 35.0

    # Spread and arbitrage
    min_spread_bps: float = 0.008
    spread_volatility_scale: float = 0.02
    arbitrage_cost_threshold: float = 0.004
    exchange_mismatch_scale: float = 0.02  # Reduced from 0.03


@dataclass(slots=True)
class MiningConfig:
    # Work income (kept low to encourage active participation)
    work_income: float = 1.0

    # Mining parameters (REDUCED profitability to encourage trading over mining)
    baseline_positive_income: float = 2.0  # Reduced to make trading more attractive
    starting_mining_power: float = 2.0
    min_starting_mining_power: float = 1.5
    max_starting_mining_power: float = 3.0
    max_mining_power: float = 10.0

    # Cost parameters (INCREASED costs to reduce mining dominance)
    base_difficulty: float = 2.0  # Increased from 0.5
    difficulty_scale: float = 0.20  # Increased from 0.05
    coin_mining_difficulty_scale: float = 0.35  # Increased from 0.10
    mined_supply_difficulty_scale: float = 0.6  # Increased from 0.3
    block_reward_value: float = 20.0  # Reduced from 100.0
    base_electricity_price: float = 1.0  # Increased from 0.20
    electricity_price_scale: float = 0.25  # Increased from 0.05

    # Upgrade mechanics
    mining_upgrade_cost: float = 25.0
    mining_upgrade_gain: float = 0.25
    mining_reinvestment_rate: float = 0.50
    general_mining_bias_reward: float = 0.25


@dataclass(slots=True)
class EconomyConfig:
    inflation_rate_per_turn: float = 0.014
    cash_drag_rate: float = 0.006
    work_decay_per_streak: float = 0.25  # Increased from 0.12 for faster decay
    min_work_income_factor: float = 0.15  # Reduced from 0.2
    mining_jackpot_share: float = 0.72
    mining_tail_share: float = 0.28
    stochastic_price_drift: float = 0.035
    capital_scaling_bias: float = 0.18
    transaction_fee_rate: float = 0.012
    holding_cost_rate: float = 0.0014
    profit_take_threshold_low: float = 0.12
    profit_take_threshold_mid: float = 0.32
    profit_take_threshold_high: float = 0.68
    panic_drop_threshold: float = 0.08
    crash_drop_threshold: float = 0.16
    fomo_rally_threshold: float = 0.08
    leverage_margin_buffer: float = 0.18


@dataclass(slots=True)
class EvolutionConfig:
    population_size: int = 130  # Increased from 120 for better evolution
    generations: int = 50  # Increased from 25 for longer evolution
    selection_ratio: float = 0.35
    mutation_rate: float = 0.85
    mutation_strength: float = 0.24
    crossover_rate: float = 0.55
    elite_count: int = 2
    reward_scale: float = 1.0
    create_reward_bonus: float = 0.12
    mine_reward_bonus: float = 0.08
    mine_loss_penalty: float = -0.2
    dump_penalty: float = -0.3
    survival_bonus: float = 0.4
    
    # Fitness calculation weights (used by new liquidity-aware fitness)
    # Note: Actual weights are hardcoded in evolution.py._fitness()
    behavior_diversity_weight: float = 0.1
    reward_fitness_weight: float = 0.45  # Used for turn-by-turn reward component
    
    # Legacy weights (kept for backwards compatibility, not used in new fitness)
    wealth_fitness_weight: float = 1.0
    cash_ratio_bonus_weight: float = 0.18
    
    # Selection mechanics
    max_turn_reward: float = 1.75
    spike_penalty_threshold: float = 0.32
    spike_penalty_scale: float = 0.7
    random_agent_fraction: float = 0.18
    niche_preservation_count: int = 1
    overcrowding_penalty: float = 0.2
    rarity_bonus: float = 0.15


@dataclass(slots=True)
class OutputConfig:
    output_dir: Path = Path("outputs")
    model_dir: Path = Path("models")
    export_csv: bool = True
    save_replay: bool = True
    save_models: bool = True


@dataclass(slots=True)
class GameConfig:
    seed: int = 7
    num_agents: int = 130  # Increased to match population_size for NEAT
    turns_per_episode: int = 100  # Increased from 80 for longer episodes
    generations: int = 50  # Increased from 25
    starting_money: float = 120.0
    debug: bool = False  # Changed to False for cleaner output
    record_live_snapshots: bool = True
    disabled_actions: set[str] = field(default_factory=set)  # Actions to disable in simulation
    fitness_objective: str = "balanced"  # cash_maximizer, balanced, or net_worth
    market: MarketConfig = field(default_factory=MarketConfig)
    mining: MiningConfig = field(default_factory=MiningConfig)
    economy: EconomyConfig = field(default_factory=EconomyConfig)
    evolution: EvolutionConfig = field(default_factory=EvolutionConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def ensure_directories(self) -> None:
        self.output.output_dir.mkdir(parents=True, exist_ok=True)
        self.output.model_dir.mkdir(parents=True, exist_ok=True)

    @property
    def episodes(self) -> int:
        return self.generations

    @episodes.setter
    def episodes(self, value: int) -> None:
        self.generations = value
        self.evolution.generations = value
