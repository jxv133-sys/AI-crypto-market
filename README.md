# CoinGame

Standalone Python simulation of a multi-agent crypto trading game with evolutionary agents (ES and NEAT), a desktop GUI, and turn-based player modes.

## Quick Start

### Install Python dependency
```bash
python3 -m pip install -r requirements.txt
```

### Run Modes

**NEAT Neural Network Training** (NEW - evolving neural networks):
```bash
python3 main.py --neat --generations 50 --agents 120
```

**GUI Player Game** (compete against trained AI):
```bash
python3 main.py --player
```

**Text-Based Player Game**:
```bash
python3 main.py --text
```

**Desktop GUI** (watch AI agents):
```bash
python3 main.py --gui
```

**Evolution Mode** (traditional ES training):
```bash
python3 main.py
```

## NEAT Neural Network Training

Train agents using **NeuroEvolution of Augmenting Topologies** - evolving neural network structures and weights.

### How NEAT Works

```
┌─────────────────────────────────────────────────────────┐
│  NEURAL NETWORK GENOME                                  │
│                                                         │
│  Input Layer (11 nodes):                               │
│  - Cash, Net Worth, Liquidity Ratio                    │
│  - Market Data (prices, trends, volatility)            │
│  - Mining Profitability, Work Income                   │
│                                                         │
│  Hidden Layers (evolving):                             │
│  - Nodes and connections mutate over generations       │
│  - Recurrent connections supported                     │
│                                                         │
│  Output Layer (12 nodes):                              │
│  - Action scores (buy, sell, mine, etc.)               │
│  - Trade fraction, coin selection                      │
└─────────────────────────────────────────────────────────┘
```

### Training Command

```bash
python3 main.py --neat --generations 50 --agents 120 --debug
```

### NEAT Features

**Evolving Topology:**
- Starts with minimal connections
- Adds nodes and connections through mutation
- Complex networks emerge naturally

**Speciation:**
- Genomes clustered by similarity
- Protects innovative structures
- Maintains behavioral diversity

**Fitness Function:**
- Same economic goals as ES (cash + assets + realized profit)
- Liquidity-based scaling applied

### Output Files

NEAT training generates:
- `neat_fitness.png` - Fitness over generations
- `neat_complexity.png` - Network complexity growth
- `neat_species.png` - Species count over time
- `neat_behavior.png` - Behavioral diversity
- `neat_statistics.json` - Detailed metrics
- `models/neat_agent_*.pkl` - Saved genomes

### Loading Trained NEAT Models

```bash
# Train
python3 main.py --neat --generations 50

# Play against trained neural networks
python3 main.py --player --load-models
```

### NEAT vs ES

| Feature | ES (Traditional) | NEAT (Neural) |
|---------|-----------------|---------------|
| Representation | 14 parameters | Neural network |
| Topology | Fixed | Evolving |
| Complexity | Constant | Grows over time |
| Diversity | Behavior-based | Speciation |
| Interpretability | High | Medium |
| Strategy Depth | Limited | Potentially unlimited |

## Player Game Mode

Compete directly against AI agents in the same economy using identical rules.

### GUI Player Mode (Recommended)
```bash
python3 main.py --player
```

**Features:**
- Live coin price charts
- Real-time net worth tracking
- Top 10 leaderboard
- Turn multiplier (1-20x) for faster gameplay
- One-click action buttons
- Choose 5-120 AI opponents

**Controls:**
- Click action buttons to execute
- Adjust turn multiplier (1-20) to repeat actions
- Watch live charts update
- Monitor leaderboard rankings

### Text-Based Player Mode
```bash
python3 main.py --text
```

**Features:**
- Same economic model as GUI
- Keyboard-based interaction
- Detailed numerical output
- Choose 5-120 AI opponents

**Controls:**
- Enter number to choose action
- `help` for action descriptions
- `m/b/s/c` for specific action categories
- `q` to quit

## GUI Features

**Evolution Mode:**
- Fast generation-based strategy evolution
- Choose between ES (Evolutionary Strategy) or NEAT (Neural Networks)
- Training speed control (1-500%)
- Live training statistics display
- Episode event summaries
- Chart export and analysis

**Visualization Mode:**
- Slow, human-observable playback
- Live actions, prices, balances, holdings
- Recent trades and highlights
- Pause/resume/step controls

**Training Statistics Panel:**
- Generation counter
- Best/Average fitness tracking
- Network complexity (NEAT only)
- Species count (NEAT only)
- Algorithm type display
- Training speed indicator

**Event Summary System:**
Automatically detects and reports major market events:
- 💥 Market Crashes (>50% price drop)
- 🚀 Price Surges (>80% gain)
- 🧨 Rug Pulls (large dumps)
- 💰 Extreme Profits (>5x return)
- 💸 Extreme Losses (>90% loss)
- ⛏️ Mining Shifts (profitability changes)
- 💀 Coin Deaths

**Training Algorithm Options:**
- **ES (Evolutionary Strategy)**: Traditional parameter evolution
- **NEAT (Neural Networks)**: Evolving neural network topologies

**Training Speed:**
- Adjust from 1% to 500%
- Higher speeds skip some visualization updates
- Use 500% for fast training, 100% for detailed monitoring

**Max Active Coins:**
- Slider controls maximum simultaneous coins (3-50)
- Lower values = more scarcity, higher prices
- Higher values = more diversity, lower competition
- Default: 10 coins

Useful flags:
```bash
python3 main.py --generations 10 --turns 80 --agents 8 --seed 42 --debug
python3 main.py --load-models --quiet
python3 main.py --gui --load-models
```

### Visualization mode tips
- Use the `Action Delay` slider to slow down or speed up visible actions
- `Pause` freezes playback
- `Resume` continues the watch run
- `Step One Turn` advances exactly one full turn while paused

### Training mode tips
- Select ES for faster, simpler evolution
- Select NEAT for evolving neural network brains
- Increase training speed for faster results
- Decrease training speed to watch learning in detail

### What gets generated
Each run creates a timestamped folder under `outputs/` containing:
- `event_log.csv`
- `episode_rewards.csv`
- `generation_fitness.csv`
- `replay.json`
- `coin_prices.png`
- `agent_net_worth.png`
- `coins_created.png`
- `mining_system.png`
- `wealth_distribution.png`
- `training_stats.png`
- `evolution_overview.png`

Saved genomes are written to `models/agent_<id>.pkl`, with the strongest saved as `models/best_agent.pkl`.

### Example debug output
```text
=== Episode 1 ===
[turn 000] active_coins=0
agent=0 action=work
market_event={"action": "work", "agent_id": 0, "income": 6.0} reward=0.6000
agent=1 action=create
[suspicious] episode=1 turn=0 reason=extreme_wealth_gain event={'agent_id': 1, 'action': 'create', 'coin_id': 'C001', 'creator_supply': 700.0}
market_event={"action": "create", "agent_id": 1, "coin_id": "C001", "creator_supply": 700.0} reward=34.6000
```

### Example completion output
```text
=== Evolution Complete ===
{'generation': 1, 'best_fitness': 910.948, 'average_fitness': 470.423, 'best_agent_id': 3, 'dominant_behavior': 'buy'}
{'generation': 2, 'best_fitness': 1256.7426, 'average_fitness': 780.3092, 'best_agent_id': 3, 'dominant_behavior': 'buy'}
Saved outputs to: outputs/simulation_20260330_170600
```
