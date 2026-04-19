"""
Player Game GUI - Graphical interface for competing against AI agents
Shows live charts, leaderboard, and allows turn multiplier
"""

from __future__ import annotations

import queue
import threading
import time
from datetime import datetime
from tkinter import BooleanVar, IntVar, StringVar, Text, Tk, ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from agent import ActionType, EvolutionAgent, StrategyGenome
from config import GameConfig
from environment import CryptoTradingEnvironment
from visualization import (
    create_behavior_mix_figure,
    create_behavior_trends_figure,
    create_coin_price_figure,
    create_net_worth_figure,
)


class PlayerGameGUI:
    def __init__(self, num_ai_agents: int = 119, load_models: bool = False) -> None:
        self.num_ai_agents = num_ai_agents
        self.total_agents = num_ai_agents + 1
        self.player_id = 0
        self.load_models = load_models
        
        self.config = GameConfig()
        self.config.num_agents = self.total_agents
        self.config.turns_per_episode = 80
        self.config.seed = 7
        
        self.env = CryptoTradingEnvironment(self.config)
        
        # Load trained models if requested
        if self.load_models:
            self.env.load_models()
        
        # Initialize environment
        self.env.reset()
        
        # Replace player agent (agent 0)
        import random
        player_genome = StrategyGenome.random(random.Random(self.config.seed + 999))
        self.player_agent = EvolutionAgent(self.player_id, player_genome, self.config.seed + 999)
        self.env.agents[self.player_id] = self.player_agent
        # Update agent_states for player
        self.env.agent_states[self.player_id] = self.env.agent_states.get(
            self.player_id,
            type('AgentState', (), {
                'cash': self.config.starting_money,
                'holdings': {},
                'mining_power': 1.0,
                'coins_created': [],
            })()
        )
        
        # Game state
        self.current_turn = 0
        self.current_episode = 0
        self.game_over = False
        self.turn_multiplier_value = 1
        
        # GUI setup
        self.root = Tk()
        self.root.title("CoinGame - Player Mode")
        self.root.geometry("1680x980")
        self.root.minsize(1400, 800)
        
        # Variables (must be created after root)
        self.message_queue: queue.Queue = queue.Queue()
        self.worker: threading.Thread | None = None
        self.action_queue: queue.Queue = queue.Queue()
        
        self.status_var = StringVar(value="Ready - Choose your action")
        self.turn_var = StringVar(value="Turn: 0")
        self.episode_var = StringVar(value="Episode: 0")
        self.player_cash_var = StringVar(value="$120.00")
        self.player_net_worth_var = StringVar(value="$120.00")
        self.player_holdings_var = StringVar(value="$0.00")
        self.turn_multiplier = IntVar(value=1)
        
        # Chart hosts
        self.chart_hosts: dict[str, ttk.Frame] = {}
        self.chart_canvases: dict[str, FigureCanvasTkAgg] = {}
        
        # Widgets
        self.leaderboard_tree: ttk.Treeview
        self.action_text: Text
        self.log_text: Text
        self.action_buttons: dict[str, ttk.Button] = {}
        
        self._build_layout()
        self.root.after(100, self._drain_queue)
        self.root.after(100, self._update_display)
    
    def _build_layout(self) -> None:
        root_frame = ttk.Frame(self.root, padding=12)
        root_frame.pack(fill="both", expand=True)
        root_frame.columnconfigure(0, weight=1)
        root_frame.columnconfigure(1, weight=2)
        root_frame.columnconfigure(2, weight=1)
        root_frame.rowconfigure(0, weight=2)
        root_frame.rowconfigure(1, weight=1)
        
        # Left panel - Player info and actions
        left_panel = ttk.Frame(root_frame, padding=(0, 0, 10, 0))
        left_panel.grid(row=0, column=0, rowspan=2, sticky="nsew")
        
        # Player stats
        stats_frame = ttk.LabelFrame(left_panel, text="Your Portfolio", padding=10)
        stats_frame.pack(fill="x", pady=(0, 10))
        
        ttk.Label(stats_frame, textvariable=self.player_cash_var, font=("Helvetica", 16, "bold")).pack(anchor="w")
        ttk.Label(stats_frame, text="Cash", foreground="#6b7280").pack(anchor="w")
        
        ttk.Separator(stats_frame, orient="horizontal").pack(fill="x", pady=8)
        
        ttk.Label(stats_frame, textvariable=self.player_net_worth_var, font=("Helvetica", 16, "bold"), foreground="#22c55e").pack(anchor="w")
        ttk.Label(stats_frame, text="Net Worth", foreground="#6b7280").pack(anchor="w")
        
        ttk.Separator(stats_frame, orient="horizontal").pack(fill="x", pady=8)
        
        ttk.Label(stats_frame, textvariable=self.player_holdings_var, font=("Helvetica", 14)).pack(anchor="w")
        ttk.Label(stats_frame, text="Holdings Value", foreground="#6b7280").pack(anchor="w")
        
        # Turn multiplier
        multiplier_frame = ttk.LabelFrame(left_panel, text="Turn Multiplier", padding=10)
        multiplier_frame.pack(fill="x", pady=(0, 10))
        
        ttk.Label(multiplier_frame, text="Execute action for next N turns:").pack(anchor="w", pady=(0, 5))
        
        multiplier_spinbox = ttk.Spinbox(
            multiplier_frame,
            from_=1,
            to=20,
            textvariable=self.turn_multiplier,
            width=10,
            font=("Helvetica", 14)
        )
        multiplier_spinbox.pack(anchor="w")
        
        # Action buttons
        actions_frame = ttk.LabelFrame(left_panel, text="Your Actions", padding=10)
        actions_frame.pack(fill="both", expand=True, pady=(0, 10))
        
        # Basic actions
        basic_frame = ttk.Frame(actions_frame)
        basic_frame.pack(fill="x", pady=(0, 5))
        
        self.action_buttons["work"] = ttk.Button(basic_frame, text="⚒️ Work", command=lambda: self._queue_action("work"))
        self.action_buttons["work"].pack(fill="x", pady=2)
        
        self.action_buttons["mine"] = ttk.Button(basic_frame, text="⛏️ Mine", command=lambda: self._queue_action("mine"))
        self.action_buttons["mine"].pack(fill="x", pady=2)
        
        self.action_buttons["hold"] = ttk.Button(basic_frame, text="⏸️ Hold", command=lambda: self._queue_action("hold"))
        self.action_buttons["hold"].pack(fill="x", pady=2)
        
        ttk.Separator(actions_frame, orient="horizontal").pack(fill="x", pady=5)
        
        # Coin actions (populated dynamically)
        self.coin_actions_frame = ttk.Frame(actions_frame)
        self.coin_actions_frame.pack(fill="both", expand=True)
        
        # Status
        ttk.Label(left_panel, textvariable=self.status_var, wraplength=300, foreground="#1d3557").pack(anchor="w", pady=(10, 0))
        ttk.Label(left_panel, textvariable=self.turn_var, foreground="#264653").pack(anchor="w")
        ttk.Label(left_panel, textvariable=self.episode_var, foreground="#264653").pack(anchor="w")
        
        # Middle panel - Charts
        middle_panel = ttk.Frame(root_frame)
        middle_panel.grid(row=0, column=1, sticky="nsew", padx=(0, 10))
        
        chart_area = ttk.Frame(middle_panel)
        chart_area.pack(fill="both", expand=True)
        chart_area.columnconfigure(0, weight=1)
        chart_area.columnconfigure(1, weight=1)
        chart_area.rowconfigure(0, weight=1)
        chart_area.rowconfigure(1, weight=1)
        
        chart_specs = [
            ("coin_prices", "Coin Prices", 0, 0),
            ("net_worth", "Net Worth", 0, 1),
            ("behavior_mix", "Behavior Mix", 1, 0),
            ("behavior_trends", "Behavior Trends", 1, 1),
        ]
        for chart_key, chart_label, row, column in chart_specs:
            host = ttk.LabelFrame(chart_area, text=chart_label, padding=6)
            host.grid(row=row, column=column, sticky="nsew", padx=3, pady=3)
            self.chart_hosts[chart_key] = host
        
        # Right panel - Leaderboard and log
        right_panel = ttk.Frame(root_frame)
        right_panel.grid(row=0, column=2, sticky="nsew")
        
        # Leaderboard
        leaderboard_frame = ttk.LabelFrame(right_panel, text="Top 10 Leaderboard", padding=6)
        leaderboard_frame.pack(fill="both", expand=True, pady=(0, 10))
        
        self.leaderboard_tree = ttk.Treeview(
            leaderboard_frame,
            columns=("rank", "agent", "net_worth", "cash"),
            show="headings",
            height=10,
        )
        for name, label, width in [
            ("rank", "#", 40),
            ("agent", "Agent", 70),
            ("net_worth", "Net Worth", 100),
            ("cash", "Cash", 90),
        ]:
            self.leaderboard_tree.heading(name, text=label)
            self.leaderboard_tree.column(name, width=width, anchor="w")
        self.leaderboard_tree.pack(fill="both", expand=True)
        
        # Action log
        log_frame = ttk.LabelFrame(right_panel, text="Recent Actions", padding=6)
        log_frame.pack(fill="both", expand=True)
        
        self.log_text = Text(log_frame, wrap="word", height=15, font=("Courier", 9))
        self.log_text.pack(fill="both", expand=True)
    
    def _queue_action(self, action_type: str, coin_id: str | None = None, fraction: float | None = None) -> None:
        """Queue player action for execution."""
        self.action_queue.put({
            "type": action_type,
            "coin_id": coin_id,
            "fraction": fraction,
        })
    
    def _update_display(self) -> None:
        """Update display with current game state."""
        # Update player stats
        player_state = self.env.agent_states[self.player_id]
        holdings_value = sum(
            units * self.env.market.coins[coin_id].price
            for coin_id, units in player_state.holdings.items()
            if coin_id in self.env.market.coins and units > 0
        )
        
        self.player_cash_var.set(f"${player_state.cash:.2f}")
        self.player_holdings_var.set(f"${holdings_value:.2f}")
        self.player_net_worth_var.set(f"${player_state.cash + holdings_value:.2f}")
        self.turn_var.set(f"Turn: {self.current_turn}")
        self.episode_var.set(f"Episode: {self.current_episode}")
        
        # Update charts
        self._refresh_charts()
        
        # Update leaderboard
        self._update_leaderboard()
        
        # Update coin action buttons
        self._update_coin_actions()
        
        # Continue updating
        if not self.game_over:
            self.root.after(500, self._update_display)
    
    def _refresh_charts(self) -> None:
        """Refresh all charts."""
        self._render_chart("coin_prices", create_coin_price_figure(self.env))
        self._render_chart("net_worth", create_net_worth_figure(self.env))
        self._render_chart("behavior_mix", create_behavior_mix_figure(self.env))
        self._render_chart("behavior_trends", create_behavior_trends_figure(self.env))
    
    def _render_chart(self, chart_key: str, figure) -> None:
        """Render a chart."""
        old_canvas = self.chart_canvases.get(chart_key)
        if old_canvas is not None:
            old_canvas.get_tk_widget().destroy()
        
        canvas = FigureCanvasTkAgg(figure, master=self.chart_hosts[chart_key])
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        self.chart_canvases[chart_key] = canvas
    
    def _update_leaderboard(self) -> None:
        """Update leaderboard display."""
        for row in self.leaderboard_tree.get_children():
            self.leaderboard_tree.delete(row)
        
        # Get all agents sorted by net worth
        agents_data = []
        for agent_id, state in self.env.agent_states.items():
            holdings_value = sum(
                units * self.env.market.coins[coin_id].price
                for coin_id, units in state.holdings.items()
                if coin_id in self.env.market.coins and units > 0
            )
            net_worth = state.cash + holdings_value
            agents_data.append({
                "agent_id": agent_id,
                "net_worth": net_worth,
                "cash": state.cash,
                "is_player": agent_id == self.player_id,
            })
        
        agents_data.sort(key=lambda x: x["net_worth"], reverse=True)
        
        for rank, agent in enumerate(agents_data[:10], 1):
            marker = "👤 " if agent["is_player"] else ""
            self.leaderboard_tree.insert(
                "",
                "end",
                values=(
                    rank,
                    f"{marker}Agent {agent['agent_id']}",
                    f"${agent['net_worth']:.2f}",
                    f"${agent['cash']:.2f}",
                ),
            )
    
    def _update_coin_actions(self) -> None:
        """Update coin-specific action buttons."""
        # Clear existing
        for widget in self.coin_actions_frame.winfo_children():
            widget.destroy()
        
        # Get coins
        coins = list(self.env.market.coins.values())[:5]  # Show top 5 coins
        
        if not coins:
            ttk.Label(self.coin_actions_frame, text="No coins yet").pack()
            return
        
        ttk.Label(self.coin_actions_frame, text="Coin Actions:", font=("Helvetica", 10, "bold")).pack(anchor="w", pady=(5, 0))
        
        for coin in coins:
            coin_frame = ttk.LabelFrame(self.coin_actions_frame, text=f"{coin.coin_id} - ${coin.price:.2f}", padding=5)
            coin_frame.pack(fill="x", pady=2)
            
            # Buy button
            buy_btn = ttk.Button(
                coin_frame,
                text="📈 Buy (35%)",
                command=lambda c=coin.coin_id: self._queue_action("buy", coin_id=c, fraction=0.35)
            )
            buy_btn.pack(side="left", padx=2)
            
            # Sell button (if player has holdings)
            player_holdings = self.env.agent_states[self.player_id].holdings.get(coin.coin_id, 0)
            if player_holdings > 0:
                sell_btn = ttk.Button(
                    coin_frame,
                    text="📉 Sell (50%)",
                    command=lambda c=coin.coin_id: self._queue_action("sell", coin_id=c, fraction=0.5)
                )
                sell_btn.pack(side="left", padx=2)
            
            # Trend button
            trend_btn = ttk.Button(
                coin_frame,
                text="📊 Trend",
                command=lambda c=coin.coin_id: self._queue_action("trend", coin_id=c)
            )
            trend_btn.pack(side="left", padx=2)
    
    def _drain_queue(self) -> None:
        """Process action queue."""
        try:
            action = self.action_queue.get_nowait()
            self._execute_player_action(action)
        except queue.Empty:
            pass
        
        self.root.after(100, self._drain_queue)
    
    def _execute_player_action(self, action: dict) -> None:
        """Execute player action for N turns."""
        from agent import Action
        
        action_type = action["type"]
        coin_id = action.get("coin_id")
        fraction = action.get("fraction")
        multiplier = self.turn_multiplier.get()  # Get value from IntVar
        
        self.status_var.set(f"Executing {action_type} for {multiplier} turns...")
        
        # Execute action for N turns
        for _ in range(multiplier):
            if self.game_over:
                break
            
            action_obj = Action(
                action_type=ActionType(action_type),
                coin_id=coin_id,
                fraction=fraction,
            )
            
            # Execute
            if action_type == "mine":
                mining_actions = [(self.player_id, action_obj)]
                results = self.env._resolve_mining_actions(mining_actions)
                if self.player_id in results:
                    reward, event = results[self.player_id]
                    self.player_agent.record_reward(reward)
                    self._log_action(f"You: {action_type} profit=${event.get('profit', 0):.2f}")
            else:
                reward, event = self.env.apply_action(self.player_id, action_obj)
                self.player_agent.record_reward(reward)
                if "profit" in event:
                    self._log_action(f"You: {action_type} profit=${event['profit']:.2f}")
                elif "income" in event:
                    self._log_action(f"You: {action_type} earned=${event['income']:.2f}")
                else:
                    self._log_action(f"You: {action_type}")
            
            # Process AI actions
            self._process_ai_actions()
            
            # Advance turn
            self.current_turn += 1
            if self.current_turn >= self.config.turns_per_episode:
                self._end_episode()
        
        self.status_var.set("Ready - Choose your action")
    
    def _process_ai_actions(self) -> None:
        """Process actions for all AI agents."""
        for agent in self.env.agents:
            if agent.agent_id == self.player_id:
                continue
            
            ai_actions = self.env.available_actions(agent.agent_id)
            if not ai_actions:
                continue
            
            state = self.env.observe(agent.agent_id)
            chosen_actions = agent.select_actions(state, ai_actions, training=False)
            
            if not chosen_actions:
                continue
            
            action = chosen_actions[0]
            
            if action.action_type == ActionType.MINE:
                mining_actions = [(agent.agent_id, action)]
                results = self.env._resolve_mining_actions(mining_actions)
                if agent.agent_id in results:
                    reward, event = results[agent.agent_id]
                    agent.record_reward(reward)
                    self.env.event_log.append({
                        "agent_id": agent.agent_id,
                        "action": action.key,
                        "profit": event.get("profit", 0),
                    })
            else:
                reward, event = self.env.apply_action(agent.agent_id, action)
                agent.record_reward(reward)
                event["agent_id"] = agent.agent_id
                self.env.event_log.append(event)
    
    def _end_episode(self) -> None:
        """Handle end of episode."""
        self.current_episode += 1
        self.current_turn = 0
        self.env.reset()
        
        if self.current_episode >= self.config.episodes:
            self.game_over = True
            self.status_var.set("Game Over!")
    
    def _log_action(self, message: str) -> None:
        """Add message to action log."""
        self.log_text.insert("end", message + "\n")
        self.log_text.see("end")
    
    def run(self) -> None:
        """Start the GUI."""
        self.root.mainloop()


def launch_player_gui(num_ai_agents: int = 119, load_models: bool = False) -> None:
    """Launch the player game GUI."""
    PlayerGameGUI(num_ai_agents=num_ai_agents, load_models=load_models).run()
