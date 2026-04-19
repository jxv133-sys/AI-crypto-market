from __future__ import annotations

import queue
import random
import threading
import time
from dataclasses import replace
from datetime import datetime
from tkinter import BooleanVar, IntVar, StringVar, Text, Tk, ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from config import GameConfig
from environment import CryptoTradingEnvironment
from evolution import EvolutionTrainer
from visualization import (
    create_behavior_mix_figure,
    create_behavior_trends_figure,
    create_coin_ownership_figure,
    create_coin_price_figure,
    create_net_worth_figure,
    plot_simulation_metrics,
)


class CoinGameGUI:
    MAX_TEXT_LINES = 1200
    QUEUE_BATCH_SIZE = 50
    LIVE_CHART_UPDATE_MS = 500
    AGENT_ACTION_COLORS = {
        0: "#22c55e",
        1: "#60a5fa",
        2: "#f59e0b",
        3: "#f87171",
        4: "#a78bfa",
        5: "#f472b6",
        6: "#2dd4bf",
        7: "#fb923c",
    }

    def __init__(self, base_config: GameConfig) -> None:
        self.base_config = base_config
        self.root = Tk()
        self.root.title("CoinGame Evolution Simulator")
        self.root.geometry("1680x980")
        self.root.minsize(1380, 860)

        self.message_queue: queue.Queue[tuple[str, object]] = queue.Queue()
        self.worker: threading.Thread | None = None
        self.current_env: CryptoTradingEnvironment | None = None
        self.pause_event = threading.Event()
        self.pause_event.set()
        self.stop_after_turns = 0
        self.control_lock = threading.Lock()
        self.current_mode = "training"
        self.visual_delay_seconds = 0.15

        self.mode_var = StringVar(value="training")
        self.algorithm_var = StringVar(value="es")  # es or neat
        self.generations_var = IntVar(value=base_config.generations)
        self.turns_var = IntVar(value=base_config.turns_per_episode)
        self.agents_var = IntVar(value=base_config.num_agents)
        self.seed_var = IntVar(value=base_config.seed)
        self.money_var = StringVar(value=f"{base_config.starting_money:.0f}")
        self.debug_var = BooleanVar(value=False)
        self.load_models_var = BooleanVar(value=False)
        self.price_bleed_var = BooleanVar(value=base_config.market.passive_price_bleed_enabled)
        self.max_coins_var = IntVar(value=base_config.market.max_active_coins)
        self.max_coins_label_var = StringVar(value=f"Max {base_config.market.max_active_coins} active coins")
        self.train_speed_var = IntVar(value=100)
        self.train_speed_label_var = StringVar(value="100% training speed")
        self.status_var = StringVar(value="Ready")
        self.current_action_var = StringVar(value="Current action: waiting")
        self.current_turn_var = StringVar(value="Turn: -")

        # Fitness objective selection
        self.fitness_objective_var = StringVar(value="balanced")

        # Action selection variables (True = enabled, False = disabled)
        self.action_vars: dict[str, BooleanVar] = {
            "CREATE": BooleanVar(value=True),
            "BUY": BooleanVar(value=True),
            "SELL": BooleanVar(value=True),
            "WORK": BooleanVar(value=True),
            "MINE": BooleanVar(value=True),
            "TREND": BooleanVar(value=True),
            "SPECULATE": BooleanVar(value=True),
            "ARBITRAGE": BooleanVar(value=True),
            "LEVERAGE": BooleanVar(value=True),
            "HOLD": BooleanVar(value=True),
            "MARKET_MAKE": BooleanVar(value=True),
        }

        self.summary_text: Text
        self.stats_tree: ttk.Treeview
        self.leaderboard_tree: ttk.Treeview
        self.highlights_text: Text
        self.market_tree: ttk.Treeview
        self.chart_hosts: dict[str, ttk.Frame] = {}
        self.chart_canvases: dict[str, FigureCanvasTkAgg] = {}
        self.chart_figures: dict[str, Figure] = {}
        self.chart_axes: dict[str, list] = {}
        self.pause_button: ttk.Button
        self.resume_button: ttk.Button
        self.step_button: ttk.Button
        self.current_action_label: ttk.Label
        self.last_chart_refresh_at = 0.0
        self.action_delay_timer_id: int | None = None

        self._build_layout()
        self._set_visual_controls_enabled(False)
        self.root.after(50, self._drain_queue)

    def _build_layout(self) -> None:
        root_frame = ttk.Frame(self.root, padding=12)
        root_frame.pack(fill="both", expand=True)
        root_frame.columnconfigure(0, weight=0)
        root_frame.columnconfigure(1, weight=2)
        root_frame.columnconfigure(2, weight=2)
        root_frame.rowconfigure(0, weight=1)

        sidebar = ttk.Frame(root_frame, padding=(0, 0, 12, 0))
        sidebar.grid(row=0, column=0, sticky="ns")

        ttk.Label(sidebar, text="Simulation Controls", font=("Helvetica", 16, "bold")).pack(anchor="w")
        mode_frame = ttk.LabelFrame(sidebar, text="Mode", padding=8)
        mode_frame.pack(fill="x", pady=(10, 8))
        ttk.Radiobutton(
            mode_frame,
            text="Evolution Mode",
            value="training",
            variable=self.mode_var,
            command=self._on_mode_selected,
        ).pack(anchor="w")
        ttk.Radiobutton(
            mode_frame,
            text="Visualization Mode",
            value="visualization",
            variable=self.mode_var,
            command=self._on_mode_selected,
        ).pack(anchor="w", pady=(4, 0))

        # Algorithm selection
        algorithm_frame = ttk.LabelFrame(sidebar, text="Training Algorithm", padding=8)
        algorithm_frame.pack(fill="x", pady=(8, 8))
        ttk.Radiobutton(
            algorithm_frame,
            text="ES (Evolutionary Strategy)",
            value="es",
            variable=self.algorithm_var,
        ).pack(anchor="w")
        ttk.Radiobutton(
            algorithm_frame,
            text="NEAT (Neural Networks)",
            value="neat",
            variable=self.algorithm_var,
        ).pack(anchor="w", pady=(4, 0))

        control_frame = ttk.Frame(sidebar)
        control_frame.pack(fill="x", pady=(8, 10))
        self._add_spinbox(control_frame, "Generations", self.generations_var)
        self._add_spinbox(control_frame, "Turns / Simulation", self.turns_var)
        self._add_spinbox(control_frame, "Population Size", self.agents_var)
        self._add_spinbox(control_frame, "Seed", self.seed_var)
        self._add_entry(control_frame, "Starting Money", self.money_var)

        # Max coins slider (replaces action delay)
        ttk.Label(control_frame, text="Max Active Coins").pack(anchor="w", pady=(8, 0))
        max_coins_scale = ttk.Scale(control_frame, from_=3, to=50, orient="horizontal", command=self._on_max_coins_change)
        max_coins_scale.set(self.max_coins_var.get())
        max_coins_scale.pack(fill="x", pady=(0, 6))
        ttk.Label(control_frame, textvariable=self.max_coins_label_var, foreground="#6b7280").pack(anchor="w", pady=(0, 6))

        ttk.Label(control_frame, text="Training Speed").pack(anchor="w", pady=(8, 0))
        train_speed_scale = ttk.Scale(control_frame, from_=1, to=500, orient="horizontal", command=self._on_train_speed_change)
        train_speed_scale.set(self.train_speed_var.get())
        train_speed_scale.pack(fill="x", pady=(0, 6))
        ttk.Label(control_frame, textvariable=self.train_speed_label_var, foreground="#6b7280").pack(anchor="w", pady=(0, 6))

        ttk.Checkbutton(control_frame, text="Verbose debug logging", variable=self.debug_var).pack(anchor="w", pady=4)
        ttk.Checkbutton(control_frame, text="Load saved genomes", variable=self.load_models_var).pack(anchor="w", pady=4)
        ttk.Checkbutton(control_frame, text="Enable passive coin price bleed", variable=self.price_bleed_var).pack(anchor="w", pady=4)

        # Fitness objective dropdown
        objective_frame = ttk.LabelFrame(sidebar, text="Fitness Objective", padding=8)
        objective_frame.pack(fill="x", pady=(8, 8))
        ttk.Label(objective_frame, text="Select what agents are rewarded for:", wraplength=280).pack(anchor="w", pady=(0, 6))
        objective_combo = ttk.Combobox(
            objective_frame,
            textvariable=self.fitness_objective_var,
            values=["cash_maximizer", "balanced", "net_worth"],
            state="readonly",
            width=25
        )
        objective_combo.pack(anchor="w")
        objective_combo.set("balanced")
        # Add descriptions
        ttk.Label(
            objective_frame,
            text="• Cash: Maximize cash, no market penalty\n• Balanced: Good cash + market balance\n• Net Worth: Total wealth focused",
            wraplength=280,
            foreground="#6b7280",
            font=("TkDefaultFont", 8)
        ).pack(anchor="w", pady=(6, 0))

        # Action selection frame
        action_frame = ttk.LabelFrame(sidebar, text="Enabled Actions", padding=8)
        action_frame.pack(fill="x", pady=(8, 8))
        
        # Create two columns of action checkboxes
        actions_left = ["WORK", "MINE", "BUY", "SELL", "CREATE", "HOLD"]
        actions_right = ["TREND", "SPECULATE", "ARBITRAGE", "LEVERAGE", "MARKET_MAKE"]
        
        left_frame = ttk.Frame(action_frame)
        left_frame.pack(side="left", padx=(0, 10))
        right_frame = ttk.Frame(action_frame)
        right_frame.pack(side="left")
        
        for action in actions_left:
            ttk.Checkbutton(left_frame, text=action.capitalize(), variable=self.action_vars[action]).pack(anchor="w")
        for action in actions_right:
            ttk.Checkbutton(right_frame, text=action.capitalize(), variable=self.action_vars[action]).pack(anchor="w")

        button_frame = ttk.Frame(sidebar)
        button_frame.pack(fill="x", pady=(8, 10))
        ttk.Button(button_frame, text="Start Selected Mode", command=self.start_run).pack(fill="x")
        self.pause_button = ttk.Button(button_frame, text="Pause", command=self.pause_visualization)
        self.pause_button.pack(fill="x", pady=(6, 0))
        self.resume_button = ttk.Button(button_frame, text="Resume", command=self.resume_visualization)
        self.resume_button.pack(fill="x", pady=(6, 0))
        self.step_button = ttk.Button(button_frame, text="Step One Turn", command=self.step_visualization)
        self.step_button.pack(fill="x", pady=(6, 0))

        ttk.Label(sidebar, textvariable=self.status_var, wraplength=320, foreground="#1d3557").pack(anchor="w", pady=(2, 8))
        ttk.Label(sidebar, textvariable=self.current_turn_var, foreground="#264653").pack(anchor="w")
        self.current_action_label = ttk.Label(sidebar, textvariable=self.current_action_var, wraplength=320, foreground="#f59e0b")
        self.current_action_label.pack(anchor="w", pady=(0, 12))

        ttk.Label(sidebar, text="Run Summary", font=("Helvetica", 13, "bold")).pack(anchor="w")
        self.summary_text = Text(sidebar, width=42, height=16, wrap="word")
        self.summary_text.pack(fill="x", pady=(6, 12))

        middle_panel = ttk.Frame(root_frame)
        middle_panel.grid(row=0, column=1, sticky="nsew", padx=(0, 8))
        middle_panel.columnconfigure(0, weight=1)
        middle_panel.rowconfigure(0, weight=3)
        middle_panel.rowconfigure(1, weight=1)

        chart_area = ttk.Frame(middle_panel)
        chart_area.grid(row=0, column=0, sticky="nsew")
        chart_area.columnconfigure(0, weight=1)
        chart_area.columnconfigure(1, weight=1)
        chart_area.rowconfigure(0, weight=1)
        chart_area.rowconfigure(1, weight=1)

        chart_specs = [
            ("coin_prices", "Coin Prices", 0, 0),
            ("net_worth", "Net Worth", 0, 1),
            ("coin_ownership", "Coin Ownership", 1, 0),
            ("behavior_mix", "Agent Behaviors", 1, 1),
        ]
        for chart_key, chart_label, row, column in chart_specs:
            host = ttk.LabelFrame(chart_area, text=chart_label, padding=6)
            host.grid(row=row, column=column, sticky="nsew", padx=6, pady=6)
            self.chart_hosts[chart_key] = host

        event_row = ttk.Frame(middle_panel)
        event_row.grid(row=1, column=0, sticky="nsew", pady=(8, 0))
        event_row.columnconfigure(0, weight=1)
        event_row.columnconfigure(1, weight=1)
        
        # Training stats panel (replaces market activity)
        stats_frame = ttk.LabelFrame(event_row, text="Training Statistics", padding=6)
        stats_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        
        # Create treeview for training stats
        self.stats_tree = ttk.Treeview(
            stats_frame,
            columns=("metric", "value"),
            show="headings",
            height=8,
        )
        self.stats_tree.heading("metric", text="Metric")
        self.stats_tree.heading("value", text="Value")
        self.stats_tree.column("metric", width=120, anchor="w")
        self.stats_tree.column("value", width=100, anchor="e")
        self.stats_tree.pack(fill="both", expand=True)
        
        # Behavior trends chart (replaces highlighted events)
        trends_frame = ttk.LabelFrame(event_row, text="Behavior Trends", padding=6)
        trends_frame.grid(row=0, column=1, sticky="nsew", padx=(6, 0))
        self.chart_hosts["behavior_trends"] = trends_frame

        right_panel = ttk.Frame(root_frame)
        right_panel.grid(row=0, column=2, sticky="nsew")
        right_panel.columnconfigure(0, weight=1)
        right_panel.rowconfigure(0, weight=1)  # Leaderboard
        right_panel.rowconfigure(1, weight=1)  # Market
        right_panel.rowconfigure(2, weight=1)  # Highlights

        leaderboard_frame = ttk.LabelFrame(right_panel, text="Leaderboard", padding=6)
        leaderboard_frame.grid(row=0, column=0, sticky="nsew", pady=(0, 8))
        self.leaderboard_tree = ttk.Treeview(
            leaderboard_frame,
            columns=("rank", "agent", "worth", "cash"),
            show="headings",
            height=8,
        )
        for name, label, width in [
            ("rank", "#", 40),
            ("agent", "Agent", 70),
            ("worth", "Cash (Primary)", 100),  # Will update based on objective
            ("cash", "Cash", 90),
        ]:
            self.leaderboard_tree.heading(name, text=label)
            self.leaderboard_tree.column(name, width=width, anchor="w")
        self.leaderboard_tree.pack(fill="both", expand=True)

        # Event Highlights panel (replaces agent balances)
        highlights_frame = ttk.LabelFrame(right_panel, text="📊 Episode Highlights", padding=6)
        highlights_frame.grid(row=2, column=0, sticky="nsew")
        self.highlights_text = Text(highlights_frame, wrap="word", height=16, font=("Courier", 9))
        self.highlights_text.pack(fill="both", expand=True)

        market_frame = ttk.LabelFrame(right_panel, text="Market: Prices and Trends", padding=6)
        market_frame.grid(row=1, column=0, sticky="nsew", pady=(0, 8))
        self.market_tree = ttk.Treeview(
            market_frame,
            columns=("coin", "price", "trend", "volume", "lifecycle", "age"),
            show="headings",
            height=12,
        )
        for name, label, width in [
            ("coin", "Coin", 80),
            ("price", "Price", 90),
            ("trend", "Trend", 80),
            ("volume", "Volume", 100),
            ("lifecycle", "Lifecycle", 110),
            ("age", "Age", 60),
        ]:
            self.market_tree.heading(name, text=label)
            self.market_tree.column(name, width=width, anchor="w")
        self.market_tree.pack(fill="both", expand=True)

    def _add_spinbox(self, parent: ttk.Frame, label: str, variable: IntVar) -> None:
        ttk.Label(parent, text=label).pack(anchor="w")
        spinbox = ttk.Spinbox(parent, from_=1, to=500, textvariable=variable, width=12)
        spinbox.pack(anchor="w", pady=(0, 8))

    def _add_entry(self, parent: ttk.Frame, label: str, variable: StringVar) -> None:
        ttk.Label(parent, text=label).pack(anchor="w")
        ttk.Entry(parent, textvariable=variable, width=14).pack(anchor="w", pady=(0, 8))

    def _on_max_coins_change(self, value: str) -> None:
        max_coins = max(3, int(float(value)))
        self.max_coins_var.set(max_coins)
        self.max_coins_label_var.set(f"Max {max_coins} active coins")
        # Update config immediately
        self.base_config.market.max_active_coins = max_coins

    def _on_train_speed_change(self, value: str) -> None:
        percent = max(1, int(float(value)))
        self.train_speed_var.set(percent)
        self.train_speed_label_var.set(f"{percent}% training speed")

    def _on_mode_selected(self) -> None:
        self.current_mode = self.mode_var.get()

    def _append_text(self, widget: Text, message: str) -> None:
        widget.insert("end", message + "\n")
        line_count = int(widget.index("end-1c").split(".")[0])
        if line_count > self.MAX_TEXT_LINES:
            trim_to = line_count - self.MAX_TEXT_LINES + 1
            widget.delete("1.0", f"{trim_to}.0")
        widget.see("end")

    def _render_chart(self, chart_key: str, figure: Figure) -> None:
        old_canvas = self.chart_canvases.get(chart_key)
        if old_canvas is not None:
            old_canvas.get_tk_widget().destroy()
        canvas = FigureCanvasTkAgg(figure, master=self.chart_hosts[chart_key])
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        self.chart_canvases[chart_key] = canvas
        self.chart_figures[chart_key] = figure

    def _refresh_charts(self, env: CryptoTradingEnvironment) -> None:
        self._render_chart("coin_prices", create_coin_price_figure(env))
        self._render_chart("net_worth", create_net_worth_figure(env))
        self._render_chart("coin_ownership", create_coin_ownership_figure(env))
        self._render_chart("behavior_mix", create_behavior_mix_figure(env))
        self._render_chart("behavior_trends", create_behavior_trends_figure(env))
        self.last_chart_refresh_at = time.time()

    def _refresh_charts_if_due(self, env: CryptoTradingEnvironment, force: bool = False) -> None:
        if force or (time.time() - self.last_chart_refresh_at) * 1000.0 >= self.LIVE_CHART_UPDATE_MS:
            self._refresh_charts(env)

    def _set_visual_controls_enabled(self, enabled: bool) -> None:
        state = "normal" if enabled else "disabled"
        self.pause_button.configure(state=state)
        self.resume_button.configure(state=state)
        self.step_button.configure(state=state)

    def _get_disabled_actions(self) -> set[str]:
        """Return set of action types that are disabled by user."""
        return {
            action_type for action_type, var in self.action_vars.items()
            if not var.get()
        }

    def _build_config(self) -> GameConfig:
        config = replace(self.base_config)
        config.generations = max(1, self.generations_var.get())
        config.evolution.generations = config.generations
        config.turns_per_episode = max(1, self.turns_var.get())
        config.num_agents = max(2, self.agents_var.get())
        config.evolution.population_size = config.num_agents
        config.seed = self.seed_var.get()
        config.starting_money = float(self.money_var.get())
        config.debug = self.debug_var.get()
        config.market.passive_price_bleed_enabled = self.price_bleed_var.get()
        config.disabled_actions = self._get_disabled_actions()
        config.fitness_objective = self.fitness_objective_var.get()
        config.ensure_directories()
        return config

    def _reset_panels(self) -> None:
        self.summary_text.delete("1.0", "end")
        for row in self.stats_tree.get_children():
            self.stats_tree.delete(row)
        for tree in (self.market_tree, self.leaderboard_tree):
            for row in tree.get_children():
                tree.delete(row)
        self.highlights_text.delete("1.0", "end")
        self.current_action_var.set("Current action: waiting")
        self.current_turn_var.set("Turn: -")
        self.current_action_label.configure(foreground="#f59e0b")
        self.last_chart_refresh_at = 0.0
        for chart_key in list(self.chart_canvases.keys()):
            old_canvas = self.chart_canvases.pop(chart_key)
            old_canvas.get_tk_widget().destroy()
        self.chart_figures.clear()
        
        # Initialize training stats
        self.training_stats = {
            'generation': 0,
            'best_fitness': 0.0,
            'average_fitness': 0.0,
            'complexity': 0.0,
            'species': 0,
        }
        self._update_training_stats()

    def start_run(self) -> None:
        if self.worker is not None and self.worker.is_alive():
            self.status_var.set("Simulation already running")
            return

        self._reset_panels()
        config = self._build_config()
        self.current_mode = self.mode_var.get()
        self.current_env = CryptoTradingEnvironment(config)
        if self.load_models_var.get():
            self.current_env.load_models()
        self.current_env.log_listener = lambda message: self.message_queue.put(("log", message))
        self._apply_snapshot(self.current_env.live_snapshot())
        self._refresh_charts(self.current_env)
        self.pause_event.set()
        with self.control_lock:
            self.stop_after_turns = 0
        self._set_visual_controls_enabled(self.current_mode == "visualization")
        if self.current_mode == "visualization":
            self.status_var.set("Visualization mode running")
            self.worker = threading.Thread(target=self._run_visualization, args=(self.current_env,), daemon=True)
        else:
            self.status_var.set("Evolution mode running")
            self.worker = threading.Thread(target=self._run_training, args=(self.current_env,), daemon=True)
        self.worker.start()

    def pause_visualization(self) -> None:
        if self.current_mode != "visualization":
            return
        self.pause_event.clear()
        self.status_var.set("Visualization paused")

    def resume_visualization(self) -> None:
        if self.current_mode != "visualization":
            return
        self.pause_event.set()
        with self.control_lock:
            self.stop_after_turns = 0
        self.status_var.set("Visualization resumed")

    def step_visualization(self) -> None:
        if self.current_mode != "visualization":
            return
        with self.control_lock:
            self.stop_after_turns += 1
        self.pause_event.set()
        self.status_var.set("Advancing one turn")

    def _run_training(self, env: CryptoTradingEnvironment) -> None:
        try:
            algorithm = self.algorithm_var.get()
            train_speed = self.train_speed_var.get() / 100.0  # Convert to multiplier
            
            if algorithm == "neat":
                from neat_trainer import NEATTrainer
                trainer = NEATTrainer(env.config)
                algorithm_name = "NEAT neural networks"
            else:
                trainer = EvolutionTrainer(env.config)
                algorithm_name = "ES evolutionary strategies"
            
            if self.load_models_var.get():
                trainer.load_population(env.config.output.model_dir)
            
            self.message_queue.put(("status", f"Training {algorithm_name}..."))
            
            def progress_callback(summary: dict[str, object], generation_env: CryptoTradingEnvironment) -> None:
                # Apply training speed - skip some generations if speed > 100%
                if train_speed < 1.0 and random.random() > train_speed:
                    return  # Skip some updates for faster training
                self.message_queue.put(("generation_update", {"summary": summary, "env": generation_env}))

            summaries, best_env = trainer.run(progress_callback=progress_callback)
            self.current_env = best_env
            for summary in summaries:
                self.message_queue.put(("episode_complete", summary))
            self.message_queue.put(("status", "Saving evolved genomes..."))
            trainer.save_population(env.config.output.model_dir)
            self._finalize_run(best_env, summaries, f"{algorithm.upper()} training finished")
        except Exception as exc:
            self.message_queue.put(("error", str(exc)))

    def _handle_turn_step_boundary(self) -> None:
        with self.control_lock:
            if self.stop_after_turns > 0:
                self.stop_after_turns -= 1
                if self.stop_after_turns == 0:
                    self.pause_event.clear()

    def _run_visualization(self, env: CryptoTradingEnvironment) -> None:
        try:
            summaries = []
            for _ in range(env.config.episodes):
                for payload in env.run_episode_iter(training=False):
                    kind = payload["kind"]
                    if kind == "action":
                        self.message_queue.put(("visual_action", payload))
                        # Non-blocking delay using threading.Event
                        deadline = time.time() + self.visual_delay_seconds
                        while time.time() < deadline:
                            if not self.pause_event.is_set():
                                self.pause_event.wait()
                                deadline = time.time() + self.visual_delay_seconds
                            else:
                                time.sleep(0.01)
                        # Check for step boundary after delay
                        self._handle_turn_step_boundary()
                    elif kind == "turn_end":
                        self.message_queue.put(("turn_end", payload))
                    elif kind == "episode_complete":
                        summary = payload["summary"]
                        summaries.append(summary)
                        self.message_queue.put(("episode_complete", summary))
                if len(summaries) < env.config.episodes:
                    self.pause_event.set()
            self._finalize_run(env, summaries, "Visualization finished")
        except Exception as exc:
            self.message_queue.put(("error", str(exc)))

    def _finalize_run(self, env: CryptoTradingEnvironment, summaries: list[dict[str, object]], status: str) -> None:
        self.message_queue.put(("status", "Exporting CSV, replay, and charts..."))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = env.config.output.output_dir / f"simulation_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)
        env.export_csv(output_dir)
        env.export_replay(output_dir)
        
        # Defer chart saving to main thread to avoid matplotlib/Tk threading crash
        self.message_queue.put(
            (
                "save_charts",
                {
                    "env": env,
                    "output_dir": output_dir,
                    "status": status,
                    "summaries": summaries,
                },
            )
        )

    def _format_holdings(self, holdings: dict[str, float]) -> str:
        if not holdings:
            return "-"
        return ", ".join(f"{coin}:{units:.1f}" for coin, units in holdings.items())

    def _update_highlights(self, snapshot: dict[str, object]) -> None:
        """Update event highlights display."""
        self.highlights_text.delete("1.0", "end")
        
        # Get major events from environment
        if self.current_env is not None:
            major_events = self.current_env.metrics.get("major_events", [])
            event_summaries = self.current_env.metrics.get("event_summaries", [])
            
            # Combine and show more events (top 15)
            all_events = event_summaries[:5] + major_events[:10]
            
            if all_events:
                self.highlights_text.insert("1.0", "📊 MAJOR EVENTS:\n\n")
                for i, event in enumerate(all_events, 1):
                    self.highlights_text.insert("end", f"{i}. {event}\n\n")
            else:
                self.highlights_text.insert("1.0", "No major events yet...\n\n")
                self.highlights_text.insert("end", "Watch for:\n")
                self.highlights_text.insert("end", "  💥 Market crashes\n")
                self.highlights_text.insert("end", "  🚀 Price surges\n")
                self.highlights_text.insert("end", "  🧨 Rug pulls\n")
                self.highlights_text.insert("end", "  💰 Extreme profits\n")
                self.highlights_text.insert("end", "  💸 Extreme losses\n")
                self.highlights_text.insert("end", "  💀 Coin deaths\n")

    def _update_leaderboard(self, snapshot: dict[str, object]) -> None:
        for row in self.leaderboard_tree.get_children():
            self.leaderboard_tree.delete(row)
        
        objective = self.fitness_objective_var.get()
        
        # Sort by cash for cash_maximizer objective, otherwise by net worth
        if objective == "cash_maximizer":
            leaders = sorted(snapshot["agents"], key=lambda agent: agent["cash"], reverse=True)
            self.leaderboard_tree.heading("worth", text="💵 Cash")
        elif objective == "balanced":
            leaders = sorted(snapshot["agents"], key=lambda agent: agent["net_worth"], reverse=True)
            self.leaderboard_tree.heading("worth", text="⚖️ Net Worth")
        else:  # net_worth
            leaders = sorted(snapshot["agents"], key=lambda agent: agent["net_worth"], reverse=True)
            self.leaderboard_tree.heading("worth", text="📈 Net Worth")
        
        for rank, agent in enumerate(leaders[:8], start=1):
            sort_value = agent["cash"] if objective == "cash_maximizer" else agent["net_worth"]
            self.leaderboard_tree.insert(
                "",
                "end",
                values=(rank, f"Agent {agent['agent_id']}", f"${sort_value:.2f}", f"${agent['cash']:.2f}"),
            )

    def _update_market_table(self, snapshot: dict[str, object]) -> None:
        for row in self.market_tree.get_children():
            self.market_tree.delete(row)
        for coin in snapshot["market"]:
            lifecycle = coin.get("lifecycle_stage", "unknown")
            lifecycle_icon = {"birth": "🆕", "growth": "📈", "peak": "🔝", "decline": "📉", "death": "💀"}.get(lifecycle, "")
            self.market_tree.insert(
                "",
                "end",
                values=(
                    coin["coin_id"],
                    f"${coin['price']:.3f}",
                    f"{coin['trend']:+.2f}",
                    f"${coin['volume']:.2f}",
                    f"{lifecycle_icon} {lifecycle}",
                    coin["age"],
                ),
            )

    def _update_training_stats(self) -> None:
        """Update training statistics display."""
        for row in self.stats_tree.get_children():
            self.stats_tree.delete(row)
        
        stats = [
            ("Generation", self.training_stats.get('generation', 0)),
            ("Best Fitness", f"{self.training_stats.get('best_fitness', 0):.4f}"),
            ("Avg Fitness", f"{self.training_stats.get('average_fitness', 0):.4f}"),
            ("Complexity", f"{self.training_stats.get('complexity', 0):.1f}"),
            ("Species", self.training_stats.get('species', 0)),
            ("Algorithm", self.algorithm_var.get().upper()),
            ("Speed", f"{self.train_speed_var.get()}%"),
        ]
        
        for metric, value in stats:
            self.stats_tree.insert("", "end", values=(metric, value))

    def _apply_snapshot(self, snapshot: dict[str, object], description: str | None = None) -> None:
        if self.current_env is None:
            return
        self._update_leaderboard(snapshot)
        self._update_market_table(snapshot)
        self._update_highlights(snapshot)

    def _drain_queue(self) -> None:
        processed = 0
        while processed < self.QUEUE_BATCH_SIZE:
            try:
                event_type, payload = self.message_queue.get_nowait()
            except queue.Empty:
                break
            processed += 1

            if event_type == "log":
                continue
            elif event_type == "status":
                self.status_var.set(str(payload))
            elif event_type == "visual_action":
                action_payload = payload
                self.current_turn_var.set(f"Turn: {action_payload['turn']}")
                agent_id = action_payload["agent_id"]
                self.current_action_label.configure(
                    foreground=self.AGENT_ACTION_COLORS.get(agent_id % len(self.AGENT_ACTION_COLORS), "#f59e0b")
                )
                self._apply_snapshot(action_payload["snapshot"], action_payload["description"])
                if self.current_env is not None:
                    self._refresh_charts_if_due(self.current_env, force=False)
            elif event_type == "turn_end":
                turn_payload = payload
                self.current_turn_var.set(f"Turn: {turn_payload['turn']} complete")
                self._apply_snapshot(turn_payload["snapshot"])
                if self.current_env is not None:
                    self._refresh_charts_if_due(self.current_env, force=True)
            elif event_type == "generation_update":
                generation_payload = payload
                self.current_env = generation_payload["env"]
                summary = generation_payload["summary"]
                self.current_turn_var.set(f"Generation: {summary['generation']}")
                self.current_action_var.set(
                    f"Current action: best fitness {summary['best_fitness']:.3f}, dominant {summary['dominant_behavior']}"
                )
                self.current_action_label.configure(foreground="#2563eb")
                self._append_text(self.summary_text, str(summary))
                self._apply_snapshot(self.current_env.live_snapshot())
                self._refresh_charts_if_due(self.current_env, force=True)
                self.status_var.set(f"Completed generation {summary['generation']}")
                
                # Update training stats
                self.training_stats['generation'] = summary.get('generation', 0)
                self.training_stats['best_fitness'] = summary.get('best_fitness', 0)
                self.training_stats['average_fitness'] = summary.get('average_fitness', 0)
                self.training_stats['complexity'] = summary.get('avg_complexity', 0)
                self.training_stats['species'] = summary.get('num_species', 0)
                self._update_training_stats()
            elif event_type == "episode_complete":
                summary = payload
                if self.current_env is not None:
                    self._refresh_charts_if_due(self.current_env, force=False)

                    # Display event summaries
                    event_summaries = self.current_env.metrics.get("event_summaries", [])
                    major_events = self.current_env.metrics.get("major_events", [])
                    
                    if event_summaries or major_events:
                        self._append_text(self.summary_text, "\n📊 MAJOR EVENTS THIS EPISODE:")
                        events_to_show = event_summaries[:5] if event_summaries else major_events[:5]
                        for event_summary in events_to_show:
                            self._append_text(self.summary_text, f"  • {event_summary}")

                if "generation" in summary:
                    self.status_var.set(f"Completed generation {summary['generation']}")
                else:
                    self.status_var.set(f"Completed episode {summary['episode']}")
            elif event_type == "save_charts":
                # Save charts in main thread to avoid matplotlib/Tk crash
                result = payload
                chart_paths = plot_simulation_metrics(result["env"], result["output_dir"])
                self.message_queue.put(
                    (
                        "done",
                        {
                            "status": result["status"],
                            "summaries": result["summaries"],
                            "output_dir": str(result["output_dir"]),
                            "chart_paths": [str(path) for path in chart_paths],
                            "snapshot": result["env"].live_snapshot(),
                        },
                    )
                )
                self.worker = None
            elif event_type == "done":
                result = payload
                self.status_var.set(f"{result['status']}. Outputs saved to {result['output_dir']}")
                self._append_text(self.summary_text, f"Saved outputs to {result['output_dir']}")
                for chart_path in result["chart_paths"]:
                    self._append_text(self.summary_text, f"Chart: {chart_path}")
                self._set_visual_controls_enabled(False)
                if self.current_env is not None:
                    self._apply_snapshot(result["snapshot"])
                    self._refresh_charts(self.current_env)
            elif event_type == "error":
                self.status_var.set("Simulation failed")
                self._append_text(self.summary_text, f"Error: {payload}")
                self._set_visual_controls_enabled(False)
                self.worker = None

        self.root.after(50, self._drain_queue)

    def run(self) -> None:
        self.root.mainloop()


def launch_gui(config: GameConfig) -> None:
    CoinGameGUI(config).run()
