from __future__ import annotations

import argparse
from datetime import datetime

from config import GameConfig
from evolution import EvolutionTrainer
from visualization import plot_simulation_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-agent crypto trading simulation")
    parser.add_argument("--generations", type=int, default=None, help="Number of evolutionary generations")
    parser.add_argument("--turns", type=int, default=None, help="Turns per simulation")
    parser.add_argument("--agents", type=int, default=None, help="Population size")
    parser.add_argument("--starting-money", type=float, default=None, help="Starting cash per agent")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--debug", action="store_true", help="Enable verbose turn logs")
    parser.add_argument("--quiet", action="store_true", help="Disable verbose turn logs")
    parser.add_argument("--gui", action="store_true", help="Launch the desktop GUI with embedded graphs")
    parser.add_argument("--player", action="store_true", help="Launch GUI player game mode")
    parser.add_argument("--text", action="store_true", help="Launch text-based player game")
    parser.add_argument("--neat", action="store_true", help="Use NEAT (neural network evolution) instead of ES")
    parser.add_argument("--load-models", action="store_true", help="Load previously saved genomes before evolution")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> GameConfig:
    config = GameConfig()
    if args.generations is not None:
        config.generations = args.generations
        config.evolution.generations = args.generations
    if args.turns is not None:
        config.turns_per_episode = args.turns
    if args.agents is not None:
        config.num_agents = args.agents
        config.evolution.population_size = args.agents
    if args.starting_money is not None:
        config.starting_money = args.starting_money
    if args.seed is not None:
        config.seed = args.seed
    if args.debug:
        config.debug = True
    if args.quiet:
        config.debug = False
    config.ensure_directories()
    return config


def main() -> None:
    args = parse_args()
    config = build_config(args)
    
    # Player game modes
    if args.player:
        # Ask for number of AI agents
        try:
            num_ai = int(input("Number of AI opponents (5-120, default 119): ") or "119")
            num_ai = max(5, min(120, num_ai))
        except ValueError:
            num_ai = 119
        
        from player_gui import launch_player_gui
        launch_player_gui(num_ai_agents=num_ai, load_models=args.load_models)
        return
    
    if args.text:
        from text_game import run_game
        run_game()
        return
    
    if args.gui:
        from gui import launch_gui

        launch_gui(config)
        return
    
    # NEAT training mode
    if args.neat:
        from neat_trainer import NEATTrainer
        from neat_visualization import plot_neat_metrics
        
        trainer = NEATTrainer(config)
        
        if args.load_models:
            trainer.load_population(config.output.model_dir)
        
        summaries, env = trainer.run()
        trainer.save_population(config.output.model_dir)
        
        # Export NEAT-specific metrics
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = config.output.output_dir / f"neat_simulation_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        env.export_csv(output_dir)
        env.export_replay(output_dir)
        
        # Save NEAT charts
        neat_chart_paths = plot_neat_metrics(trainer.get_training_metrics(), output_dir)
        
        print("\n=== NEAT Training Complete ===")
        for summary in summaries[-3:]:
            print(f"Gen {summary['generation']}: best={summary['best_fitness']:.4f}, "
                  f"avg={summary['average_fitness']:.4f}, "
                  f"complexity={summary['avg_complexity']:.1f}, "
                  f"species={summary['num_species']}")
        print(f"Saved outputs to: {output_dir}")
        for chart_path in neat_chart_paths:
            print(f"NEAT Chart: {chart_path}")
        return

    # Standard evolution mode (ES)
    trainer = EvolutionTrainer(config)
    if args.load_models:
        trainer.load_population(config.output.model_dir)

    summaries, env = trainer.run()
    trainer.save_population(config.output.model_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = config.output.output_dir / f"simulation_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    env.export_csv(output_dir)
    env.export_replay(output_dir)
    chart_paths = plot_simulation_metrics(env, output_dir)

    print("\n=== Evolution Complete ===")
    for summary in summaries[-3:]:
        print(summary)
    print(f"Saved outputs to: {output_dir}")
    for chart_path in chart_paths:
        print(f"Chart: {chart_path}")


if __name__ == "__main__":
    main()
