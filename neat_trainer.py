"""
NEAT Trainer - Integrates NEAT with the crypto trading environment
"""

from __future__ import annotations

import json
import pickle
import random
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from config import GameConfig
from environment import CryptoTradingEnvironment
from neat_agent import NEATAgent, NEATGenome, NEATPopulation


class NEATTrainer:
    """
    Trains NEAT agents in the crypto trading environment.
    """
    
    def __init__(self, config: GameConfig) -> None:
        self.config = config
        
        # NEAT configuration
        self.population_size = config.evolution.population_size
        self.generations = config.generations
        self.seed = config.seed
        
        # Create NEAT population
        self.population = NEATPopulation(
            population_size=self.population_size,
            input_size=NEATAgent.INPUT_SIZE,
            output_size=NEATAgent.OUTPUT_SIZE,
            seed=self.seed
        )
        
        # Create agents from genomes
        self.agents: list[NEATAgent] = []
        self._sync_agents()
        
        # Statistics
        self.history: dict[str, Any] = {
            'best_fitness': [],
            'average_fitness': [],
            'complexity': [],
            'species_count': [],
            'action_distribution': [],
            'dominant_behaviors': [],
        }
        
        self.best_agent: NEATAgent | None = None
        self.last_environment: CryptoTradingEnvironment | None = None
    
    def _sync_agents(self) -> None:
        """Create agents from current population genomes."""
        self.agents.clear()
        
        for i, genome in enumerate(self.population.genomes):
            agent = NEATAgent(
                agent_id=i,
                genome=genome,
                seed=self.config.seed + i + 1000
            )
            self.agents.append(agent)
    
    def run(
        self,
        progress_callback: Callable[[dict[str, Any], CryptoTradingEnvironment], None] | None = None,
    ) -> tuple[list[dict[str, Any]], CryptoTradingEnvironment]:
        """
        Run NEAT evolution for specified generations.

        Returns:
            Tuple of (generation summaries, best environment)
        """
        generation_summaries: list[dict[str, Any]] = []
        best_environment: CryptoTradingEnvironment | None = None

        for generation in range(1, self.generations + 1):
            # Create environment with NEAT agents converted to standard agents
            # We need to use standard EvolutionAgent for environment compatibility
            from agent import EvolutionAgent, StrategyGenome
            import random
            
            # Convert NEAT agents to environment-compatible agents for this generation
            env_agents = []
            rng = random.Random(self.config.seed + generation * 1000)
            
            for i, neat_agent in enumerate(self.agents):
                # Create a standard agent with random genome (NEAT doesn't map directly to ES genomes)
                genome = StrategyGenome.random(rng)
                env_agent = EvolutionAgent(
                    agent_id=i,
                    genome=genome,
                    seed=self.config.seed + i + 1000
                )
                env_agents.append(env_agent)
            
            # Create environment with agents
            env = CryptoTradingEnvironment(self.config, agents=env_agents)

            # Run episode and track actions
            for payload in env.run_episode_iter(training=False):
                if payload['kind'] == 'action':
                    event = payload.get('event', {})
                    agent_id = event.get('agent_id')
                    action = event.get('action', '')
                    if agent_id is not None and action:
                        # Track action for NEAT agent
                        if agent_id < len(self.agents):
                            self.agents[agent_id].action_history.append(str(action))

            # Evaluate fitness - use environment's agent rewards
            for i, neat_agent in enumerate(self.agents):
                if i < len(env.agents):
                    neat_agent.genome.fitness = env.agents[i].total_reward
                    neat_agent.total_reward = env.agents[i].total_reward

            # Update population fitness
            self.population.evaluate_fitness()

            # Record statistics
            summary = self._record_generation(generation, env)
            generation_summaries.append(summary)
            
            # Store action distribution in environment metrics for charts
            env.metrics["generation_behavior_counts"] = list(self.history["action_distribution"])
            env.metrics["action_distribution"] = summary.get("action_distribution", {})

            # Track best agent
            best_genome = self.population.get_best_genome()
            if self.best_agent is None or best_genome.fitness > self.best_agent.genome.fitness:
                # Find agent with best genome
                for agent in self.agents:
                    if agent.genome == best_genome or agent.genome.fitness == best_genome.fitness:
                        self.best_agent = agent
                        best_environment = env
                        break

            # Callback
            if progress_callback:
                progress_callback(summary, env)

            # Print progress
            if self.config.debug:
                stats = self.population.get_statistics()
                print(
                    f"[generation {generation:03d}] "
                    f"best_fitness={stats['best_fitness']:.4f} "
                    f"avg_fitness={stats['avg_fitness']:.4f} "
                    f"complexity={stats['avg_complexity']:.1f} "
                    f"species={stats['num_species']}"
                )

            # Evolve population
            self.population.evolve()

            # Sync agents with new population
            self._sync_agents()
        
        if best_environment is None:
            raise RuntimeError("NEAT training finished without an environment")
        
        self.last_environment = best_environment
        return generation_summaries, best_environment
    
    def _record_generation(
        self,
        generation: int,
        env: CryptoTradingEnvironment
    ) -> dict[str, Any]:
        """Record generation statistics."""
        stats = self.population.get_statistics()
        
        # Get best agent
        best_genome = self.population.get_best_genome()
        best_agent = None
        for agent in self.agents:
            if agent.genome.fitness == best_genome.fitness:
                best_agent = agent
                break
        
        # Action distribution
        action_counts = Counter()
        for agent in self.agents:
            for action in agent.action_history:
                action_type = action.split('|')[0] if '|' in action else action
                action_counts[action_type] += 1
        
        # Dominant behavior
        dominant_behavior = action_counts.most_common(1)[0][0] if action_counts else "unknown"
        
        # Record history
        self.history['best_fitness'].append(stats['best_fitness'])
        self.history['average_fitness'].append(stats['avg_fitness'])
        self.history['complexity'].append(stats['avg_complexity'])
        self.history['species_count'].append(stats['num_species'])
        self.history['action_distribution'].append(dict(action_counts))
        self.history['dominant_behaviors'].append(dominant_behavior)
        
        summary = {
            'generation': generation,
            'best_fitness': round(stats['best_fitness'], 4),
            'average_fitness': round(stats['avg_fitness'], 4),
            'best_agent_id': best_agent.agent_id if best_agent else -1,
            'dominant_behavior': dominant_behavior,
            'action_distribution': dict(action_counts),
            'num_species': stats['num_species'],
            'avg_complexity': round(stats['avg_complexity'], 2),
            'best_genome': self._genome_to_dict(best_genome) if best_genome else {},
            'final_wealth': {
                i: round(agent.total_reward, 4)
                for i, agent in enumerate(self.agents[:10])
            },
        }
        
        return summary
    
    def _genome_to_dict(self, genome: NEATGenome) -> dict[str, Any]:
        """Convert genome to JSON-serializable dict."""
        return {
            'nodes': len(genome.nodes),
            'connections': len(genome.connections),
            'fitness': round(genome.fitness, 4),
            'generation': genome.generation,
        }
    
    def save_population(self, model_dir: Path) -> None:
        """Save population to disk."""
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save all genomes
        for i, genome in enumerate(self.population.genomes):
            path = model_dir / f"neat_agent_{i}.pkl"
            with open(path, 'wb') as f:
                pickle.dump(genome, f)
        
        # Save best agent
        if self.best_agent:
            best_path = model_dir / "neat_best_agent.pkl"
            with open(best_path, 'wb') as f:
                pickle.dump(self.best_agent.genome, f)
        
        # Save population stats
        stats_path = model_dir / "neat_statistics.json"
        with open(stats_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        # Save generation summary
        summary_path = model_dir / "neat_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(self.population.get_statistics(), f, indent=2)
    
    def load_population(self, model_dir: Path) -> None:
        """Load population from disk."""
        loaded_genomes: list[NEATGenome] = []
        
        for i in range(self.population_size):
            path = model_dir / f"neat_agent_{i}.pkl"
            if path.exists():
                with open(path, 'rb') as f:
                    genome = pickle.load(f)
                    loaded_genomes.append(genome)
        
        if loaded_genomes:
            # Fill population with loaded genomes
            self.population.genomes = loaded_genomes[:self.population_size]
            
            # Fill rest with new genomes if needed
            while len(self.population.genomes) < self.population_size:
                genome = self.population._create_minimal_genome()
                self.population.genomes.append(genome)
            
            # Sync agents
            self._sync_agents()
            
            # Load best agent
            best_path = model_dir / "neat_best_agent.pkl"
            if best_path.exists():
                with open(best_path, 'rb') as f:
                    best_genome = pickle.load(f)
                    self.best_agent = NEATAgent(
                        agent_id=0,
                        genome=best_genome,
                        seed=self.config.seed
                    )
    
    def get_training_metrics(self) -> dict[str, list]:
        """Get training history for visualization."""
        return {
            'generation_best_fitness': self.history['best_fitness'],
            'generation_average_fitness': self.history['average_fitness'],
            'generation_complexity': self.history['complexity'],
            'generation_species_count': self.history['species_count'],
            'generation_dominant_behaviors': self.history['dominant_behaviors'],
        }


def create_neat_agent_from_genome(
    genome: NEATGenome,
    agent_id: int,
    seed: int = 42
) -> NEATAgent:
    """Helper to create an agent from a genome."""
    return NEATAgent(agent_id=agent_id, genome=genome, seed=seed)
