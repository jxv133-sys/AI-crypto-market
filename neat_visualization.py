"""
NEAT Visualization - Charts for neuroevolution training
"""

from pathlib import Path
from typing import Any

from matplotlib.figure import Figure


def create_neat_fitness_figure(history: dict[str, Any]) -> Figure:
    """Create fitness over generations chart."""
    fig = Figure(figsize=(8, 5))
    ax = fig.add_subplot(111)
    
    best_fitness = history.get('generation_best_fitness', [])
    avg_fitness = history.get('generation_average_fitness', [])
    
    if best_fitness:
        generations = list(range(1, len(best_fitness) + 1))
        ax.plot(generations, best_fitness, label='Best Fitness', linewidth=2, color='#22c55e')
        ax.plot(generations, avg_fitness, label='Avg Fitness', linewidth=2, color='#60a5fa')
        
        ax.set_xlabel('Generation')
        ax.set_ylabel('Fitness')
        ax.set_title('NEAT Fitness Over Generations')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
    
    return fig


def create_neat_complexity_figure(history: dict[str, Any]) -> Figure:
    """Create network complexity over generations chart."""
    fig = Figure(figsize=(8, 5))
    ax = fig.add_subplot(111)
    
    complexity = history.get('generation_complexity', [])
    
    if complexity:
        generations = list(range(1, len(complexity) + 1))
        ax.plot(generations, complexity, linewidth=2, color='#f59e0b')
        
        ax.set_xlabel('Generation')
        ax.set_ylabel('Avg Nodes + Connections')
        ax.set_title('NEAT Network Complexity')
        ax.grid(True, alpha=0.3)
    
    return fig


def create_neat_species_figure(history: dict[str, Any]) -> Figure:
    """Create species count over generations chart."""
    fig = Figure(figsize=(8, 5))
    ax = fig.add_subplot(111)
    
    species_count = history.get('generation_species_count', [])
    
    if species_count:
        generations = list(range(1, len(species_count) + 1))
        ax.bar(generations, species_count, color='#a78bfa', alpha=0.7)
        
        ax.set_xlabel('Generation')
        ax.set_ylabel('Number of Species')
        ax.set_title('NEAT Speciation Over Time')
        ax.grid(True, alpha=0.3, axis='y')
    
    return fig


def create_neat_behavior_figure(history: dict[str, Any]) -> Figure:
    """Create dominant behavior over generations chart."""
    fig = Figure(figsize=(10, 5))
    ax = fig.add_subplot(111)
    
    behaviors = history.get('generation_dominant_behaviors', [])
    
    if behaviors:
        generations = list(range(1, len(behaviors) + 1))
        behavior_ids = [hash(b) % 10 for b in behaviors]
        
        ax.scatter(generations, behavior_ids, c=behavior_ids, cmap='viridis', s=50, alpha=0.6)
        
        ax.set_xlabel('Generation')
        ax.set_ylabel('Behavior Cluster')
        ax.set_title('NEAT Behavioral Diversity Over Generations')
        ax.grid(True, alpha=0.3)
    
    return fig


def plot_neat_metrics(
    history: dict[str, Any],
    output_dir: Path
) -> list[Path]:
    """
    Save all NEAT visualization charts.
    
    Returns:
        List of saved chart paths
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    charts = {
        'neat_fitness.png': create_neat_fitness_figure(history),
        'neat_complexity.png': create_neat_complexity_figure(history),
        'neat_species.png': create_neat_species_figure(history),
        'neat_behavior.png': create_neat_behavior_figure(history),
    }
    
    paths: list[Path] = []
    for filename, figure in charts.items():
        path = output_dir / filename
        figure.savefig(path, dpi=150, bbox_inches='tight')
        paths.append(path)
    
    return paths
