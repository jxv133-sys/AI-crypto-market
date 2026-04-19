"""
NEAT (NeuroEvolution of Augmenting Topologies) Agent System
Neural network-based agents with evolving topologies
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any

from agent import ActionType


@dataclass
class NEATGenome:
    """
    NEAT genome representing a neural network.
    Contains nodes and connections with innovation tracking.
    """
    # Network structure
    nodes: dict[int, dict[str, Any]] = field(default_factory=dict)
    connections: dict[tuple[int, int], dict[str, Any]] = field(default_factory=dict)
    
    # Innovation tracking for speciation
    innovation_history: list[int] = field(default_factory=list)
    
    # Fitness tracking
    fitness: float = 0.0
    adjusted_fitness: float = 0.0
    
    # Metadata
    generation: int = 0
    species_id: int = 0
    
    def copy(self) -> NEATGenome:
        """Create a deep copy of this genome."""
        import copy
        return copy.deepcopy(self)


class NEATNeuralNetwork:
    """
    Neural network that executes a NEAT genome.
    Supports arbitrary topology with recurrent connections.
    """
    
    def __init__(self, genome: NEATGenome, input_size: int, output_size: int) -> None:
        self.genome = genome
        self.input_size = input_size
        self.output_size = output_size
        self.node_values: dict[int, float] = {}
        
    def activate(self, inputs: list[float]) -> list[float]:
        """
        Forward pass through the network.
        
        Args:
            inputs: List of input values (should match input_size)
            
        Returns:
            List of output values
        """
        # Clear node values
        self.node_values.clear()
        
        # Set input nodes (nodes 0 to input_size-1)
        for i, value in enumerate(inputs[:self.input_size]):
            self.node_values[i] = self._sigmoid(value)
        
        # Get hidden and output nodes sorted by depth (for feedforward)
        # or just iterate multiple times for recurrent
        node_ids = sorted(self.genome.nodes.keys())
        
        # Multiple passes for recurrent connections
        for _ in range(3):
            for node_id in node_ids:
                if node_id < self.input_size:
                    continue  # Skip input nodes
                    
                node = self.genome.nodes.get(node_id, {})
                node_type = node.get('type', 'hidden')
                bias = node.get('bias', 0.0)
                
                # Sum incoming connections
                total_input = bias
                for (in_id, out_id), conn in self.genome.connections.items():
                    if out_id == node_id and conn.get('enabled', True):
                        in_value = self.node_values.get(in_id, 0.0)
                        weight = conn.get('weight', 1.0)
                        total_input += in_value * weight
                
                # Apply activation
                if node_type == 'output':
                    self.node_values[node_id] = self._sigmoid(total_input)
                else:
                    self.node_values[node_id] = self._tanh(total_input)
        
        # Collect outputs (last output_size nodes)
        outputs = []
        output_nodes = sorted(
            [nid for nid, n in self.genome.nodes.items() 
             if n.get('type') == 'output']
        )[-self.output_size:]
        
        for node_id in output_nodes:
            outputs.append(self.node_values.get(node_id, 0.0))
        
        return outputs
    
    @staticmethod
    def _sigmoid(x: float) -> float:
        """Sigmoid activation function."""
        return 1.0 / (1.0 + math.exp(-max(-500, min(500, x))))
    
    @staticmethod
    def _tanh(x: float) -> float:
        """Tanh activation function."""
        return math.tanh(max(-5, min(5, x)))


class NEATAgent:
    """
    Agent that uses a NEAT neural network for decision making.
    """
    
    # Input feature indices
    INPUT_CASH = 0
    INPUT_NET_WORTH = 1
    INPUT_LIQUIDITY_RATIO = 2
    INPUT_NUM_COINS = 3
    INPUT_AVG_PRICE = 4
    INPUT_AVG_TREND = 5
    INPUT_AVG_VOLATILITY = 6
    INPUT_MINING_PROFITABILITY = 7
    INPUT_HOLDINGS_VALUE = 8
    INPUT_WORK_INCOME = 9
    INPUT_BIAS = 10  # Always 1.0
    
    INPUT_SIZE = 11
    
    # Output indices
    OUTPUT_BUY = 0
    OUTPUT_SELL = 1
    OUTPUT_MINE = 2
    OUTPUT_CREATE = 3
    OUTPUT_HOLD = 4
    OUTPUT_TREND = 5
    OUTPUT_SPECULATE = 6
    OUTPUT_ARBITRAGE = 7
    OUTPUT_MARKET_MAKE = 8
    OUTPUT_LEVERAGE = 9
    OUTPUT_FRACTION = 10  # Trade fraction (0-1)
    OUTPUT_COIN_IDX = 11  # Coin selection
    
    OUTPUT_SIZE = 12
    
    def __init__(
        self,
        agent_id: int,
        genome: NEATGenome,
        seed: int = 42
    ) -> None:
        self.agent_id = agent_id
        self.genome = genome
        self.seed = seed
        self.random = random.Random(seed)
        
        # Create neural network
        self.network = NEATNeuralNetwork(
            genome=genome,
            input_size=self.INPUT_SIZE,
            output_size=self.OUTPUT_SIZE
        )
        
        # Tracking
        self.total_reward = 0.0
        self.action_history: list[str] = []
        self.generation = 0
        
    def select_actions(
        self,
        state: dict[str, Any],
        available_actions: list,
        training: bool = True
    ) -> list:
        """
        Select actions using neural network inference.
        
        Args:
            state: Environment observation
            available_actions: List of available Action objects
            training: Whether in training mode
            
        Returns:
            List of selected actions
        """
        # Prepare inputs
        inputs = self._prepare_inputs(state)
        
        # Forward pass
        outputs = self.network.activate(inputs)
        
        # Decode outputs into actions
        actions = self._decode_outputs(outputs, available_actions)
        
        # Record actions
        for action in actions:
            self.action_history.append(str(action))
        
        return actions
    
    def _prepare_inputs(self, state: dict[str, Any]) -> list[float]:
        """
        Prepare normalized input features from state.
        
        Args:
            state: Environment observation dict
            
        Returns:
            List of normalized input values
        """
        # Extract state values
        cash = state.get('cash', 0.0)
        holdings_value = state.get('holdings_value', 0.0)
        net_worth = cash + holdings_value
        liquidity_ratio = cash / max(net_worth, 1.0)
        
        coins = state.get('visible_coins', [])
        num_coins = len(coins)
        
        # Market averages
        avg_price = 0.0
        avg_trend = 0.0
        avg_volatility = 0.0
        
        if coins:
            prices = [c.get('price', 1.0) for c in coins]
            trends = [c.get('trend', 0.0) for c in coins]
            
            avg_price = sum(prices) / len(prices)
            avg_trend = sum(trends) / len(trends)
            
            # Simple volatility estimate
            if len(prices) > 1:
                avg_volatility = max(prices) - min(prices)
        
        # Mining profitability (simplified)
        mining_profitability = state.get('mining_profitability', 0.5)
        
        # Work income
        work_income = state.get('work_income', 1.0)
        
        # Normalize inputs to 0-1 range
        inputs = [
            self._normalize(cash, 0, 500),           # Cash
            self._normalize(net_worth, 0, 1000),     # Net worth
            liquidity_ratio,                          # Liquidity ratio (already 0-1)
            self._normalize(num_coins, 0, 20),       # Number of coins
            self._normalize(avg_price, 0, 50),       # Average price
            self._normalize(avg_trend + 1, -1, 1),   # Average trend
            self._normalize(avg_volatility, 0, 10),  # Volatility
            mining_profitability,                     # Mining profitability
            self._normalize(holdings_value, 0, 500), # Holdings value
            self._normalize(work_income, 0, 10),     # Work income
            1.0,                                      # Bias
        ]
        
        return inputs
    
    def _decode_outputs(
        self,
        outputs: list[float],
        available_actions: list
    ) -> list:
        """
        Decode neural network outputs into actions.
        
        Args:
            outputs: Network output values (0-1)
            available_actions: Available actions
            
        Returns:
            List of selected actions
        """
        if not available_actions:
            return []
        
        # Get action scores from outputs
        action_scores = {
            ActionType.BUY: outputs[self.OUTPUT_BUY] if len(outputs) > self.OUTPUT_BUY else 0.0,
            ActionType.SELL: outputs[self.OUTPUT_SELL] if len(outputs) > self.OUTPUT_SELL else 0.0,
            ActionType.MINE: outputs[self.OUTPUT_MINE] if len(outputs) > self.OUTPUT_MINE else 0.0,
            ActionType.CREATE: outputs[self.OUTPUT_CREATE] if len(outputs) > self.OUTPUT_CREATE else 0.0,
            ActionType.HOLD: outputs[self.OUTPUT_HOLD] if len(outputs) > self.OUTPUT_HOLD else 0.0,
            ActionType.TREND: outputs[self.OUTPUT_TREND] if len(outputs) > self.OUTPUT_TREND else 0.0,
            ActionType.SPECULATE: outputs[self.OUTPUT_SPECULATE] if len(outputs) > self.OUTPUT_SPECULATE else 0.0,
            ActionType.ARBITRAGE: outputs[self.OUTPUT_ARBITRAGE] if len(outputs) > self.OUTPUT_ARBITRAGE else 0.0,
            ActionType.MARKET_MAKE: outputs[self.OUTPUT_MARKET_MAKE] if len(outputs) > self.OUTPUT_MARKET_MAKE else 0.0,
            ActionType.LEVERAGE: outputs[self.OUTPUT_LEVERAGE] if len(outputs) > self.OUTPUT_LEVERAGE else 0.0,
        }
        
        # Get fraction and coin index
        fraction = outputs[self.OUTPUT_FRACTION] if len(outputs) > self.OUTPUT_FRACTION else 0.5
        coin_idx = int(outputs[self.OUTPUT_COIN_IDX] * 10) if len(outputs) > self.OUTPUT_COIN_IDX else 0
        
        # Select action with highest score that's available
        sorted_actions = sorted(
            action_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        selected_actions = []
        
        for action_type, score in sorted_actions[:2]:  # Top 2 actions
            # Find matching available action
            for action in available_actions:
                if action.action_type == action_type:
                    # Apply fraction if applicable
                    if fraction and hasattr(action, 'fraction'):
                        selected_actions.append(action)
                        break
                    elif action_type in {ActionType.HOLD, ActionType.MINE, ActionType.CREATE}:
                        selected_actions.append(action)
                        break
        
        # Default to hold if nothing selected
        if not selected_actions:
            for action in available_actions:
                if action.action_type == ActionType.HOLD:
                    selected_actions.append(action)
                    break
        
        return selected_actions[:1]  # Return single best action
    
    @staticmethod
    def _normalize(value: float, min_val: float, max_val: float) -> float:
        """Normalize value to 0-1 range."""
        if max_val == min_val:
            return 0.5
        normalized = (value - min_val) / (max_val - min_val)
        return max(0.0, min(1.0, normalized))
    
    def record_reward(self, reward: float) -> None:
        """Record reward for fitness calculation."""
        self.total_reward += reward
        self.genome.fitness = self.total_reward
    
    def reset(self) -> None:
        """Reset agent state for new episode."""
        self.total_reward = 0.0
        self.action_history.clear()


class NEATPopulation:
    """
    Manages a population of NEAT genomes with speciation.
    """
    
    def __init__(
        self,
        population_size: int = 120,
        input_size: int = NEATAgent.INPUT_SIZE,
        output_size: int = NEATAgent.OUTPUT_SIZE,
        seed: int = 42
    ) -> None:
        self.population_size = population_size
        self.input_size = input_size
        self.output_size = output_size
        self.seed = seed
        self.random = random.Random(seed)
        
        # Population
        self.genomes: list[NEATGenome] = []
        self.species: dict[int, list[NEATGenome]] = {}
        
        # Innovation tracking
        self.innovation_counter = 0
        self.connection_innovations: dict[tuple[int, int], int] = {}
        
        # Statistics
        self.generation = 0
        self.best_fitness_history: list[float] = []
        self.avg_fitness_history: list[float] = []
        self.complexity_history: list[float] = []
        
        # Initialize population
        self._initialize_population()
    
    def _initialize_population(self) -> None:
        """Create initial population with minimal networks."""
        self.genomes.clear()
        
        for i in range(self.population_size):
            genome = self._create_minimal_genome()
            genome.generation = 0
            self.genomes.append(genome)
    
    def _create_minimal_genome(self) -> NEATGenome:
        """Create a minimal genome with just inputs and outputs."""
        genome = NEATGenome()
        
        # Create input nodes
        for i in range(self.input_size):
            genome.nodes[i] = {'type': 'input', 'bias': 0.0}
        
        # Create output nodes
        for i in range(self.output_size):
            node_id = self.input_size + i
            genome.nodes[node_id] = {'type': 'output', 'bias': 0.0}
        
        # Create initial connections (all inputs to all outputs)
        for in_id in range(self.input_size):
            for out_id in range(self.input_size, self.input_size + self.output_size):
                conn_id = (in_id, out_id)
                innovation = self._get_innovation(conn_id)
                genome.connections[conn_id] = {
                    'weight': self.random.gauss(0, 1),
                    'enabled': True,
                    'innovation': innovation,
                }
        
        return genome
    
    def _get_innovation(self, connection: tuple[int, int]) -> int:
        """Get or create innovation number for a connection."""
        if connection not in self.connection_innovations:
            self.innovation_counter += 1
            self.connection_innovations[connection] = self.innovation_counter
        return self.connection_innovations[connection]
    
    def evaluate_fitness(self) -> None:
        """Calculate adjusted fitness with speciation."""
        # Reset species
        self.species.clear()
        
        # Speciate genomes
        for genome in self.genomes:
            species_id = self._find_species(genome)
            genome.species_id = species_id
            
            if species_id not in self.species:
                self.species[species_id] = []
            self.species[species_id].append(genome)
        
        # Calculate adjusted fitness
        for species_id, species_genomes in self.species.items():
            avg_fitness = sum(g.fitness for g in species_genomes) / len(species_genomes)
            
            for genome in species_genomes:
                genome.adjusted_fitness = genome.fitness / max(len(species_genomes), 1)
    
    def _find_species(self, genome: NEATGenome) -> int:
        """Find or create species for a genome."""
        for species_id, species_genomes in self.species.items():
            if species_genomes:
                representative = species_genomes[0]
                distance = self._genome_distance(genome, representative)
                
                if distance < 3.0:  # Compatibility threshold
                    return species_id
        
        # Create new species
        new_id = max(self.species.keys(), default=0) + 1
        return new_id
    
    def _genome_distance(
        self,
        genome_a: NEATGenome,
        genome_b: NEATGenome
    ) -> float:
        """
        Calculate compatibility distance between two genomes.
        Used for speciation.
        """
        # Get innovation numbers
        innovations_a = set(c.get('innovation', 0) for c in genome_a.connections.values())
        innovations_b = set(c.get('innovation', 0) for c in genome_b.connections.values())

        # Count excess and disjoint genes
        excess = len(innovations_a.symmetric_difference(innovations_b))
        disjoint = len(innovations_a.intersection(innovations_b))

        # Average weight difference
        common_innovations = innovations_a.intersection(innovations_b)
        weight_diff = 0.0

        if common_innovations:
            for conn_a in genome_a.connections.values():
                for conn_b in genome_b.connections.values():
                    if conn_a.get('innovation') in common_innovations and \
                       conn_b.get('innovation') == conn_a.get('innovation'):
                        weight_diff += abs(conn_a.get('weight', 1.0) - conn_b.get('weight', 1.0))
            weight_diff /= max(len(common_innovations), 1)

        # Normalize by genome size
        N = max(len(genome_a.connections), len(genome_b.connections), 1)
        
        # Distance formula from NEAT paper
        c1 = 1.0  # Excess genes coefficient
        c2 = 1.0  # Disjoint genes coefficient
        c3 = 0.4  # Weight difference coefficient
        
        distance = (c1 * excess / N) + (c2 * disjoint / N) + (c3 * weight_diff)
        return distance
    
    def evolve(self) -> None:
        """
        Create next generation through selection, crossover, and mutation.
        """
        self.generation += 1
        
        # Evaluate and speciate
        self.evaluate_fitness()
        
        # Record statistics
        fitnesses = [g.fitness for g in self.genomes]
        self.best_fitness_history.append(max(fitnesses))
        self.avg_fitness_history.append(sum(fitnesses) / len(fitnesses))
        
        avg_nodes = sum(len(g.nodes) for g in self.genomes) / len(self.genomes)
        avg_conns = sum(len(g.connections) for g in self.genomes) / len(self.genomes)
        self.complexity_history.append((avg_nodes + avg_conns) / 2)
        
        # Sort by adjusted fitness
        self.genomes.sort(key=lambda g: g.adjusted_fitness, reverse=True)
        
        # Create new population
        new_genomes: list[NEATGenome] = []
        
        # Elitism: Keep top 2 unchanged
        elite_count = min(2, len(self.genomes))
        for i in range(elite_count):
            new_genomes.append(self.genomes[i].copy())
        
        # Fill rest of population
        while len(new_genomes) < self.population_size:
            # Select parents from species (fitness sharing)
            parent1 = self._select_parent()
            parent2 = self._select_parent()
            
            # Crossover
            if self.random.random() < 0.75:
                child = self._crossover(parent1, parent2)
            else:
                child = parent1.copy()
            
            # Mutate
            self._mutate(child)
            
            child.generation = self.generation
            new_genomes.append(child)
        
        self.genomes = new_genomes
    
    def _select_parent(self) -> NEATGenome:
        """Select parent using tournament selection within species."""
        # Tournament selection
        tournament_size = 5
        candidates = self.random.sample(self.genomes, min(tournament_size, len(self.genomes)))
        return max(candidates, key=lambda g: g.adjusted_fitness)
    
    def _crossover(
        self,
        parent1: NEATGenome,
        parent2: NEATGenome
    ) -> NEATGenome:
        """
        Perform crossover between two parents.
        Follows NEAT crossover rules (match by innovation).
        """
        child = NEATGenome()
        
        # Copy nodes from fitter parent
        fitter = parent1 if parent1.fitness > parent2.fitness else parent2
        child.nodes = fitter.nodes.copy()

        # Match connections by innovation
        p1_by_innov = {c.get('innovation'): (k, c) for k, c in parent1.connections.items()}
        p2_by_innov = {c.get('innovation'): (k, c) for k, c in parent2.connections.items()}

        all_innovations = set(p1_by_innov.keys()) | set(p2_by_innov.keys())

        for innov in all_innovations:
            item1 = p1_by_innov.get(innov)
            item2 = p2_by_innov.get(innov)

            if item1 and item2:
                # Both have gene - pick one
                key1, conn1 = item1
                key2, conn2 = item2
                if self.random.random() < 0.5:
                    child.connections[key1] = conn1.copy()
                else:
                    child.connections[key2] = conn2.copy()
            elif item1:
                # Only parent 1 has gene
                key1, conn1 = item1
                if parent1.fitness >= parent2.fitness:
                    child.connections[key1] = conn1.copy()
            elif item2:
                # Only parent 2 has gene
                key2, conn2 = item2
                if parent2.fitness > parent1.fitness:
                    child.connections[key2] = conn2.copy()

        return child
    
    def _mutate(self, genome: NEATGenome) -> None:
        """Apply mutations to a genome."""
        # Weight mutation (85% chance)
        if self.random.random() < 0.85:
            for conn in genome.connections.values():
                if self.random.random() < 0.8:
                    # Perturb weight
                    conn['weight'] += self.random.gauss(0, 0.1)
                else:
                    # Replace weight
                    conn['weight'] = self.random.gauss(0, 1)
        
        # Add node mutation (3% chance)
        if self.random.random() < 0.03:
            self._mutate_add_node(genome)
        
        # Add connection mutation (5% chance)
        if self.random.random() < 0.05:
            self._mutate_add_connection(genome)
        
        # Enable/disable mutation (2% chance)
        if self.random.random() < 0.02:
            self._mutate_enable_disable(genome)
    
    def _mutate_add_node(self, genome: NEATGenome) -> None:
        """Add a new node to the network."""
        if not genome.connections:
            return
        
        # Select random connection to split
        conn_key = self.random.choice(list(genome.connections.keys()))
        conn = genome.connections[conn_key]
        
        if not conn.get('enabled', True):
            return
        
        in_id, out_id = conn_key
        
        # Create new node
        new_node_id = max(genome.nodes.keys()) + 1
        genome.nodes[new_node_id] = {'type': 'hidden', 'bias': 0.0}
        
        # Disable old connection
        conn['enabled'] = False
        
        # Add new connections
        innov1 = self._get_innovation((in_id, new_node_id))
        genome.connections[(in_id, new_node_id)] = {
            'weight': 1.0,
            'enabled': True,
            'innovation': innov1,
            'in_id': in_id,
            'out_id': new_node_id,
        }
        
        innov2 = self._get_innovation((new_node_id, out_id))
        genome.connections[(new_node_id, out_id)] = {
            'weight': conn['weight'],
            'enabled': True,
            'innovation': innov2,
            'in_id': new_node_id,
            'out_id': out_id,
        }
    
    def _mutate_add_connection(self, genome: NEATGenome) -> None:
        """Add a new connection to the network."""
        # Get possible connections
        input_nodes = [nid for nid, n in genome.nodes.items() if n.get('type') == 'input']
        other_nodes = [nid for nid, n in genome.nodes.items() if n.get('type') != 'input']
        
        if not input_nodes or not other_nodes:
            return
        
        # Try to add connection
        for _ in range(10):
            in_id = self.random.choice(input_nodes)
            out_id = self.random.choice(other_nodes)
            
            if in_id == out_id:
                continue
            
            conn_key = (in_id, out_id)
            if conn_key not in genome.connections:
                innov = self._get_innovation(conn_key)
                genome.connections[conn_key] = {
                    'weight': self.random.gauss(0, 1),
                    'enabled': True,
                    'innovation': innov,
                    'in_id': in_id,
                    'out_id': out_id,
                }
                break
            elif not genome.connections[conn_key].get('enabled', True):
                # Re-enable disabled connection
                genome.connections[conn_key]['enabled'] = True
                break
    
    def _mutate_enable_disable(self, genome: NEATGenome) -> None:
        """Enable or disable a connection."""
        if not genome.connections:
            return
        
        conn_key = self.random.choice(list(genome.connections.keys()))
        conn = genome.connections[conn_key]
        
        # Only disable, don't enable (enabling handled in add_connection)
        if conn.get('enabled', True):
            conn['enabled'] = False
    
    def get_best_genome(self) -> NEATGenome:
        """Get the best genome in the population."""
        return max(self.genomes, key=lambda g: g.fitness)
    
    def get_statistics(self) -> dict[str, Any]:
        """Get population statistics."""
        fitnesses = [g.fitness for g in self.genomes]
        complexities = [len(g.nodes) + len(g.connections) for g in self.genomes]
        
        return {
            'generation': self.generation,
            'best_fitness': max(fitnesses),
            'avg_fitness': sum(fitnesses) / len(fitnesses),
            'best_complexity': max(complexities),
            'avg_complexity': sum(complexities) / len(complexities),
            'num_species': len(self.species),
            'species_sizes': {k: len(v) for k, v in self.species.items()},
        }
