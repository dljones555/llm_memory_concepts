"""
⚡ Hardware-Inspired Memory Architecture
Simulating brain-like memory systems with modern computing concepts
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import threading
import time
import asyncio
from collections import defaultdict, deque
import weakref
import math

class NeuralOscillation(Enum):
    """Brain wave patterns that affect memory consolidation."""
    GAMMA = "gamma"      # 30-100 Hz - binding, attention
    BETA = "beta"        # 13-30 Hz - active thinking
    ALPHA = "alpha"      # 8-13 Hz - relaxed awareness
    THETA = "theta"      # 4-8 Hz - memory consolidation
    DELTA = "delta"      # 0.5-4 Hz - deep sleep, long-term storage

class SynapticStrength(Enum):
    """Synaptic connection strengths."""
    POTENTIATED = 1.5    # Long-term potentiation
    NORMAL = 1.0         # Baseline strength
    DEPRESSED = 0.5      # Long-term depression
    SILENT = 0.1         # Nearly inactive

@dataclass
class NeuralCluster:
    """Represents a cluster of neurons with similar functionality."""
    cluster_id: str
    neurons: Dict[str, 'ArtificialNeuron'] = field(default_factory=dict)
    activation_threshold: float = 0.7
    lateral_inhibition: float = 0.3
    cluster_type: str = "memory"
    synchronization_frequency: float = 40.0  # Hz
    
    def add_neuron(self, neuron: 'ArtificialNeuron'):
        """Add a neuron to this cluster."""
        self.neurons[neuron.neuron_id] = neuron
        neuron.parent_cluster = self.cluster_id
    
    def compute_cluster_activation(self) -> float:
        """Compute overall cluster activation."""
        if not self.neurons:
            return 0.0
        
        activations = [n.current_activation for n in self.neurons.values()]
        # Apply lateral inhibition
        max_activation = max(activations) if activations else 0.0
        avg_activation = np.mean(activations)
        
        # Winner-takes-more with lateral inhibition
        return max_activation * (1 - self.lateral_inhibition) + avg_activation * self.lateral_inhibition
    
    def synchronize_oscillation(self, global_time: float):
        """Synchronize neural oscillations within the cluster."""
        phase = 2 * math.pi * self.synchronization_frequency * global_time
        sync_factor = (math.sin(phase) + 1) / 2  # 0 to 1
        
        for neuron in self.neurons.values():
            neuron.oscillation_phase = phase
            neuron.sync_modulation = sync_factor

@dataclass
class ArtificialNeuron:
    """Brain-inspired artificial neuron with realistic properties."""
    neuron_id: str
    neuron_type: str = "pyramidal"  # pyramidal, interneuron, etc.
    resting_potential: float = -70.0  # mV
    threshold_potential: float = -55.0  # mV
    current_potential: float = -70.0
    current_activation: float = 0.0
    refractory_period: float = 2.0  # ms
    last_spike_time: float = 0.0
    
    # Synaptic properties
    incoming_synapses: Dict[str, 'Synapse'] = field(default_factory=dict)
    outgoing_synapses: Dict[str, 'Synapse'] = field(default_factory=dict)
    
    # Memory-specific properties
    memory_trace_strength: float = 0.0
    consolidation_factor: float = 0.0
    interference_resistance: float = 0.5
    
    # Neural oscillation properties
    oscillation_phase: float = 0.0
    sync_modulation: float = 1.0
    parent_cluster: Optional[str] = None
    
    def receive_input(self, current_time: float, input_current: float):
        """Receive synaptic input and update membrane potential."""
        # Check refractory period
        if current_time - self.last_spike_time < self.refractory_period:
            return False
        
        # Update membrane potential
        decay_factor = 0.95  # Passive decay
        self.current_potential = (self.current_potential * decay_factor + 
                                input_current * self.sync_modulation)
        
        # Check for spike
        if self.current_potential >= self.threshold_potential:
            self.spike(current_time)
            return True
        
        # Update activation based on potential
        self.current_activation = max(0, (self.current_potential - self.resting_potential) / 
                                    (self.threshold_potential - self.resting_potential))
        return False
    
    def spike(self, current_time: float):
        """Generate an action potential."""
        self.last_spike_time = current_time
        self.current_potential = self.resting_potential
        self.current_activation = 1.0
        
        # Propagate spike to connected neurons
        for synapse in self.outgoing_synapses.values():
            synapse.transmit_spike(current_time)
        
        # Strengthen memory trace
        self.memory_trace_strength = min(1.0, self.memory_trace_strength + 0.1)

@dataclass
class Synapse:
    """Synaptic connection between neurons."""
    synapse_id: str
    presynaptic_neuron: str
    postsynaptic_neuron: str
    weight: float = 1.0
    delay: float = 1.0  # ms
    plasticity_rule: str = "hebbian"
    
    # Plasticity properties
    last_pre_spike: float = 0.0
    last_post_spike: float = 0.0
    eligibility_trace: float = 0.0
    
    # Neurotransmitter simulation
    vesicle_pool: int = 100
    release_probability: float = 0.3
    
    def transmit_spike(self, spike_time: float):
        """Transmit spike across synapse with delay."""
        self.last_pre_spike = spike_time
        
        # Simulate neurotransmitter release
        if np.random.random() < self.release_probability and self.vesicle_pool > 0:
            self.vesicle_pool -= 1
            # Schedule postsynaptic current (would be handled by scheduler)
            return self.weight
        return 0.0
    
    def update_plasticity(self, learning_rate: float = 0.01):
        """Update synaptic strength based on spike timing."""
        if self.plasticity_rule == "hebbian":
            # Simple Hebbian learning
            if self.last_pre_spike > 0 and self.last_post_spike > 0:
                time_diff = abs(self.last_post_spike - self.last_pre_spike)
                if time_diff < 20:  # 20ms window
                    # Strengthen if spikes are correlated
                    self.weight = min(2.0, self.weight + learning_rate)
                else:
                    # Weaken if uncorrelated
                    self.weight = max(0.1, self.weight - learning_rate * 0.1)
        
        elif self.plasticity_rule == "stdp":  # Spike-timing dependent plasticity
            if self.last_pre_spike > 0 and self.last_post_spike > 0:
                dt = self.last_post_spike - self.last_pre_spike
                if -20 < dt < 20:  # 20ms window
                    if dt > 0:  # Post after pre - potentiation
                        self.weight = min(2.0, self.weight + learning_rate * np.exp(-dt/10))
                    else:  # Pre after post - depression
                        self.weight = max(0.1, self.weight - learning_rate * np.exp(dt/10))

class HippocampalMemorySystem:
    """Simplified hippocampal memory system with realistic neural dynamics."""
    
    def __init__(self):
        # Neural clusters representing hippocampal regions
        self.ca3_cluster = NeuralCluster("CA3", synchronization_frequency=40)  # Pattern completion
        self.ca1_cluster = NeuralCluster("CA1", synchronization_frequency=8)   # Output to cortex
        self.dg_cluster = NeuralCluster("DG", synchronization_frequency=60)    # Pattern separation
        
        # Time and oscillation tracking
        self.global_time = 0.0
        self.current_oscillation = NeuralOscillation.BETA
        
        # Memory consolidation queue
        self.consolidation_queue = deque()
        
        # Pattern separation and completion
        self.pattern_storage: Dict[str, np.ndarray] = {}
        self.pattern_associations: Dict[str, List[str]] = defaultdict(list)
        
    def create_memory_engram(self, content: Any, importance: float = 1.0) -> str:
        """Create a distributed memory representation (engram)."""
        # Generate unique pattern ID
        pattern_id = f"engram_{len(self.pattern_storage)}"
        
        # Create sparse distributed representation
        pattern_size = 1000
        sparsity = 0.05  # 5% of neurons active
        active_neurons = int(pattern_size * sparsity)
        
        # Create pattern with some neurons more strongly activated
        pattern = np.zeros(pattern_size)
        active_indices = np.random.choice(pattern_size, active_neurons, replace=False)
        pattern[active_indices] = np.random.exponential(importance, active_neurons)
        
        # Normalize pattern
        pattern = pattern / np.max(pattern) if np.max(pattern) > 0 else pattern
        
        self.pattern_storage[pattern_id] = pattern
        
        # Create neural representation in hippocampal regions
        self._encode_in_hippocampus(pattern_id, pattern, importance)
        
        return pattern_id
    
    def _encode_in_hippocampus(self, pattern_id: str, pattern: np.ndarray, importance: float):
        """Encode pattern in hippocampal neural clusters."""
        # Dentate Gyrus - Pattern separation
        dg_neurons = min(50, int(len(pattern) * 0.05))  # Sparse coding
        for i in range(dg_neurons):
            neuron_id = f"DG_{pattern_id}_{i}"
            neuron = ArtificialNeuron(
                neuron_id=neuron_id,
                neuron_type="granule",
                memory_trace_strength=importance * np.random.random()
            )
            self.dg_cluster.add_neuron(neuron)
        
        # CA3 - Recurrent network for pattern completion
        ca3_neurons = min(30, int(len(pattern) * 0.03))
        for i in range(ca3_neurons):
            neuron_id = f"CA3_{pattern_id}_{i}"
            neuron = ArtificialNeuron(
                neuron_id=neuron_id,
                neuron_type="pyramidal",
                memory_trace_strength=importance * np.random.random()
            )
            self.ca3_cluster.add_neuron(neuron)
        
        # CA1 - Output to neocortex
        ca1_neurons = min(40, int(len(pattern) * 0.04))
        for i in range(ca1_neurons):
            neuron_id = f"CA1_{pattern_id}_{i}"
            neuron = ArtificialNeuron(
                neuron_id=neuron_id,
                neuron_type="pyramidal",
                memory_trace_strength=importance * np.random.random()
            )
            self.ca1_cluster.add_neuron(neuron)
        
        # Create synaptic connections
        self._create_hippocampal_connections(pattern_id)
    
    def _create_hippocampal_connections(self, pattern_id: str):
        """Create realistic hippocampal connectivity patterns."""
        # DG -> CA3 (mossy fibers)
        dg_neurons = [n for n in self.dg_cluster.neurons.values() 
                     if pattern_id in n.neuron_id]
        ca3_neurons = [n for n in self.ca3_cluster.neurons.values() 
                      if pattern_id in n.neuron_id]
        
        for dg_neuron in dg_neurons:
            # Each DG neuron connects to few CA3 neurons (sparse but strong)
            target_ca3 = np.random.choice(ca3_neurons, size=min(3, len(ca3_neurons)), replace=False)
            for ca3_neuron in target_ca3:
                synapse = Synapse(
                    synapse_id=f"{dg_neuron.neuron_id}_to_{ca3_neuron.neuron_id}",
                    presynaptic_neuron=dg_neuron.neuron_id,
                    postsynaptic_neuron=ca3_neuron.neuron_id,
                    weight=2.0,  # Strong connection
                    plasticity_rule="hebbian"
                )
                dg_neuron.outgoing_synapses[synapse.synapse_id] = synapse
                ca3_neuron.incoming_synapses[synapse.synapse_id] = synapse
        
        # CA3 -> CA1 (Schaffer collaterals)
        ca1_neurons = [n for n in self.ca1_cluster.neurons.values() 
                      if pattern_id in n.neuron_id]
        
        for ca3_neuron in ca3_neurons:
            # Each CA3 neuron connects to many CA1 neurons
            target_ca1 = np.random.choice(ca1_neurons, 
                                        size=min(len(ca1_neurons), int(len(ca1_neurons) * 0.7)), 
                                        replace=False)
            for ca1_neuron in target_ca1:
                synapse = Synapse(
                    synapse_id=f"{ca3_neuron.neuron_id}_to_{ca1_neuron.neuron_id}",
                    presynaptic_neuron=ca3_neuron.neuron_id,
                    postsynaptic_neuron=ca1_neuron.neuron_id,
                    weight=1.0,
                    plasticity_rule="stdp"
                )
                ca3_neuron.outgoing_synapses[synapse.synapse_id] = synapse
                ca1_neuron.incoming_synapses[synapse.synapse_id] = synapse
        
        # CA3 -> CA3 recurrent connections for pattern completion
        for i, ca3_source in enumerate(ca3_neurons):
            for j, ca3_target in enumerate(ca3_neurons):
                if i != j and np.random.random() < 0.3:  # 30% connectivity
                    synapse = Synapse(
                        synapse_id=f"{ca3_source.neuron_id}_to_{ca3_target.neuron_id}",
                        presynaptic_neuron=ca3_source.neuron_id,
                        postsynaptic_neuron=ca3_target.neuron_id,
                        weight=0.5,  # Moderate recurrent weight
                        plasticity_rule="hebbian"
                    )
                    ca3_source.outgoing_synapses[synapse.synapse_id] = synapse
                    ca3_target.incoming_synapses[synapse.synapse_id] = synapse
    
    def recall_pattern(self, cue_pattern: np.ndarray, confidence_threshold: float = 0.3) -> List[Tuple[str, float]]:
        """Recall patterns based on partial cue (pattern completion)."""
        matches = []
        
        for pattern_id, stored_pattern in self.pattern_storage.items():
            # Compute similarity (cosine similarity)
            dot_product = np.dot(cue_pattern, stored_pattern)
            norm_product = np.linalg.norm(cue_pattern) * np.linalg.norm(stored_pattern)
            
            if norm_product > 0:
                similarity = dot_product / norm_product
                
                # Apply neural dynamics for pattern completion
                completion_strength = self._simulate_pattern_completion(pattern_id, similarity)
                
                if completion_strength >= confidence_threshold:
                    matches.append((pattern_id, completion_strength))
        
        # Sort by completion strength
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches
    
    def _simulate_pattern_completion(self, pattern_id: str, initial_similarity: float) -> float:
        """Simulate neural dynamics for pattern completion in CA3."""
        ca3_neurons = [n for n in self.ca3_cluster.neurons.values() 
                      if pattern_id in n.neuron_id]
        
        if not ca3_neurons:
            return initial_similarity
        
        # Initialize neuron activations based on similarity
        for neuron in ca3_neurons:
            neuron.current_activation = initial_similarity * neuron.memory_trace_strength
        
        # Run recurrent dynamics for several time steps
        for step in range(10):  # 10 time steps
            new_activations = {}
            
            for neuron in ca3_neurons:
                input_current = 0.0
                
                # Sum inputs from recurrent connections
                for synapse in neuron.incoming_synapses.values():
                    if synapse.presynaptic_neuron in [n.neuron_id for n in ca3_neurons]:
                        presynaptic = next(n for n in ca3_neurons 
                                         if n.neuron_id == synapse.presynaptic_neuron)
                        input_current += synapse.weight * presynaptic.current_activation
                
                # Apply sigmoid activation
                new_activation = 1 / (1 + np.exp(-(input_current - 2.0)))
                new_activations[neuron.neuron_id] = new_activation
            
            # Update activations
            for neuron in ca3_neurons:
                neuron.current_activation = new_activations[neuron.neuron_id]
        
        # Compute final completion strength
        final_activations = [n.current_activation for n in ca3_neurons]
        return np.mean(final_activations) if final_activations else 0.0
    
    def simulate_sleep_consolidation(self, duration_minutes: float = 60):
        """Simulate sleep-based memory consolidation."""
        print(f"🌙 Simulating {duration_minutes} minutes of sleep consolidation...")
        
        # Switch to theta oscillations for consolidation
        original_oscillation = self.current_oscillation
        self.current_oscillation = NeuralOscillation.THETA
        
        # Identify memories for consolidation
        consolidation_candidates = []
        
        for cluster in [self.ca3_cluster, self.ca1_cluster]:
            for neuron in cluster.neurons.values():
                if neuron.memory_trace_strength > 0.5:  # Strong memories
                    consolidation_candidates.append(neuron)
        
        # Simulate consolidation process
        for minute in range(int(duration_minutes)):
            # Update global time
            self.global_time += 60  # 60 seconds per minute
            
            # Synchronize oscillations
            for cluster in [self.ca3_cluster, self.ca1_cluster, self.dg_cluster]:
                cluster.synchronize_oscillation(self.global_time)
            
            # Strengthen synapses of important memories
            for neuron in consolidation_candidates:
                # Strengthen outgoing synapses
                for synapse in neuron.outgoing_synapses.values():
                    synapse.update_plasticity(learning_rate=0.02)
                
                # Increase consolidation factor
                neuron.consolidation_factor = min(1.0, neuron.consolidation_factor + 0.01)
        
        # Restore original oscillation
        self.current_oscillation = original_oscillation
        
        print(f"✅ Consolidation complete. Strengthened {len(consolidation_candidates)} memory engrams.")
    
    def get_neural_statistics(self) -> Dict:
        """Get comprehensive statistics about the neural system."""
        stats = {}
        
        for cluster_name, cluster in [
            ("CA3", self.ca3_cluster), 
            ("CA1", self.ca1_cluster), 
            ("DG", self.dg_cluster)
        ]:
            neuron_count = len(cluster.neurons)
            avg_activation = np.mean([n.current_activation for n in cluster.neurons.values()]) if neuron_count > 0 else 0
            avg_memory_strength = np.mean([n.memory_trace_strength for n in cluster.neurons.values()]) if neuron_count > 0 else 0
            
            synapse_count = sum(len(n.outgoing_synapses) for n in cluster.neurons.values())
            avg_synapse_weight = np.mean([
                s.weight for n in cluster.neurons.values() 
                for s in n.outgoing_synapses.values()
            ]) if synapse_count > 0 else 0
            
            stats[cluster_name] = {
                'neuron_count': neuron_count,
                'synapse_count': synapse_count,
                'avg_activation': avg_activation,
                'avg_memory_strength': avg_memory_strength,
                'avg_synapse_weight': avg_synapse_weight,
                'cluster_activation': cluster.compute_cluster_activation()
            }
        
        stats['total_patterns'] = len(self.pattern_storage)
        stats['current_oscillation'] = self.current_oscillation.value
        
        return stats

def demo_hardware_inspired_memory():
    """Demonstrate the hardware-inspired memory system."""
    print("⚡ Hardware-Inspired Memory Architecture Demo")
    print("=" * 60)
    
    # Create hippocampal memory system
    hippocampus = HippocampalMemorySystem()
    
    print("1. Encoding memories as neural engrams...")
    
    # Encode different types of memories
    memories = [
        ("I love Mexican tacos with cilantro", 0.9),
        ("Neural networks use backpropagation", 0.8),
        ("Python is a programming language", 0.7),
        ("Tacos originated in Mexico", 0.85),
        ("Deep learning requires large datasets", 0.75)
    ]
    
    pattern_ids = []
    for content, importance in memories:
        # Create a simple pattern representation
        content_hash = hash(content) % 1000
        pattern = np.zeros(1000)
        # Set some neurons based on content hash
        for i in range(0, 50):  # 50 active neurons
            idx = (content_hash + i * 17) % 1000  # Pseudo-random but deterministic
            pattern[idx] = np.random.exponential(importance)
        
        pattern_id = hippocampus.create_memory_engram(content, importance)
        pattern_ids.append((pattern_id, content, pattern))
        print(f"   Encoded: {content} (importance: {importance})")
    
    print(f"\n2. Neural system statistics after encoding:")
    stats = hippocampus.get_neural_statistics()
    for region, data in stats.items():
        if isinstance(data, dict) and 'neuron_count' in data:
            print(f"   {region}: {data['neuron_count']} neurons, "
                  f"{data['synapse_count']} synapses, "
                  f"activation: {data['avg_activation']:.2f}")
    
    print(f"\n3. Testing pattern completion (recall from partial cues)...")
    
    # Test recall with partial cues
    test_cues = [
        "tacos Mexico",
        "neural networks", 
        "Python programming"
    ]
    
    for cue in test_cues:
        print(f"\n   Cue: '{cue}'")
        
        # Create cue pattern
        cue_hash = hash(cue) % 1000
        cue_pattern = np.zeros(1000)
        for i in range(0, 20):  # Fewer active neurons for partial cue
            idx = (cue_hash + i * 17) % 1000
            cue_pattern[idx] = 0.5
        
        # Recall patterns
        matches = hippocampus.recall_pattern(cue_pattern, confidence_threshold=0.2)
        
        for pattern_id, strength in matches[:3]:  # Top 3 matches
            # Find original content
            original_content = next(content for pid, content, _ in pattern_ids 
                                  if pid == pattern_id)
            print(f"     Recalled: {original_content} (strength: {strength:.3f})")
    
    print(f"\n4. Simulating sleep-based memory consolidation...")
    hippocampus.simulate_sleep_consolidation(duration_minutes=30)
    
    print(f"\n5. Post-consolidation statistics:")
    final_stats = hippocampus.get_neural_statistics()
    for region, data in final_stats.items():
        if isinstance(data, dict) and 'avg_synapse_weight' in data:
            print(f"   {region}: avg synapse weight: {data['avg_synapse_weight']:.3f}, "
                  f"avg memory strength: {data['avg_memory_strength']:.3f}")
    
    print(f"\n6. Testing recall after consolidation...")
    
    # Test the same cues after consolidation
    cue = "tacos Mexico"
    cue_hash = hash(cue) % 1000
    cue_pattern = np.zeros(1000)
    for i in range(0, 20):
        idx = (cue_hash + i * 17) % 1000
        cue_pattern[idx] = 0.5
    
    post_consolidation_matches = hippocampus.recall_pattern(cue_pattern, confidence_threshold=0.2)
    print(f"   Post-consolidation recall for '{cue}':")
    for pattern_id, strength in post_consolidation_matches[:3]:
        original_content = next(content for pid, content, _ in pattern_ids 
                              if pid == pattern_id)
        print(f"     {original_content} (strength: {strength:.3f})")
    
    print(f"\n✅ Hardware-inspired memory demo completed!")
    print(f"   Total patterns stored: {final_stats['total_patterns']}")
    print(f"   Current brain state: {final_stats['current_oscillation']} waves")

if __name__ == "__main__":
    demo_hardware_inspired_memory()
