"""
🧠 Advanced Memory Architecture for LLM Systems
Next-generation memory constructs inspired by hardware and cognitive science
"""

import asyncio
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Union, Callable, Any
from collections import defaultdict, deque
import threading
import time
from enum import Enum
import weakref
import pickle
import hashlib
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

class MemoryType(Enum):
    """Different types of memory with distinct characteristics."""
    WORKING = "working"          # Fast, volatile, limited capacity
    EPISODIC = "episodic"        # Event-based, temporal sequences
    SEMANTIC = "semantic"        # Factual knowledge, associations
    PROCEDURAL = "procedural"    # Skills, patterns, how-to knowledge
    EMOTIONAL = "emotional"      # Affective associations and valence
    METACOGNITIVE = "metacognitive"  # Self-awareness, learning about learning

class AttentionMechanism(Enum):
    """Different attention patterns for memory retrieval."""
    FOCUSED = "focused"          # Single-point high precision
    DIFFUSE = "diffuse"          # Broad, associative scanning
    SELECTIVE = "selective"      # Filter-based attention
    DIVIDED = "divided"          # Multi-task attention splitting
    SUSTAINED = "sustained"      # Long-duration focus maintenance

@dataclass
class MemoryTrace:
    """Advanced memory representation with neural-inspired features."""
    content: Any
    embedding: Optional[np.ndarray] = None
    strength: float = 1.0
    access_frequency: int = 0
    last_accessed: float = field(default_factory=time.time)
    decay_rate: float = 0.1
    consolidation_level: float = 0.0
    memory_type: MemoryType = MemoryType.WORKING
    associations: Set[str] = field(default_factory=set)
    emotional_valence: float = 0.0
    confidence: float = 1.0
    source_reliability: float = 1.0
    interference_resistance: float = 0.5
    
    def __post_init__(self):
        """Generate unique identifier and compute initial embedding."""
        self.trace_id = hashlib.md5(str(self.content).encode()).hexdigest()[:16]
        if self.embedding is None:
            self.embedding = self._compute_embedding()
    
    def _compute_embedding(self) -> np.ndarray:
        """Compute a simple embedding for the content."""
        # In production, you'd use a proper embedding model
        content_str = str(self.content)
        # Simple hash-based embedding for demo
        hash_obj = hashlib.sha256(content_str.encode())
        hash_bytes = hash_obj.digest()
        return np.frombuffer(hash_bytes[:128], dtype=np.uint8).astype(np.float32) / 255.0
    
    def decay(self) -> float:
        """Apply memory decay based on time and access patterns."""
        current_time = time.time()
        time_delta = current_time - self.last_accessed
        
        # Forgetting curve with interference
        decay_factor = np.exp(-self.decay_rate * time_delta / 3600)  # Hours
        self.strength *= decay_factor * self.interference_resistance
        
        return self.strength
    
    def strengthen(self, amount: float = 0.1):
        """Strengthen memory through repetition or rehearsal."""
        self.strength = min(1.0, self.strength + amount)
        self.access_frequency += 1
        self.last_accessed = time.time()
        
        # Spacing effect - distributed practice strengthens more
        if self.access_frequency > 1:
            spacing_bonus = 1.0 / np.sqrt(self.access_frequency)
            self.consolidation_level = min(1.0, self.consolidation_level + spacing_bonus)

class HierarchicalMemory:
    """Multi-level memory hierarchy mimicking biological memory systems."""
    
    def __init__(self, capacity_limits: Dict[MemoryType, int] = None):
        self.capacity_limits = capacity_limits or {
            MemoryType.WORKING: 7,        # Miller's 7±2 rule
            MemoryType.EPISODIC: 10000,   # Large but finite
            MemoryType.SEMANTIC: 100000,  # Very large factual store
            MemoryType.PROCEDURAL: 1000,  # Skills and patterns
            MemoryType.EMOTIONAL: 5000,   # Emotionally significant events
            MemoryType.METACOGNITIVE: 500 # Self-knowledge
        }
        
        self.memory_stores: Dict[MemoryType, Dict[str, MemoryTrace]] = {
            mem_type: {} for mem_type in MemoryType
        }
        
        self.working_memory_queue = deque(maxlen=self.capacity_limits[MemoryType.WORKING])
        self.consolidation_queue = deque()
        self.attention_focus = set()
        
        # Background processes
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.running = True
        self.start_background_processes()
    
    def start_background_processes(self):
        """Start background memory maintenance processes."""
        self.executor.submit(self._memory_consolidation_process)
        self.executor.submit(self._decay_process)
        self.executor.submit(self._association_strengthening_process)
    
    def store_memory(self, content: Any, memory_type: MemoryType = MemoryType.WORKING, 
                    metadata: Dict = None) -> str:
        """Store a new memory with automatic type management."""
        trace = MemoryTrace(
            content=content,
            memory_type=memory_type,
            **(metadata or {})
        )
        
        # Check capacity and manage overflow
        store = self.memory_stores[memory_type]
        if len(store) >= self.capacity_limits[memory_type]:
            self._evict_weakest_memory(memory_type)
        
        store[trace.trace_id] = trace
        
        # Add to working memory if not already there
        if memory_type == MemoryType.WORKING:
            self.working_memory_queue.append(trace.trace_id)
        
        # Queue for potential consolidation
        if memory_type == MemoryType.WORKING and trace.strength > 0.7:
            self.consolidation_queue.append(trace.trace_id)
        
        return trace.trace_id
    
    def retrieve_memory(self, query: Any, attention_type: AttentionMechanism = AttentionMechanism.FOCUSED,
                       memory_types: List[MemoryType] = None, top_k: int = 5) -> List[MemoryTrace]:
        """Advanced memory retrieval with attention mechanisms."""
        if memory_types is None:
            memory_types = list(MemoryType)
        
        # Compute query embedding
        query_embedding = self._compute_query_embedding(query)
        
        candidates = []
        for mem_type in memory_types:
            for trace in self.memory_stores[mem_type].values():
                similarity = self._compute_similarity(query_embedding, trace.embedding)
                
                # Apply attention mechanism
                attention_weight = self._apply_attention(trace, attention_type)
                
                # Compute retrieval strength
                retrieval_strength = (
                    similarity * 0.4 +
                    trace.strength * 0.3 +
                    attention_weight * 0.2 +
                    trace.consolidation_level * 0.1
                )
                
                candidates.append((retrieval_strength, trace))
        
        # Sort and return top candidates
        candidates.sort(key=lambda x: x[0], reverse=True)
        retrieved = [trace for _, trace in candidates[:top_k]]
        
        # Strengthen retrieved memories
        for trace in retrieved:
            trace.strengthen(0.05)
        
        return retrieved
    
    def _compute_query_embedding(self, query: Any) -> np.ndarray:
        """Compute embedding for query (placeholder implementation)."""
        query_str = str(query)
        hash_obj = hashlib.sha256(query_str.encode())
        hash_bytes = hash_obj.digest()
        return np.frombuffer(hash_bytes[:128], dtype=np.uint8).astype(np.float32) / 255.0
    
    def _compute_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Compute cosine similarity between embeddings."""
        return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    
    def _apply_attention(self, trace: MemoryTrace, attention_type: AttentionMechanism) -> float:
        """Apply attention mechanism to modulate retrieval."""
        if attention_type == AttentionMechanism.FOCUSED:
            return 1.0 if trace.trace_id in self.attention_focus else 0.1
        elif attention_type == AttentionMechanism.DIFFUSE:
            return 0.5 + 0.5 * trace.emotional_valence
        elif attention_type == AttentionMechanism.SELECTIVE:
            return trace.confidence * trace.source_reliability
        elif attention_type == AttentionMechanism.DIVIDED:
            return 0.3 + 0.4 * trace.consolidation_level
        else:  # SUSTAINED
            return min(1.0, trace.access_frequency / 10.0)
    
    def _evict_weakest_memory(self, memory_type: MemoryType):
        """Remove the weakest memory when capacity is exceeded."""
        store = self.memory_stores[memory_type]
        if not store:
            return
        
        weakest_id = min(store.keys(), key=lambda tid: store[tid].strength)
        del store[weakest_id]
    
    def _memory_consolidation_process(self):
        """Background process for memory consolidation."""
        while self.running:
            try:
                if self.consolidation_queue:
                    trace_id = self.consolidation_queue.popleft()
                    self._consolidate_memory(trace_id)
                time.sleep(1)  # Check every second
            except Exception as e:
                print(f"Consolidation error: {e}")
    
    def _consolidate_memory(self, trace_id: str):
        """Move memory from working to long-term storage."""
        # Find trace in working memory
        working_store = self.memory_stores[MemoryType.WORKING]
        if trace_id not in working_store:
            return
        
        trace = working_store[trace_id]
        
        # Determine best long-term memory type
        target_type = self._determine_consolidation_target(trace)
        
        # Move to long-term memory
        del working_store[trace_id]
        trace.memory_type = target_type
        trace.consolidation_level = 1.0
        
        self.memory_stores[target_type][trace_id] = trace
    
    def _determine_consolidation_target(self, trace: MemoryTrace) -> MemoryType:
        """Determine which long-term memory type is most appropriate."""
        content_str = str(trace.content).lower()
        
        if any(word in content_str for word in ['how', 'process', 'step', 'method']):
            return MemoryType.PROCEDURAL
        elif any(word in content_str for word in ['fact', 'definition', 'concept']):
            return MemoryType.SEMANTIC
        elif trace.emotional_valence != 0.0:
            return MemoryType.EMOTIONAL
        else:
            return MemoryType.EPISODIC
    
    def _decay_process(self):
        """Background process for memory decay."""
        while self.running:
            try:
                for store in self.memory_stores.values():
                    for trace in list(store.values()):
                        if trace.decay() < 0.01:  # Remove very weak memories
                            del store[trace.trace_id]
                time.sleep(60)  # Run every minute
            except Exception as e:
                print(f"Decay error: {e}")
    
    def _association_strengthening_process(self):
        """Background process for strengthening memory associations."""
        while self.running:
            try:
                # Strengthen associations between recently accessed memories
                recent_traces = []
                for store in self.memory_stores.values():
                    for trace in store.values():
                        if time.time() - trace.last_accessed < 3600:  # Last hour
                            recent_traces.append(trace)
                
                # Create associations between co-occurring memories
                for i, trace1 in enumerate(recent_traces):
                    for trace2 in recent_traces[i+1:]:
                        similarity = self._compute_similarity(trace1.embedding, trace2.embedding)
                        if similarity > 0.7:
                            trace1.associations.add(trace2.trace_id)
                            trace2.associations.add(trace1.trace_id)
                
                time.sleep(300)  # Run every 5 minutes
            except Exception as e:
                print(f"Association error: {e}")
    
    def set_attention_focus(self, trace_ids: List[str]):
        """Set current attention focus for retrieval bias."""
        self.attention_focus = set(trace_ids)
    
    def get_memory_statistics(self) -> Dict:
        """Get comprehensive memory system statistics."""
        stats = {}
        for mem_type in MemoryType:
            store = self.memory_stores[mem_type]
            if store:
                strengths = [trace.strength for trace in store.values()]
                stats[mem_type.value] = {
                    'count': len(store),
                    'capacity': self.capacity_limits[mem_type],
                    'utilization': len(store) / self.capacity_limits[mem_type],
                    'avg_strength': np.mean(strengths),
                    'total_strength': np.sum(strengths)
                }
            else:
                stats[mem_type.value] = {'count': 0, 'capacity': self.capacity_limits[mem_type]}
        
        return stats
    
    def shutdown(self):
        """Clean shutdown of background processes."""
        self.running = False
        self.executor.shutdown(wait=True)

# Advanced query language for memory operations
class MemoryQuery:
    """Declarative query language for complex memory operations."""
    
    def __init__(self, memory_system: HierarchicalMemory):
        self.memory = memory_system
        self.filters = []
        self.sorters = []
        self.transformers = []
    
    def where(self, predicate: Callable[[MemoryTrace], bool]) -> 'MemoryQuery':
        """Add a filter predicate."""
        self.filters.append(predicate)
        return self
    
    def order_by(self, key_func: Callable[[MemoryTrace], float], reverse: bool = True) -> 'MemoryQuery':
        """Add a sorting criterion."""
        self.sorters.append((key_func, reverse))
        return self
    
    def transform(self, transformer: Callable[[MemoryTrace], Any]) -> 'MemoryQuery':
        """Add a transformation function."""
        self.transformers.append(transformer)
        return self
    
    def execute(self, limit: int = None) -> List[Any]:
        """Execute the query and return results."""
        # Collect all memories
        all_traces = []
        for store in self.memory.memory_stores.values():
            all_traces.extend(store.values())
        
        # Apply filters
        for filter_func in self.filters:
            all_traces = [t for t in all_traces if filter_func(t)]
        
        # Apply sorting
        for key_func, reverse in self.sorters:
            all_traces.sort(key=key_func, reverse=reverse)
        
        # Apply limit
        if limit:
            all_traces = all_traces[:limit]
        
        # Apply transformations
        results = all_traces
        for transformer in self.transformers:
            results = [transformer(item) for item in results]
        
        return results

def demo_advanced_memory():
    """Demonstrate the advanced memory system."""
    print("🧠 Advanced Memory Architecture Demo")
    print("=" * 50)
    
    # Create memory system
    memory = HierarchicalMemory()
    
    # Store various types of memories
    print("📥 Storing memories...")
    
    # Working memory
    memory.store_memory("Current task: analyzing tacos", MemoryType.WORKING)
    memory.store_memory("User asked about Mexican food", MemoryType.WORKING)
    
    # Semantic knowledge
    memory.store_memory("Tacos are a traditional Mexican dish", MemoryType.SEMANTIC)
    memory.store_memory("AI models use attention mechanisms", MemoryType.SEMANTIC)
    
    # Episodic memories
    memory.store_memory("Yesterday's conversation about neural networks", MemoryType.EPISODIC, 
                       {"emotional_valence": 0.8})
    
    # Procedural knowledge
    memory.store_memory("How to generate embeddings: 1) tokenize 2) encode 3) normalize", 
                       MemoryType.PROCEDURAL)
    
    time.sleep(2)  # Let background processes run
    
    # Demonstrate retrieval with different attention mechanisms
    print("\n🔍 Testing retrieval with different attention mechanisms...")
    
    query = "Tell me about tacos and Mexican cuisine"
    
    for attention_type in AttentionMechanism:
        results = memory.retrieve_memory(query, attention_type, top_k=3)
        print(f"\n{attention_type.value.upper()} attention:")
        for i, trace in enumerate(results, 1):
            print(f"  {i}. [{trace.memory_type.value}] {str(trace.content)[:60]}...")
            print(f"     Strength: {trace.strength:.2f}, Confidence: {trace.confidence:.2f}")
    
    # Demonstrate query language
    print("\n🔎 Advanced Query Language Demo:")
    
    query_engine = MemoryQuery(memory)
    
    # Find all high-strength semantic memories
    semantic_memories = (query_engine
                        .where(lambda t: t.memory_type == MemoryType.SEMANTIC)
                        .where(lambda t: t.strength > 0.5)
                        .order_by(lambda t: t.strength)
                        .execute())
    
    print(f"High-strength semantic memories: {len(semantic_memories)}")
    for trace in semantic_memories:
        print(f"  - {trace.content}")
    
    # Memory statistics
    print("\n📊 Memory System Statistics:")
    stats = memory.get_memory_statistics()
    for mem_type, data in stats.items():
        if data['count'] > 0:
            print(f"  {mem_type}: {data['count']}/{data['capacity']} "
                  f"({data['utilization']:.1%} utilized, avg strength: {data['avg_strength']:.2f})")
    
    # Cleanup
    memory.shutdown()
    print("\n✅ Demo completed!")

if __name__ == "__main__":
    demo_advanced_memory()
