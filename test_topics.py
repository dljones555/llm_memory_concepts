#!/usr/bin/env python3
"""
🧪 Test improved topic detection
"""

from memory_stream_agent import MemoryStreamAgent

def test_topic_detection():
    print("🧪 Testing LLM-Powered Topic Detection")
    print("=" * 40)
    
    agent = MemoryStreamAgent()
    
    # Test messages with clear topics
    test_messages = [
        "I love eating tacos, especially al pastor and carnitas from Mexico",
        "Neural networks and transformers are fascinating AI architectures", 
        "Planning a trip to Japan next month, need to book flights",
        "The weather is terrible today, lots of rain and wind",
        "Working on a Python project with machine learning models"
    ]
    
    print("🔍 Testing topic detection...")
    for i, msg in enumerate(test_messages, 1):
        result = agent.add_message(msg, f"test_conv_{i}")
        print(f"\n{i}. Message: \"{msg}\"")
        print(f"   Topic: {result['topic']}")
        print(f"   Keywords: {result['keywords'][:5]}")
    
    print(f"\n📊 Generated {len(agent.topic_clusters)} topics:")
    for topic, keywords in agent.topic_clusters.items():
        print(f"  - {topic}: {keywords[:5]}...")

if __name__ == "__main__":
    test_topic_detection()
