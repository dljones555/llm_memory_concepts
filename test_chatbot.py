#!/usr/bin/env python3
"""
🧪 Chatbot Test Demo
Shows how the memory-enhanced chatbot works
"""

from chatbot import MemoryChatBot
import time

def demo_conversation():
    """Demonstrate the chatbot capabilities."""
    print("🧪 Chatbot Demo")
    print("=" * 30)
    
    # Create chatbot instance
    bot = MemoryChatBot()
    
    # Start a demo conversation
    conv_id = bot.start_chat_session("demo_user", "Chatbot Demo Session")
    
    # Simulate a conversation
    demo_messages = [
        "Hello! I'm interested in learning about neural networks.",
        "What are the main types of neural networks?",
        "Can you explain how memory works in AI systems?",
        "That's fascinating! How does this relate to human memory?"
    ]
    
    print("\n🎭 Simulating conversation...")
    
    for i, message in enumerate(demo_messages, 1):
        print(f"\n--- Message {i} ---")
        print(f"👤 User: {message}")
        
        # Add user message
        bot.tracker.add_user_message(conv_id, message)
        
        # Get context
        context = bot.tracker.get_context_for_response(conv_id, message)
        
        # Generate response
        response = bot.generate_ai_response(message, context)
        
        # Add assistant response
        bot.tracker.add_assistant_message(conv_id, response)
        
        print(f"🤖 Assistant: {response}")
        
        if context:
            print(f"💡 Context used: {len(context)} relevant memories")
        
        time.sleep(1)  # Pause for readability
    
    # Show conversation insights
    print("\n📊 Conversation Analysis:")
    insights = bot.tracker.get_conversation_insights(conv_id)
    
    if insights and not insights.get("error"):
        print(f"  Duration: {insights.get('duration_minutes', 0):.1f} minutes")
        print(f"  Messages: {insights.get('message_count', insights.get('total_messages', 0))}")
        print(f"  Topics: {', '.join(insights.get('topics_discussed', []))}")
    else:
        print("  ⚠️  Could not retrieve conversation insights")
        print(f"  Error: {insights.get('error', 'Unknown error')}")
    
    # Generate summary
    print("\n🤖 AI-Generated Summary:")
    summary = bot.tracker.generate_ai_summary(conv_id)
    print(f"  {summary}")
    
    # End conversation
    bot.tracker.end_conversation(conv_id)
    print(f"\n✅ Demo completed! Conversation saved as: {conv_id}")

if __name__ == "__main__":
    demo_conversation()
