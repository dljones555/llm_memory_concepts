#!/usr/bin/env python3
"""
🧪 Quick Test for Fixed Chatbot
"""

from chatbot import MemoryChatBot

def test_chatbot_fixes():
    print("🧪 Testing Chatbot Fixes")
    print("=" * 30)
    
    try:
        # Create chatbot
        bot = MemoryChatBot()
        
        # Start conversation
        conv_id = bot.start_chat_session("test_user", "Fix Test Session")
        
        # Test a few messages
        test_messages = [
            "I want to learn about Mexican tacos",
            "What about AI and neural networks?"
        ]
        
        for msg in test_messages:
            print(f"\n--- Testing: {msg} ---")
            bot.process_user_input(msg)
        
        # Test the topics command
        print("\n--- Testing 'topics' command ---")
        bot.show_topics()
        
        # Test summary
        print("\n--- Testing 'summary' command ---")
        bot.show_conversation_summary()
        
        print("\n✅ All tests passed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_chatbot_fixes()
