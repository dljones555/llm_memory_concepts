#!/usr/bin/env python3
"""
🚀 Chatbot Launcher
Quick launcher for the Memory-Enhanced Chatbot
"""

from chatbot import MemoryChatBot

def main():
    print("🤖 Starting Memory-Enhanced Chatbot...")
    print("⚡ Loading memory systems...")
    
    try:
        bot = MemoryChatBot()
        bot.run()
    except Exception as e:
        print(f"❌ Failed to start chatbot: {e}")
        print("💡 Make sure you have set up your .env file with GH_TOKEN")

if __name__ == "__main__":
    main()
