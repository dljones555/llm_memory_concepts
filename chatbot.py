#!/usr/bin/env python3
"""
🤖 Memory-Enhanced Chatbot
A simple chatbot that uses the MemoryStreamAgent for context-aware conversations.
"""

import os
import sys
import datetime
from typing import List, Dict, Optional
from conversation_tracker import ConversationTracker
import requests
import json

class MemoryChatBot:
    def __init__(self):
        """Initialize the chatbot with memory tracking capabilities."""
        self.tracker = ConversationTracker()
        self.current_conversation_id = None
        self.user_id = None
        self.github_token = os.getenv('GH_TOKEN')
        
        if not self.github_token:
            print("⚠️  Warning: No GitHub token found. AI responses will be limited.")
            print("   Set GH_TOKEN in your .env file for full functionality.")

    def start_chat_session(self, user_id: str = None, session_name: str = None) -> str:
        """Start a new chat session."""
        self.user_id = user_id or input("👤 Enter your user ID: ").strip()
        
        if not session_name:
            default_name = f"Chat Session {datetime.datetime.now().strftime('%H:%M')}"
            session_name = input(f"💬 Session name (default: {default_name}): ").strip()
            if not session_name:
                session_name = default_name
        
        self.current_conversation_id = self.tracker.start_conversation(self.user_id, session_name)
        
        print(f"\n🚀 Started new chat session: {session_name}")
        print(f"📝 Session ID: {self.current_conversation_id}")
        print("💡 Type 'help' for commands, 'quit' to exit")
        print("-" * 50)
        
        return self.current_conversation_id

    def generate_ai_response(self, user_message: str, context: List[Dict] = None) -> str:
        """Generate an AI response using GitHub Models API."""
        if not self.github_token:
            return "I don't have access to AI models right now. Please set up your GitHub token!"
        
        # Build context from memory
        context_text = ""
        if context:
            context_text = "\n\nRelevant conversation history:\n"
            for item in context[:3]:  # Use top 3 most relevant
                context_text += f"- {item['content']}\n"
        
        # Create the prompt
        system_prompt = f"""You are a helpful AI assistant with memory capabilities. 
You can remember previous conversations and provide contextual responses.
Current time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Keep responses conversational, helpful, and refer to previous context when relevant.{context_text}"""

        user_prompt = f"User message: {user_message}"
        
        # GitHub Models API call
        url = "https://models.inference.ai.azure.com/chat/completions"
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.github_token}"
        }
        
        payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "max_tokens": 500,
            "temperature": 0.7
        }
        
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            result = response.json()
            return result['choices'][0]['message']['content']
        except requests.exceptions.Timeout:
            return self.generate_fallback_response(user_message, context)
        except requests.exceptions.RequestException as e:
            return self.generate_fallback_response(user_message, context)
        except (KeyError, IndexError) as e:
            return f"I got a confusing response from my AI brain. Error: {str(e)}"

    def generate_fallback_response(self, user_message: str, context: List[Dict] = None) -> str:
        """Generate a simple fallback response when AI is unavailable."""
        responses = [
            "That's interesting! I'd love to discuss that more when my AI connection is working better.",
            "I hear you! Let me think about that... (My AI brain is having connectivity issues right now)",
            "Good point! I'm having some technical difficulties, but I'm following our conversation.",
            "Thanks for sharing that! I'm experiencing some connection issues but I'm still here listening.",
            "I appreciate you telling me that! My AI responses are limited right now, but I'm tracking our conversation."
        ]
        
        # Simple keyword-based responses
        user_lower = user_message.lower()
        if any(word in user_lower for word in ['hello', 'hi', 'hey']):
            return "Hello! Nice to chat with you. (I'm having some AI connectivity issues, but I'm here!)"
        elif any(word in user_lower for word in ['thank', 'thanks']):
            return "You're welcome! (My AI brain is a bit slow today, but I appreciate the conversation!)"
        elif '?' in user_message:
            return "That's a great question! I wish I could give you a better answer right now - my AI connection is having issues."
        else:
            import random
            return random.choice(responses)

    def process_user_input(self, user_input: str) -> bool:
        """Process user input and generate appropriate response. Returns False if user wants to quit."""
        user_input = user_input.strip()
        
        # Handle special commands
        if user_input.lower() in ['quit', 'exit', 'bye']:
            return False
        elif user_input.lower() == 'help':
            self.show_help()
            return True
        elif user_input.lower() == 'summary':
            self.show_conversation_summary()
            return True
        elif user_input.lower() == 'topics':
            self.show_topics()
            return True
        elif user_input.lower() == 'stats':
            self.show_stats()
            return True
        elif user_input.lower() == 'clear':
            self.clear_screen()
            return True
        elif user_input.lower().startswith('save'):
            self.save_conversation()
            return True
        
        # Regular conversation
        if not user_input:
            print("💭 Silent treatment? Say something!")
            return True
        
        # Add user message to memory
        self.tracker.add_user_message(self.current_conversation_id, user_input)
        
        # Get relevant context
        context = self.tracker.get_context_for_response(self.current_conversation_id, user_input)
        
        print("🤖 Thinking...")
        
        # Generate AI response
        ai_response = self.generate_ai_response(user_input, context)
        
        # Add AI response to memory
        self.tracker.add_assistant_message(self.current_conversation_id, ai_response)
        
        # Display response with context info
        print(f"\n🤖 Assistant: {ai_response}")
        
        if context:
            print(f"\n💡 (Used {len(context)} relevant memories)")
        
        # Check for topic drift
        drift = self.tracker.agent.emit_topic_drift_event()
        if drift:
            print(f"\n🔄 {drift}")
        
        return True

    def show_help(self):
        """Display available commands."""
        print("\n🆘 Available Commands:")
        print("  help     - Show this help message")
        print("  summary  - Show conversation summary")
        print("  topics   - Show discussed topics")
        print("  stats    - Show conversation statistics")
        print("  clear    - Clear the screen")
        print("  save     - Save conversation")
        print("  quit     - Exit the chat")
        print("-" * 30)

    def show_conversation_summary(self):
        """Show a summary of the current conversation."""
        if not self.current_conversation_id:
            print("❌ No active conversation to summarize.")
            return
        
        insights = self.tracker.get_conversation_insights(self.current_conversation_id)
        
        if not insights or insights.get("error"):
            print("❌ Could not retrieve conversation summary.")
            print(f"Error: {insights.get('error', 'Unknown error') if insights else 'No data'}")
            return
        
        print(f"\n📊 Conversation Summary:")
        print(f"  Messages: {insights.get('message_count', insights.get('total_messages', 0))}")
        print(f"  Duration: {insights.get('duration_minutes', 0):.1f} minutes")
        print(f"  Topics: {', '.join(insights.get('topics_discussed', []))}")
        
        # Generate AI summary
        print("\n🤖 AI Summary:")
        try:
            summary = self.tracker.generate_ai_summary(self.current_conversation_id)
            print(f"  {summary}")
        except Exception as e:
            print(f"  ⚠️  Could not generate AI summary: {e}")

    def show_topics(self):
        """Show topics discussed in the conversation."""
        if not self.current_conversation_id:
            print("❌ No active conversation.")
            return
        
        # Get insights and handle potential missing data
        insights = self.tracker.get_conversation_insights(self.current_conversation_id)
        
        if not insights or insights.get("error"):
            print("❌ Could not retrieve topics.")
            return
        
        # Try multiple ways to get topics
        topics = (insights.get('topics_discussed') or 
                 insights.get('topics') or 
                 [])
        
        if not topics:
            # Fall back to getting topics from the memory agent directly
            conv_summary = self.tracker.agent.get_conversation_summary(self.current_conversation_id)
            topics = conv_summary.get('topics', [])
        
        if topics:
            print(f"\n🏷️  Topics Discussed ({len(topics)}):")
            for i, topic in enumerate(topics, 1):
                print(f"  {i}. {topic}")
        else:
            print("\n🏷️  No topics detected yet. Keep chatting!")

    def show_stats(self):
        """Show detailed conversation statistics."""
        if not self.current_conversation_id:
            print("❌ No active conversation.")
            return
        
        insights = self.tracker.get_conversation_insights(self.current_conversation_id)
        
        if not insights or insights.get("error"):
            print("❌ Could not retrieve conversation statistics.")
            print(f"Error: {insights.get('error', 'Unknown error') if insights else 'No data'}")
            return
        
        print(f"\n📈 Detailed Statistics:")
        print(f"  Session ID: {self.current_conversation_id}")
        print(f"  User ID: {insights.get('user_id', 'Unknown')}")
        print(f"  Session Name: {insights.get('session_name', 'Unknown')}")
        print(f"  Start Time: {insights.get('start_time', 'Unknown')}")
        print(f"  Duration: {insights.get('duration_minutes', 0):.2f} minutes")
        print(f"  Messages: {insights.get('message_count', insights.get('total_messages', 0))}")
        print(f"  Topics: {len(insights.get('topics_discussed', []))}")
        print(f"  Active: {insights.get('is_active', insights.get('active', False))}")

    def save_conversation(self):
        """Save the current conversation."""
        if self.current_conversation_id:
            self.tracker.end_conversation(self.current_conversation_id)
            print("💾 Conversation saved!")
        else:
            print("❌ No active conversation to save.")

    def clear_screen(self):
        """Clear the terminal screen."""
        os.system('cls' if os.name == 'nt' else 'clear')
        print("🤖 Memory-Enhanced Chatbot")
        print("=" * 30)

    def run(self):
        """Main chat loop."""
        print("🤖 Memory-Enhanced Chatbot")
        print("=" * 30)
        print("Welcome! This chatbot remembers our conversations.")
        
        # Start a new chat session
        self.start_chat_session()
        
        print("\n💬 Start chatting! (Type 'help' for commands)")
        
        try:
            while True:
                # Get user input
                user_input = input(f"\n👤 {self.user_id}: ").strip()
                
                # Process input
                should_continue = self.process_user_input(user_input)
                
                if not should_continue:
                    break
        
        except KeyboardInterrupt:
            print("\n\n⚡ Chat interrupted!")
        
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")
        
        finally:
            # End the conversation
            if self.current_conversation_id:
                self.tracker.end_conversation(self.current_conversation_id)
                print(f"\n👋 Chat session ended. Conversation saved!")
                
                # Show final summary
                insights = self.tracker.get_conversation_insights(self.current_conversation_id)
                print(f"📊 Final stats: {insights['message_count']} messages, {insights['duration_minutes']:.1f} minutes")
            
            print("🚪 Goodbye!")


def main():
    """Main entry point."""
    chatbot = MemoryChatBot()
    chatbot.run()


if __name__ == "__main__":
    main()
