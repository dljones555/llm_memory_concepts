"""
Real-time Conversation Tracker using MemoryStreamAgent

This module provides a practical interface for t        # Update session info
        session = self.active_sessions[conversation_id]
        session["message_count"] += 1
        if result["topic"] not in session["topics_discussed"]:
            session["topics_discussed"].append(result["topic"])
        session["last_activity"] = datetime.datetime.now().isoformat()
        
        print(f"🤖 Assistant [{result['topic']}]: {message}")
        print(f"   Keywords: {', '.join(result['keywords'][:5])}")  # Show first 5 keywordslive conversations
between users and AI assistants, storing both sides of the dialogue and
providing intelligent memory-based context retrieval.
"""

import datetime
import json
from typing import Dict, List, Optional, Tuple
from memory_stream_agent import MemoryStreamAgent


class ConversationTracker:
    """
    Real-time conversation tracker that monitors both user and assistant messages,
    providing memory-based context and conversation analytics.
    """
    
    def __init__(self, agent: MemoryStreamAgent = None, save_file: str = "conversations.json"):
        """
        Initialize the conversation tracker.
        
        Args:
            agent: Optional MemoryStreamAgent instance. Creates new one if None.
            save_file: File to persist conversation history
        """
        self.agent = agent or MemoryStreamAgent()
        self.save_file = save_file
        self.active_sessions: Dict[str, Dict] = {}
        self.load_conversations()
    
    def start_conversation(self, user_id: str, session_name: str = None) -> str:
        """
        Start a new conversation session.
        
        Args:
            user_id: Unique identifier for the user
            session_name: Optional descriptive name for the session
            
        Returns:
            conversation_id: Unique identifier for this conversation
        """
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        conversation_id = f"{user_id}_{timestamp}"
        
        session_info = {
            "user_id": user_id,
            "session_name": session_name or f"Chat_{timestamp}",
            "start_time": datetime.datetime.now().isoformat(),
            "message_count": 0,
            "topics_discussed": [],  # Use list instead of set for JSON serialization
            "active": True
        }
        
        self.active_sessions[conversation_id] = session_info
        print(f"🚀 Started conversation: {conversation_id}")
        if session_name:
            print(f"   Session: {session_name}")
        
        return conversation_id
    
    def add_user_message(self, conversation_id: str, message: str) -> Dict:
        """
        Add a user message to the conversation.
        
        Args:
            conversation_id: The conversation to add to
            message: The user's message
            
        Returns:
            Analysis results including topic and keywords
        """
        if conversation_id not in self.active_sessions:
            raise ValueError(f"Conversation {conversation_id} not found. Start conversation first.")
        
        # Add message with user prefix for clarity
        formatted_message = f"User: {message}"
        result = self.agent.add_message(formatted_message, conversation_id=conversation_id)
        
        # Update session info
        session = self.active_sessions[conversation_id]
        session["message_count"] += 1
        if result["topic"] not in session["topics_discussed"]:
            session["topics_discussed"].append(result["topic"])
        session["last_activity"] = datetime.datetime.now().isoformat()
        
        print(f"👤 User [{result['topic']}]: {message}")
        print(f"   Keywords: {', '.join(result['keywords'][:5])}")  # Show first 5 keywords
        
        return result
    
    def add_assistant_message(self, conversation_id: str, message: str) -> Dict:
        """
        Add an assistant message to the conversation.
        
        Args:
            conversation_id: The conversation to add to
            message: The assistant's response
            
        Returns:
            Analysis results including topic and keywords
        """
        if conversation_id not in self.active_sessions:
            raise ValueError(f"Conversation {conversation_id} not found. Start conversation first.")
        
        # Add message with assistant prefix for clarity
        formatted_message = f"Assistant: {message}"
        result = self.agent.add_message(formatted_message, conversation_id=conversation_id)
        
        # Update session info
        session = self.active_sessions[conversation_id]
        session["message_count"] += 1
        if result["topic"] not in session["topics_discussed"]:
            session["topics_discussed"].append(result["topic"])
        session["last_activity"] = datetime.datetime.now().isoformat()
        
        print(f"🤖 Assistant [{result['topic']}]: {message}")
        print(f"   Keywords: {', '.join(result['keywords'][:5])}")
        
        return result
    
    def get_context_for_response(self, conversation_id: str, user_query: str, 
                                include_count: int = 3) -> List[Dict]:
        """
        Get relevant context from conversation history to help generate responses.
        
        Args:
            conversation_id: The conversation to search
            user_query: The current user query
            include_count: Number of relevant memories to return
            
        Returns:
            List of relevant memory items with context
        """
        # Get relevant context from the agent
        context = self.agent.get_relevant_context(user_query, min_score=0.1)
        
        # Filter to only include messages from this conversation and limit results
        conversation_context = [
            item for item in context 
            if conversation_id in str(item.get('content', ''))
        ][:include_count]
        
        if conversation_context:
            print(f"\n🔍 Relevant context for response (top {len(conversation_context)}):")
            for i, item in enumerate(conversation_context, 1):
                # Clean up the content display
                content = item['content'].replace('User: ', '').replace('Assistant: ', '')
                print(f"   {i}. [{item['topic']}] {content[:80]}... (score: {item['score']})")
        
        return conversation_context
    
    def check_topic_drift(self) -> Optional[str]:
        """Check if there's been topic drift in the conversation."""
        drift = self.agent.emit_topic_drift_event()
        if drift:
            print(f"\n🌊 {drift}")
        return drift
    
    def get_conversation_insights(self, conversation_id: str) -> Dict:
        """
        Get detailed insights about a conversation.
        
        Args:
            conversation_id: The conversation to analyze
            
        Returns:
            Comprehensive conversation analysis
        """
        if conversation_id not in self.active_sessions:
            return {"error": "Conversation not found"}
        
        # Get basic summary from agent
        summary = self.agent.get_conversation_summary(conversation_id)
        
        # Add session-specific insights
        session = self.active_sessions[conversation_id]
        
        insights = {
            **summary,
            "session_name": session.get("session_name"),
            "user_id": session.get("user_id"), 
            "duration_minutes": self._calculate_duration(session),
            "message_count": summary.get("total_messages", 0),  # Add this for compatibility
            "topics_count": len(session["topics_discussed"]),
            "messages_per_topic": summary.get("total_messages", 0) / max(len(session["topics_discussed"]), 1),
            "is_active": session.get("active", False)
        }
        
        return insights
    
    def generate_ai_summary(self, conversation_id: str, model: str = "gpt-4o-mini") -> str:
        """
        Generate an AI summary of the conversation.
        
        Args:
            conversation_id: The conversation to summarize
            model: The model to use for summarization
            
        Returns:
            AI-generated summary of the conversation
        """
        summary = self.agent.create_openai_summary(conversation_id, model=model)
        
        print(f"\n📝 AI Summary of {conversation_id}:")
        print(f"   {summary}")
        
        return summary
    
    def end_conversation(self, conversation_id: str) -> Dict:
        """
        End a conversation and generate final analytics.
        
        Args:
            conversation_id: The conversation to end
            
        Returns:
            Final conversation analytics
        """
        if conversation_id not in self.active_sessions:
            return {"error": "Conversation not found"}
        
        session = self.active_sessions[conversation_id]
        session["active"] = False
        session["end_time"] = datetime.datetime.now().isoformat()
        
        # Generate final insights
        insights = self.get_conversation_insights(conversation_id)
        
        print(f"\n🏁 Conversation ended: {conversation_id}")
        print(f"   Duration: {insights.get('duration_minutes', 0):.1f} minutes")
        print(f"   Messages: {insights.get('total_messages', 0)}")
        print(f"   Topics: {insights.get('topics_count', 0)}")
        
        # Save conversation history
        self.save_conversations()
        
        return insights
    
    def list_active_conversations(self) -> List[Dict]:
        """List all currently active conversations."""
        active = [
            {
                "conversation_id": conv_id,
                "session_name": session.get("session_name"),
                "user_id": session.get("user_id"),
                "message_count": session.get("message_count", 0),
                "topics": len(session.get("topics_discussed", set())),
                "last_activity": session.get("last_activity")
            }
            for conv_id, session in self.active_sessions.items()
            if session.get("active", False)
        ]
        
        if active:
            print(f"\n📋 Active conversations ({len(active)}):")
            for conv in active:
                print(f"   • {conv['conversation_id']} - {conv['session_name']} "
                      f"({conv['message_count']} msgs, {conv['topics']} topics)")
        
        return active
    
    def _calculate_duration(self, session: Dict) -> float:
        """Calculate conversation duration in minutes."""
        try:
            start = datetime.datetime.fromisoformat(session["start_time"])
            end_time = session.get("end_time") or session.get("last_activity")
            
            if end_time:
                end = datetime.datetime.fromisoformat(end_time)
            else:
                end = datetime.datetime.now()
            
            return (end - start).total_seconds() / 60
        except:
            return 0.0
    
    def save_conversations(self):
        """Save conversation history to file."""
        try:
            # Convert sets to lists for JSON serialization
            serializable_sessions = {}
            for conv_id, session in self.active_sessions.items():
                session_copy = session.copy()
                if "topics_discussed" in session_copy:
                    session_copy["topics_discussed"] = list(session_copy["topics_discussed"])
                serializable_sessions[conv_id] = session_copy
            
            with open(self.save_file, 'w') as f:
                json.dump(serializable_sessions, f, indent=2)
        except Exception as e:
            print(f"⚠️ Could not save conversations: {e}")
    
    def load_conversations(self):
        """Load conversation history from file."""
        try:
            with open(self.save_file, 'r') as f:
                loaded_sessions = json.load(f)
                
            # Convert lists back to sets
            for conv_id, session in loaded_sessions.items():
                if "topics_discussed" in session:
                    session["topics_discussed"] = set(session["topics_discussed"])
                self.active_sessions[conv_id] = session
                
            print(f"📂 Loaded {len(self.active_sessions)} conversation sessions")
        except FileNotFoundError:
            print("📂 No previous conversations found - starting fresh")
        except Exception as e:
            print(f"⚠️ Could not load conversations: {e}")


# Demo usage
if __name__ == "__main__":
    # Initialize the tracker
    tracker = ConversationTracker()
    
    # Start a conversation
    conv_id = tracker.start_conversation("user123", "AI Research Discussion")
    
    # Simulate a realistic conversation
    conversation_flow = [
        ("user", "I'm interested in learning about large language models and their memory capabilities"),
        ("assistant", "LLMs have fascinating memory architectures! They use attention mechanisms to remember context within their training window. However, they lack persistent memory between conversations."),
        ("user", "That's interesting. How do researchers approach giving LLMs better memory?"),
        ("assistant", "There are several approaches: retrieval-augmented generation (RAG), external memory stores, and episodic memory systems. Some systems maintain conversation history in vector databases."),
        ("user", "Can you tell me more about vector databases for this purpose?"),
        ("assistant", "Vector databases store embeddings of text chunks, allowing semantic similarity search. When you ask a question, the system finds relevant past conversations and includes them as context."),
        ("user", "What about the challenges with this approach?"),
        ("assistant", "Main challenges include: determining relevance, managing context window limits, handling temporal decay of information, and maintaining conversation coherence across topics.")
    ]
    
    print("\n" + "="*60)
    print("🎭 SIMULATING REALISTIC CONVERSATION")
    print("="*60)
    
    for speaker, message in conversation_flow:
        if speaker == "user":
            tracker.add_user_message(conv_id, message)
        else:
            tracker.add_assistant_message(conv_id, message)
        
        # Check for topic drift periodically
        if len([m for m in conversation_flow[:conversation_flow.index((speaker, message))+1]]) % 3 == 0:
            tracker.check_topic_drift()
        
        print()  # Add spacing between messages
    
    # Demonstrate context retrieval
    print("\n" + "="*60)
    print("🔍 CONTEXT RETRIEVAL DEMO")
    print("="*60)
    
    # User asks a follow-up question
    follow_up = "How does temporal decay work in memory systems?"
    print(f"👤 User: {follow_up}")
    
    # Get context for generating a response
    context = tracker.get_context_for_response(conv_id, follow_up)
    
    # Generate AI summary
    print("\n" + "="*60)
    print("📊 CONVERSATION ANALYTICS")
    print("="*60)
    
    insights = tracker.get_conversation_insights(conv_id)
    print("\n📈 Conversation Insights:")
    for key, value in insights.items():
        if key not in ['start_time', 'end_time', 'keywords']:  # Skip verbose fields
            print(f"   {key}: {value}")
    
    # Generate AI summary
    tracker.generate_ai_summary(conv_id)
    
    # List active conversations
    tracker.list_active_conversations()
    
    # End the conversation
    final_insights = tracker.end_conversation(conv_id)
    
    print(f"\n✨ Demo completed! Check '{tracker.save_file}' for saved conversation data.")
