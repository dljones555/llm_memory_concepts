import datetime
import json
from dataclasses import dataclass, field
from typing import List, Set, Dict, Tuple, Optional
from collections import defaultdict, Counter
import re
import os
import requests
try:
    from dotenv import load_dotenv
    load_dotenv()  # Load environment variables from .env file
except ImportError:
    pass  # dotenv not installed, will use system environment variables

@dataclass
class MemoryItem:
    content: str
    keywords: Set[str]
    timestamp: datetime.datetime
    access_count: int = 0
    relevance_score: float = 1.0
    topic_tags: Set[str] = field(default_factory=set)
    conversation_id: Optional[str] = None

    def age_in_hours(self) -> float:
        return (datetime.datetime.now() - self.timestamp).total_seconds() / 3600

class MemoryStreamAgent:
    def __init__(self, topic_config_file: str = "topic_keywords.json", github_token: str = None):
        self.memories: List[MemoryItem] = []
        self.topic_transitions: Dict[str, List[str]] = defaultdict(list)
        self.keyword_pairs: Dict[Tuple[str, str], int] = defaultdict(int)
        self.current_topic: Optional[str] = None
        self.conversation_sessions: Dict[str, List[MemoryItem]] = defaultdict(list)
        self.last_topic_emitted = None
        self.load_topic_clusters(topic_config_file)
        self.topic_config_file = topic_config_file
        self.github_token = github_token or os.getenv('GH_TOKEN')
        if not self.github_token:
            raise ValueError("GitHub token is required. Set GH_TOKEN environment variable or pass it to constructor.")

    def load_topic_clusters(self, path):
        if os.path.exists(path):
            with open(path, 'r') as f:
                self.topic_clusters = json.load(f)
        else:
            self.topic_clusters = {}

    def save_topic_clusters(self):
        with open(self.topic_config_file, 'w') as f:
            json.dump(self.topic_clusters, f, indent=2)

    def extract_keywords(self, text: str) -> Set[str]:
        stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "is", "are", "was", "were", "be", "been", "have", "has", "had", "do", "does", "did", "will", "would", "could", "should", "may", "might", "can", "this", "that", "these", "those", "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them"}
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        return {word for word in words if word not in stop_words}

    def detect_topic(self, text: str, keywords: Optional[Set[str]] = None) -> str:
        if keywords is None:
            keywords = self.extract_keywords(text)

        scores = {topic: len(keywords & set(words)) for topic, words in self.topic_clusters.items()}
        if scores:
            top_topic, top_score = max(scores.items(), key=lambda x: x[1])
            if top_score > 0:
                return top_topic

        # Create new topic if not found
        new_topic = f"topic_{len(self.topic_clusters)+1}"
        self.topic_clusters[new_topic] = list(keywords)
        self.save_topic_clusters()
        return new_topic

    def calculate_similarity(self, kw1: Set[str], kw2: Set[str]) -> float:
        if not kw1 or not kw2:
            return 0.0
        return len(kw1 & kw2) / len(kw1 | kw2)

    def add_message(self, text: str, conversation_id: Optional[str] = None) -> Dict:
        keywords = self.extract_keywords(text)
        topic = self.detect_topic(text, keywords)

        if self.current_topic and self.current_topic != topic:
            self.topic_transitions[self.current_topic].append(topic)
        self.current_topic = topic

        for i, w1 in enumerate(keywords):
            for w2 in list(keywords)[i+1:]:
                self.keyword_pairs[tuple(sorted([w1, w2]))] += 1

        mem = MemoryItem(
            content=text,
            keywords=keywords,
            timestamp=datetime.datetime.now(),
            topic_tags={topic},
            conversation_id=conversation_id
        )
        self.memories.append(mem)
        if conversation_id:
            self.conversation_sessions[conversation_id].append(mem)

        return {"topic": topic, "keywords": list(keywords)}

    def get_relevant_context(self, user_input: str, min_score=0.1) -> List[Dict]:
        query_keywords = self.extract_keywords(user_input)
        query_topic = self.detect_topic(user_input, query_keywords)

        results = []
        for mem in self.memories:
            score = self.calculate_similarity(query_keywords, mem.keywords)
            if query_topic in mem.topic_tags:
                score += 0.3
            score *= max(0.1, 1.0 - (mem.age_in_hours() / 168))
            if score >= min_score:
                mem.access_count += 1
                results.append({"content": mem.content, "topic": list(mem.topic_tags)[0], "score": round(score, 3)})

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:5]

    def emit_topic_drift_event(self) -> Optional[str]:
        if len(self.topic_transitions) < 2:
            return None
        recent = []
        for transitions in list(self.topic_transitions.values())[-3:]:
            recent.extend(transitions[-2:])
        topic_counts = Counter(recent)
        most_common = topic_counts.most_common(1)[0][0]
        if most_common != self.current_topic and most_common != self.last_topic_emitted:
            self.last_topic_emitted = most_common
            return f"Topic drift: {self.current_topic} → {most_common}"
        return None

    def get_conversation_summary(self, conversation_id: str) -> Dict:
        conv = self.conversation_sessions.get(conversation_id, [])
        if not conv:
            return {}
        summary = {
            "total_messages": len(conv),
            "topics": list({t for m in conv for t in m.topic_tags}),
            "keywords": list({k for m in conv for k in m.keywords}),
            "start_time": min(m.timestamp for m in conv).isoformat(),
            "end_time": max(m.timestamp for m in conv).isoformat()
        }
        return summary

    def split_conversation(self, conversation_id: str, split_index: int) -> Tuple[str, str]:
        original = self.conversation_sessions.get(conversation_id, [])
        if len(original) < split_index:
            return (conversation_id, None)

        new_id = f"{conversation_id}_part2"
        self.conversation_sessions[conversation_id] = original[:split_index]
        self.conversation_sessions[new_id] = original[split_index:]
        for m in self.conversation_sessions[new_id]:
            m.conversation_id = new_id
        return (conversation_id, new_id)

    def create_openai_summary(self, conversation_id: str, model="gpt-4o-mini") -> str:
        """Create a summary using GitHub Models API."""
        conv = self.conversation_sessions.get(conversation_id, [])
        if not conv:
            return "No conversation found."

        chat_log = "\n".join(f"User: {m.content}" for m in conv)
        prompt = f"Summarize the following conversation:\n{chat_log}"
        
        # GitHub Models API endpoint
        url = "https://models.inference.ai.azure.com/chat/completions"
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.github_token}"
        }
        
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant summarizing user conversations."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 500,
            "temperature": 0.7
        }
        
        try:
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()
            return result['choices'][0]['message']['content']
        except requests.exceptions.RequestException as e:
            return f"Error generating summary: {str(e)}"
        except (KeyError, IndexError) as e:
            return f"Error parsing response: {str(e)}"

    def split_conversation_by_topic(self, conversation_id: str) -> Dict[str, List[MemoryItem]]:
        conv = self.conversation_sessions.get(conversation_id, [])
        by_topic = defaultdict(list)
        for item in conv:
            topic = list(item.topic_tags)[0]
            by_topic[topic].append(item)
        return by_topic

    def create_split_chats_from_conversation(self, conversation_id: str, model="gpt-4o-mini") -> Dict[str, str]:
        topic_groups = self.split_conversation_by_topic(conversation_id)
        new_chat_ids = {}
        
        for topic, items in topic_groups.items():
            new_id = f"{conversation_id}_{topic}"
            self.conversation_sessions[new_id] = items
            for m in items:
                m.conversation_id = new_id
            
            summary = self.create_openai_summary(new_id, model=model)
            new_chat_ids[new_id] = summary

        return new_chat_ids


