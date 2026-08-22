"""Multi-turn conversation manager."""

import uuid
from datetime import datetime
from typing import Optional


class ConversationSession:
    """Manages a single conversation session with history."""

    def __init__(self, session_id: Optional[str] = None):
        self.session_id = session_id or str(uuid.uuid4())
        self.messages: list[dict] = []
        self.created_at = datetime.now()
        self.last_active = datetime.now()
        # Track topics for context window management
        self.active_topics: list[str] = []

    def add_user_message(self, content: str):
        """Add a user message to the conversation."""
        self.messages.append({
            "role": "user",
            "content": content,
            "timestamp": datetime.now().isoformat(),
        })
        self.last_active = datetime.now()

    def add_assistant_message(self, content: str, sources: list[str] = None):
        """Add an assistant response to the conversation."""
        self.messages.append({
            "role": "assistant",
            "content": content,
            "sources": sources or [],
            "timestamp": datetime.now().isoformat(),
        })
        self.last_active = datetime.now()

    def get_history(self, max_turns: int = 10) -> list[dict]:
        """Get conversation history for the LLM.
        
        Returns messages in OpenAI format, limited to max_turns.
        """
        # Take the last N turns (user + assistant pairs)
        recent = self.messages[-(max_turns * 2):]
        return [{"role": m["role"], "content": m["content"]} for m in recent]

    def get_context_summary(self) -> str:
        """Get a brief summary of the conversation context."""
        if not self.messages:
            return "New conversation."
        
        topics = []
        for msg in self.messages:
            if msg["role"] == "user":
                content = msg["content"].lower()
                if any(w in content for w in ["return", "refund", "exchange"]):
                    topics.append("returns/refunds")
                elif any(w in content for w in ["ship", "deliver", "track"]):
                    topics.append("shipping/delivery")
                elif any(w in content for w in ["order", "ord-"]):
                    topics.append("order inquiry")
                elif any(w in content for w in ["warranty", "defect", "broken"]):
                    topics.append("warranty/defects")
                elif any(w in content for w in ["cancel"]):
                    topics.append("cancellation")
        
        unique_topics = list(dict.fromkeys(topics))  # preserve order, deduplicate
        if unique_topics:
            return f"Conversation topics: {', '.join(unique_topics)}"
        return "General inquiry."

    def clear(self):
        """Clear conversation history."""
        self.messages = []
        self.active_topics = []
        self.created_at = datetime.now()
        self.last_active = datetime.now()


class SessionManager:
    """Manages multiple conversation sessions."""

    def __init__(self):
        self.sessions: dict[str, ConversationSession] = {}

    def get_or_create(self, session_id: Optional[str] = None) -> ConversationSession:
        """Get an existing session or create a new one."""
        if session_id and session_id in self.sessions:
            return self.sessions[session_id]
        
        session = ConversationSession(session_id)
        self.sessions[session.session_id] = session
        return session

    def get_session(self, session_id: str) -> Optional[ConversationSession]:
        """Get a session by ID."""
        return self.sessions.get(session_id)

    def clear_session(self, session_id: str):
        """Clear a session's history."""
        if session_id in self.sessions:
            self.sessions[session_id].clear()


# Global session manager
session_manager = SessionManager()
