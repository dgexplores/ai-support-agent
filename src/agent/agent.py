"""Core agent logic - orchestrates retrieval, tools, and Groq LLM responses."""

import json
import re
import time
from typing import Optional
from datetime import datetime

from groq import Groq

from src.config import GROQ_API_KEY, CHAT_MODEL, SNAPSHOT_TIME
from src.knowledge_base.retriever import KnowledgeRetriever
from src.tools.order_lookup import lookup_order, get_order_tool_description
from src.agent.prompts import build_system_prompt_with_context
from src.agent.conversation import ConversationSession, session_manager
from src.observability.logger import (
    ConversationTracer, DebugPanel, logger,
)

# Initialize Groq client
groq_client = Groq(api_key=GROQ_API_KEY)

# Tool definition for Groq function calling
GROQ_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "lookup_order",
            "description": get_order_tool_description(),
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "The order ID to look up, e.g., 'ORD-1007'",
                    }
                },
                "required": ["order_id"],
            },
        },
    }
]


def _groq_call_with_retry(messages, tools=None, max_retries=3, base_delay=2):
    """Call Groq API with retry logic for rate limits."""
    for attempt in range(max_retries):
        try:
            kwargs = {
                "model": CHAT_MODEL,
                "messages": messages,
                "temperature": 0,
                "max_tokens": 1024,
            }
            if tools:
                kwargs["tools"] = tools
                kwargs["tool_choice"] = "auto"

            return groq_client.chat.completions.create(**kwargs)
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "rate" in error_str.lower() or "limit" in error_str.lower():
                delay = base_delay * (2 ** attempt)
                logger.warning(f"Rate limited, retrying in {delay}s (attempt {attempt + 1}/{max_retries})")
                time.sleep(delay)
                continue
            raise


class SupportAgent:
    """Main support agent that orchestrates RAG, tools, and conversation."""

    def __init__(self):
        self.retriever = KnowledgeRetriever()
        self.debug = False

    def handle_message(
        self,
        user_message: str,
        session_id: Optional[str] = None,
        debug: bool = False,
    ) -> dict:
        """Process a user message and return the agent's response."""
        self.debug = debug
        session = session_manager.get_or_create(session_id)
        tracer = ConversationTracer(session.session_id)
        tracer.log_event("user_message", {"content": user_message})

        # Add user message to session
        session.add_user_message(user_message)

        # Get conversation history
        history = session.get_history()

        # Step 1: Retrieve relevant passages
        passages = self.retriever.retrieve(user_message)
        tracer.log_retrieval(passages)

        if self.debug:
            DebugPanel.show_retrieval(passages)

        # Step 2: Detect source conflicts
        conflicts = self.retriever.detect_source_conflicts(passages)

        # Step 3: Build system prompt with context
        system_prompt = build_system_prompt_with_context(
            retrieved_passages=passages,
            conversation_history=history,
            conflicts=conflicts,
        )

        # Step 4: Call Groq with tools
        messages = [{"role": "system", "content": system_prompt}]
        # Add conversation history
        for msg in history:
            messages.append({"role": msg["role"], "content": msg["content"]})
        # Add current user message
        messages.append({"role": "user", "content": user_message})

        tool_results = []
        final_text = ""
        max_tool_rounds = 3

        try:
            for round_num in range(max_tool_rounds):
                response = _groq_call_with_retry(messages, tools=GROQ_TOOLS)
                choice = response.choices[0]
                message = choice.message

                # Check if the model wants to call a tool
                if message.tool_calls:
                    # Add assistant message with tool calls to history
                    assistant_msg = {"role": "assistant", "content": message.content or ""}
                    if message.tool_calls:
                        assistant_msg["tool_calls"] = [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments,
                                },
                            }
                            for tc in message.tool_calls
                        ]
                    messages.append(assistant_msg)

                    for tool_call in message.tool_calls:
                        tool_name = tool_call.function.name
                        try:
                            tool_args = json.loads(tool_call.function.arguments)
                        except json.JSONDecodeError:
                            tool_args = {}

                        # Execute the tool
                        if tool_name == "lookup_order":
                            order_id = tool_args.get("order_id", "")
                            result = lookup_order(order_id)
                        else:
                            result = {"error": f"Unknown tool: {tool_name}"}

                        tool_results.append({
                            "tool": tool_name,
                            "arguments": tool_args,
                            "result": result,
                        })

                        tracer.log_tool_call(tool_name, tool_args, result)

                        if self.debug:
                            DebugPanel.show_tool_call(tool_name, tool_args, result)

                        # Add tool result to messages
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": json.dumps(result),
                        })

                    # Continue loop - model will respond with tool results
                    continue

                # No tool calls - extract final text response
                final_text = message.content or ""
                break

        except Exception as e:
            logger.error(f"Groq error: {e}")
            tracer.log_event("error", {"message": str(e)})
            final_text = "I'm sorry, I'm experiencing a technical issue. Please try again or contact our support team."

        # Extract sources mentioned in the response
        sources = self._extract_sources(final_text, passages)

        # Check if handoff is recommended
        handoff = self._detect_handoff_needed(final_text, passages, tool_results)

        # Add assistant response to session
        session.add_assistant_message(final_text, sources)

        tracer.log_event("agent_response", {
            "content": final_text,
            "sources": sources,
            "handoff": handoff,
        })

        trace = tracer.get_trace()

        if self.debug:
            DebugPanel.show_response(final_text, sources)

        return {
            "response": final_text,
            "sources": sources,
            "tool_calls": [tr["tool"] for tr in tool_results],
            "handoff": handoff,
            "session_id": session.session_id,
            "trace": trace,
        }

    def _extract_sources(self, response_text: str, passages: list[dict]) -> list[str]:
        """Extract source filenames mentioned in the response."""
        sources = set()

        # Look for explicit source references in the response
        source_patterns = [
            r'\[Source:\s*([\w\d._-]+\.md)',
            r'\(([ \w\d._-]*\.md)\)',
            r'from\s+([\w\d._-]+\.md)',
            r'according to\s+([\w\d._-]+\.md)',
        ]

        for pattern in source_patterns:
            matches = re.findall(pattern, response_text, re.IGNORECASE)
            for match in matches:
                source = match.strip().strip('"').strip("'")
                if source.endswith('.md'):
                    sources.add(source)

        # Also add sources from passages that were likely used
        for p in passages:
            if p.get("score", 0) > 0.5:
                sources.add(p["source"])

        return sorted(sources)

    def _detect_handoff_needed(
        self, response: str, passages: list[dict], tool_results: list[dict]
    ) -> bool:
        """Detect if human handoff is recommended."""
        handoff_indicators = [
            "contact our support team",
            "human support",
            "reach out to",
            "speak with a representative",
            "customer support team",
            "recommend contacting",
            "would be best to contact",
            "escalate",
            "conflicting information",
        ]
        response_lower = response.lower()
        return any(indicator in response_lower for indicator in handoff_indicators)


# Global agent instance
support_agent = SupportAgent()
