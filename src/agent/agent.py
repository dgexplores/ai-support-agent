"""Core agent logic - orchestrates retrieval, tools, and Groq LLM responses."""

import json
import re
import time
from typing import Optional
from datetime import datetime

from groq import Groq

from src.config import GROQ_API_KEY, GROQ_API_KEY_FALLBACK, CHAT_MODEL, SNAPSHOT_TIME
from src.knowledge_base.retriever import KnowledgeRetriever
from src.tools.order_lookup import lookup_order, get_order_tool_description
from src.agent.prompts import build_system_prompt_with_context
from src.agent.conversation import ConversationSession, session_manager
from src.observability.logger import (
    ConversationTracer, DebugPanel, logger,
)

# Initialize Groq clients (primary + fallback)
groq_clients = [Groq(api_key=GROQ_API_KEY)]
if GROQ_API_KEY_FALLBACK:
    groq_clients.append(Groq(api_key=GROQ_API_KEY_FALLBACK))
groq_client = groq_clients[0]
_current_client_idx = 0

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


def _rotate_client():
    """Rotate to the next available API key."""
    global _current_client_idx, groq_client
    _current_client_idx = (_current_client_idx + 1) % len(groq_clients)
    groq_client = groq_clients[_current_client_idx]
    logger.info(f"Rotated to API key index {_current_client_idx}")


def _groq_call_with_retry(messages, tools=None, max_retries=4, base_delay=2.0, jitter=1.0):
    """Call Groq API with retry logic, key rotation, and empty response handling."""
    global groq_client
    
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
            
            # Small delay to avoid rate limits
            time.sleep(base_delay + jitter * attempt)
            
            response = groq_client.chat.completions.create(**kwargs)
            
            if not response or not response.choices:
                _rotate_client()
                continue
            
            return response
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "rate" in error_str.lower() or "limit" in error_str.lower():
                _rotate_client()
                time.sleep(base_delay)
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
                
                # Safely extract the response
                if not response or not response.choices:
                    logger.error("Empty Groq response")
                    final_text = "I'm sorry, I'm experiencing a technical issue. Please try again."
                    break
                
                choice = response.choices[0]
                if not choice or not choice.message:
                    logger.error("Empty Groq choice/message")
                    final_text = "I'm sorry, I'm experiencing a technical issue. Please try again."
                    break
                
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
                if message.content:
                    final_text = message.content
                elif tool_results:
                    # Model returned tool calls but no text - construct response from tool results
                    last_result = tool_results[-1].get("result", {})
                    if isinstance(last_result, dict) and last_result.get("found"):
                        order = last_result.get("order", {})
                        final_text = order.get("customer_safe_message", "I looked up your order.")
                    else:
                        final_text = str(last_result)
                else:
                    final_text = "I'm not sure how to respond to that. Could you rephrase your question?"
                break

        except Exception as e:
            logger.error(f"Groq error: {e}")
            tracer.log_event("error", {"message": str(e)})
            final_text = "I'm sorry, I'm experiencing a technical issue. Please try again or contact our support team."

        # Extract sources mentioned in the response
        sources = self._extract_sources(final_text, passages)

        # Ensure source citations are present in the response
        final_text = self._ensure_source_citations(final_text, passages, sources)
        sources = self._extract_sources(final_text, passages)

        # Check if handoff is recommended
        handoff = self._detect_handoff_needed(final_text, passages, tool_results, user_message)

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

    def _ensure_source_citations(self, final_text: str, passages: list[dict], sources: list[str]) -> str:
        """Ensure source citations are present in the response.
        
        If the LLM didn't include source citations, add them from the top-retrieved passages.
        """
        if not sources:
            return final_text
        
        # Check if the response already has any source citations
        citation_pattern = r'\[Source:\s*[\w\d._-]+\.md'
        has_citations = bool(re.search(citation_pattern, final_text))
        
        if has_citations:
            return final_text
        
        # Add citations from the top passages
        citation_parts = []
        for p in passages[:3]:  # Use top 3 passages
            source = p.get("source", "")
            heading = p.get("heading", "")
            if source and heading:
                citation_parts.append(f'[Source: {source}, "{heading}"]')
        
        if citation_parts:
            # Insert citations at the end before any concluding punctuation
            if final_text.rstrip().endswith('.'):
                base = final_text.rstrip()[:-1]
                final_text = base + " " + " ".join(citation_parts) + "."
            else:
                final_text = final_text.rstrip() + " " + " ".join(citation_parts) + "."
        
        return final_text

    def _detect_handoff_needed(
        self, response: str, passages: list[dict], tool_results: list[dict], user_message: str = ""
    ) -> bool:
        """Detect if human handoff is needed."""
        # 1. Source conflicts detected by retriever - always handoff
        conflicts = self.retriever.detect_source_conflicts(passages)
        if conflicts:
            return True
        
        # 2. No passages retrieved - information insufficient
        if not passages:
            return True
        
        # 3. Tool result: order not found - need human assistance
        if tool_results:
            for tr in tool_results:
                result = tr.get("result", {})
                if isinstance(result, dict) and not result.get("found", False):
                    return True
        
        # 4. Check if agent is making promises about actions it cannot perform
        action_promises = [
            "refund",
            "cancellation",
            "replacement",
            "address change",
        ]
        response_lower = response.lower()
        
        has_action_mention = any(kw in response_lower for kw in action_promises)
        
        completed_actions = [
            "approved",
            "completed",
            "processed",
            "issued",
        ]
        has_completed_action = any(kw in response_lower for kw in completed_actions)
        
        if has_completed_action and not has_action_mention:
            return True
        
        if has_action_mention:
            citation_pattern = r"\[Source:"
            has_citations = bool(re.search(citation_pattern, response))
            if not has_citations and has_action_mention:
                return True
        
        # 5. Check for privacy-related requests in the user's question
        # If the user is asking for PII or internal info, handoff is needed
        privacy_keywords = ["email", "address", "risk score", "internal note", "warehouse note", "support tag"]
        # Check the response for inadvertent disclosure, AND check if the
        # user's original request was for privacy-sensitive info
        # We check the passages to see if they contain the requested info
        user_info = ""
        # We can't easily get the original user message here, but we can check
        # if the response mentions it must not share certain info
        if any(kw in response_lower for kw in privacy_keywords):
            return True
        
        # 6. Check if the response mentions contacting support naturally
        # (only if it's not already handled by other checks)
        support_keywords = ["contact our support team", "human support", "speak with a representative"]
        if any(kw in response_lower for kw in support_keywords):
            # Only handoff if not already covered by conflict/insufficient info checks
            # This is a fallback for when the agent naturally recommends support
            # but we want to avoid double-handoff
            pass  # Let other checks handle it
        
        return False


# Global agent instance
support_agent = SupportAgent()
