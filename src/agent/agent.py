"""Core agent logic - orchestrates retrieval, tools, and Gemini LLM responses."""

import json
import re
import time
from typing import Optional
from datetime import datetime

from google import genai
from google.genai import types
from google.api_core import exceptions as google_exceptions

from src.config import GEMINI_API_KEY, CHAT_MODEL, SNAPSHOT_TIME
from src.knowledge_base.retriever import KnowledgeRetriever
from src.tools.order_lookup import lookup_order, get_order_tool_description
from src.agent.prompts import build_system_prompt_with_context
from src.agent.conversation import ConversationSession, session_manager
from src.observability.logger import (
    ConversationTracer, DebugPanel, logger,
)

# Configure Gemini client
client = genai.Client(api_key=GEMINI_API_KEY)


def _gemini_call_with_retry(func, *args, max_retries=3, base_delay=10, **kwargs):
    """Call Gemini API with retry logic for rate limits."""
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_str = str(e)
            if '429' in error_str or 'RESOURCE_EXHAUSTED' in error_str or 'rate' in error_str.lower():
                delay = base_delay * (2 ** attempt)  # Exponential backoff
                logger.warning(f"Rate limited, retrying in {delay}s (attempt {attempt + 1}/{max_retries})")
                time.sleep(delay)
                continue
            raise  # Re-raise non-rate-limit errors
    raise Exception(f"Gemini API rate limit exceeded after {max_retries} retries")


def lookup_order_tool(order_id: str) -> str:
    """Look up an order by ID and return customer-safe information."""
    result = lookup_order(order_id)
    return json.dumps(result)


def create_model():
    """Create a Gemini model instance."""
    return client.models


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

        # Step 4: Call Gemini with tools
        tool_results = []
        final_text = ""

        try:
            # Build the full prompt with system instruction and user message
            full_prompt = f"{system_prompt}\n\n---\n\nUser message: {user_message}"

            # Add conversation context if available
            if history:
                context_parts = []
                for msg in history[-6:]:  # Last 3 turns
                    role = "Customer" if msg["role"] == "user" else "Agent"
                    context_parts.append(f"{role}: {msg['content']}")
                context_header = "## Conversation History\n\n" + "\n".join(context_parts) + "\n\n---\n\n"
                full_prompt = f"{system_prompt}\n\n{context_header}User message: {user_message}"

            # Use generate_content with tools
            response = _gemini_call_with_retry(
                client.models.generate_content,
                model=CHAT_MODEL,
                contents=full_prompt,
                config=types.GenerateContentConfig(
                    temperature=0,
                    max_output_tokens=1024,
                    tools=[types.Tool(
                        function_declarations=[
                            types.FunctionDeclaration(
                                name="lookup_order",
                                description=get_order_tool_description(),
                                parameters=types.Schema(
                                    type=types.Type.OBJECT,
                                    properties={
                                        "order_id": types.Schema(
                                            type=types.Type.STRING,
                                            description="The order ID to look up, e.g., 'ORD-1007'",
                                        ),
                                    },
                                    required=["order_id"],
                                ),
                            )
                        ]
                    )],
                ),
            )

            # Handle tool calls in a loop
            max_rounds = 3
            for round_num in range(max_rounds):
                # Check for function calls in the response
                if hasattr(response, 'candidates') and response.candidates:
                    candidate = response.candidates[0]
                    if hasattr(candidate, 'content') and candidate.content:
                        parts = candidate.content.parts
                        has_function_call = False

                        for part in parts:
                            if hasattr(part, 'function_call') and part.function_call:
                                has_function_call = True
                                fc = part.function_call
                                func_name = fc.name
                                func_args = dict(fc.args) if fc.args else {}

                                # Execute the tool
                                if func_name == "lookup_order":
                                    order_id = func_args.get("order_id", "")
                                    result = lookup_order(order_id)
                                else:
                                    result = {"error": f"Unknown tool: {func_name}"}

                                tool_results.append({
                                    "tool": func_name,
                                    "arguments": func_args,
                                    "result": result,
                                })

                                tracer.log_tool_call(func_name, func_args, result)

                                if self.debug:
                                    DebugPanel.show_tool_call(func_name, func_args, result)

                        if has_function_call:
                            # Send tool results back
                            tool_result_parts = []
                            for tr in tool_results[-1:]:
                                tool_result_parts.append(json.dumps(tr["result"]))

                            response = _gemini_call_with_retry(
                                client.models.generate_content,
                                model=CHAT_MODEL,
                                contents=[
                                    types.Content(
                                        role="user",
                                        parts=[types.Part(text=full_prompt)]
                                    ),
                                    types.Content(
                                        role="model",
                                        parts=[types.Part(text="I'll look that up for you.")]
                                    ),
                                    types.Content(
                                        role="user",
                                        parts=[types.Part(text="\n".join(tool_result_parts))]
                                    ),
                                ],
                                config=types.GenerateContentConfig(
                                    temperature=0,
                                    max_output_tokens=1024,
                                ),
                            )
                            continue

                        # Extract text response
                        for part in parts:
                            if hasattr(part, 'text') and part.text:
                                final_text += part.text
                        break

            # Fallback if no text was extracted
            if not final_text and response:
                if hasattr(response, 'text'):
                    final_text = response.text
                else:
                    final_text = str(response)

        except Exception as e:
            logger.error(f"Gemini error: {e}")
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
        # Match formats like [Source: filename.md, "Heading"] or [Source: filename.md]
        source_patterns = [
            r'\[Source:\s*([\w\d._-]+\.md)',
            r'\(([\w\d._-]+\.md)\)',
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
