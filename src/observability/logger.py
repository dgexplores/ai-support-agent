"""Debug and trace logging for observability."""

import json
import logging
import os
from datetime import datetime
from typing import Any, Optional

from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table

console = Console()

# Configure structured logging
logging.basicConfig(
    level=logging.DEBUG if os.environ.get("DEBUG", "false").lower() == "true" else logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, rich_tracebacks=True)],
)
logger = logging.getLogger("support-agent")


class ConversationTracer:
    """Traces a single conversation turn for debugging."""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.turn_id = datetime.now().strftime("%H%M%S%f")
        self.events: list[dict[str, Any]] = []
        self.start_time = datetime.now()

    def log_event(self, event_type: str, data: dict[str, Any]):
        """Log a structured event."""
        event = {
            "timestamp": datetime.now().isoformat(),
            "type": event_type,
            **data,
        }
        self.events.append(event)
        if event_type == "user_message":
            logger.info(f"[Turn {self.turn_id}] User: {data.get('content', '')}")
        elif event_type == "retrieved_passages":
            sources = [p.get("source", "unknown") for p in data.get("passages", [])]
            logger.info(f"[Turn {self.turn_id}] Retrieved: {sources}")
        elif event_type == "tool_call":
            logger.info(f"[Turn {self.turn_id}] Tool: {data.get('tool', '')}({data.get('arguments', {})})")
        elif event_type == "tool_result":
            logger.info(f"[Turn {self.turn_id}] Tool result: {str(data.get('result', ''))[:100]}...")
        elif event_type == "agent_response":
            logger.info(f"[Turn {self.turn_id}] Agent: {data.get('content', '')[:150]}...")
        elif event_type == "error":
            logger.error(f"[Turn {self.turn_id}] Error: {data.get('message', '')}")

    def log_retrieval(self, passages: list[dict]):
        """Log retrieved passages with metadata."""
        self.log_event("retrieved_passages", {
            "passages": [
                {
                    "source": p.get("source", "unknown"),
                    "heading": p.get("heading", ""),
                    "score": p.get("score", 0),
                    "preview": p.get("content", "")[:100],
                }
                for p in passages
            ]
        })

    def log_tool_call(self, tool_name: str, arguments: dict, result: Any):
        """Log a tool call and its result."""
        self.log_event("tool_call", {
            "tool": tool_name,
            "arguments": arguments,
        })
        # Sanitize result for logging - remove internal fields
        safe_result = result
        if isinstance(result, dict):
            safe_result = {k: v for k, v in result.items() if k not in ("internal", "customer")}
        self.log_event("tool_result", {
            "tool": tool_name,
            "result": str(safe_result)[:500],
        })

    def get_trace(self) -> dict:
        """Return the full trace for this turn."""
        duration = (datetime.now() - self.start_time).total_seconds()
        return {
            "session_id": self.session_id,
            "turn_id": self.turn_id,
            "duration_seconds": duration,
            "events": self.events,
        }


class DebugPanel:
    """Rich debug panels for terminal output."""

    @staticmethod
    def show_retrieval(passages: list[dict]):
        """Display retrieved passages in a table."""
        table = Table(title="Retrieved Passages")
        table.add_column("Source", style="cyan")
        table.add_column("Heading", style="green")
        table.add_column("Score", justify="right", style="yellow")
        table.add_column("Preview", max_width=50)
        for p in passages:
            table.add_row(
                p.get("source", "?"),
                p.get("heading", "?"),
                f"{p.get('score', 0):.3f}",
                p.get("content", "")[:80] + "...",
            )
        console.print(table)

    @staticmethod
    def show_tool_call(tool_name: str, arguments: dict, result: Any):
        """Display tool call info."""
        console.print(Panel(
            f"[bold]Tool:[/bold] {tool_name}\n"
            f"[bold]Args:[/bold] {json.dumps(arguments, indent=2)}\n"
            f"[bold]Result:[/bold] {json.dumps(result, indent=2)[:500]}",
            title="Tool Call",
            border_style="blue",
        ))

    @staticmethod
    def show_response(response: str, sources: list[str] = None):
        """Display agent response."""
        content = f"[bold]Response:[/bold]\n{response}"
        if sources:
            content += f"\n\n[bold]Sources:[/bold] {', '.join(sources)}"
        console.print(Panel(content, title="Agent Response", border_style="green"))

    @staticmethod
    def show_error(error: str):
        """Display error."""
        console.print(Panel(f"[red]{error}[/red]", title="Error", border_style="red"))
