"""CLI entry point for the AI Support Agent."""

import sys
import json
import argparse
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

from src.config import DEBUG
from src.knowledge_base.indexer import KnowledgeBaseIndexer
from src.agent.agent import support_agent
from src.agent.conversation import session_manager
from src.observability.logger import DebugPanel, console


def index_knowledge_base():
    """Index all knowledge base documents."""
    console.print("[bold blue]Indexing knowledge base...[/bold blue]")
    indexer = KnowledgeBaseIndexer()
    count = indexer.index_all_documents()
    console.print(f"[bold green]Indexed {count} chunks successfully.[/bold green]")


def run_cli():
    """Run the interactive CLI chat."""
    console.print(Panel(
        "[bold]Aster & Row Support Agent[/bold]\n"
        "Type your message and press Enter. Type 'quit' to exit.\n"
        "Type '/debug' to toggle debug mode. Type '/new' for new session.",
        title="Welcome",
        border_style="blue",
    ))
    
    session = session_manager.get_or_create()
    debug_mode = DEBUG
    
    while True:
        try:
            user_input = console.input("\n[bold cyan]You:[/bold cyan] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[bold]Goodbye![/bold]")
            break
        
        if not user_input:
            continue
        
        if user_input.lower() in ("quit", "exit", "q"):
            console.print("[bold]Goodbye![/bold]")
            break
        
        if user_input.lower() == "/debug":
            debug_mode = not debug_mode
            console.print(f"[bold]Debug mode: {'ON' if debug_mode else 'OFF'}[/bold]")
            continue
        
        if user_input.lower() == "/new":
            session = session_manager.get_or_create()
            console.print("[bold]New session started.[/bold]")
            continue
        
        # Process message
        result = support_agent.handle_message(
            user_message=user_input,
            session_id=session.session_id,
            debug=debug_mode,
        )
        
        # Display response
        console.print("\n[bold green]Agent:[/bold green]")
        console.print(Markdown(result["response"]))
        
        if result["sources"]:
            console.print(f"\n[dim]Sources: {', '.join(result['sources'])}[/dim]")
        
        if result["handoff"]:
            console.print(Panel(
                "[yellow]This issue may require human support. "
                "Please contact our support team for further assistance.[/yellow]",
                title="Recommendation",
                border_style="yellow",
            ))
        
        if result["tool_calls"]:
            console.print(f"[dim]Tools used: {', '.join(result['tool_calls'])}[/dim]")


def main():
    parser = argparse.ArgumentParser(description="Aster & Row Support Agent")
    parser.add_argument("--cli", action="store_true", help="Run in CLI mode")
    parser.add_argument("--web", action="store_true", help="Run web server")
    parser.add_argument("--index", action="store_true", help="Index knowledge base")
    parser.add_argument("--eval", action="store_true", help="Run evaluation suite")
    args = parser.parse_args()
    
    if args.index:
        index_knowledge_base()
    elif args.cli:
        index_knowledge_base()  # Ensure index is fresh
        run_cli()
    elif args.web:
        index_knowledge_base()  # Ensure index is fresh
        import uvicorn
        from src.web.app import app
        from src.config import HOST, PORT
        console.print(f"[bold]Starting web server on http://{HOST}:{PORT}[/bold]")
        uvicorn.run(app, host=HOST, port=PORT)
    elif args.eval:
        from evaluation.run_eval import run_evaluation
        run_evaluation()
    else:
        # Default: show help
        parser.print_help()


if __name__ == "__main__":
    main()
