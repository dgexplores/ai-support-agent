"""FastAPI web application for the AI support agent."""

import json
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import Optional

from src.agent.agent import support_agent
from src.agent.conversation import session_manager
from src.config import BASE_DIR

app = FastAPI(title="Aster & Row Support Agent")
templates = Jinja2Templates(directory=str(BASE_DIR / "src" / "web" / "templates"))


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    debug: Optional[bool] = False


class ChatResponse(BaseModel):
    response: str
    sources: list[str] = []
    tool_calls: list[str] = []
    handoff: bool = False
    session_id: str = ""
    trace: Optional[dict] = None


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Serve the chat interface."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Handle a chat message."""
    result = support_agent.handle_message(
        user_message=request.message,
        session_id=request.session_id,
        debug=request.debug,
    )
    return ChatResponse(**result)


@app.post("/api/session/new")
async def new_session():
    """Create a new conversation session."""
    session = session_manager.get_or_create()
    return {"session_id": session.session_id}


@app.post("/api/session/clear")
async def clear_session(session_id: str):
    """Clear a conversation session."""
    session_manager.clear_session(session_id)
    return {"status": "cleared"}


@app.get("/api/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "service": "aster-row-support-agent"}
