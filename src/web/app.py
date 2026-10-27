"""FastAPI web application for the AI support agent."""

from threading import Lock
from fastapi import HTTPException, Response
from pydantic import Field
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import Optional

from src.agent.agent import support_agent
from src.agent.conversation import session_manager
from src.config import BASE_DIR, LLM_PROVIDER, CHAT_MODEL
from src.web.demo_guard import DemoGuard

app = FastAPI(title="Aster & Row Support Agent")
guard = DemoGuard()
agent_lock = Lock()
COOKIE = "aster_demo"

templates = Jinja2Templates(directory=str(BASE_DIR / "src" / "web" / "templates"))


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=1500)
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
    return templates.TemplateResponse(request=request, name="index.html", context={})


def owned_session(request: Request, session_id=None):
    token = request.cookies.get(COOKIE)
    if not token or not session_manager.get_session(token) or (session_id and token != session_id):
        raise HTTPException(403, "Start your own demo session first.")
    return token


@app.middleware("http")
async def demo_boundary(request: Request, call_next):
    if request.method == "POST":
        origin = request.headers.get("origin")
        if origin and origin != str(request.base_url).rstrip("/"):
            return JSONResponse({"detail": "Cross-origin requests are not allowed."}, status_code=403)
        body = await request.body()
        if len(body) > 8192:
            return JSONResponse({"detail": "Request is too large."}, status_code=413)
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-store"
    response.headers["X-Content-Type-Options"] = "nosniff"
    return response


@app.post("/api/chat", response_model=ChatResponse)
def chat(payload: ChatRequest, request: Request):
    token = owned_session(request, payload.session_id)
    actor = request.client.host if request.client else "unknown"
    with agent_lock:
        if not guard.admit(actor):
            raise HTTPException(429, "Demo request limit reached. Please try later.")
        result = support_agent.handle_message(payload.message, token, debug=False)
    if result.get("provider_failed"):
        raise HTTPException(503, "The AI provider is temporarily unavailable. Please try later.")
    result["trace"] = None
    return ChatResponse(**result)


@app.post("/api/session/new")
def new_session(request: Request, response: Response):
    actor = request.client.host if request.client else "unknown"
    if not guard.admit(actor, inference=False):
        raise HTTPException(429, "Please wait before starting another session.")
    with agent_lock:
        # Bound retained conversations; this deployment intentionally uses one worker.
        from datetime import datetime
        session_manager.sessions = {k: v for k, v in session_manager.sessions.items()
                                    if (datetime.now() - v.last_active).total_seconds() < 3600}
        if len(session_manager.sessions) >= 128:
            raise HTTPException(429, "The demo is busy. Please try later.")
        session = session_manager.get_or_create()
    response.set_cookie(COOKIE, session.session_id, httponly=True, secure=request.url.scheme == "https",
                        samesite="strict", max_age=3600)
    return {"session_id": session.session_id}


@app.post("/api/session/clear")
def clear_session(request: Request, session_id: str):
    token = owned_session(request, session_id)
    with agent_lock:
        session_manager.clear_session(token)
    return {"status": "cleared"}


@app.get("/api/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "service": "aster-row-support-agent", "provider": LLM_PROVIDER, "model": CHAT_MODEL}
