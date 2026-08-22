"""Configuration management for the AI Support Agent."""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Paths
BASE_DIR = Path(__file__).parent.parent
KNOWLEDGE_BASE_DIR = BASE_DIR / "knowledge-base"
DATA_DIR = BASE_DIR / "data"
ORDERS_FILE = DATA_DIR / "orders.json"
CHROMA_DIR = BASE_DIR / "chroma_db"

# Groq Configuration (primary + fallback)
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_API_KEY_FALLBACK = os.getenv("GROQ_API_KEY_FALLBACK", "")
CHAT_MODEL = os.getenv("CHAT_MODEL", "openai/gpt-oss-120b")

# Server Configuration
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
DEBUG = os.getenv("DEBUG", "false").lower() == "true"

# Chunking Configuration
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

# Retrieval Configuration
TOP_K_RESULTS = 8
MIN_RELEVANCE_SCORE = 0.3

# Snapshot time for deterministic evaluation
SNAPSHOT_TIME = "2026-08-15T12:00:00Z"
