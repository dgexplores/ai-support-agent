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

# Gemini Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gemini-2.0-flash")

# Server Configuration
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
DEBUG = os.getenv("DEBUG", "false").lower() == "true"

# Chunking Configuration
CHUNK_SIZE = 800  # characters per chunk
CHUNK_OVERLAP = 150  # overlap between chunks

# Retrieval Configuration
TOP_K_RESULTS = 8  # number of chunks to retrieve
MIN_RELEVANCE_SCORE = 0.3  # minimum similarity score

# Snapshot time for deterministic evaluation (from orders.json)
SNAPSHOT_TIME = "2026-08-15T12:00:00Z"
