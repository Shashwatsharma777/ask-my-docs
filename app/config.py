"""Central configuration loaded from environment variables."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Storage paths
CHROMA_DIR = PROJECT_ROOT / "chroma_db"
UPLOAD_DIR = PROJECT_ROOT / "data" / "uploads"

# Models
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.5-flash")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# Chunking (token counts, not characters)
CHUNK_TOKENS = int(os.getenv("CHUNK_TOKENS", "600"))
CHUNK_OVERLAP_TOKENS = int(os.getenv("CHUNK_OVERLAP_TOKENS", "100"))

# Retrieval
TOP_K_RETRIEVE = int(os.getenv("TOP_K_RETRIEVE", "20"))
TOP_K_RERANK = int(os.getenv("TOP_K_RERANK", "5"))
RRF_K = 60  # standard constant for Reciprocal Rank Fusion
