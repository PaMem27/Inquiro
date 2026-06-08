from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PAPERS_DIR = PROJECT_ROOT / "papers"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "gemini-2.5-flash"
