import os
from pathlib import Path

RAG_HOME = Path(os.environ.get("RAG_HOME", Path(__file__).resolve().parent.parent))
DB_PATH = Path(os.environ.get("RAG_DB", RAG_HOME / "data" / "index.db"))

# Эмбеддинги: мультиязычная модель, хорошо работает с русским, до 8k токенов контекста.
MODEL_NAME = os.environ.get("RAG_MODEL", "BAAI/bge-m3")
MAX_SEQ_LEN = 1024

# Размеры чанков в символах (~3.5 символа на токен для русского текста).
CHUNK_TARGET = 1500
CHUNK_MAX = 2200
CHUNK_MIN = 400

SUPPORTED = {
    ".docx", ".pdf", ".md", ".markdown", ".txt", ".html", ".htm",
    ".xlsx", ".xlsm", ".csv", ".pptx",
    ".doc", ".rtf", ".odt",  # через macOS textutil
}
SKIP_DIRS = {".git", ".venv", "venv", "node_modules", "__pycache__", ".idea", ".vscode"}
