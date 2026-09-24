import os
import sys
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
}
# Старые форматы Word/OpenOffice читаются утилитой textutil, она есть только в macOS.
# На Windows/Linux такие файлы нужно пересохранить в .docx.
if sys.platform == "darwin":
    SUPPORTED |= {".doc", ".rtf", ".odt"}
SKIP_DIRS = {".git", ".venv", "venv", "node_modules", "__pycache__", ".idea", ".vscode"}
