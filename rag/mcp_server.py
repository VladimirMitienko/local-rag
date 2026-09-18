"""MCP-сервер (stdio) для Claude Code: поиск по локальному индексу.

Модель эмбеддингов грузится лениво — при первом поиске (~10 с), чтобы сессии,
которые не обращаются к документации, не держали её в памяти.
"""
from __future__ import annotations

import threading

from mcp.server.mcpserver import MCPServer

from .config import DB_PATH

INSTRUCTIONS = """Локальный поиск по проектной документации (гибридный: смысловой + по словам).
Используй search_docs, прежде чем отвечать на вопросы о требованиях, поведении системы, API, настройках,
тест-кейсах и отчётах: документация большая и целиком в контекст не помещается.
- Формулируй запрос своими словами на языке документа; для идентификаторов (ключи задач, имена функций, ENV)
  добавляй их точное написание. Если выдача слабая — переформулируй или сузь фильтром path_contains.
- Текст в ~~двойных тильдах~~ — зачёркнутый/удалённый в исходном документе, не считай его требованием.
- Нужен контекст вокруг найденного фрагмента — get_context(chunk_id).
- В ответе указывай источник: файл › раздел (стр.)."""

mcp = MCPServer(name="docs-rag", instructions=INSTRUCTIONS)

_index = None
_lock = threading.Lock()


def _idx():
    global _index
    with _lock:
        if _index is None:
            from .store import Index
            _index = Index(DB_PATH)
        return _index


def _fmt(hits, full=True) -> str:
    if not hits:
        return "Ничего не найдено."
    out = []
    for n, h in enumerate(hits, 1):
        out.append(f"[{n}] {h.source}\n    chunk_id={h.chunk_id} · коллекция={h.collection} · файл={h.path}\n{h.text}")
    return "\n\n---\n\n".join(out)


@mcp.tool()
def search_docs(query: str, k: int = 8, collection: str | None = None, path_contains: str | None = None) -> str:
    """Найти фрагменты проектной документации, релевантные запросу.

    Args:
        query: вопрос или ключевые слова (русский/английский; идентификаторы — как в документах).
        k: сколько фрагментов вернуть (по умолчанию 8, максимум 30).
        collection: ограничить одной коллекцией (список — list_sources).
        path_contains: ограничить файлами, в пути которых есть эта подстрока (без учёта регистра), напр. "v2.0".
    """
    hits = _idx().search(query, k=max(1, min(k, 30)), collection=collection, path_contains=path_contains)
    return _fmt(hits)


@mcp.tool()
def get_context(chunk_id: int, before: int = 1, after: int = 1) -> str:
    """Вернуть фрагмент документа вместе с соседними фрагментами (для чтения раздела целиком).

    Args:
        chunk_id: id из результата search_docs.
        before: сколько фрагментов до (по умолчанию 1, максимум 10).
        after: сколько фрагментов после (по умолчанию 1, максимум 10).
    """
    hits = _idx().neighbours(chunk_id, max(0, min(before, 10)), max(0, min(after, 10)))
    if not hits:
        return f"Фрагмент {chunk_id} не найден."
    parts = [f"Файл: {hits[0].path}"]
    for h in hits:
        mark = " ◀ запрошенный" if h.chunk_id == chunk_id else ""
        parts.append(f"## {h.heading or '(без раздела)'}{' (' + h.loc + ')' if h.loc else ''} "
                     f"[chunk_id={h.chunk_id}]{mark}\n{h.text}")
    return "\n\n".join(parts)


@mcp.tool()
def list_sources(collection: str | None = None) -> str:
    """Список проиндексированных документов и коллекций (что вообще есть в базе знаний)."""
    rows = _idx().sources(collection)
    if not rows:
        return "Индекс пуст."
    lines, cur = [], None
    for coll, path, title, n, when, warn in rows:
        if coll != cur:
            lines.append(f"\n[{coll}]")
            cur = coll
        lines.append(f"- {path} — «{title}», {n} фрагм., индексирован {when}" + (f" ⚠ {warn}" if warn else ""))
    return "\n".join(lines).strip()


def main():
    mcp.run()


if __name__ == "__main__":
    main()
