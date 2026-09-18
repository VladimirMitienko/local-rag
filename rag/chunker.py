"""Блоки → чанки с учётом структуры: чанк не пересекает границу раздела верхнего уровня,
мелкие соседние подразделы склеиваются, длинные таблицы режутся по строкам с повтором шапки."""
from __future__ import annotations

import re
from dataclasses import dataclass

from .config import CHUNK_MAX, CHUNK_MIN, CHUNK_TARGET
from .loaders import Block

SEP = " › "


@dataclass
class Chunk:
    heading: str
    loc: str | None
    text: str


def _split_long(text: str, maxlen: int) -> list[str]:
    if len(text) <= maxlen:
        return [text]
    sentences = re.split(r"(?<=[.!?…;])\s+|\n+", text)
    out, cur = [], ""
    for s in sentences:
        while len(s) > maxlen:  # «предложение» без точек — режем по пробелу
            cut = s.rfind(" ", 0, maxlen)
            cut = cut if cut > maxlen // 2 else maxlen
            if cur:
                out.append(cur)
                cur = ""
            out.append(s[:cut].strip())
            s = s[cut:].strip()
        if cur and len(cur) + len(s) + 1 > CHUNK_TARGET:
            out.append(cur)
            cur = s
        else:
            cur = f"{cur} {s}".strip()
    if cur:
        out.append(cur)
    return out


def _units(b: Block) -> list[str]:
    if b.kind != "table":
        return _split_long(b.text, CHUNK_MAX)
    rows = b.text.split("\n")
    pieces, cur = [], []
    for r in rows:
        r = _split_long(r, CHUNK_MAX)[0] if len(r) > CHUNK_MAX else r
        size = sum(len(x) + 1 for x in cur)
        if cur and size + len(r) > CHUNK_TARGET:
            pieces.append(cur)
            cur = [b.header] if b.header and b.header != r else []
        cur.append(r)
    if cur:
        pieces.append(cur)
    return ["\n".join(p) for p in pieces]


def _common_prefix(a: tuple, b: tuple) -> tuple:
    n = 0
    while n < min(len(a), len(b)) and a[n] == b[n]:
        n += 1
    return a[:n]


def _loc_range(locs: list[str]) -> str | None:
    locs = [l for l in locs if l]
    if not locs:
        return None
    first, last = locs[0], locs[-1]
    if first == last:
        return first
    m1, m2 = re.search(r"\d+", first), re.search(r"\d+", last)
    if first.startswith("стр.") and m1 and m2:
        return f"стр. {m1.group()}–{m2.group()}"
    return f"{first} – {last}"


def chunk_blocks(blocks: list[Block]) -> list[Chunk]:
    chunks: list[Chunk] = []
    cur_path: tuple | None = None   # путь, общий для всего текущего чанка
    last_path: tuple | None = None  # путь последнего добавленного блока
    parts: list[str] = []
    locs: list[str] = []
    size = 0

    def flush():
        nonlocal parts, locs, size
        if parts:
            chunks.append(Chunk(SEP.join(cur_path or ()), _loc_range(locs), "\n".join(parts).strip()))
        parts, locs, size = [], [], 0

    for b in blocks:
        for u in _units(b):
            if cur_path is None:
                cur_path, last_path = b.path, b.path
            if b.path == last_path:
                if parts and size + len(u) > CHUNK_TARGET and size >= CHUNK_MIN or size + len(u) > CHUNK_MAX:
                    flush()
                    cur_path = b.path
            else:
                common = _common_prefix(cur_path, b.path)
                sub = b.path[len(common):]
                marker = f"### {SEP.join(sub)}" if sub else ""
                same_top = bool(common) or (not cur_path and not b.path)
                if parts and same_top and size + len(marker) + len(u) <= CHUNK_TARGET:
                    # мелкий соседний подраздел — приклеиваем с заголовком внутри текста
                    if cur_path != common and parts:
                        own = cur_path[len(common):]
                        if own and not parts[0].startswith("### "):
                            parts.insert(0, f"### {SEP.join(own)}")
                    cur_path = common
                    if marker:
                        parts.append(marker)
                        size += len(marker)
                else:
                    flush()
                    cur_path = b.path
            last_path = b.path
            parts.append(u)
            locs.append(b.loc)
            size += len(u) + 1
    flush()
    return chunks
