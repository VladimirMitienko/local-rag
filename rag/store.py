"""Индекс: SQLite (документы, чанки, эмбеддинги float16) + FTS5 по стеммированному тексту.
Поиск гибридный: косинус по эмбеддингам + BM25, объединение через Reciprocal Rank Fusion."""
from __future__ import annotations

import hashlib
import os
import re
import sqlite3
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import DB_PATH, MAX_SEQ_LEN, MODEL_NAME, SKIP_DIRS, SUPPORTED

# ---------------------------------------------------------------- стемминг для BM25

_WORD = re.compile(r"\w+")
_CAMEL = re.compile(r"[A-ZА-ЯЁ]?[a-zа-яё]+|[A-ZА-ЯЁ]+(?![a-zа-яё])|\d+")
_CYR = re.compile(r"[а-я]")
STOP = set("""
и в во не что он на я с со как а то все она так его но да ты к у же вы за бы по только ее мне было вот от меня
еще нет о из ему теперь когда даже ну ли если уже или ни быть был него до вас нибудь опять уж вам ведь там потом
себя ничего ей может они тут где есть надо ней для мы тебя их чем была сам чтоб без будто чего раз тоже себе под
будет ж тогда кто этот того потому этого какой какая какие каких совсем ним здесь этом один почти мой тем чтобы нее
сейчас были куда зачем всех никогда можно при об другой хоть после над больше тот через эти нас про всего них много
эту моя свою этой перед лучше том нельзя такой им более всегда между это как есть ли
the a an of to in is are be for on and or with how what which where when does do it this that by from as at
""".split())

_stemmers = {}


def _stem(word: str, lang: str) -> str:
    if lang not in _stemmers:
        import Stemmer
        _stemmers[lang] = Stemmer.Stemmer(lang)
    return _stemmers[lang].stemWord(word)


def stems(text: str, query: bool = False) -> list[str]:
    out = []
    for w in _WORD.findall(text):
        parts = [w]
        sub = _CAMEL.findall(w)
        if len(sub) > 1:
            parts += sub  # SumNodeFlows → sumnodeflows + sum, node, flows
        for p in parts:
            l = p.lower().replace("ё", "е")
            if query and l in STOP:
                continue
            if _CYR.search(l):
                l = _stem(l, "russian")
            elif l.isascii() and l.isalpha():
                l = _stem(l, "english")
            out.append(l)
    return out


# ---------------------------------------------------------------- эмбеддинги

class Embedder:
    def __init__(self, model_name: str = MODEL_NAME):
        self.model_name = model_name
        self._model = None

    def _load(self):
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        cache = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
        if (cache / f"models--{self.model_name.replace('/', '--')}").exists():
            os.environ.setdefault("HF_HUB_OFFLINE", "1")  # модель уже скачана — в сеть не ходим
        import torch
        import transformers
        from sentence_transformers import SentenceTransformer

        transformers.logging.set_verbosity_error()
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        m = SentenceTransformer(self.model_name, device=device)
        if device == "mps":
            m = m.half()
        m.max_seq_length = MAX_SEQ_LEN
        self._model = m

    def encode(self, texts: list[str], progress: bool = False, batch_size: int = 16) -> np.ndarray:
        if self._model is None:
            self._load()
        v = self._model.encode(texts, batch_size=batch_size, show_progress_bar=progress, convert_to_numpy=True)
        v = v.astype(np.float32)
        v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
        return v


# ---------------------------------------------------------------- индекс

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS roots(path TEXT PRIMARY KEY, collection TEXT NOT NULL, added_at TEXT);
CREATE TABLE IF NOT EXISTS documents(
    id INTEGER PRIMARY KEY, path TEXT UNIQUE NOT NULL, collection TEXT NOT NULL, title TEXT,
    sha1 TEXT, mtime REAL, size INTEGER, indexed_at TEXT, n_chunks INTEGER, warning TEXT);
CREATE TABLE IF NOT EXISTS chunks(
    id INTEGER PRIMARY KEY, doc_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    ord INTEGER NOT NULL, heading TEXT, loc TEXT, text TEXT NOT NULL, embedding BLOB NOT NULL);
CREATE INDEX IF NOT EXISTS chunks_doc ON chunks(doc_id, ord);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    stems, content='', contentless_delete=1, tokenize='unicode61 remove_diacritics 2');
"""


@dataclass
class Hit:
    chunk_id: int
    score: float
    path: str
    title: str
    collection: str
    heading: str
    loc: str | None
    text: str
    dense_rank: int | None
    kw_rank: int | None

    @property
    def source(self) -> str:
        s = Path(self.path).name
        if self.heading:
            s += f" › {self.heading}"
        if self.loc:
            s += f" ({self.loc})"
        return s


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _sha1(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


_UNDER = "(path=? OR substr(path, 1, ?)=?)"


def _under_args(root: Path) -> tuple:
    """Аргументы для _UNDER: сам путь или всё, что лежит внутри него (без LIKE — в путях бывает «_»)."""
    prefix = str(root).rstrip("/") + "/"
    return (str(root), len(prefix), prefix)


def _log(msg: str):
    print(msg, file=sys.stderr, flush=True)


class Index:
    def __init__(self, db_path: Path = DB_PATH, embedder: Embedder | None = None):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(db_path), check_same_thread=False)
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript(SCHEMA)
        self.embedder = embedder or Embedder()
        self._matrix: tuple[str, np.ndarray, np.ndarray] | None = None  # (version, ids, vectors)
        model = self._meta("model")
        if model is None:
            self._set_meta("model", self.embedder.model_name)
        elif model != self.embedder.model_name:
            raise SystemExit(f"Индекс построен моделью {model}, а сейчас задана {self.embedder.model_name}. "
                             f"Удалите {db_path} и переиндексируйте.")

    # -- meta
    def _meta(self, key):
        r = self.db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return r[0] if r else None

    def _set_meta(self, key, value):
        self.db.execute("INSERT OR REPLACE INTO meta VALUES(?,?)", (key, str(value)))

    def _bump(self):
        self._set_meta("version", f"{time.time():.6f}")

    # -- индексирование
    def add_root(self, path: Path, collection: str):
        self.db.execute("INSERT OR REPLACE INTO roots VALUES(?,?,?)", (str(path), collection, _now()))
        self.db.commit()

    def roots(self) -> list[tuple[str, str]]:
        return self.db.execute("SELECT path, collection FROM roots ORDER BY path").fetchall()

    @staticmethod
    def discover(root: Path) -> list[Path]:
        if root.is_file():
            return [root] if root.suffix.lower() in SUPPORTED else []
        files = []
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")]
            for fn in filenames:
                if fn.startswith((".", "~$")):
                    continue
                p = Path(dirpath) / fn
                if p.suffix.lower() in SUPPORTED:
                    files.append(p)
        return sorted(files)

    def index_root(self, root: Path, collection: str, force: bool = False) -> dict:
        from .chunker import chunk_blocks
        from .loaders import load

        root = root.resolve()
        if not root.exists():
            raise FileNotFoundError(f"{root} не существует (индекс не тронут)")
        try:
            if root.is_dir():
                os.listdir(root)
            else:
                open(root, "rb").close()
        except PermissionError:
            raise PermissionError(
                f"Нет доступа к {root}. macOS не даёт этому приложению читать папку — "
                "запустите индексацию из Терминала или выдайте доступ в «Конфиденциальность и безопасность» "
                "(индекс не тронут)") from None
        files = self.discover(root)
        known = {r[0]: r for r in self.db.execute(
            "SELECT path, id, sha1, mtime, size FROM documents WHERE " + _UNDER, _under_args(root))}
        stats = {"files": len(files), "new": 0, "updated": 0, "unchanged": 0, "removed": 0, "failed": 0, "chunks": 0}

        todo = []
        for f in files:
            st = f.stat()
            k = known.get(str(f))
            if k and not force and k[3] == st.st_mtime and k[4] == st.st_size:
                stats["unchanged"] += 1
                continue
            sha = _sha1(f)
            if k and not force and k[2] == sha:
                self.db.execute("UPDATE documents SET mtime=?, size=? WHERE id=?", (st.st_mtime, st.st_size, k[1]))
                stats["unchanged"] += 1
                continue
            todo.append((f, st, sha, k))

        for path in set(known) - {str(f) for f in files}:
            self._delete_doc(known[path][1])
            stats["removed"] += 1
            _log(f"  − удалён из индекса: {path}")

        for i, (f, st, sha, k) in enumerate(todo, 1):
            rel = f.relative_to(root) if f != root else f.name
            try:
                parsed = load(f)
                chunks = chunk_blocks(parsed.blocks)
            except Exception as e:  # битый/защищённый файл не должен ронять всю индексацию
                stats["failed"] += 1
                _log(f"  ! [{i}/{len(todo)}] {rel}: не удалось разобрать — {type(e).__name__}: {e}")
                continue
            t0 = time.time()
            vecs = self.embedder.encode(
                [f"{parsed.title}\n{c.heading}\n\n{c.text}" for c in chunks],
                progress=len(chunks) > 64,
            ) if chunks else np.zeros((0, 1024), np.float32)
            if k:
                self._delete_doc(k[1])
            cur = self.db.execute(
                "INSERT INTO documents(path, collection, title, sha1, mtime, size, indexed_at, n_chunks, warning) "
                "VALUES(?,?,?,?,?,?,?,?,?)",
                (str(f), collection, parsed.title, sha, st.st_mtime, st.st_size, _now(), len(chunks), parsed.warning))
            doc_id = cur.lastrowid
            for ord_, (c, v) in enumerate(zip(chunks, vecs)):
                cid = self.db.execute(
                    "INSERT INTO chunks(doc_id, ord, heading, loc, text, embedding) VALUES(?,?,?,?,?,?)",
                    (doc_id, ord_, c.heading, c.loc, c.text, v.astype(np.float16).tobytes())).lastrowid
                self.db.execute("INSERT INTO chunks_fts(rowid, stems) VALUES(?,?)",
                                (cid, " ".join(stems(f"{parsed.title} {c.heading} {c.text}"))))
            self._bump()
            self.db.commit()
            stats["updated" if k else "new"] += 1
            stats["chunks"] += len(chunks)
            warn = f"  ⚠ {parsed.warning}" if parsed.warning else ""
            _log(f"  + [{i}/{len(todo)}] {rel}: {len(chunks)} чанков, {time.time() - t0:.1f} с{warn}")
        self._bump()
        self.db.commit()
        return stats

    def _delete_doc(self, doc_id: int):
        ids = [r[0] for r in self.db.execute("SELECT id FROM chunks WHERE doc_id=?", (doc_id,))]
        self.db.executemany("DELETE FROM chunks_fts WHERE rowid=?", [(i,) for i in ids])
        self.db.execute("DELETE FROM documents WHERE id=?", (doc_id,))
        self._bump()

    def remove(self, path: Path) -> int:
        p = path.resolve()
        docs = self.db.execute("SELECT id FROM documents WHERE " + _UNDER, _under_args(p)).fetchall()
        for (d,) in docs:
            self._delete_doc(d)
        self.db.execute("DELETE FROM roots WHERE " + _UNDER, _under_args(p))
        self.db.commit()
        return len(docs)

    # -- поиск
    def _vectors(self):
        ver = self._meta("version") or "0"
        if self._matrix is None or self._matrix[0] != ver:
            rows = self.db.execute("SELECT id, embedding FROM chunks ORDER BY id").fetchall()
            ids = np.array([r[0] for r in rows], dtype=np.int64)
            mat = (np.frombuffer(b"".join(r[1] for r in rows), dtype=np.float16).reshape(len(rows), -1)
                   .astype(np.float32) if rows else np.zeros((0, 1024), np.float32))
            self._matrix = (ver, ids, mat)
        return self._matrix[1], self._matrix[2]

    def _allowed(self, collection: str | None, path_contains: str | None) -> set[int] | None:
        if not collection and not path_contains:
            return None
        docs = self._docs_matching(collection, path_contains)
        return {r[0] for r in self.db.execute(
            f"SELECT id FROM chunks WHERE doc_id IN ({','.join('?' * len(docs))})", docs)}

    def _docs_matching(self, collection: str | None, path_contains: str | None) -> list[int]:
        needle = (path_contains or "").casefold()
        return [d for d, coll, path in self.db.execute("SELECT id, collection, path FROM documents")
                if (not collection or coll == collection) and needle in path.casefold()]

    def _dense(self, query: str, n: int, allowed: set[int] | None) -> list[int]:
        ids, mat = self._vectors()
        if not len(ids):
            return []
        q = self.embedder.encode([query])[0]
        sims = mat @ q
        if allowed is not None:
            sims = np.where(np.isin(ids, list(allowed)), sims, -np.inf)
        top = np.argsort(-sims)[:n]
        return [int(ids[i]) for i in top if np.isfinite(sims[i])]

    def _keyword(self, query: str, n: int, allowed: set[int] | None) -> list[int]:
        terms = list(dict.fromkeys(stems(query, query=True)))
        if not terms:
            return []
        match = " OR ".join('"' + t.replace('"', "") + '"' for t in terms)
        rows = self.db.execute(
            "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ? ORDER BY bm25(chunks_fts) LIMIT ?",
            (match, n * 4 if allowed is not None else n)).fetchall()
        res = [r[0] for r in rows if allowed is None or r[0] in allowed]
        return res[:n]

    def search(self, query: str, k: int = 8, collection: str | None = None, path_contains: str | None = None,
               mode: str = "hybrid", candidates: int = 50) -> list[Hit]:
        allowed = self._allowed(collection, path_contains)
        dense = self._dense(query, candidates, allowed) if mode in ("hybrid", "dense") else []
        kw = self._keyword(query, candidates, allowed) if mode in ("hybrid", "keyword") else []
        d_rank = {cid: i for i, cid in enumerate(dense, 1)}
        k_rank = {cid: i for i, cid in enumerate(kw, 1)}
        scores: dict[int, float] = {}
        for ranks in (d_rank, k_rank):
            for cid, r in ranks.items():
                scores[cid] = scores.get(cid, 0.0) + 1.0 / (60 + r)
        best = sorted(scores, key=scores.get, reverse=True)[:k]
        return [self._hit(cid, scores[cid], d_rank.get(cid), k_rank.get(cid)) for cid in best]

    def _hit(self, cid: int, score: float = 0.0, dr=None, kr=None) -> Hit:
        r = self.db.execute(
            "SELECT d.path, d.title, d.collection, c.heading, c.loc, c.text FROM chunks c "
            "JOIN documents d ON d.id=c.doc_id WHERE c.id=?", (cid,)).fetchone()
        return Hit(cid, score, *r, dr, kr)

    def neighbours(self, chunk_id: int, before: int = 1, after: int = 1) -> list[Hit]:
        r = self.db.execute("SELECT doc_id, ord FROM chunks WHERE id=?", (chunk_id,)).fetchone()
        if not r:
            return []
        ids = self.db.execute("SELECT id FROM chunks WHERE doc_id=? AND ord BETWEEN ? AND ? ORDER BY ord",
                              (r[0], r[1] - before, r[1] + after)).fetchall()
        return [self._hit(i) for (i,) in ids]

    def document_chunks(self, path_contains: str) -> list[Hit]:
        docs = self._docs_matching(None, path_contains)
        ids = self.db.execute(
            f"SELECT c.id FROM chunks c JOIN documents d ON d.id=c.doc_id WHERE d.id IN ({','.join('?' * len(docs))}) "
            "ORDER BY d.path, c.ord", docs).fetchall()
        return [self._hit(i) for (i,) in ids]

    def sources(self, collection: str | None = None):
        sql = "SELECT collection, path, title, n_chunks, indexed_at, warning FROM documents"
        args = []
        if collection:
            sql += " WHERE collection=?"
            args.append(collection)
        return self.db.execute(sql + " ORDER BY collection, path", args).fetchall()

    def stats(self) -> dict:
        q = lambda s: self.db.execute(s).fetchone()[0]
        return {
            "model": self._meta("model"),
            "documents": q("SELECT count(*) FROM documents"),
            "chunks": q("SELECT count(*) FROM chunks"),
            "collections": self.db.execute(
                "SELECT collection, count(*), sum(n_chunks) FROM documents GROUP BY collection").fetchall(),
            "roots": self.roots(),
        }
