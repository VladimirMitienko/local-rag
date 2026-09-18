"""rag — командная строка локального RAG.

  rag index <папка|файл>... [-c коллекция]   добавить источники и проиндексировать
  rag update                                  переиндексировать изменённое во всех источниках
  rag search "вопрос" [-k 8] [-c ...] [-p ...] [--full] [--mode hybrid|dense|keyword]
  rag show <id> [--around N]                  показать чанк и соседей
  rag doc <часть_пути>                        вывести документ целиком (как его видит индекс)
  rag sources [-c ...]                        список документов
  rag stats                                   сводка по индексу
  rag remove <папка|файл>                     убрать из индекса
"""
from __future__ import annotations

import argparse
import json
import sys
import textwrap
import time
from pathlib import Path

from .config import DB_PATH

TTY = sys.stdout.isatty()


def _b(s: str) -> str:
    return f"\033[1m{s}\033[0m" if TTY else s


def _dim(s: str) -> str:
    return f"\033[2m{s}\033[0m" if TTY else s


def _index():
    from .store import Index
    return Index(DB_PATH)


def cmd_index(a):
    idx = _index()
    for p in a.paths:
        root = Path(p).expanduser().resolve()
        coll = a.collection or (root.name if root.is_dir() else root.parent.name)
        print(f"{_b('Индексирую')} {root}  [коллекция: {coll}]", file=sys.stderr)
        t = time.time()
        st = idx.index_root(root, coll, force=a.force)
        idx.add_root(root, coll)
        _report(st, time.time() - t)


def cmd_update(a):
    idx = _index()
    roots = idx.roots()
    if not roots:
        sys.exit("Источников нет. Сначала: rag index <папка>")
    for path, coll in roots:
        print(f"{_b('Обновляю')} {path}  [{coll}]", file=sys.stderr)
        t = time.time()
        try:
            st = idx.index_root(Path(path), coll, force=a.force)
        except (FileNotFoundError, PermissionError) as e:
            print(f"  ! {e}", file=sys.stderr)
            continue
        _report(st, time.time() - t)


def _report(st: dict, dt: float):
    print(f"  файлов: {st['files']}, новых: {st['new']}, обновлено: {st['updated']}, без изменений: "
          f"{st['unchanged']}, удалено: {st['removed']}, ошибок: {st['failed']}, чанков записано: {st['chunks']}"
          f"  ({dt:.1f} с)", file=sys.stderr)


def cmd_search(a):
    idx = _index()
    t = time.time()
    hits = idx.search(" ".join(a.query), k=a.k, collection=a.collection, path_contains=a.path, mode=a.mode)
    dt = time.time() - t
    if a.json:
        print(json.dumps([h.__dict__ for h in hits], ensure_ascii=False, indent=2))
        return
    if not hits:
        print("Ничего не найдено.")
        return
    for n, h in enumerate(hits, 1):
        ranks = f"вектор #{h.dense_rank or '—'}, слова #{h.kw_rank or '—'}"
        print(f"{_b(f'[{n}]')} {_b(h.source)}  {_dim(f'id={h.chunk_id} · {ranks}')}")
        body = h.text if a.full else textwrap.shorten(h.text.replace("\n", " "), 400, placeholder=" …")
        print(textwrap.indent(body, "    "), end="\n\n")
    print(_dim(f"{len(hits)} результатов за {dt:.2f} с. Полный текст: rag show <id> --around 1"))


def cmd_show(a):
    idx = _index()
    hits = idx.neighbours(a.id, a.around, a.around)
    if not hits:
        sys.exit(f"Чанк {a.id} не найден")
    print(_b(hits[0].path), end="\n\n")
    for h in hits:
        mark = "▶ " if h.chunk_id == a.id else "  "
        print(f"{mark}{_b(h.heading or '(без раздела)')}{' (' + h.loc + ')' if h.loc else ''}  {_dim(f'id={h.chunk_id}')}")
        print(textwrap.indent(h.text, "    "), end="\n\n")


def cmd_doc(a):
    idx = _index()
    hits = idx.document_chunks(a.path)
    if not hits:
        sys.exit("Документ не найден")
    paths = sorted({h.path for h in hits})
    if len(paths) > 1:
        print("Подходит несколько документов, уточните:\n" + "\n".join(f"  {p}" for p in paths))
        return
    print(_b(hits[0].title), "\n")
    for h in hits:
        print(f"{_dim(f'[{h.chunk_id}]')} {_b(h.heading or '')}{' (' + h.loc + ')' if h.loc else ''}")
        print(h.text, end="\n\n")


def cmd_sources(a):
    idx = _index()
    cur = None
    for coll, path, title, n, when, warn in idx.sources(a.collection):
        if coll != cur:
            print(_b(f"\n[{coll}]"))
            cur = coll
        print(f"  {n:>5} чанков  {path}" + (f"  ⚠ {warn}" if warn else ""))


def cmd_stats(a):
    s = _index().stats()
    print(f"Индекс: {DB_PATH}\nМодель: {s['model']}\nДокументов: {s['documents']}, чанков: {s['chunks']}")
    for coll, docs, chunks in s["collections"]:
        print(f"  [{coll}] документов: {docs}, чанков: {chunks}")
    print("Источники (rag update обходит их):")
    for path, coll in s["roots"]:
        print(f"  [{coll}] {path}")


def cmd_remove(a):
    n = _index().remove(Path(a.path).expanduser())
    print(f"Удалено из индекса документов: {n} (сами файлы не тронуты)")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="rag", description="Локальный RAG по документации",
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("index", help="добавить папки/файлы и проиндексировать")
    p.add_argument("paths", nargs="+")
    p.add_argument("-c", "--collection", help="имя коллекции (по умолчанию — имя папки)")
    p.add_argument("--force", action="store_true", help="переиндексировать всё, даже без изменений")
    p.set_defaults(fn=cmd_index)

    p = sub.add_parser("update", help="обновить индекс по всем источникам")
    p.add_argument("--force", action="store_true")
    p.set_defaults(fn=cmd_update)

    p = sub.add_parser("search", aliases=["s"], help="поиск")
    p.add_argument("query", nargs="+")
    p.add_argument("-k", type=int, default=8, help="сколько результатов (8)")
    p.add_argument("-c", "--collection")
    p.add_argument("-p", "--path", help="фильтр: подстрока пути файла")
    p.add_argument("--mode", choices=["hybrid", "dense", "keyword"], default="hybrid")
    p.add_argument("--full", action="store_true", help="печатать чанки целиком")
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_search)

    p = sub.add_parser("show", help="показать чанк по id с соседями")
    p.add_argument("id", type=int)
    p.add_argument("--around", type=int, default=1)
    p.set_defaults(fn=cmd_show)

    p = sub.add_parser("doc", help="вывести документ целиком")
    p.add_argument("path")
    p.set_defaults(fn=cmd_doc)

    p = sub.add_parser("sources", help="список документов")
    p.add_argument("-c", "--collection")
    p.set_defaults(fn=cmd_sources)

    sub.add_parser("stats", help="сводка").set_defaults(fn=cmd_stats)

    p = sub.add_parser("remove", help="убрать папку/файл из индекса")
    p.add_argument("path")
    p.set_defaults(fn=cmd_remove)

    a = ap.parse_args(argv)
    try:
        a.fn(a)
    except (FileNotFoundError, PermissionError) as e:
        sys.exit(f"Ошибка: {e}")
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":
    main()
