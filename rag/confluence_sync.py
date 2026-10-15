"""Выгрузка дерева страниц Confluence в HTML для RAG и индексация изменённого.

  rag-sync [--config sync.json] [--no-index] [--full]

Каждая страница под корневой сохраняется отдельным файлом «<название> [<pageId>].html»; в <title> —
цепочка разделов от корня. Скачиваются только новые/изменённые страницы (по номеру версии и пути),
удалённые из Confluence — удаляются с диска. Затем индексируется только эта папка.

Токен (PAT) берётся из переменной CONFLUENCE_TOKEN или из Связки ключей macOS (keychain_service
из конфига). В аргументы процессов и лог он не попадает: curl получает заголовок через stdin.
Запросы идут через curl, а не urllib: системный curl доверяет корпоративному CA из Связки ключей.
"""
from __future__ import annotations

import argparse
import html
import json
import os
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlencode

from .config import RAG_HOME

DEFAULT_CONFIG = RAG_HOME / "sync.json"
MANIFEST = ".sync-manifest.json"
# Версия формата файлов: при её смене все страницы перегенерируются (например, добавили извлечение PlantUML).
FORMAT = 2
NOTIFY_TITLE = "RAG: выгрузка Confluence"
WORKERS = 4
# Если страниц стало заметно меньше, чем в прошлый раз, — это скорее сбой поиска/прав, чем
# массовое удаление: файлы не удаляем, пока человек не посмотрит.
MAX_DROP = 0.3


def log(msg: str):
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}", file=sys.stderr, flush=True)


def notify(msg: str):
    """Системное уведомление macOS — только о сбоях; на других ОС просто пропускаем."""
    if sys.platform != "darwin":
        return
    esc = lambda t: t.replace("\\", "\\\\").replace('"', '\\"')
    subprocess.run(["osascript", "-e", f'display notification "{esc(msg[:220])}" with title "{esc(NOTIFY_TITLE)}"'],
                   capture_output=True)


_PUML = re.compile(r'<ac:structured-macro[^>]*ac:name="plantuml"[^>]*>(.*?)</ac:structured-macro>', re.S)
_PUML_TITLE = re.compile(r'<ac:parameter ac:name="title">(.*?)</ac:parameter>', re.S)
_PUML_BODY = re.compile(r"<ac:plain-text-body><!\[CDATA\[(.*?)\]\]></ac:plain-text-body>", re.S)


def _plantuml_sources(storage: str) -> list[tuple[str, str]]:
    out = []
    for m in _PUML.finditer(storage):
        body = _PUML_BODY.search(m.group(1))
        if body and body.group(1).strip():
            title = _PUML_TITLE.search(m.group(1))
            out.append((html.unescape(title.group(1)) if title else "", body.group(1).strip()))
    return out


def _inject_plantuml(view: str, storage: str) -> str:
    """export_view отдаёт PlantUML только картинкой. Подставляем исходник диаграммы текстом
    (<pre> сразу после абзаца с картинкой), чтобы сущности, поля и связи находились поиском."""
    sources = _plantuml_sources(storage)
    if not sources:
        return view
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(view, "lxml")
    imgs = [i for i in soup.find_all("img") if "plantuml" in (i.get("src") or "")]

    def block(title, code):
        pre = soup.new_tag("pre")
        pre.string = (f"Диаграмма PlantUML «{title}» (исходник):\n" if title else
                      "Диаграмма PlantUML (исходник):\n") + code
        return pre

    if len(imgs) == len(sources):
        for img, (title, code) in zip(imgs, sources):
            anchor = img.find_parent(["p", "li", "td", "th"]) or img
            if anchor.name == "p":
                anchor.insert_after(block(title, code))
            else:  # в ячейке/списке — прямо на место картинки
                img.replace_with(block(title, code))
    else:  # не удалось сопоставить картинки с макросами — все исходники в конец страницы
        body = soup.body or soup
        h = soup.new_tag("h1")
        h.string = "Диаграммы PlantUML"
        body.append(h)
        for title, code in sources:
            body.append(block(title, code))
    return "".join(str(c) for c in (soup.body.contents if soup.body else soup.contents))


def get_token(cfg: dict) -> str:
    tok = os.environ.get("CONFLUENCE_TOKEN")
    if tok:
        return tok.strip()
    if sys.platform != "darwin":
        raise RuntimeError("задайте токен в переменной окружения CONFLUENCE_TOKEN")
    r = subprocess.run(["security", "find-generic-password", "-s", cfg["keychain_service"], "-w"],
                       capture_output=True, text=True)
    if r.returncode != 0 or not r.stdout.strip():
        raise RuntimeError(f"токен не найден в Связке ключей (service «{cfg['keychain_service']}»): "
                           f"security add-generic-password -s {cfg['keychain_service']} -a \"$USER\" -w")
    return r.stdout.strip()


class Confluence:
    def __init__(self, base_url: str, token: str):
        self.base = base_url.rstrip("/")
        self._curl = shutil.which("curl") or "/usr/bin/curl"
        self._headers = (f'header = "Authorization: Bearer {token}"\n'
                         'header = "Accept: application/json"\n')

    def get(self, path: str, params: dict | None = None, retries: int = 3) -> dict:
        url = f"{self.base}/rest/api{path}" + ("?" + urlencode(params) if params else "")
        for attempt in range(1, retries + 1):
            r = subprocess.run([self._curl, "-sS", "--fail-with-body", "--max-time", "120", "-K", "-", url],
                               input=self._headers, capture_output=True, text=True)
            if r.returncode == 0:
                return json.loads(r.stdout)
            err = " ".join((r.stderr or r.stdout).split())[:300]
            if attempt == retries:
                raise RuntimeError(f"GET {path}: {err}")
            time.sleep(3 * attempt)

    def page_url(self, page_id: str) -> str:
        return f"{self.base}/pages/viewpage.action?pageId={page_id}"

    def list_tree(self, root_id: str) -> list[dict]:
        """Все страницы под корнем (CQL ancestor), с версией и предками. Повтор, если при
        постраничной выдаче что-то потерялось (страницы правят прямо во время обхода)."""
        for attempt in range(2):
            seen: dict[str, dict] = {}
            start, total = 0, None
            while True:
                d = self.get("/content/search", {"cql": f"ancestor={root_id} and type=page", "limit": 200,
                                                 "start": start, "expand": "version,ancestors"})
                total = d.get("totalSize", total)
                for p in d["results"]:
                    seen[p["id"]] = p
                if d.get("size", 0) == 0 or not d.get("_links", {}).get("next"):
                    break
                start += d["size"]
            if total is None or len(seen) >= total:
                return list(seen.values())
            log(f"  получено {len(seen)} из {total} страниц — повторяю обход")
        raise RuntimeError(f"не удалось получить полный список страниц ({len(seen)} из {total})")


def _safe_name(title: str, page_id: str) -> str:
    name = re.sub(r'[\\/:*?"<>|\n\r\t]', "_", title).strip(" .")
    return f"{name[:150]} [{page_id}].html"


def _breadcrumb(page: dict, root_id: str) -> list[str]:
    anc = page.get("ancestors", [])
    ids = [a["id"] for a in anc]
    below = anc[ids.index(root_id) + 1:] if root_id in ids else anc
    return [a["title"] for a in below] + [page["title"]]


def _render(cf: Confluence, page: dict, crumbs: list[str], body: str) -> str:
    v = page["version"]
    when = v.get("when", "")[:10]
    head = " › ".join(crumbs)
    return (f'<!DOCTYPE html>\n<html><head><meta charset="utf-8"><title>{html.escape(head)}</title></head>\n'
            f'<body>\n<p>Страница Confluence «{html.escape(page["title"])}», pageId {page["id"]}, '
            f'версия {v["number"]} от {when}: {cf.page_url(page["id"])}</p>\n{body}\n</body></html>\n')


def sync_root(cf: Confluence, root: dict, out: Path, full: bool = False) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    mf_path = out / MANIFEST
    old = json.loads(mf_path.read_text(encoding="utf-8")) if mf_path.exists() else {}
    old_pages: dict = old.get("pages", {})
    if old.get("format") != FORMAT:
        full = True  # формат файлов поменялся — перегенерируем всё
        if old_pages:
            log(f"  формат выгрузки обновлён до {FORMAT} — страницы будут перегенерированы")

    pages = cf.list_tree(root["page_id"])
    if not pages:
        raise RuntimeError("под корневой страницей не найдено ни одной страницы (файлы не тронуты)")

    new_pages, todo = {}, []
    for p in pages:
        crumbs = _breadcrumb(p, root["page_id"])
        rec = {"title": p["title"], "version": p["version"]["number"], "breadcrumb": crumbs,
               "file": _safe_name(p["title"], p["id"]), "when": p["version"].get("when", "")}
        new_pages[p["id"]] = rec
        prev = old_pages.get(p["id"])
        if (full or not prev or prev["version"] != rec["version"] or prev["file"] != rec["file"]
                or prev["breadcrumb"] != rec["breadcrumb"] or not (out / rec["file"]).exists()):
            todo.append((p, rec, prev))

    def fetch(item):
        p, rec, prev = item
        d = cf.get(f"/content/{p['id']}", {"expand": "body.export_view,body.storage,version"})
        body = _inject_plantuml(d["body"]["export_view"]["value"], d["body"]["storage"]["value"])
        tmp = out / f".{rec['file']}.tmp"
        tmp.write_text(_render(cf, d, rec["breadcrumb"], body), encoding="utf-8")
        os.replace(tmp, out / rec["file"])
        if prev and prev["file"] != rec["file"]:
            (out / prev["file"]).unlink(missing_ok=True)  # страницу переименовали
        return rec

    stats = {"pages": len(new_pages), "new": 0, "updated": 0, "removed": 0, "failed": 0,
             "failed_ids": [], "new_failures": [], "drop_guard": False}
    with ThreadPoolExecutor(WORKERS) as ex:
        for (p, rec, prev), fut in zip(todo, [ex.submit(fetch, t) for t in todo]):
            try:
                fut.result()
                stats["updated" if prev else "new"] += 1
            except Exception as e:  # одна битая страница не должна срывать всю выгрузку
                stats["failed"] += 1
                stats["failed_ids"].append(p["id"])
                if p["id"] not in old.get("failed", []):
                    stats["new_failures"].append(p["title"])
                log(f"  ! {p['title']} [{p['id']}]: {e}")
                if prev:
                    new_pages[p["id"]] = prev  # оставляем прошлую версию, попробуем в следующий раз
                else:
                    new_pages.pop(p["id"], None)

    gone = set(old_pages) - set(new_pages)
    if old_pages and len(gone) > MAX_DROP * len(old_pages):
        log(f"  ⚠ пропало {len(gone)} из {len(old_pages)} страниц — похоже на сбой, удаление пропущено")
        stats["drop_guard"] = True
        for pid in gone:
            new_pages[pid] = old_pages[pid]
    else:
        for pid in gone:
            (out / old_pages[pid]["file"]).unlink(missing_ok=True)
            stats["removed"] += 1
            log(f"  − {old_pages[pid]['title']} [{pid}]")

    tmp = out / f"{MANIFEST}.tmp"
    tmp.write_text(json.dumps({"format": FORMAT, "root_id": root["page_id"],
                               "synced_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "failed": stats["failed_ids"],
                               "pages": new_pages}, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, mf_path)
    stats["unchanged"] = stats["pages"] - stats["new"] - stats["updated"] - stats["failed"]
    return stats


def _lock(path: Path):
    """Не даём двум выгрузкам идти одновременно (ручной запуск во время планового)."""
    try:
        import fcntl
    except ImportError:  # Windows
        return None
    f = open(path, "w")
    try:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("выгрузка уже идёт в другом процессе")
    return f


def main(argv=None):
    ap = argparse.ArgumentParser(prog="rag-sync", description="Выгрузка страниц Confluence в RAG")
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--no-index", action="store_true", help="только скачать, не индексировать")
    ap.add_argument("--full", action="store_true", help="перекачать все страницы, даже без изменений")
    a = ap.parse_args(argv)

    if not a.config.exists():
        sys.exit(f"Нет конфига {a.config} (образец — sync.example.json)")
    cfg = json.loads(a.config.read_text(encoding="utf-8"))
    out_base = (RAG_HOME / cfg["out_dir"]).resolve()
    out_base.mkdir(parents=True, exist_ok=True)
    lock = _lock(out_base / ".sync.lock")

    t0 = time.time()
    log("Синхронизация Confluence → RAG")
    try:
        cf = Confluence(cfg["base_url"], get_token(cfg))
        results = []
        for root in cfg["roots"]:
            out = out_base / root["name"]
            log(f"[{root['name']}] pageId {root['page_id']}")
            st = sync_root(cf, root, out, full=a.full)
            log(f"  страниц: {st['pages']}, новых: {st['new']}, обновлено: {st['updated']}, "
                f"удалено: {st['removed']}, без изменений: {st['unchanged']}, ошибок: {st['failed']}")
            results.append((out, st))
    except Exception as e:
        log(f"Ошибка: {e}. Индекс не тронут, повтор при следующем запуске.")
        notify(f"Не удалась: {e}"[:220])
        sys.exit(1)

    # Уведомляем только о том, что требует внимания: новые недоступные страницы (не те, что падают
    # каждый день) и срабатывание защиты от массового удаления.
    for out, st in results:
        if st["new_failures"]:
            notify(f"{out.name}: не скачалось страниц — {len(st['new_failures'])}: "
                   + "; ".join(st["new_failures"][:3]))
        if st["drop_guard"]:
            notify(f"{out.name}: из Confluence пропала заметная часть страниц — удаление пропущено, см. лог")

    if not a.no_index:
        from .store import Index
        idx = Index()
        for out, _ in results:
            try:
                st = idx.index_root(out, cfg["collection"])
            except Exception as e:
                log(f"  Ошибка индексации {out.name}: {e}")
                notify(f"{out.name}: страницы скачаны, но индексация не удалась: {e}"[:220])
                sys.exit(1)
            if st["failed"]:
                notify(f"{out.name}: не удалось проиндексировать файлов — {st['failed']}, см. лог")
            idx.add_root(out, cfg["collection"])
            log(f"  индекс [{cfg['collection']}]: новых {st['new']}, обновлено {st['updated']}, "
                f"удалено {st['removed']}, ошибок {st['failed']}, чанков {st['chunks']}")
    log(f"Готово за {time.time() - t0:.0f} с")
    if lock:
        lock.close()


if __name__ == "__main__":
    main()
