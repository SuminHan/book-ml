#!/usr/bin/env python3
"""Local, zero-dependency web tool for page-by-page slide review.

Browse each deck's PDF one page at a time and leave a short instruction
("이 페이지에 이런 내용/이미지 넣어줘") per page. Notes are saved next to
each deck's PDF as `<deck>.review.json`, keyed by page number, so they can
be read back later and applied to the .tex source.

/board is a second view (storyboard): existing pages as draggable thumbnail
cards, reorderable, with an inline "+ 다음에 추가" to drop in a blank
placeholder card carrying just a text prompt for content that doesn't exist
yet, and a "삭제" toggle on each page card to mark it for removal. None of
this touches the .tex/PDF -- it's all staged in `<deck>.plan.json` and the
board visibly marks anything not yet applied: moved cards (an item's id
encodes the page it was at when last "committed"; a mismatch with its
current computed page means it's been dragged since), insert placeholders,
and delete-marked cards all render as pending, plus a banner if any exist.

Applying a plan (by Claude, in a later session):
    1. Read `<deck>.review.json` for the pending per-page notes.
    2. Read `<deck>.plan.json` for the target page order, any pending
       "insert" placeholders (blank-page prompts), and any page items with
       "deleted": true.
    3. Edit the .tex source: reorder frames to match the plan's page-item
       order, delete the frames marked deleted, write a new frame for each
       insert prompt at its position, satisfy per-page notes. Rebuild.
    4. Resolve each applied item so it stops showing as pending and the
       request is permanently logged to `<deck>.review_log.jsonl` (log
       written *before* the pending entry is removed, so nothing is lost if
       something goes wrong mid-apply):
         - POST /api/resolve {"id","page","action"} per note
         - POST /api/plan/resolve_insert {"id","item_id","action"} per
           inserted placeholder now baked into a real frame
         - POST /api/plan/resolve_delete {"id","item_id","action"} per
           deleted frame
         - POST /api/plan/commit {"id","action"} once, last, after the
           rebuilt deck's order actually matches the plan -- this resets
           every remaining page item's origin id to its new position, which
           is what clears the "moved (미반영)" pending indicator on the
           board. Skipping this leaves cards permanently flagged as moved
           even though the reorder is already live.

"반영하기" button (/deck and /board headers): shows a preview of the pending
items and the exact prompt, then on 실행 spawns a fresh headless
`claude -p --model <haiku|sonnet|opus>` session in the deck folder that does
the steps above for that deck only. Progress streams into the page's log
panel; the page reloads when the session ends.

On startup, every deck's pages are pre-rendered in the background (pass
--no-warm to skip) so clicking into any deck's /deck or /board is instant
instead of waiting on pdftoppm for that deck's first view.

Usage:
    python3 tools/slide_review_server.py [--port 8765] [--no-warm]
    open http://localhost:8765
"""
import argparse
import base64
import json
import shutil
import os
import re
import secrets
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from string import Template
from urllib.parse import parse_qs, quote, unquote, urlparse

ROOT = Path(__file__).resolve().parent.parent
SLIDES_DIR = ROOT / "slides"
CACHE_DIR = ROOT / ".review_cache"
EXCLUDE_DIR_NAMES = {"exams", "quizzes"}  # sensitive drafts, skip in the review tool
RENDER_DPI = 200
WATCH_INTERVAL = 0.5  # seconds -- how often to poll for tex file changes

_render_locks: dict[str, threading.Lock] = {}
_render_locks_guard = threading.Lock()

# did -> (pdf_mtime_when_counted, page_count). Avoids shelling out to pdfinfo
# (a fresh subprocess) on every single page click once a deck's page count is
# known -- only re-checked when the PDF's own mtime moves (i.e. a rebuild).
_page_count_cache: dict[str, tuple[float, int]] = {}
_page_count_cache_guard = threading.Lock()

# tex file -> mtime at last check. The watcher polls this and rebuilds any
# deck whose tex files have a newer mtime than recorded.
_tex_watch_state: dict[Path, float] = {}
_tex_watch_lock = threading.Lock()


def find_decks() -> list[str]:
    decks = []
    for pdf in sorted(SLIDES_DIR.rglob("*.pdf")):
        rel = pdf.relative_to(SLIDES_DIR)
        if EXCLUDE_DIR_NAMES & set(rel.parts):
            continue
        if "figs" in rel.parts or "assets" in rel.parts or "gen" in rel.parts:
            continue
        decks.append(str(rel.with_suffix("")))
    return decks


def deck_pdf(did: str) -> Path:
    return SLIDES_DIR / f"{did}.pdf"


def deck_tex(did: str) -> Path:
    return SLIDES_DIR / f"{did}.tex"


def deck_pages_dir(did: str) -> Path:
    return deck_tex(did).parent / "pages"


def collect_tex_files() -> dict[str, Path]:
    """Map did -> main .tex file for all decks. Used by the file watcher."""
    result = {}
    for did in find_decks():
        tex = deck_tex(did)
        if tex.exists():
            result[did] = tex
    return result


def start_tex_watcher() -> threading.Thread:
    """Start a background thread that polls for .tex file changes and
    automatically rebuilds affected decks. Returns the daemon thread."""
    def watch_loop():
        last_seen = {}  # did -> max mtime of tex + pages/*.tex
        print("Tex file watcher started (interval=%.1fs)" % WATCH_INTERVAL)
        while True:
            time.sleep(WATCH_INTERVAL)
            try:
                decks = find_decks()
                for did in decks:
                    tex = deck_tex(did)
                    if not tex.exists():
                        continue
                    job = _apply_jobs.get(did)
                    if job and job["status"] == "running":
                        continue  # the apply session builds it itself; rebuild once after
                    # Collect mtime of main tex + all pages/*.tex
                    max_mtime = tex.stat().st_mtime
                    pages_dir = deck_pages_dir(did)
                    if pages_dir.is_dir():
                        for pf in pages_dir.glob("p*.tex"):
                            try:
                                max_mtime = max(max_mtime, pf.stat().st_mtime)
                            except OSError:
                                pass
                    prev = last_seen.get(did, 0.0)
                    if max_mtime > prev:
                        last_seen[did] = max_mtime
                        print(f"[watcher] tex changed: {did}")
                        try:
                            ok, log = rebuild_deck(did)
                            if ok:
                                print(f"[watcher] rebuild OK: {did}")
                            else:
                                print(f"[watcher] rebuild FAILED: {did}\n{log[-2000:]}")
                        except Exception as e:
                            print(f"[watcher] rebuild error: {did}: {e}")
            except Exception as e:
                print(f"[watcher] scan error: {e}")

    t = threading.Thread(target=watch_loop, daemon=True)
    t.start()
    return t


def page_source_files(did: str) -> list:
    """Sorted list of a migrated deck's pages/pNN.tex files (zero-padded, so
    lexicographic sort == numeric page order). Empty if not migrated."""
    d = deck_pages_dir(did)
    if not d.is_dir():
        return []
    return sorted(d.glob("p*.tex"))


def page_source_path(did: str, page: int):
    """The pages/*.tex file that produced PDF page `page`, from the build's
    .nav (see page_file_map). Sorting the pages/ folder by name is NOT the
    page order: auto section-divider pages, files dropped from \\input but
    left on disk, and inserted p11b/p09g1-style files all shift it."""
    hit = page_file_map(did).get(page)
    return deck_tex(did).parent / hit[0] if hit else None


def source_file_by_name(did: str, rel: str):
    """A pages/*.tex path the client loaded earlier, validated to stay inside pages/."""
    pages = deck_pages_dir(did).resolve()
    path = (deck_tex(did).parent / rel).resolve()
    if path.parent != pages or path.suffix != ".tex" or not path.exists():
        return None
    return path


def rebuild_deck(did: str) -> tuple:
    """Recompile a deck's main .tex with tectonic. Returns (ok, log_tail).

    Passes --keep-intermediates so the beamer .nav file survives the build --
    that's where refresh_sections() reads each \\section{}'s start page from.
    """
    tex = deck_tex(did)
    proc = subprocess.run(
        ["tectonic", "-X", "compile", "--keep-intermediates", tex.name],
        cwd=tex.parent,
        capture_output=True,
        text=True,
    )
    ok = proc.returncode == 0
    log = (proc.stdout or "") + (proc.stderr or "")
    if ok:
        refresh_sections(did)
        # The rebuild just invalidated this deck's page images. Re-render them now,
        # in the background, so the next click doesn't pay a synchronous pdftoppm
        # pass over the whole deck.
        threading.Thread(
            target=lambda: ensure_rendered(did), daemon=True
        ).start()
    return ok, log[-4000:]  # tail only -- tectonic logs can be long


def sections_path(did: str) -> Path:
    return SLIDES_DIR / f"{did}.sections.json"


_NAV_SECTION_RE = re.compile(
    r"\\sectionentry\s*\{\d+\}\{(.*?)\}\{(\d+)\}\{.*?\}\{\d+\}"
)


def refresh_sections(did: str) -> list:
    """Parse <deck>.nav (written by the --keep-intermediates build) for
    \\sectionentry marks and cache {title, page} per section as
    <deck>.sections.json, next to the existing .review.json/.plan.json.

    Each deck's 3 \\section{} calls in pages/ are its 3 차시 (class
    sessions) -- this is how the viewer groups the jumplist by session
    without any separate per-deck session config to maintain."""
    nav = deck_tex(did).with_suffix(".nav")
    sections = []
    if nav.exists():
        for m in _NAV_SECTION_RE.finditer(nav.read_text(encoding="utf-8", errors="replace")):
            title, page = m.group(1), int(m.group(2))
            sections.append({"title": title, "page": page})
    sections_path(did).write_text(
        json.dumps(sections, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return sections


def load_sections(did: str) -> list:
    p = sections_path(did)
    if not p.exists():
        return []
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []


def write_page_source(did: str, page: int, content: str, rebuild: bool = True, file: str = "") -> tuple:
    """Overwrite one page's source file, optionally rebuilding the deck.

    With rebuild=True (the default): on a failed build, restores the
    previous content so the deck stays in a known-good, buildable state --
    the caller's edit is never silently left half-applied.

    With rebuild=False: just writes the file and returns immediately --
    the PDF/cached images are left as-is (showing the pre-edit content)
    until a later save actually rebuilds. Useful for iterating on a page
    without paying the tectonic cost on every keystroke-adjacent save.

    `file` (the pages/… path returned when the source was loaded) wins over
    `page`, so a rebuild that shifts page numbers between load and save can
    never redirect the write into a different frame's file.

    Returns (ok, error_message)."""
    path = source_file_by_name(did, file) if file else page_source_path(did, page)
    if path is None:
        return False, "이 쪽의 소스 파일을 찾지 못함 (빌드 기록과 소스가 어긋남 — 재빌드 후 다시 열어 보세요)"
    old = path.read_text(encoding="utf-8")
    if not content.endswith("\n"):
        content += "\n"
    path.write_text(content, encoding="utf-8")
    if not rebuild:
        return True, ""
    ok, log = rebuild_deck(did)
    if not ok:
        path.write_text(old, encoding="utf-8")  # revert -- keep the deck buildable
        rebuild_deck(did)  # restore the PDF to match the reverted source
        return False, log
    return True, ""


def deck_figs_dir(did: str) -> Path:
    return (SLIDES_DIR / did).parent / "figs"


_DATA_URL_RE = re.compile(r"^data:image/(?P<ext>png|jpeg|jpg|gif|webp);base64,(?P<b64>.+)$", re.DOTALL)


def save_pasted_image(did: str, page: int, data_url: str) -> str:
    """Save a clipboard-pasted image straight into the deck's figs/ dir (the
    same place URL-downloaded reference images end up) so it's immediately
    \\includegraphics-able. Returns the saved filename."""
    m = _DATA_URL_RE.match(data_url.strip())
    if not m:
        raise ValueError("unsupported image data URL")
    ext = "jpg" if m.group("ext") == "jpeg" else m.group("ext")
    raw = base64.b64decode(m.group("b64"))
    fdir = deck_figs_dir(did)
    fdir.mkdir(parents=True, exist_ok=True)
    name = f"pasted_p{page}_{secrets.token_hex(4)}.{ext}"
    (fdir / name).write_bytes(raw)
    return name


def notes_path(did: str) -> Path:
    return SLIDES_DIR / f"{did}.review.json"


def log_path(did: str) -> Path:
    return SLIDES_DIR / f"{did}.review_log.jsonl"


def plan_path(did: str) -> Path:
    return SLIDES_DIR / f"{did}.plan.json"


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def cache_dir_for(did: str) -> Path:
    # DPI baked into the path so bumping RENDER_DPI auto-invalidates old renders
    return CACHE_DIR / f"{did.replace('/', '__')}__{RENDER_DPI}dpi"


def pdf_page_count(pdf: Path) -> int:
    out = subprocess.run(
        ["pdfinfo", str(pdf)], capture_output=True, text=True, check=True
    ).stdout
    m = re.search(r"Pages:\s+(\d+)", out)
    return int(m.group(1))


# Page images are double-buffered: a new build renders into "<cache>.tmp" in
# the background and is swapped in only when complete, so while a deck
# rebuilds the viewer keeps serving the previous render instead of blocking
# every page click on a full-deck pdftoppm. meta.json records which PDF
# (mtime) a cache dir holds.
_render_running: set = set()


def _cache_meta(did: str):
    try:
        m = json.loads((cache_dir_for(did) / "meta.json").read_text())
        return m["mtime"], m["n"]
    except (OSError, ValueError, KeyError):
        return None


def _render_into_cache(did: str) -> bool:
    """Render the current PDF into a temp dir, then swap it in. False if the
    PDF was still being written (or changed mid-render) -- caller retries."""
    pdf = deck_pdf(did)
    mtime = pdf.stat().st_mtime
    cache = cache_dir_for(did)
    tmp = cache.with_name(cache.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    try:
        n = pdf_page_count(pdf)
        subprocess.run(["pdftoppm", "-png", "-r", str(RENDER_DPI), str(pdf), str(tmp / "page")],
                       check=True, capture_output=True)
    except (subprocess.CalledProcessError, AttributeError):
        return False
    if pdf.stat().st_mtime != mtime:
        return False
    (tmp / "meta.json").write_text(json.dumps({"mtime": mtime, "n": n}))
    old = cache.with_name(cache.name + ".old")
    if old.exists():
        shutil.rmtree(old)
    if cache.exists():
        cache.rename(old)
    tmp.rename(cache)
    if old.exists():
        shutil.rmtree(old)
    with _page_count_cache_guard:
        _page_count_cache[did] = (mtime, n)
    return True


def _render_lock(did: str) -> threading.Lock:
    with _render_locks_guard:
        return _render_locks.setdefault(did, threading.Lock())


def _render_until_current(did: str) -> None:
    pdf = deck_pdf(did)
    with _render_lock(did):
        for _ in range(60):
            meta = _cache_meta(did)
            if meta and meta[0] == pdf.stat().st_mtime:
                return
            # let tectonic finish writing before reading the PDF
            m0 = pdf.stat().st_mtime
            time.sleep(0.6)
            if pdf.stat().st_mtime != m0:
                continue
            if not _render_into_cache(did):
                time.sleep(1.0)


def _kick_render(did: str) -> None:
    with _render_locks_guard:
        if did in _render_running:
            return
        _render_running.add(did)

    def run():
        try:
            _render_until_current(did)
        finally:
            with _render_locks_guard:
                _render_running.discard(did)

    threading.Thread(target=run, daemon=True).start()


def _adopt_legacy_cache(did: str) -> None:
    """Caches from before meta.json: trust them if their last page is newer than the PDF."""
    pdf = deck_pdf(did)
    cache = cache_dir_for(did)
    pngs = sorted(cache.glob("page-*.png")) if cache.is_dir() else []
    if pngs and pngs[-1].stat().st_mtime >= pdf.stat().st_mtime:
        (cache / "meta.json").write_text(json.dumps({"mtime": pdf.stat().st_mtime, "n": len(pngs)}))


def ensure_rendered(did: str) -> int:
    """Page count of the render being served. Never blocks while an older
    render exists; only the very first render of a deck is synchronous."""
    pdf = deck_pdf(did)
    if not pdf.exists():
        raise FileNotFoundError(did)
    meta = _cache_meta(did)
    if meta is None:
        _adopt_legacy_cache(did)
        meta = _cache_meta(did)
    if meta and meta[0] == pdf.stat().st_mtime:
        return meta[1]
    if meta:
        _kick_render(did)
        return meta[1]
    _render_until_current(did)
    meta = _cache_meta(did)
    if not meta:
        raise FileNotFoundError(did)
    return meta[1]


def render_status(did: str) -> dict:
    meta = _cache_meta(did)
    pdf = deck_pdf(did)
    current = bool(meta) and meta[0] == pdf.stat().st_mtime
    if not current:
        _kick_render(did)
    return {"version": meta[0] if meta else 0, "n": meta[1] if meta else 0,
            "updating": not current or did in _render_running}


def page_image_path(did: str, page: int, total: int) -> Path:
    width = len(str(total))
    return cache_dir_for(did) / f"page-{page:0{width}d}.png"


def _normalize_entry(entry) -> dict:
    # tolerate the original plain-string format (pre-log-feature files)
    if isinstance(entry, str):
        return {"note": entry, "created_at": None, "updated_at": None}
    return entry


def load_notes(did: str) -> dict:
    p = notes_path(did)
    if not p.exists():
        return {}
    raw = json.loads(p.read_text(encoding="utf-8"))
    return {k: _normalize_entry(v) for k, v in raw.items()}


def save_note(did: str, page: int, note: str) -> None:
    p = notes_path(did)
    data = load_notes(did)
    note = note.strip()
    key = str(page)
    if note:
        existing = data.get(key) or {}
        data[key] = {
            "note": note,
            "created_at": existing.get("created_at") or now_iso(),
            "updated_at": now_iso(),
        }
    else:
        data.pop(key, None)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def resolve_note(did: str, page: int, action: str) -> bool:
    """Permanently log what was done about a pending note, then remove it.

    Log-append happens first and is the durable record: if anything below it
    fails, the note is still sitting in review.json (worst case: reprocessed
    once more), never silently gone with no trace.
    """
    data = load_notes(did)
    key = str(page)
    entry = data.get(key)
    if entry is None:
        return False
    log_entry = {
        "page": page,
        "note": entry.get("note", ""),
        "created_at": entry.get("created_at"),
        "updated_at": entry.get("updated_at"),
        "resolved_at": now_iso(),
        "action": action.strip(),
    }
    lp = log_path(did)
    lp.parent.mkdir(parents=True, exist_ok=True)
    with lp.open("a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
    data.pop(key, None)
    p = notes_path(did)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return True


def _origin_page(item: dict):
    """The real, current PDF page an item's id was assigned at last commit."""
    m = re.fullmatch(r"p(\d+)", item.get("id", ""))
    return int(m.group(1)) if m else None


def identity_plan(total: int) -> list:
    return [{"id": f"p{n}", "type": "page"} for n in range(1, total + 1)]


def load_plan(did: str) -> list:
    """Return the stored plan as-is (plus a "page" field, always freshly
    re-derived from each page item's id -- never trusted from storage).

    "page" is the item's stable reference to a real PDF page: it stays put
    while the item is only dragged around (the plan array's order is a
    separate, purely positional thing -- computed live wherever it's
    needed, never stored). Only commit_plan() ever changes "page" /
    the underlying id, and only after the .tex has actually been edited to
    match and rebuilt.
    """
    p = plan_path(did)
    if p.exists():
        stored = json.loads(p.read_text(encoding="utf-8"))
    else:
        stored = identity_plan(ensure_rendered(did))
    result = []
    for it in stored:
        if it.get("type") == "page":
            it = {**it, "page": _origin_page(it)}
        result.append(it)
    return result


def save_plan(did: str, items: list) -> None:
    p = plan_path(did)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")


def resolve_insert(did: str, item_id: str, action: str) -> bool:
    """Permanently log what was done about an insert placeholder, then drop it.

    Same durability order as resolve_note: log first, remove second.
    """
    p = plan_path(did)
    if not p.exists():
        return False
    stored = json.loads(p.read_text(encoding="utf-8"))
    target = None
    remaining = []
    for it in stored:
        if target is None and it.get("type") == "insert" and it.get("id") == item_id:
            target = it
        else:
            remaining.append(it)
    if target is None:
        return False
    log_entry = {
        "kind": "insert_resolved",
        "item_id": item_id,
        "prompt": target.get("prompt", ""),
        "created_at": target.get("created_at"),
        "resolved_at": now_iso(),
        "action": action.strip(),
    }
    lp = log_path(did)
    lp.parent.mkdir(parents=True, exist_ok=True)
    with lp.open("a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
    p.write_text(json.dumps(remaining, ensure_ascii=False, indent=2), encoding="utf-8")
    return True


def resolve_delete(did: str, item_id: str, action: str) -> bool:
    """Permanently log a page that was marked for deletion, then drop it.

    Only removes an item that's actually marked deleted=true (the client
    toggles this per page card) -- same log-first durability order.
    """
    p = plan_path(did)
    if not p.exists():
        return False
    stored = json.loads(p.read_text(encoding="utf-8"))
    target = None
    remaining = []
    for it in stored:
        if (
            target is None
            and it.get("type") == "page"
            and it.get("id") == item_id
            and it.get("deleted")
        ):
            target = it
        else:
            remaining.append(it)
    if target is None:
        return False
    log_entry = {
        "kind": "page_deleted",
        "item_id": item_id,
        "resolved_at": now_iso(),
        "action": action.strip(),
    }
    lp = log_path(did)
    lp.parent.mkdir(parents=True, exist_ok=True)
    with lp.open("a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
    p.write_text(json.dumps(remaining, ensure_ascii=False, indent=2), encoding="utf-8")
    return True


def commit_plan(did: str, action: str) -> bool:
    """Reset each page item's origin id to its current array position.

    Call this after the .tex has actually been reordered to match the plan
    and rebuilt: at that point the plan's Nth page-item really is the new
    physical PDF page N, so its id (the "which real page is this" stamp)
    should become "pN". Clears the "moved, not yet applied" drift signal.
    """
    p = plan_path(did)
    if not p.exists():
        return False
    plan = load_plan(did)
    new_items = []
    seq = 0
    for it in plan:
        if it.get("type") == "page":
            seq += 1
            was_deleted = it.get("deleted", False)  # shouldn't normally happen at commit time
            it = {"id": f"p{seq}", "type": "page"}
            if was_deleted:
                it["deleted"] = True
        new_items.append(it)
    log_entry = {
        "kind": "plan_committed",
        "resolved_at": now_iso(),
        "item_count": len(new_items),
        "action": action.strip(),
    }
    lp = log_path(did)
    lp.parent.mkdir(parents=True, exist_ok=True)
    with lp.open("a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
    save_plan(did, new_items)
    return True


def reset_plan(did: str) -> bool:
    """Discard the staged plan entirely (no log entry -- nothing was ever applied)."""
    p = plan_path(did)
    if not p.exists():
        return False
    p.unlink()
    return True


def set_page_deleted(did: str, page: int, deleted: bool) -> bool:
    """Toggle the delete-mark on one page from outside the storyboard (e.g. the
    single-page /deck viewer), by real page number rather than array position."""
    items = load_plan(did)  # "page" is freshly derived here, safe to match on
    found = False
    clean = []
    for it in items:
        it = dict(it)
        if it.get("type") == "page" and it.get("page") == page:
            found = True
            if deleted:
                it["deleted"] = True
            else:
                it.pop("deleted", None)
        it.pop("page", None)  # derived convenience field -- don't persist it
        clean.append(it)
    if found:
        save_plan(did, clean)
    return found


# ---------------------------------------------------------------------------
# "반영하기": one fresh, headless Claude Code session per click
# ---------------------------------------------------------------------------
# The button spawns `claude -p` in the deck's folder (so slides/CLAUDE.md
# applies), hands it only this deck's pending notes / plan inline, and lets it
# edit pages/*.tex, rebuild, and resolve each item through this server's API.
# A new session per click keeps token use small: nothing from earlier applies
# rides along. The watcher skips a deck while its job runs, then rebuilds it
# once more after the job ends so page images refresh.

APPLY_MODELS = ["haiku", "sonnet", "opus"]  # first = default
APPLY_ALLOWED_TOOLS = [
    "Read", "Edit", "Write", "Glob", "Grep",
    "Bash(tectonic:*)", "Bash(curl:*)", "Bash(pdftotext:*)", "Bash(pdfinfo:*)",
    "Bash(pdftoppm:*)", "Bash(rsvg-convert:*)", "Bash(cp:*)", "Bash(ls:*)", "Bash(grep:*)",
    # 그림(make_figs.py) · 노트북(make_lab.py) 을 다시 만드는 파이썬, v2 덱 빌드 도구
    "Bash(~/Documents/manim_shorts/_env/v/bin/python:*)",
    f"Bash({os.path.expanduser('~')}/Documents/manim_shorts/_env/v/bin/python:*)",
    "Bash(python3 slides/v2/tools/deck.py:*)", f"Bash(python3 {ROOT}/slides/v2/tools/deck.py:*)",
]
# acceptEdits auto-approves some filesystem commands (mv was seen running), so
# block the destructive ones explicitly.
APPLY_DISALLOWED_TOOLS = ["Bash(rm:*)", "Bash(mv:*)", "Bash(git:*)", "Bash(rmdir:*)"]
SERVER_PORT = 8765  # set in main(); the prompt tells the session where to POST resolves

_apply_jobs: dict[str, dict] = {}
_apply_lock = threading.Lock()


def page_file_map(did: str) -> dict:
    """PDF page -> (pages/pNN.tex, kind) for the deck as last built.

    Walks the main .tex's \\input{pages/..} order, expanding each file into
    the frames it emits (an auto section-divider frame per \\section, then one
    per \\begin{frame}), and zips that with the .nav's \\beamer@framepages
    ranges. Returns {} if the counts disagree (stale build, odd source) so
    callers fall back to the old "find it yourself" instruction.
    """
    tex = deck_tex(did)
    nav = tex.with_suffix(".nav")
    if not nav.exists():
        return {}
    ranges = [(int(a), int(b)) for a, b in
              re.findall(r"\\beamer@framepages \{(\d+)\}\{(\d+)\}", nav.read_text(encoding="utf-8"))]
    owners = []
    for line in tex.read_text(encoding="utf-8").splitlines():
        line = line.split("%", 1)[0] if not line.lstrip().startswith("\\%") else line
        m = re.search(r"\\input\{(pages/[^}]+)\}", line)
        if not m:
            continue
        rel = m.group(1) if m.group(1).endswith(".tex") else m.group(1) + ".tex"
        src = tex.parent / rel
        if not src.exists():
            return {}
        body = "\n".join(l.split("%", 1)[0] for l in src.read_text(encoding="utf-8").splitlines())
        owners += [(rel, "섹션 시작 쪽(\\section 이 자동으로 만드는 쪽)")] * len(re.findall(r"\\section\b", body))
        owners += [(rel, "프레임")] * len(re.findall(r"\\begin\{frame\}", body))
    if len(owners) != len(ranges):
        return {}
    out = {}
    for (rel, kind), (a, b) in zip(owners, ranges):
        for pg in range(a, b + 1):
            out[pg] = (rel, kind)
    return out


def plan_pending(did: str) -> dict:
    """Summary of storyboard changes not yet baked into the .tex."""
    items = load_plan(did)
    inserts = [it for it in items if it.get("type") == "insert"]
    deleted = [it for it in items if it.get("type") == "page" and it.get("deleted")]
    pages = [it.get("page") for it in items if it.get("type") == "page"]
    moved = pages != sorted(pages)
    return {"inserts": len(inserts), "deleted": len(deleted), "moved": moved}


def pending_count(did: str) -> int:
    pp = plan_pending(did)
    return len(load_notes(did)) + pp["inserts"] + pp["deleted"] + (1 if pp["moved"] else 0)


def apply_prompt(did: str, extra: str = "") -> str:
    tex = deck_tex(did)
    notes = load_notes(did)
    pp = plan_pending(did)
    items = load_plan(did)
    pmap = page_file_map(did)
    base = f"http://localhost:{SERVER_PORT}"

    def where(pg):
        if not deck_pages_dir(did).is_dir():          # v2: tex 한 파일 → 하단 번호와 첫 줄(제목)로 알려 준다
            try:
                first = subprocess.run(["pdftotext", "-f", str(pg), "-l", str(pg), str(deck_pdf(did)), "-"],
                                       capture_output=True, text=True).stdout.strip().splitlines()[0]
            except (IndexError, OSError):
                first = "?"
            return f"하단 번호 {pg - 1} · 제목 '{first}' (deck.py where 로 tex 줄 확인)"
        hit = pmap.get(pg)
        return f"{hit[0]} ({hit[1]})" if hit else "파일 미확인 — pdftotext 로 직접 찾을 것"
    is_v2 = not deck_pages_dir(did).is_dir()       # slides/v2: tex 한 파일 + make_figs.py + lab/
    parts = [
        f"슬라이드 덱 `{tex.name}` (폴더 {tex.parent}) 에 아래 요청을 반영해라.",
        "규칙은 slides/CLAUDE.md 를 따른다: 프레임 하나 = pages/pNN.tex 하나. "
        "pNN 번호는 PDF 쪽 번호와 다를 수 있으니(p09g1 같은 파일), 쪽 번호로 지시된 요청은 "
        f"`pdftotext -f N -l N {tex.with_suffix('.pdf').name} - | head` 로 글자를 보고 grep 으로 파일을 찾는다.",
        "요청받은 부분만 고친다. 요청과 상관없는 쪽은 건드리지 않는다. "
        "파일 삭제·이동(rm, mv)과 git 은 금지. 붙여넣은 그림(figs/pasted_*)은 그 이름 그대로 쓴다.",
        "",
        "## 쪽별 메모",
    ]
    if is_v2:
        parts[1] = ("이 덱은 slides/v2 형식이다: 덱 전체가 tex 한 파일(pages/ 없음). 규칙은 스킬 slide-v2 와 slides/v2/README.md. "
                    "그림은 손으로 고치지 말고 make_figs.py 를 고친 뒤 `~/Documents/manim_shorts/_env/v/bin/python make_figs.py` 로 다시 만든다. "
                    f"빌드는 `python3 {ROOT}/slides/v2/tools/deck.py build {tex.parent}`, "
                    f"쪽 확인은 `python3 {ROOT}/slides/v2/tools/deck.py show {tex.parent} N` (N = 하단 번호 = PDF 쪽 − 1). "
                    "아래 메모의 쪽 번호는 PDF 쪽 번호다. 슬라이드 내용을 바꾸면 그 프레임의 \\note{} 도 같이 고친다.")
        parts = [x for x in parts if not x.startswith("쪽 번호 → 파일 대응은")]
        if pp["deleted"] or pp["inserts"] or pp["moved"]:
            parts.append("쪽 삭제·순서 변경은 tex 안의 \\begin{frame}...\\end{frame} 블록을 옮기거나 지워서 한다.")
    if notes:
        for k, v in sorted(notes.items(), key=lambda kv: int(kv[0])):
            parts.append(f"- PDF {k}쪽 = {where(int(k))}: {v.get('note', '')}")
    else:
        parts.append("(없음)")
    if pmap:
        parts.insert(2, "쪽 번호 → 파일 대응은 서버가 빌드 기록(.nav)으로 계산해 아래에 적어 두었다. "
                        "그대로 믿고 쓰고, 쪽 번호를 다시 추측하지 않는다(그림만 있는 쪽은 글자 검색으로 못 찾는다).")
    if extra.strip():
        parts += [
            "",
            "## 주차 전체 지시 (덱 전체에 적용, 필요한 쪽을 직접 찾아 고친다)",
            extra.strip(),
        ]
    if pp["deleted"]:
        parts += ["", "## 지울 쪽 (해당 \\input 줄만 뺀다, 파일은 남김. 섹션 시작 쪽이면 그 \\section 줄을 뺀다)"]
        for it in items:
            if it.get("type") == "page" and it.get("deleted"):
                parts.append(f"- item_id={it['id']} · PDF {it.get('page')}쪽 = {where(it.get('page'))}")
    if pp["inserts"] or pp["moved"]:
        parts += [
            "",
            "## 스토리보드 계획 (순서 변경 / 새 쪽)",
            "아래가 목표 순서다. 새 쪽은 그 자리에 새 pages 파일로 만들고 \\input 을 넣는다.",
        ]
        for i, it in enumerate(items, 1):
            if it.get("type") == "insert":
                parts.append(f"{i}. [새 쪽 item_id={it['id']}] {it.get('prompt', '')}")
            elif not it.get("deleted"):
                parts.append(f"{i}. PDF {it.get('page')}쪽 = {where(it.get('page'))}")
    parts += [
        "",
        "## 끝내는 법",
        (f"1. `python3 {ROOT}/slides/v2/tools/deck.py build {tex.parent}` 로 빌드하고 오류·Overfull 이 없을 때까지 고친다."
         if is_v2 else
         f"1. 덱 폴더에서 `tectonic -X compile --keep-intermediates {tex.name}` 로 빌드하고 오류가 없을 때까지 고친다."),
        "2. 반영한 항목마다 아래를 호출한다(action = 한 일 한 줄, 한국어):",
        f"   - 메모: curl -s -X POST {base}/api/resolve -H 'Content-Type: application/json' "
        f"-d '{{\"id\":\"{did}\",\"page\":N,\"action\":\"...\"}}'",
        f"   - insert: {base}/api/plan/resolve_insert  body {{\"id\",\"item_id\",\"action\"}}",
        f"   - delete: {base}/api/plan/resolve_delete  body {{\"id\",\"item_id\",\"action\"}}",
        f"   - 계획을 반영했다면 마지막에 한 번: {base}/api/plan/commit  body {{\"id\",\"action\"}}",
        "   각 curl 응답에 \"ok\": true 가 있는지 확인한다. 아니면 item_id 를 다시 보고 재시도한다.",
        "   쪽 삭제·추가·순서 변경을 하나라도 했다면, 다 끝난 뒤 plan/commit 을 꼭 한 번 호출한다"
        "(안 하면 쪽 번호가 어긋난 채로 남는다).",
        "   반영하지 못한 항목은 resolve 하지 말고 남겨 둔다.",
        "3. 마지막 답은 항목마다 한 줄씩: `✅ p.N 무엇을 함` 또는 `⚠️ p.N 못 한 이유`.",
    ]
    return "\n".join(parts)


def _stamp(line: str) -> str:
    return time.strftime("[%H:%M:%S] ") + line


def _summarize_event(ev: dict):
    """One short log line per interesting stream-json event, else None."""
    t = ev.get("type")
    if t == "assistant":
        out = []
        for block in ev.get("message", {}).get("content", []):
            if block.get("type") == "text" and block.get("text", "").strip():
                out.append(block["text"].strip())
            elif block.get("type") == "tool_use":
                inp = block.get("input", {})
                arg = inp.get("file_path") or inp.get("command") or inp.get("pattern") or ""
                arg = str(arg).replace(str(ROOT) + "/", "").strip().split("\n")[0]
                out.append(f"→ {block.get('name')} {arg[:120]}")
        return "\n".join(out) or None
    if t == "result":
        cost = ev.get("total_cost_usd")
        secs = (ev.get("duration_ms") or 0) / 1000
        tail = f" · ${cost:.3f}" if isinstance(cost, (int, float)) else ""
        return f"[끝] {ev.get('subtype', '')} · {secs:.0f}초{tail}"
    return None


def start_apply(did: str, model: str, extra: str = "") -> tuple:
    with _apply_lock:
        job = _apply_jobs.get(did)
        if job and job["status"] == "running":
            return False, "이미 반영 중입니다"
        if pending_count(did) == 0 and not extra.strip():
            return False, "반영할 항목이 없습니다"
        job = {"status": "running", "model": model, "started": now_iso(),
               "lines": [_stamp(f"[시작] 모델 {model} · 새 Claude Code 세션")], "exit": None}
        if extra.strip():
            job["lines"].append(_stamp("[주차 전체 지시] " + extra.strip()))
        _apply_jobs[did] = job

    def run():
        cmd = ["claude", "-p", apply_prompt(did, extra), "--model", model,
               "--permission-mode", "acceptEdits",
               "--allowedTools", *APPLY_ALLOWED_TOOLS,
               "--disallowedTools", *APPLY_DISALLOWED_TOOLS,
               "--output-format", "stream-json", "--verbose",
               "--no-session-persistence"]
        try:
            proc = subprocess.Popen(cmd, cwd=deck_tex(did).parent, stdin=subprocess.DEVNULL,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for raw in proc.stdout:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    line = _summarize_event(json.loads(raw))
                except json.JSONDecodeError:
                    line = raw[:300]
                if line:
                    job["lines"].append(_stamp(line))
            job["exit"] = proc.wait()
            job["status"] = "done" if job["exit"] == 0 else "failed"
        except Exception as e:
            job["lines"].append(_stamp(f"[오류] {e}"))
            job["status"] = "failed"

    threading.Thread(target=run, daemon=True).start()
    return True, ""


def apply_preview(did: str) -> dict:
    """What a click would send, for the confirm dialog: items + the exact prompt."""
    items = load_plan(did)
    pp = plan_pending(did)
    notes = load_notes(did)
    return {
        "notes": [{"page": int(k), "note": v.get("note", "")}
                  for k, v in sorted(notes.items(), key=lambda kv: int(kv[0]))],
        "inserts": [{"pos": i + 1, "prompt": it.get("prompt", "")}
                    for i, it in enumerate(items) if it.get("type") == "insert"],
        "deleted": [it.get("page") for it in items if it.get("type") == "page" and it.get("deleted")],
        "moved": pp["moved"],
        "prompt": apply_prompt(did),
    }


def apply_parts(did: str) -> dict:
    return {"apply_html": APPLY_HTML, "apply_panel": APPLY_PANEL,
            "apply_js": APPLY_JS.replace("DID_JSON", json.dumps(did))}


def apply_status(did: str) -> dict:
    job = _apply_jobs.get(did)
    base = {"pending": pending_count(did), "models": APPLY_MODELS}
    if not job:
        return {**base, "status": "idle", "lines": []}
    return {**base, "status": job["status"], "model": job["model"], "lines": job["lines"][-200:]}


APPLY_HTML = """
<span class="apply-box">
  <select id="applyModel" title="반영에 쓸 모델"></select>
  <button id="applyBtn">반영하기</button>
</span>
"""

APPLY_PANEL = """<div id="applyBar" class="apply-bar" hidden>
  <span id="applyNow"></span>
  <button id="applyLogBtn" type="button">▾ 전체</button>
</div>
<pre id="applyLog" class="apply-log" hidden></pre>
<dialog id="applyDialog" class="apply-dialog">
  <h3>반영할 항목 미리보기</h3>
  <div id="applyItems"></div>
  <label class="apply-extra-label" for="applyExtra">주차 전체 지시 (선택)</label>
  <textarea id="applyExtra" class="apply-extra" placeholder="예: 이 주차 전체에서 수식 기호를 \\boldsymbol{w} 로 통일 / 모든 예제 숫자에 단위 붙이기"></textarea>
  <details><summary>세션에 보낼 지시문 전체</summary><pre id="applyPrompt"></pre></details>
  <div class="apply-dialog-row">
    <span id="applyModelLabel"></span>
    <button id="applyCancel" class="secondary">취소</button>
    <button id="applyGo">실행</button>
  </div>
</dialog>"""

APPLY_JS = """
<script>
(function () {
  const did = DID_JSON;
  const sel = document.getElementById('applyModel');
  const btn = document.getElementById('applyBtn');
  const logBtn = document.getElementById('applyLogBtn');
  const logBox = document.getElementById('applyLog');
  let saved = 'haiku';
  try { saved = localStorage.getItem('applyModel') || 'haiku'; } catch (e) {}
  let timer = null, wasRunning = false;
  function paint(s) {
    if (!sel.options.length) {
      s.models.forEach(function (m) { const o = document.createElement('option'); o.value = o.textContent = m; sel.appendChild(o); });
      sel.value = s.models.indexOf(saved) >= 0 ? saved : s.models[0];
    }
    const running = s.status === 'running';
    btn.disabled = running;
    btn.textContent = running ? '반영 중… (' + (s.model || '') + ')' : '반영하기 (' + s.pending + '건)';
    const bar = document.getElementById('applyBar');
    bar.hidden = !s.lines.length;
    // one line only: the latest event, first line of it (final answers are multi-line)
    const last = s.lines.length ? s.lines[s.lines.length - 1] : '';
    document.getElementById('applyNow').textContent = (running ? '⏳ ' : '') + last.split('\\n')[0];
    logBox.textContent = s.lines.join('\\n');
    if (!logBox.hidden) { logBox.scrollTop = logBox.scrollHeight; }
    if (wasRunning && !running) { setTimeout(function () { location.reload(); }, 1500); }
    wasRunning = running;
    clearTimeout(timer);
    timer = setTimeout(poll, running ? 1500 : 5000);
  }
  function poll() {
    fetch('/api/apply/status?id=' + encodeURIComponent(did)).then(function (r) { return r.json(); }).then(paint)
      .catch(function () { timer = setTimeout(poll, 5000); });
  }
  sel.addEventListener('change', function () { try { localStorage.setItem('applyModel', sel.value); } catch (e) {} });
  logBtn.addEventListener('click', function () {
    logBox.hidden = !logBox.hidden;
    logBtn.textContent = logBox.hidden ? '▾ 전체' : '▴ 접기';
    logBox.scrollTop = logBox.scrollHeight;
  });
  const dlg = document.getElementById('applyDialog');
  const itemsBox = document.getElementById('applyItems');
  function row(tag, text, page) {
    const d = document.createElement('div'); d.className = 'apply-item';
    if (page) {
      d.classList.add('clickable');
      d.title = page + '쪽 보기';
      d.addEventListener('click', function () {
        if (typeof goTo === 'function') { dlg.close(); goTo(page); }
        else { location.href = '/deck?id=' + encodeURIComponent(did) + '&start=' + page; }
      });
    }
    const b = document.createElement('b'); b.textContent = tag; d.appendChild(b);
    const t = document.createElement('span'); t.textContent = text; d.appendChild(t);
    itemsBox.appendChild(d);
  }
  btn.addEventListener('click', function () {
    fetch('/api/apply/preview?id=' + encodeURIComponent(did)).then(function (r) { return r.json(); }).then(function (pv) {
      itemsBox.textContent = '';
      pv.notes.forEach(function (n) { row('p.' + n.page, n.note, n.page); });
      pv.inserts.forEach(function (n) { row('추가 @' + n.pos, n.prompt || '(빈 프롬프트)'); });
      pv.deleted.forEach(function (pg) { row('삭제', 'p.' + pg, pg); });
      if (pv.moved) { row('순서', '스토리보드에서 쪽 순서가 바뀜'); }
      if (!itemsBox.childNodes.length) { row('', '쪽별 메모 없음 — 아래에 주차 전체 지시를 적으면 실행할 수 있습니다.'); }
      document.getElementById('applyPrompt').textContent = pv.prompt;
      document.getElementById('applyModelLabel').textContent = '모델: ' + sel.value;
      dlg.showModal();
    });
  });
  document.getElementById('applyCancel').addEventListener('click', function () { dlg.close(); });
  document.getElementById('applyGo').addEventListener('click', function () {
    dlg.close();
    btn.disabled = true;
    fetch('/api/apply', { method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ id: did, model: sel.value, extra: document.getElementById('applyExtra').value }) })
      .then(function (r) { return r.json(); })
      .then(function (r) { if (!r.ok) { alert(r.error); } poll(); });
  });
  poll();
})();
</script>
"""

APPLY_CSS = """
.apply-box { margin-left: auto; display: flex; gap: 6px; align-items: center; }
.apply-box select, .apply-box button { font-size: 13px; padding: 5px 10px; border-radius: 6px; border: 1px solid #c9c9c4; background: #fff; cursor: pointer; }
.apply-box #applyBtn { background: #2354a5; color: #fff; border-color: #2354a5; font-weight: 600; }
.apply-box #applyBtn:disabled { opacity: .55; cursor: default; }
.apply-dialog { width: min(720px, 92vw); border: 1px solid #c9c9c4; border-radius: 10px; padding: 18px 20px; }
.apply-dialog h3 { margin: 0 0 12px; font-size: 15px; }
.apply-item { display: flex; gap: 10px; padding: 7px 0; border-bottom: 1px solid #eee; font-size: 13px; white-space: pre-wrap; }
.apply-item b { min-width: 64px; color: #2354a5; }
.apply-item.clickable { cursor: pointer; border-radius: 6px; padding-left: 6px; padding-right: 6px; }
.apply-item.clickable:hover { background: #eef3fb; }
.apply-item.clickable b { text-decoration: underline; }
.apply-dialog details { margin-top: 12px; font-size: 12px; }
.apply-extra-label { display: block; margin-top: 14px; font-size: 13px; font-weight: 600; color: #444; }
.apply-extra { width: 100%; min-height: 80px; margin-top: 6px; padding: 8px; font-size: 13px; border: 1px solid #c9c9c4; border-radius: 6px; font-family: inherit; }
.apply-dialog details pre { max-height: 260px; overflow: auto; background: #f4f4f1; padding: 10px; white-space: pre-wrap; }
.apply-dialog-row { display: flex; gap: 8px; align-items: center; margin-top: 14px; }
.apply-dialog-row span { margin-right: auto; color: #666; font-size: 12px; }
.apply-dialog-row button { font-size: 13px; padding: 6px 14px; border-radius: 6px; border: 1px solid #c9c9c4; background: #fff; cursor: pointer; }
.apply-dialog-row #applyGo { background: #2354a5; color: #fff; border-color: #2354a5; font-weight: 600; }
.apply-bar { display: flex; align-items: center; gap: 10px; padding: 6px 20px; background: #111; color: #d6f5d6;
  font: 12px ui-monospace, Menlo, monospace; border-bottom: 1px solid #333; }
.apply-bar #applyNow { flex: 1; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.apply-bar button { background: #222; color: #d6f5d6; border: 1px solid #444; border-radius: 5px; padding: 2px 8px; font-size: 12px; cursor: pointer; }
.apply-log { margin: 0; max-height: 260px; overflow: auto; padding: 10px 20px; background: #111; color: #d6f5d6;
  font-size: 12px; line-height: 1.45; white-space: pre-wrap; border-bottom: 1px solid #333; }
"""


PAGE_CSS = """
:root { color-scheme: light dark; }
* { box-sizing: border-box; }
body {
  margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Apple SD Gothic Neo", sans-serif;
  background: #fafaf8; color: #1c1c1c;
}
header {
  padding: 14px 20px; border-bottom: 1px solid #e2e2df; display: flex;
  align-items: center; gap: 14px; background: #fff; position: sticky; top: 0; z-index: 2;
}
header a { color: #2354a5; text-decoration: none; font-weight: 600; }
header h1 { font-size: 15px; margin: 0; font-weight: 600; color: #444; }
main { max-width: 1400px; margin: 0 auto; padding: 20px; }
.deck-list { list-style: none; padding: 0; margin: 0; display: grid; gap: 8px; }
.deck-list li { border: 1px solid #e2e2df; border-radius: 8px; background: #fff; display: flex; align-items: stretch; }
.deck-list li > a:first-child { flex: 1; display: flex; justify-content: space-between; padding: 12px 16px; text-decoration: none; color: #1c1c1c; }
.deck-list li > a:not(:first-child) { display: flex; align-items: center; padding: 12px 16px; text-decoration: none; color: #2354a5; font-size: 13px; white-space: nowrap; }
.deck-list a:hover { background: #f3f3ef; }
.deck-id { font-family: ui-monospace, monospace; font-size: 13px; color: #555; }
.badge { font-size: 12px; padding: 2px 8px; border-radius: 999px; background: #eef1fb; color: #2354a5; }
.badge.zero { background: #f0f0ee; color: #999; }
.viewer { display: grid; grid-template-columns: 1fr 340px; gap: 20px; align-items: start; }
.page-wrap { background: #fff; border: 1px solid #e2e2df; border-radius: 10px; padding: 14px; text-align: center; }
.page-wrap img { max-width: 100%; border: 1px solid #ddd; border-radius: 4px; }
.nav { display: flex; justify-content: center; align-items: center; gap: 10px; margin-top: 12px; flex-wrap: wrap; }
.nav button { padding: 8px 14px; border-radius: 6px; border: 1px solid #ccc; background: #fff; cursor: pointer; font-size: 14px; }
.nav button:hover { background: #f3f3ef; }
.nav button:disabled { opacity: .4; cursor: default; }
.nav input { width: 54px; text-align: center; padding: 6px; border-radius: 6px; border: 1px solid #ccc; }
.nav a { font-size: 13px; color: #2354a5; text-decoration: none; margin-left: 6px; }
.nav a:hover { text-decoration: underline; }
.del-row { display: flex; justify-content: center; align-items: center; gap: 8px; margin-top: 8px; }
.del-row button {
  padding: 6px 12px; border-radius: 6px; border: 1px solid #e0b4ae; background: #fff;
  color: #c0392b; cursor: pointer; font-size: 13px;
}
.del-row button:hover { background: #fdf1f0; }
.del-row button.active { background: #c0392b; color: #fff; border-color: #c0392b; }
.page-wrap.deleted { opacity: .5; filter: grayscale(1); border-color: #c0392b; border-style: dashed; }
.side { position: sticky; top: 70px; display: flex; flex-direction: column; gap: 10px; }
textarea {
  width: 100%; min-height: 220px; padding: 10px; border-radius: 8px; border: 1px solid #ccc;
  font-size: 14px; line-height: 1.5; resize: vertical; font-family: inherit;
}
.save-row { display: flex; align-items: center; gap: 10px; }
.save-row button { padding: 8px 16px; border-radius: 6px; border: none; background: #2354a5; color: #fff; cursor: pointer; font-weight: 600; }
.save-row button:hover { background: #1d478c; }
.paste-row { min-height: 18px; }
.paste-previews { display: flex; flex-wrap: wrap; gap: 6px; }
.paste-previews img {
  width: 64px; height: 64px; object-fit: cover; border-radius: 6px; border: 1px solid #ccc;
}
.status { font-size: 13px; color: #888; }
.status.ok { color: #2a8a4a; }
.status.err { color: #c0392b; }
.source-panel { grid-column: 1; margin-top: 16px; }
.source-panel textarea {
  min-height: 320px; font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
  font-size: 13px; white-space: pre; tab-size: 2;
}
.source-toggle-row { display: flex; justify-content: center; margin-top: 8px; }
.source-toggle-row button {
  padding: 6px 12px; border-radius: 6px; border: 1px solid #ccc; background: #fff;
  color: #555; cursor: pointer; font-size: 13px;
}
.source-toggle-row button:hover { background: #f3f3ef; }
.source-save-row { display: flex; align-items: center; gap: 10px; margin-top: 8px; }
.source-save-row button { padding: 8px 16px; border-radius: 6px; border: none; background: #2354a5; color: #fff; cursor: pointer; font-weight: 600; }
.source-save-row button:hover { background: #1d478c; }
.source-save-row button:disabled { opacity: .5; cursor: default; }
.source-save-row button.secondary { background: #fff; color: #2354a5; border: 1px solid #2354a5; font-weight: 500; }
.source-save-row button.secondary:hover { background: #f0f4fb; }
.source-error { white-space: pre-wrap; font-family: ui-monospace, monospace; font-size: 12px; color: #c0392b; background: #fdf1f0; border-radius: 6px; padding: 8px; margin-top: 6px; max-height: 200px; overflow-y: auto; }
.jumplist { display: flex; flex-wrap: wrap; gap: 5px; max-height: 220px; overflow-y: auto; }
.jumplist button {
  width: 30px; height: 26px; font-size: 11px; border-radius: 4px; border: 1px solid #ccc;
  background: #fff; cursor: pointer;
}
.jumplist button.current { background: #2354a5; color: #fff; border-color: #2354a5; }
.jumplist button.noted { border-color: #2a8a4a; color: #2a8a4a; font-weight: 700; }
.jumplist button.marked-del { border-color: #c0392b; border-style: dashed; color: #c0392b; text-decoration: line-through; }
.jumplist button.current.noted { color: #fff; }
.jumplist-session {
  flex-basis: 100%; margin: 6px 0 1px; font-size: 11px; font-weight: 700;
  color: #2354a5; border-top: 1px solid #d8dce6; padding-top: 5px;
}
.jumplist-session:first-child { margin-top: 0; border-top: none; padding-top: 0; }

.board-toolbar { display: flex; align-items: center; gap: 12px; margin-bottom: 14px; }
.board-toolbar button { padding: 7px 12px; border-radius: 6px; border: 1px solid #ccc; background: #fff; cursor: pointer; }
.board-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(170px, 1fr)); gap: 14px; }
.card {
  border: 1px solid #e2e2df; border-radius: 8px; background: #fff; padding: 8px;
  cursor: grab; display: flex; flex-direction: column;
}
.card.dragging { opacity: .35; }
.card img { width: 100%; border-radius: 4px; border: 1px solid #ddd; display: block; cursor: pointer; }
.card .pagenum { font-size: 11px; color: #888; text-align: center; margin-top: 5px; }
.card.noted { border-color: #2a8a4a; }
.card.noted .pagenum { color: #2a8a4a; font-weight: 600; }
.card.insert { border-style: dashed; border-color: #2354a5; background: #f5f8ff; min-height: 150px; }
.card.insert textarea {
  flex: 1; width: 100%; border: 1px solid #ccc; border-radius: 6px; font-size: 12px;
  padding: 6px; resize: none; font-family: inherit; min-height: 90px;
}
.card-actions { display: flex; justify-content: flex-end; gap: 4px; margin-top: 6px; }
.card-actions button {
  font-size: 11px; padding: 3px 7px; border-radius: 4px; border: 1px solid #ccc;
  background: #fff; cursor: pointer;
}
.card-actions button:hover { background: #f3f3ef; }
#endZone { height: 36px; }
.card.moved { border-left: 4px solid #d68910; }
.card.moved .pagenum { color: #a8710a; }
.card.deleted { opacity: .45; filter: grayscale(1); border-color: #c0392b; border-style: dashed; }
.card.deleted .pagenum { color: #c0392b; font-weight: 600; text-decoration: line-through; }
.card-actions button.danger { color: #c0392b; border-color: #e0b4ae; }
.card-actions button.danger:hover { background: #fdf1f0; }
.card-actions button.undo { color: #2354a5; border-color: #b9c9e6; }
.pending-banner {
  display: flex; align-items: center; gap: 10px; padding: 10px 14px; margin-bottom: 14px;
  border-radius: 8px; background: #fff8e6; border: 1px solid #f0d68a; color: #7a5b00; font-size: 13px;
}
.pending-banner button {
  margin-left: auto; padding: 5px 10px; border-radius: 6px; border: 1px solid #d8b94a;
  background: #fff; cursor: pointer; font-size: 12px; white-space: nowrap;
}
"""

FAVICON_LINK = (
    '<link rel="icon" href="data:image/svg+xml;base64,'
    'PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAxMDAgMTAwIj'
    '48dGV4dCB5PSIuOWVtIiBmb250LXNpemU9IjkwIj7wn46sPC90ZXh0Pjwvc3ZnPg==">'
)

INDEX_TMPL = Template("""<!doctype html><html><head><meta charset="utf-8">
""" + FAVICON_LINK + """
<title>Slide Review</title><style>$css</style></head><body>
<header><h1>Slide Review · book-ml</h1></header>
<main>
<ul class="deck-list">
$rows
</ul>
</main></body></html>""")

VIEWER_TMPL = Template("""<!doctype html><html><head><meta charset="utf-8">
""" + FAVICON_LINK + """
<title>$did</title><style>$css</style></head><body>
<header>
  <a href="/">&larr; 목록</a>
  <a href="/board?id=$did_quoted">스토리보드</a>
  <h1>$did &nbsp;·&nbsp; <span id="pageLabel"></span> / $total</h1>
  $apply_html
</header>
$apply_panel
<main>
<div class="viewer">
  <div class="page-wrap">
    <img id="pageImg" src="" alt="page">
    <div class="nav">
      <button id="prevBtn">&larr; 이전</button>
      <input id="pageInput" type="number" min="1" max="$total" value="1">
      <button id="nextBtn">다음 &rarr;</button>
      <a id="pdfLink" href="#" target="_blank" rel="noopener">원본 PDF로 보기 (텍스트 선택 가능)</a>
    </div>
    <div class="del-row">
      <button id="deleteBtn">이 페이지 삭제 표시</button>
      <span class="status" id="deleteStatus"></span>
    </div>
  </div>
  <div class="side">
    <textarea id="noteBox" placeholder="이 페이지에 넣고 싶은 내용/이미지 지시사항을 적어줘... (Ctrl/Cmd+V로 클립보드 이미지도 붙여넣기 가능)"></textarea>
    <div class="paste-row">
      <span class="status" id="pasteStatus"></span>
    </div>
    <div class="paste-previews" id="pastePreviews"></div>
    <div class="save-row">
      <button id="saveBtn">지금 저장</button>
      <span class="status" id="statusLabel">자동 저장 켜짐</span>
    </div>
    <div class="jumplist" id="jumplist"></div>
  </div>
  <div class="source-panel">
    <div class="source-toggle-row">
      <button id="sourceToggleBtn">이 페이지 코드 보기/수정</button>
    </div>
    <div id="sourcePanelBody" hidden>
      <textarea id="sourceBox" spellcheck="false"></textarea>
      <div class="source-save-row">
        <button id="sourceSaveOnlyBtn" class="secondary">저장만</button>
        <button id="sourceSaveBtn">저장 &amp; 재빌드</button>
        <span class="status" id="sourceStatus"></span>
      </div>
      <div class="source-error" id="sourceError" hidden></div>
    </div>
  </div>
</div>
</main>
<script>
const did = $did_json;
const total = $total;
const notes = $notes_json;
const deletedPages = new Set($deleted_json);
const sections = $sections_json;  // [{title, page}] -- one per 차시 (class session)
let page = 1;

const img = document.getElementById('pageImg');
const label = document.getElementById('pageLabel');
const pdfLink = document.getElementById('pdfLink');
const input = document.getElementById('pageInput');
const box = document.getElementById('noteBox');
const status = document.getElementById('statusLabel');
const jumplist = document.getElementById('jumplist');
const prevBtn = document.getElementById('prevBtn');
const nextBtn = document.getElementById('nextBtn');
const deleteBtn = document.getElementById('deleteBtn');
const deleteStatus = document.getElementById('deleteStatus');
const pageWrap = document.querySelector('.page-wrap');
const pasteStatus = document.getElementById('pasteStatus');
const pastePreviews = document.getElementById('pastePreviews');
const sourceToggleBtn = document.getElementById('sourceToggleBtn');
const sourcePanelBody = document.getElementById('sourcePanelBody');
const sourceBox = document.getElementById('sourceBox');
const sourceSaveBtn = document.getElementById('sourceSaveBtn');
const sourceSaveOnlyBtn = document.getElementById('sourceSaveOnlyBtn');
const sourceStatus = document.getElementById('sourceStatus');
const sourceError = document.getElementById('sourceError');
let sourcePanelOpen = false;
let sourceEditable = false;
let sourceLoadedPage = null;
let sourceUnbuilt = false; // true once "저장만" has saved something the PDF doesn't reflect yet

let sourceLoadedFile = '';
function loadSource() {
  sourceError.hidden = true;
  sourceStatus.textContent = '불러오는 중...';
  sourceStatus.className = 'status';
  sourceBox.value = '';
  sourceBox.disabled = true;
  sourceSaveBtn.disabled = true;
  sourceSaveOnlyBtn.disabled = true;
  fetch(`/api/source?id=$${encodeURIComponent(did)}&page=$${page}`)
    .then(r => r.json())
    .then(res => {
      sourceLoadedPage = page;
      sourceEditable = !!res.editable;
      if (!sourceEditable) {
        sourceBox.value = '(이 쪽의 소스 파일을 찾지 못함 — 빌드 기록과 소스가 어긋난 상태. 재빌드가 끝난 뒤 다시 열어 보세요)';
        sourceLoadedFile = '';
        sourceStatus.textContent = '';
        return;
      }
      sourceBox.value = res.content;
      sourceBox.disabled = false;
      sourceSaveBtn.disabled = false;
      sourceSaveOnlyBtn.disabled = false;
      sourceLoadedFile = res.file || '';
      sourceStatus.textContent = res.file + (res.kind && res.kind !== '프레임' ? ' · ' + res.kind : '') + ' · 자동 저장 켜짐 (재빌드는 버튼으로)';
      sourceStatus.className = 'status';
      sourceUnbuilt = false;
    })
    .catch(() => { sourceStatus.textContent = '불러오기 실패'; sourceStatus.className = 'status err'; });
}

sourceToggleBtn.onclick = () => {
  sourcePanelOpen = !sourcePanelOpen;
  sourcePanelBody.hidden = !sourcePanelOpen;
  sourceToggleBtn.textContent = sourcePanelOpen ? '코드 패널 닫기' : '이 페이지 코드 보기/수정';
  if (sourcePanelOpen && sourceLoadedPage !== page) loadSource();
};

const INDENT = '  '; // matches this project's 2-space LaTeX indent convention

sourceBox.addEventListener('keydown', (e) => {
  if (e.key !== 'Tab') return;
  e.preventDefault();
  const val = sourceBox.value;
  const start = sourceBox.selectionStart;
  const end = sourceBox.selectionEnd;
  if (!e.shiftKey && start === end) {
    // no selection: just insert indent and move the cursor past it
    sourceBox.value = val.slice(0, start) + INDENT + val.slice(end);
    sourceBox.selectionStart = sourceBox.selectionEnd = start + INDENT.length;
    return;
  }
  // selection spanning one or more lines (or shift+tab with no selection):
  // indent/outdent every touched line, keeping the selection over the same text
  let lineStart = val.lastIndexOf('\\n', start - 1) + 1;
  const beforeLines = val.slice(lineStart, end);
  const lines = beforeLines.split('\\n');
  let delta = 0;
  const newLines = lines.map(line => {
    if (e.shiftKey) {
      if (line.startsWith(INDENT)) { delta -= INDENT.length; return line.slice(INDENT.length); }
      if (line.startsWith(' ')) { delta -= 1; return line.slice(1); }
      return line;
    }
    delta += INDENT.length;
    return INDENT + line;
  });
  const replaced = newLines.join('\\n');
  sourceBox.value = val.slice(0, lineStart) + replaced + val.slice(end);
  sourceBox.selectionStart = lineStart;
  sourceBox.selectionEnd = lineStart + replaced.length;
});

function saveSource(rebuild) {
  sourceSaveBtn.disabled = true;
  sourceSaveOnlyBtn.disabled = true;
  sourceStatus.textContent = rebuild ? '저장 & 재빌드 중... (몇 초 걸릴 수 있음)' : '저장 중...';
  sourceStatus.className = 'status';
  sourceError.hidden = true;
  const savedPage = page;
  fetch('/api/source', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({id: did, page: savedPage, file: sourceLoadedFile, content: sourceBox.value, rebuild}),
  }).then(r => r.json()).then(res => {
    sourceSaveBtn.disabled = false;
    sourceSaveOnlyBtn.disabled = false;
    if (!res.ok) {
      sourceStatus.textContent = rebuild ? '빌드 실패 -- 이전 내용으로 되돌림' : '저장 실패';
      sourceStatus.className = 'status err';
      sourceError.hidden = false;
      sourceError.textContent = res.error || '';
      return;
    }
    if (rebuild) {
      sourceUnbuilt = false;
      sourceStatus.textContent = '저장 & 재빌드 완료';
      sourceStatus.className = 'status ok';
      if (savedPage === page) img.src = `/img?id=$${encodeURIComponent(did)}&page=$${page}&t=$${Date.now()}`;
    } else {
      sourceUnbuilt = true;
      sourceStatus.textContent = '저장됨 (재빌드 전이라 위 이미지는 아직 이전 상태)';
      sourceStatus.className = 'status ok';
    }
  }).catch(() => {
    sourceSaveBtn.disabled = false;
    sourceSaveOnlyBtn.disabled = false;
    sourceStatus.textContent = '저장 실패 (네트워크)';
    sourceStatus.className = 'status err';
  });
}

sourceSaveBtn.onclick = () => { clearTimeout(sourceSaveTimer); sourceSaveTimer = null; saveSource(true); };
sourceSaveOnlyBtn.onclick = () => { clearTimeout(sourceSaveTimer); sourceSaveTimer = null; saveSource(false); };

// auto-save (save-only, no rebuild) while typing, same debounce pattern as the note box
let sourceSaveTimer = null;
const SOURCE_AUTOSAVE_DELAY = 900;

function flushSourcePending() {
  if (sourceSaveTimer) {
    clearTimeout(sourceSaveTimer);
    sourceSaveTimer = null;
    saveSource(false);
  }
}

sourceBox.addEventListener('input', () => {
  if (!sourceEditable) return;
  clearTimeout(sourceSaveTimer);
  sourceStatus.textContent = '입력 중...';
  sourceStatus.className = 'status';
  sourceSaveTimer = setTimeout(() => { sourceSaveTimer = null; saveSource(false); }, SOURCE_AUTOSAVE_DELAY);
});

function refreshDeleteUI() {
  const isDeleted = deletedPages.has(page);
  deleteBtn.textContent = isDeleted ? '삭제 취소' : '이 페이지 삭제 표시';
  deleteBtn.classList.toggle('active', isDeleted);
  pageWrap.classList.toggle('deleted', isDeleted);
  deleteStatus.textContent = isDeleted ? '삭제 예정 (미반영, 스토리보드에도 표시됨)' : '';
}

deleteBtn.onclick = () => {
  const wantDeleted = !deletedPages.has(page);
  deleteBtn.disabled = true;
  fetch('/api/plan/set_deleted', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({id: did, page, deleted: wantDeleted}),
  }).then(res => {
    deleteBtn.disabled = false;
    if (!res.ok) return;
    if (wantDeleted) deletedPages.add(page); else deletedPages.delete(page);
    refreshDeleteUI();
    buildJumplist();
    refreshJumplistHighlight();
  });
};

function buildJumplist() {
  jumplist.innerHTML = '';
  // sections.page marks where each 차시 starts; look up by page as we go
  // instead of once up front, so a page also matching total+1 (none do,
  // but be defensive) never double-inserts a header.
  const startsSession = new Map(sections.map((s, i) => [s.page, i + 1]));
  for (let p = 1; p <= total; p++) {
    if (startsSession.has(p)) {
      const label = document.createElement('div');
      label.className = 'jumplist-session';
      label.textContent = `$${startsSession.get(p)}차시`;
      label.title = sections.find(s => s.page === p).title;
      jumplist.appendChild(label);
    }
    const b = document.createElement('button');
    b.textContent = p;
    b.dataset.page = p;
    if (notes[p] && notes[p].note) b.classList.add('noted');
    if (deletedPages.has(p)) b.classList.add('marked-del');
    b.onclick = () => goTo(p);
    jumplist.appendChild(b);
  }
}

function refreshJumplistHighlight() {
  [...jumplist.children].forEach(b => {
    b.classList.toggle('current', Number(b.dataset.page) === page);
  });
}

let saveTimer = null;
const AUTOSAVE_DELAY = 700;

function doSave(pageToSave, note) {
  status.textContent = '저장 중...';
  status.className = 'status';
  return fetch('/api/save', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({id: did, page: pageToSave, note}),
  }).then(res => {
    if (res.ok) {
      if (note) {
        notes[pageToSave] = {note, updated_at: new Date().toISOString()};
      } else {
        delete notes[pageToSave];
      }
      status.textContent = '자동 저장됨';
      status.className = 'status ok';
      buildJumplist();
      refreshJumplistHighlight();
    } else {
      status.textContent = '저장 실패';
      status.className = 'status';
    }
  });
}

// flush any note edit that's still waiting on its debounce timer
function flushPending() {
  if (saveTimer) {
    clearTimeout(saveTimer);
    saveTimer = null;
    doSave(page, box.value);
  }
}

function scheduleAutosave() {
  clearTimeout(saveTimer);
  status.textContent = '입력 중...';
  status.className = 'status';
  saveTimer = setTimeout(() => { saveTimer = null; doSave(page, box.value); }, AUTOSAVE_DELAY);
}

function goTo(p) {
  flushPending();
  flushSourcePending();
  p = Math.max(1, Math.min(total, p));
  page = p;
  img.src = `/img?id=$${encodeURIComponent(did)}&page=$${p}&t=$${Date.now()}`;
  pdfLink.href = `/pdf?id=$${encodeURIComponent(did)}#page=$${p}`;
  label.textContent = p;
  input.value = p;
  history.replaceState(null, '', '/deck?id=' + encodeURIComponent(did) + '&start=' + p);
  box.value = (notes[p] && notes[p].note) || '';
  status.textContent = '자동 저장 켜짐';
  status.className = 'status';
  pastePreviews.innerHTML = '';
  pasteStatus.textContent = '';
  prevBtn.disabled = p === 1;
  nextBtn.disabled = p === total;
  refreshJumplistHighlight();
  refreshDeleteUI();
  sourceError.hidden = true;
  sourceStatus.textContent = '';
  if (sourcePanelOpen) loadSource();
}

box.addEventListener('paste', (e) => {
  const items = e.clipboardData && e.clipboardData.items;
  if (!items) return;
  let imageItem = null;
  for (const item of items) {
    if (item.type && item.type.startsWith('image/')) { imageItem = item; break; }
  }
  if (!imageItem) return; // let normal text paste through
  e.preventDefault();
  const file = imageItem.getAsFile();
  const reader = new FileReader();
  reader.onload = () => {
    const dataUrl = reader.result;
    pasteStatus.textContent = '이미지 업로드 중...';
    pasteStatus.className = 'status';
    fetch('/api/upload_paste', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({id: did, page, data: dataUrl}),
    }).then(r => r.json()).then(res => {
      if (!res.ok) {
        pasteStatus.textContent = '업로드 실패: ' + (res.error || '');
        pasteStatus.className = 'status';
        return;
      }
      const marker = `[이미지 첨부: $${res.filename}]`;
      const pos = box.selectionStart;
      box.value = box.value.slice(0, pos) + marker + box.value.slice(pos);
      box.dispatchEvent(new Event('input'));
      pasteStatus.textContent = '이미지 저장됨: ' + res.filename;
      pasteStatus.className = 'status ok';
      const thumb = document.createElement('img');
      thumb.src = dataUrl;
      thumb.title = res.filename;
      pastePreviews.appendChild(thumb);
    }).catch(() => {
      pasteStatus.textContent = '업로드 실패';
      pasteStatus.className = 'status';
    });
  };
  reader.readAsDataURL(file);
});

box.addEventListener('input', scheduleAutosave);
prevBtn.onclick = () => goTo(page - 1);
nextBtn.onclick = () => goTo(page + 1);
input.onchange = () => goTo(Number(input.value));
document.getElementById('saveBtn').onclick = () => { clearTimeout(saveTimer); saveTimer = null; doSave(page, box.value); };

document.addEventListener('keydown', (e) => {
  const inBox = document.activeElement === box;
  if ((e.metaKey || e.ctrlKey) && e.key === 's') { e.preventDefault(); flushPending(); flushSourcePending(); return; }
  if (inBox) return;
  if (e.key === 'ArrowLeft') goTo(page - 1);
  if (e.key === 'ArrowRight') goTo(page + 1);
});

window.addEventListener('beforeunload', () => {
  if (saveTimer) {
    clearTimeout(saveTimer);
    const payload = JSON.stringify({id: did, page, note: box.value});
    navigator.sendBeacon('/api/save', new Blob([payload], {type: 'application/json'}));
  }
  if (sourceSaveTimer && sourceEditable) {
    clearTimeout(sourceSaveTimer);
    // save-only (rebuild=false): a rebuild is too slow to trust to sendBeacon on unload
    const payload = JSON.stringify({id: did, page: sourceLoadedPage, file: sourceLoadedFile, content: sourceBox.value, rebuild: false});
    navigator.sendBeacon('/api/source', new Blob([payload], {type: 'application/json'}));
  }
});

// While a rebuild renders in the background the old page images stay up;
// when the new render lands, refresh the current image (or reload if the
// page count changed -- the URL keeps the page).
let renderVersion = null;
const buildBadge = document.createElement('span');
buildBadge.className = 'status';
buildBadge.style.marginLeft = '8px';
document.querySelector('header h1').appendChild(buildBadge);
function pollRender() {
  fetch('/api/render_status?id=' + encodeURIComponent(did)).then(r => r.json()).then(st => {
    buildBadge.textContent = st.updating ? '· 새 빌드 반영 중…' : '';
    if (renderVersion !== null && st.version !== renderVersion) {
      if (st.n !== total) { location.reload(); return; }
      img.src = '/img?id=' + encodeURIComponent(did) + '&page=' + page + '&t=' + Date.now();
    }
    renderVersion = st.version;
  }).catch(() => {}).finally(() => setTimeout(pollRender, 2000));
}
pollRender();

buildJumplist();
const startParam = Number(new URLSearchParams(location.search).get('start'));
goTo(startParam >= 1 && startParam <= total ? startParam : 1);
</script>
$apply_js
</body></html>""")

BOARD_TMPL = Template("""<!doctype html><html><head><meta charset="utf-8">
""" + FAVICON_LINK + """
<title>$did · 스토리보드</title><style>$css</style></head><body>
<header>
  <a href="/">&larr; 목록</a>
  <a href="/deck?id=$did_quoted">페이지별 보기</a>
  <h1>$did &nbsp;·&nbsp; 스토리보드</h1>
  $apply_html
</header>
$apply_panel
<main>
<div class="board-toolbar">
  <button id="addFirstBtn">+ 맨 앞에 빈 페이지 추가</button>
  <span class="status" id="boardStatus">자동 저장 켜짐</span>
</div>
<div class="pending-banner" id="pendingBanner" hidden>
  <span id="pendingText"></span>
  <button id="resetBtn">전부 되돌리기</button>
</div>
<div class="board-grid" id="grid"></div>
<div id="endZone"></div>
</main>
<script>
const did = $did_json;
const notes = $notes_json;
let plan = $plan_json;
let dragIdx = null;

const grid = document.getElementById('grid');
const endZone = document.getElementById('endZone');
const statusEl = document.getElementById('boardStatus');
const pendingBanner = document.getElementById('pendingBanner');
const pendingText = document.getElementById('pendingText');

// a "page" item's id is "p<N>" for whatever page N it was at when the plan
// was last committed (applied to the real deck) -- if that no longer
// matches its current computed page, it's been dragged since and the
// reorder isn't live yet.
function originPage(item) {
  const m = /^p(\d+)$$/.exec(item.id || '');
  return m ? Number(m[1]) : null;
}

function newInsertItem() {
  return {
    id: 'ins-' + Date.now().toString(36) + Math.random().toString(36).slice(2, 7),
    type: 'insert',
    prompt: '',
    created_at: new Date().toISOString(),
  };
}

function insertAfter(idx) {
  plan.splice(idx + 1, 0, newInsertItem());
  render();
  savePlanNow();
}

function attachDrag(card, idx) {
  card.draggable = true;
  card.addEventListener('dragstart', () => { dragIdx = idx; card.classList.add('dragging'); });
  card.addEventListener('dragend', () => card.classList.remove('dragging'));
  card.addEventListener('dragover', (e) => e.preventDefault());
  card.addEventListener('drop', (e) => {
    e.preventDefault();
    if (dragIdx === null || dragIdx === idx) return;
    const moved = plan.splice(dragIdx, 1)[0];
    plan.splice(idx, 0, moved);
    dragIdx = null;
    render();
    savePlanNow();
  });
}

function render() {
  grid.innerHTML = '';
  let pendingMoved = 0, pendingInsert = 0, pendingDeleted = 0;
  let seq = 0; // 1-indexed position among page-type items only, computed live from array order

  plan.forEach((item, idx) => {
    const card = document.createElement('div');
    card.className = 'card' + (item.type === 'insert' ? ' insert' : '');

    if (item.type === 'page') {
      seq++;
      const refPage = originPage(item); // stable real-PDF-page reference; never touched by dragging
      const noted = notes[refPage] && notes[refPage].note;
      if (noted) card.classList.add('noted');
      const moved = refPage !== null && refPage !== seq;
      if (moved && !item.deleted) { card.classList.add('moved'); pendingMoved++; }
      if (item.deleted) { card.classList.add('deleted'); pendingDeleted++; }

      const img = document.createElement('img');
      img.src = `/img?id=$${encodeURIComponent(did)}&page=$${refPage}`;
      img.loading = 'lazy';
      img.draggable = false; // otherwise the browser's native image-drag hijacks the card drag
      img.onclick = () => { location.href = `/deck?id=$${encodeURIComponent(did)}&start=$${refPage}`; };
      card.appendChild(img);

      const label = document.createElement('div');
      label.className = 'pagenum';
      let labelText = `p.$${refPage}`;
      if (noted) labelText += ' · 노트 있음';
      if (item.deleted) labelText += ' · 삭제 예정(미반영)';
      else if (moved) labelText += ' · 순서 변경됨(미반영)';
      label.textContent = labelText;
      card.appendChild(label);
    } else {
      pendingInsert++;
      const ta = document.createElement('textarea');
      ta.placeholder = '이 자리에 넣을 내용을 설명해줘 (텍스트/이미지 지시사항)...';
      ta.value = item.prompt || '';
      ta.addEventListener('click', (e) => e.stopPropagation());
      ta.addEventListener('input', () => { item.prompt = ta.value; scheduleSavePlan(); });
      card.appendChild(ta);
      const label = document.createElement('div');
      label.className = 'pagenum';
      label.textContent = '새 페이지 (예정, 미반영)';
      card.appendChild(label);
    }

    const actions = document.createElement('div');
    actions.className = 'card-actions';
    if (!item.deleted) {
      const addBtn = document.createElement('button');
      addBtn.textContent = '+ 다음에 추가';
      addBtn.addEventListener('click', (e) => { e.stopPropagation(); insertAfter(idx); });
      actions.appendChild(addBtn);
    }
    if (item.type === 'insert') {
      const delBtn = document.createElement('button');
      delBtn.textContent = '삭제';
      delBtn.className = 'danger';
      delBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        plan.splice(idx, 1);
        render();
        savePlanNow();
      });
      actions.appendChild(delBtn);
    } else {
      const toggleBtn = document.createElement('button');
      toggleBtn.textContent = item.deleted ? '삭제 취소' : '삭제';
      toggleBtn.className = item.deleted ? 'undo' : 'danger';
      toggleBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        item.deleted = !item.deleted;
        render();
        savePlanNow();
      });
      actions.appendChild(toggleBtn);
    }
    card.appendChild(actions);

    attachDrag(card, idx);
    grid.appendChild(card);
  });

  const total = pendingMoved + pendingInsert + pendingDeleted;
  if (total > 0) {
    const parts = [];
    if (pendingMoved) parts.push(`순서 변경 $${pendingMoved}건`);
    if (pendingInsert) parts.push(`새 페이지 $${pendingInsert}건`);
    if (pendingDeleted) parts.push(`삭제 예정 $${pendingDeleted}건`);
    pendingText.textContent = `⚠ 아직 실제 슬라이드에 반영되지 않았어요 ($${parts.join(', ')}) — Claude에게 반영해달라고 하면 .tex에 적용돼.`;
    pendingBanner.hidden = false;
  } else {
    pendingBanner.hidden = true;
  }
}

endZone.addEventListener('dragover', (e) => e.preventDefault());
endZone.addEventListener('drop', (e) => {
  e.preventDefault();
  if (dragIdx === null) return;
  const moved = plan.splice(dragIdx, 1)[0];
  plan.push(moved);
  dragIdx = null;
  render();
  savePlanNow();
});

let planSaveTimer = null;
function scheduleSavePlan() {
  clearTimeout(planSaveTimer);
  statusEl.textContent = '입력 중...';
  statusEl.className = 'status';
  planSaveTimer = setTimeout(savePlanNow, 700);
}

function savePlanNow() {
  clearTimeout(planSaveTimer);
  planSaveTimer = null;
  statusEl.textContent = '저장 중...';
  statusEl.className = 'status';
  fetch('/api/plan', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({id: did, items: plan}),
  }).then(res => {
    statusEl.textContent = res.ok ? '자동 저장됨' : '저장 실패';
    statusEl.className = res.ok ? 'status ok' : 'status';
  });
}

document.getElementById('addFirstBtn').addEventListener('click', () => insertAfter(-1));
document.getElementById('resetBtn').addEventListener('click', () => {
  if (!confirm('순서 변경 · 새 페이지 · 삭제 표시를 전부 취소하고 현재 실제 상태로 되돌릴까?')) return;
  fetch('/api/plan/reset', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({id: did}),
  }).then(() => location.reload());
});

render();
</script>
$apply_js
</body></html>""")


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # keep console quiet

    def _send(self, status: int, body: bytes, content_type: str):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, html: str, status: int = 200):
        html = html.replace("</style>", APPLY_CSS + "</style>", 1)
        self._send(status, html.encode("utf-8"), "text/html; charset=utf-8")

    def _send_json(self, obj, status: int = 200):
        self._send(status, json.dumps(obj).encode("utf-8"), "application/json")

    def do_GET(self):
        parsed = urlparse(self.path)
        qs = parse_qs(parsed.query)
        try:
            if parsed.path == "/":
                return self.handle_index()
            if parsed.path == "/favicon.ico":
                return self.handle_favicon()
            if parsed.path == "/deck":
                return self.handle_deck(qs.get("id", [None])[0])
            if parsed.path == "/img":
                return self.handle_img(qs.get("id", [None])[0], qs.get("page", [None])[0])
            if parsed.path == "/pdf":
                return self.handle_pdf(qs.get("id", [None])[0])
            if parsed.path == "/api/source":
                return self.handle_api_source_get(qs.get("id", [None])[0], qs.get("page", [None])[0])
            if parsed.path == "/api/notes":
                return self.handle_api_notes(qs.get("id", [None])[0])
            if parsed.path == "/board":
                return self.handle_board(qs.get("id", [None])[0])
            if parsed.path == "/api/plan":
                return self.handle_api_plan(qs.get("id", [None])[0])
            if parsed.path == "/api/render_status":
                did = qs.get("id", [None])[0]
                if did not in find_decks():
                    raise FileNotFoundError()
                return self._send_json(render_status(did))
            if parsed.path in ("/api/apply/status", "/api/apply/preview"):
                did = qs.get("id", [None])[0]
                if did not in find_decks():
                    raise FileNotFoundError()
                fn = apply_status if parsed.path.endswith("status") else apply_preview
                return self._send_json(fn(did))
        except FileNotFoundError:
            return self._send_html("<h1>404</h1>not found", 404)
        except Exception as e:
            return self._send_html(f"<h1>500</h1><pre>{e}</pre>", 500)
        self._send_html("<h1>404</h1>", 404)

    def do_POST(self):
        if self.path == "/api/apply":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length) or b"{}")
            did, model = body.get("id"), body.get("model", "haiku")
            if did not in find_decks() or model not in APPLY_MODELS:
                return self._send_json({"ok": False, "error": "bad request"}, 400)
            ok, err = start_apply(did, model, str(body.get("extra", ""))[:4000])
            return self._send_json({"ok": ok, "error": err}, 200 if ok else 409)
        if self.path == "/api/save":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length) or b"{}")
            did = body.get("id")
            page = int(body.get("page"))
            note = body.get("note", "")
            if did not in find_decks():
                return self._send_json({"ok": False, "error": "unknown deck"}, 400)
            save_note(did, page, note)
            return self._send_json({"ok": True})
        if self.path == "/api/resolve":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length) or b"{}")
            did = body.get("id")
            page = int(body.get("page"))
            action = body.get("action", "")
            if did not in find_decks():
                return self._send_json({"ok": False, "error": "unknown deck"}, 400)
            ok = resolve_note(did, page, action)
            if not ok:
                return self._send_json({"ok": False, "error": "no pending note on that page"}, 404)
            return self._send_json({"ok": True})
        if self.path == "/api/plan":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length) or b"{}")
            did = body.get("id")
            items = body.get("items")
            if did not in find_decks() or not isinstance(items, list):
                return self._send_json({"ok": False, "error": "bad request"}, 400)
            save_plan(did, items)
            return self._send_json({"ok": True})
        if self.path == "/api/plan/resolve_insert":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length) or b"{}")
            did = body.get("id")
            item_id = body.get("item_id")
            action = body.get("action", "")
            if did not in find_decks():
                return self._send_json({"ok": False, "error": "unknown deck"}, 400)
            ok = resolve_insert(did, item_id, action)
            if not ok:
                return self._send_json({"ok": False, "error": "no such insert item"}, 404)
            return self._send_json({"ok": True})
        if self.path == "/api/plan/resolve_delete":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length) or b"{}")
            did = body.get("id")
            item_id = body.get("item_id")
            action = body.get("action", "")
            if did not in find_decks():
                return self._send_json({"ok": False, "error": "unknown deck"}, 400)
            ok = resolve_delete(did, item_id, action)
            if not ok:
                return self._send_json({"ok": False, "error": "no such deleted item"}, 404)
            return self._send_json({"ok": True})
        if self.path == "/api/plan/commit":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length) or b"{}")
            did = body.get("id")
            action = body.get("action", "")
            if did not in find_decks():
                return self._send_json({"ok": False, "error": "unknown deck"}, 400)
            ok = commit_plan(did, action)
            if not ok:
                return self._send_json({"ok": False, "error": "no plan to commit"}, 404)
            return self._send_json({"ok": True})
        if self.path == "/api/plan/reset":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length) or b"{}")
            did = body.get("id")
            if did not in find_decks():
                return self._send_json({"ok": False, "error": "unknown deck"}, 400)
            reset_plan(did)
            return self._send_json({"ok": True})
        if self.path == "/api/plan/set_deleted":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length) or b"{}")
            did = body.get("id")
            page = body.get("page")
            deleted = bool(body.get("deleted"))
            if did not in find_decks():
                return self._send_json({"ok": False, "error": "unknown deck"}, 400)
            ok = set_page_deleted(did, int(page), deleted)
            if not ok:
                return self._send_json({"ok": False, "error": "no such page in plan"}, 404)
            return self._send_json({"ok": True})
        if self.path == "/api/upload_paste":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length) or b"{}")
            did = body.get("id")
            page = body.get("page")
            data_url = body.get("data", "")
            if did not in find_decks():
                return self._send_json({"ok": False, "error": "unknown deck"}, 400)
            try:
                name = save_pasted_image(did, int(page), data_url)
            except ValueError as e:
                return self._send_json({"ok": False, "error": str(e)}, 400)
            return self._send_json({"ok": True, "filename": name})
        if self.path == "/api/source":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length) or b"{}")
            did = body.get("id")
            page = body.get("page")
            content = body.get("content", "")
            rebuild = body.get("rebuild", True)
            if did not in find_decks():
                return self._send_json({"ok": False, "error": "unknown deck"}, 400)
            ok, err = write_page_source(did, int(page), content, rebuild=bool(rebuild),
                                        file=str(body.get("file") or ""))
            if not ok:
                return self._send_json({"ok": False, "error": err}, 400)
            return self._send_json({"ok": True, "rebuilt": bool(rebuild)})
        self._send_json({"ok": False}, 404)

    def handle_api_notes(self, did):
        if did is None:
            raise FileNotFoundError()
        did = unquote(did)
        if did not in find_decks():
            raise FileNotFoundError()
        self._send_json(load_notes(did))

    def handle_api_plan(self, did):
        if did is None:
            raise FileNotFoundError()
        did = unquote(did)
        if did not in find_decks():
            raise FileNotFoundError()
        self._send_json(load_plan(did))

    def handle_favicon(self):
        svg = (
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">'
            '<text y=".9em" font-size="90">\U0001F3AC</text></svg>'
        )
        self._send(200, svg.encode("utf-8"), "image/svg+xml")

    def handle_index(self):
        decks = find_decks()
        rows = []
        for did in decks:
            n = len(load_notes(did))
            badge = f'<span class="badge{" zero" if n == 0 else ""}">{n}개 주석</span>'
            rows.append(
                f'<li><a href="/deck?id={quote(did)}">'
                f'<span class="deck-id">{did}</span>{badge}</a>'
                f'<a href="/board?id={quote(did)}" style="border-left:1px solid #e2e2df;">스토리보드</a>'
                f'<a href="/pdf?id={quote(did)}" target="_blank" rel="noopener" style="border-left:1px solid #e2e2df;">PDF</a></li>'
            )
        html = INDEX_TMPL.substitute(css=PAGE_CSS, rows="\n".join(rows) or "<li>덱을 찾지 못했습니다.</li>")
        self._send_html(html)

    def handle_deck(self, did):
        if did is None:
            raise FileNotFoundError()
        did = unquote(did)
        if did not in find_decks():
            raise FileNotFoundError()
        total = ensure_rendered(did)
        notes = load_notes(did)
        deleted_pages = [
            it["page"] for it in load_plan(did)
            if it.get("type") == "page" and it.get("deleted")
        ]
        html = VIEWER_TMPL.substitute(
            css=PAGE_CSS,
            did=did,
            did_quoted=quote(did),
            did_json=json.dumps(did),
            total=total,
            notes_json=json.dumps(notes),
            deleted_json=json.dumps(deleted_pages),
            sections_json=json.dumps(load_sections(did)),
            **apply_parts(did),
        )
        self._send_html(html)

    def handle_board(self, did):
        if did is None:
            raise FileNotFoundError()
        did = unquote(did)
        if did not in find_decks():
            raise FileNotFoundError()
        ensure_rendered(did)
        notes = load_notes(did)
        plan = load_plan(did)
        html = BOARD_TMPL.substitute(
            css=PAGE_CSS,
            did=did,
            did_quoted=quote(did),
            did_json=json.dumps(did),
            notes_json=json.dumps(notes),
            plan_json=json.dumps(plan),
            **apply_parts(did),
        )
        self._send_html(html)

    def handle_img(self, did, page):
        if did is None or page is None:
            raise FileNotFoundError()
        did = unquote(did)
        if did not in find_decks():
            raise FileNotFoundError()
        total = ensure_rendered(did)
        page_n = int(page)
        path = page_image_path(did, page_n, total)
        if not path.exists():
            time.sleep(0.4)  # mid-swap of a fresh render: the new dir lands a moment later
            total = ensure_rendered(did)
            path = page_image_path(did, page_n, total)
        if not path.exists():
            raise FileNotFoundError()
        self._send(200, path.read_bytes(), "image/png")

    def handle_pdf(self, did):
        if did is None:
            raise FileNotFoundError()
        did = unquote(did)
        if did not in find_decks():
            raise FileNotFoundError()
        pdf = deck_pdf(did)
        if not pdf.exists():
            raise FileNotFoundError()
        self._send(200, pdf.read_bytes(), "application/pdf")

    def handle_api_source_get(self, did, page):
        if did is None or page is None:
            raise FileNotFoundError()
        did = unquote(did)
        if did not in find_decks():
            raise FileNotFoundError()
        path = page_source_path(did, int(page))
        if path is None:
            return self._send_json({"ok": True, "editable": False})
        kind = page_file_map(did)[int(page)][1]
        return self._send_json({"ok": True, "editable": True, "content": path.read_text(encoding="utf-8"),
                                "file": f"pages/{path.name}", "kind": kind})


def warm_cache(workers: int = 4):
    """Pre-render every deck's pages in the background so opening a deck's
    /deck or /board is instant instead of blocking on pdftoppm on first
    click. Runs on a daemon thread, decks rendered in parallel (pdftoppm is
    CPU-bound, one process per deck) -- the server answers already-cached
    decks immediately while the rest warm up behind it."""
    decks = find_decks()
    print(f"Warming page-image cache for {len(decks)} decks ({workers} at a time)...")
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(ensure_rendered, did): did for did in decks}
        for fut in as_completed(futures):
            did = futures[fut]
            try:
                fut.result()
            except Exception as e:
                detail = e.stderr.decode(errors="replace").strip() if isinstance(e, subprocess.CalledProcessError) and e.stderr else str(e)
                print(f"  skip {did}: {detail}")
            else:
                done += 1
    print(f"Cache warm-up done ({done}/{len(decks)} decks).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-warm", action="store_true", help="skip pre-rendering all decks on startup")
    ap.add_argument("--no-watch", action="store_true", help="disable automatic rebuild on .tex changes")
    args = ap.parse_args()
    global SERVER_PORT
    SERVER_PORT = args.port
    # Redirected stdout is block-buffered, which makes the warm-up progress log
    # look empty (and the server look hung) when started with `> server.log`.
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except AttributeError:
        pass
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Slide review server: http://localhost:{args.port}  (Ctrl+C to stop)")
    if not args.no_warm:
        threading.Thread(target=warm_cache, daemon=True).start()
    if not args.no_watch:
        start_tex_watcher()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
