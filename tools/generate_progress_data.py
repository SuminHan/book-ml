#!/usr/bin/env python3
"""Generate JSON progress data for the cq expansion-progress dashboard.

Korean (kor/): already split into N.1/N.2/N.3 sections -- tracked at section
granularity (96 sections total), diffed against the `pre-cq-expansion` git
tag, cross-referenced with running cq batch processes + batch_logs/*.log for
status (not_started / in_progress / done / failed) and a short DONE summary.

English (eng/): NOT YET split into sections (still one flat page per
chapter) -- tracked at chapter granularity (32 chapters), same statuses,
until/unless it gets the same N.1/N.2/N.3 restructuring kor already has.

Usage: python3 tools/generate_progress_data.py > tools/progress_data.json
"""
import base64
import json
import re
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO = Path("/home/smhan/book-ml")
BATCH_LOGS = REPO / "tools" / "batch_logs"
BASE_REF = "pre-cq-expansion"

CH_RE = re.compile(r'^- \[(?P<title>.+?)\]\((?P<book>ml[12])/chapter(?P<num>\d+)\.md\)$')
SEC_RE = re.compile(r'^    - \[(?P<title>.+?)\]\((?P<book>ml[12])/chapter(?P<cnum>\d+)/(?P<snum>\d+)\.md\)$')
DONE_RE = re.compile(r'DONE:\s*(.+?)(?:\\n\\n|\.\.\.\[\d+chars\]|$)', re.MULTILINE)
FAILED_RE = re.compile(r'FAILED:\s*(.+?)(?:\\n\\n|\.\.\.\[\d+chars\]|$)', re.MULTILINE)


def parse_kor_summary():
    text = (REPO / "kor" / "src" / "SUMMARY.md").read_text()
    chapters, sections = {}, []
    for line in text.splitlines():
        m = CH_RE.match(line)
        if m:
            chapters[(m["book"], m["num"])] = m["title"]
            continue
        m = SEC_RE.match(line)
        if m:
            sections.append({"book": m["book"], "chapter": m["cnum"],
                              "section": m["snum"], "title": m["title"]})
    for s in sections:
        s["chapter_title"] = chapters.get((s["book"], s["chapter"]), "")
    return sections


def parse_eng_chapters():
    text = (REPO / "eng" / "src" / "SUMMARY.md").read_text()
    chapters = []
    for line in text.splitlines():
        m = CH_RE.match(line)
        if m:
            chapters.append({"book": m["book"], "chapter": m["num"], "title": m["title"]})
    return chapters


def git_word_count(ref, path):
    try:
        out = subprocess.run(["git", "show", f"{ref}:{path}"], cwd=REPO,
                              capture_output=True, text=True, check=True).stdout
    except subprocess.CalledProcessError:
        return 0
    return len(out.split())


def current_word_count(path):
    p = REPO / path
    return len(p.read_text().split()) if p.exists() else 0


def current_text(path):
    p = REPO / path
    return p.read_text() if p.exists() else ""


MAIN_RE = re.compile(r"<main>(.*?)</main>", re.DOTALL)


IMG_SRC_RE = re.compile(r'src="([^"]+)"')
MIME_BY_EXT = {".svg": "image/svg+xml", ".png": "image/png",
               ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".gif": "image/gif"}


ZOOM_FIGURE_RE = re.compile(
    r'<label class="checkbox-label"><input class="checkbox-img"[^>]*>'
    r'<img[^>]*alt="([^"]*)"[^>]*>'
    r'<span class="img-wrapper">.*?</span></label>',
    re.DOTALL,
)
BARE_IMG_RE = re.compile(r'<img([^>]*)>')
ALT_RE = re.compile(r'alt="([^"]*)"')


def figure_caption(alt_text):
    return f'<p class="fig-caption">\U0001F5BC️ {alt_text}</p>' if alt_text else ""


def inline_local_images(html, base_dir):
    """Don't embed image bytes at all (base64 SVGs blow past the artifact's
    16MB cap once a couple dozen sections are done) -- local figures (the
    mdbook click-to-zoom widget, or a bare <img>) get replaced with just
    their alt-text as a caption line. External absolute srcs (Colab badge)
    are left alone since those are tiny/hosted elsewhere."""
    def zoom_repl(m):
        return figure_caption(m.group(1))
    html = ZOOM_FIGURE_RE.sub(zoom_repl, html)

    def bare_repl(m):
        attrs = m.group(1)
        if "http://" in attrs or "https://" in attrs or "data:" in attrs:
            return m.group(0)  # external (e.g. Colab badge) -- leave as-is
        alt_m = ALT_RE.search(attrs)
        return figure_caption(alt_m.group(1) if alt_m else "")
    return BARE_IMG_RE.sub(bare_repl, html)


def built_html_content(book, chapter, section):
    """Extract the rendered <main>...</main> body from the mdbook-built
    docs/kor/<book>/chapterNN/N.html for this section (real CommonMark
    rendering -- links/lists/tables all correct, unlike a hand-rolled
    markdown-lite parser). Caller must have run `mdbook build` first."""
    html_path = REPO / "docs" / "kor" / book / f"chapter{chapter}" / f"{section}.html"
    if not html_path.exists():
        return None
    m = MAIN_RE.search(html_path.read_text())
    if not m:
        return None
    return inline_local_images(m.group(1).strip(), html_path.parent)


_RESTRUCTURE_ID_RE = re.compile(r'^restructure_kor_src_(ml\d)_chapter(\d+)(?:_(\d+))?$')


def _normalize_task_id(raw_id):
    """restructure_expand_orchestrator.py names tasks after their file path
    (restructure_kor_src_ml1_chapter13_1) rather than the {book}_ch{ch}_{sec}
    convention every other orchestrator uses -- without this, a section
    being actively rewritten by that orchestrator never matches `task_id in
    running` below and its grid cell never turns "in progress" (found via a
    live screenshot: the cell just kept showing its old status while cq was
    mid-edit). Returns the equivalent {book}_ch{ch}_{sec-or-opener} id, or
    None if raw_id doesn't match that pattern."""
    if raw_id == "restructure_ml2_ch15_worldmodel":
        # one-off task covering two files at once
        return ["ml2_ch15_4", "ml2_ch15_opener"]
    if raw_id == "restructure_ml2_ch14_av_sim":
        return ["ml2_ch14_4", "ml2_ch14_opener"]
    m = _RESTRUCTURE_ID_RE.match(raw_id)
    if not m:
        return []
    book, ch, sec = m.group(1), m.group(2), m.group(3)
    return [f"{book}_ch{ch}_{sec}" if sec else f"{book}_ch{ch}_opener"]


def running_task_ids():
    try:
        out = subprocess.run(["pgrep", "-af", "cq_run_once.py"],
                              capture_output=True, text=True).stdout
    except Exception:
        return set()
    ids = set()
    for line in out.splitlines():
        for m in re.finditer(r'batch_(?:prompts|logs)/([\w]+)\.(?:txt|log)', line):
            raw_id = m.group(1)
            ids.add(raw_id)
            ids.update(_normalize_task_id(raw_id))
    return ids


def _task_kind(task_id):
    if task_id.startswith("reflib_"):
        return "레퍼런스 라이브러리"
    if task_id.startswith("capt_"):
        return "노트북 캡션 영어 번역"
    if task_id.startswith("boldfix_"):
        return "볼드/수식 렌더링 버그 수정"
    if task_id.startswith("restructure_"):
        return "커리큘럼 구조개편 (13/14/15/4장)"
    if task_id.endswith("_cite"):
        return "각주/그림 삽입"
    if "_opener" in task_id:
        return "챕터 개요 확장"
    return "절 확장/마무리"


def running_tasks_detail():
    """Richer live view for the dashboard's "지금 실행 중" panel: task_id,
    which pipeline it belongs to, and an absolute UTC start timestamp
    (derived from `ps etimes=` on the cq_run_once.py PID -- stable even
    though the process itself was launched with start_new_session=True,
    since /proc's start time isn't affected by session detachment).
    An absolute timestamp instead of a frozen elapsed-seconds count so the
    dashboard's own JS can tick it live against the *viewer's* clock --
    a static "18초" is already wrong by the time someone opens the page."""
    try:
        out = subprocess.run(
            ["ps", "-eo", "pid,etimes,args"], capture_output=True, text=True
        ).stdout
    except Exception:
        return []
    now = datetime.now(timezone.utc)
    tasks = []
    for line in out.splitlines():
        if "cq_run_once.py" not in line or "grep" in line:
            continue
        parts = line.strip().split(None, 2)
        if len(parts) < 3:
            continue
        pid, etimes, args = parts
        m = re.search(r'batch_(?:prompts|logs)/([\w]+)\.(?:txt|log)', args)
        if not m:
            continue
        task_id = m.group(1)
        started_at = (now - timedelta(seconds=int(etimes))).isoformat() if etimes.isdigit() else None
        tasks.append({
            "task_id": task_id, "pid": pid,
            "started_at": started_at,
            "kind": _task_kind(task_id),
        })
    # de-dupe (the same task_id can appear twice in `args` matches across
    # wrapper/child processes) keeping the lowest PID (the original)
    seen, out_list = {}, []
    for t in sorted(tasks, key=lambda x: x["pid"]):
        if t["task_id"] in seen:
            continue
        seen[t["task_id"]] = True
        out_list.append(t)
    return sorted(out_list, key=lambda x: x["started_at"] or "")


def _search_done_failed(text):
    m = DONE_RE.search(text)
    if m:
        return "done", unescape(m.group(1))
    m = FAILED_RE.search(text)
    if m:
        return "failed", unescape(m.group(1))
    return None, None


def _restructure_log_stems(task_id):
    """Inverse of _normalize_task_id(): given a canonical {book}_ch{ch}_{sec
    or 'opener'} task_id, name the restructure_expand_orchestrator.py (or
    standalone one-off) log stem(s) that actually cover it -- checked
    *before* the legacy `{task_id}.log` stem in log_info(), because a
    chapter that got repurposed by this session's restructuring (13/14/15
    LLM->graph-learning move, new ML2 14.4/15.4) can have a stale legacy
    log left over from before the rename, describing the *old* topic at
    that path. Returns [] if task_id doesn't look like a restructured slot."""
    special = {
        "ml2_ch15_4": ["restructure_ml2_ch15_worldmodel"],
        "ml2_ch15_opener": ["restructure_ml2_ch15_worldmodel"],
        "ml2_ch14_4": ["restructure_ml2_ch14_av_sim"],
        "ml2_ch14_opener": ["restructure_ml2_ch14_av_sim"],
    }
    if task_id in special:
        return special[task_id]
    m = re.match(r'^(ml\d)_ch(\d+)_(opener|\d+)$', task_id)
    if not m:
        return []
    book, ch, sec = m.groups()
    stem = f"restructure_kor_src_{book}_chapter{ch}"
    return [stem] if sec == "opener" else [f"{stem}_{sec}"]


def _log_info_for_stem(stem):
    """Returns (terminal_status_or_None, summary_text_or_None, timeout_hit).

    The DONE/FAILED marker can end up in either file: the main .log (if
    fmt_event's per-line truncation didn't cut it off) or only in the
    .wrapper.log's raw final `result` JSON (untruncated, but only written
    there) -- so check both, preferring whichever actually has it."""
    log = BATCH_LOGS / f"{stem}.log"
    wrapper = BATCH_LOGS / f"{stem}.wrapper.log"

    timeout_hit = False
    if log.exists():
        # subprocess output occasionally has a stray invalid byte (e.g. a
        # multi-byte UTF-8 sequence chopped in half by a mid-write kill) --
        # these are diagnostic logs, not final content, so decode leniently
        # rather than crashing the whole status pass over one bad log.
        log_text = log.read_text(encoding="utf-8", errors="replace")
        status, summary = _search_done_failed(log_text)
        if status:
            return status, summary, False
        timeout_hit = "=== TIMEOUT" in log_text

    if wrapper.exists():
        for line in wrapper.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            result_text = obj.get("result")
            if isinstance(result_text, str):
                status, summary = _search_done_failed(result_text.replace("\n", "\\n"))
                if status:
                    return status, summary, False

    return None, None, timeout_hit


def log_info(task_id):
    """Returns (terminal_status_or_None, summary_text_or_None). If this
    slot was touched by the restructuring effort (a restructure_* log
    exists for it at all), trust *only* that -- never fall through to the
    legacy `{task_id}.log` stem, even if the restructure attempt itself
    only got as far as a timeout. Falling through would otherwise surface
    a stale legacy DONE summary describing the *old* topic that used to
    live at this chapter/section slot before the restructuring moved
    content around (e.g. old chapter 13 was LLMs; legacy `ml1_ch13_1.log`
    still has a real DONE for that, which has nothing to do with the new
    13.1 "Random Walk" content now at that path)."""
    restructure_stems = _restructure_log_stems(task_id)
    touched_by_restructure = any(
        (BATCH_LOGS / f"{stem}.log").exists() or (BATCH_LOGS / f"{stem}.wrapper.log").exists()
        for stem in restructure_stems
    )
    stems = restructure_stems if touched_by_restructure else [task_id]

    any_timeout = False
    for stem in stems:
        status, summary, timeout_hit = _log_info_for_stem(stem)
        if status:
            return status, summary
        any_timeout = any_timeout or timeout_hit
    if any_timeout:
        return "failed", "타임아웃"
    return None, None


def unescape(s):
    return s.replace("\\n", " ").strip()


def notebook_exists_for(book, chapter, section=None):
    nb_dir = REPO / "notebooks" / book
    if not nb_dir.exists():
        return False
    pat = f"chapter{chapter}_{section}_" if section else f"chapter{chapter}"
    return any(p.name.startswith(pat) for p in nb_dir.glob("*.ipynb"))


def built_chapter_html_content(book, chapter):
    """Same as built_html_content() but for the chapter-opener page itself
    (docs/kor/<book>/chapterNN.html), not a numbered N.1/N.2/N.3 section."""
    html_path = REPO / "docs" / "kor" / book / f"chapter{chapter}.html"
    if not html_path.exists():
        return None
    m = MAIN_RE.search(html_path.read_text())
    if not m:
        return None
    return inline_local_images(m.group(1).strip(), html_path.parent)


def build_chapter_opener_items():
    """Chapter-opener pages (kor/src/<book>/chapterNN.md) get the same
    word-count-diff + log_info() status treatment as a regular section, but
    at chapter granularity and with no notebook dependency -- represented
    as a synthetic section="0" item so the dashboard can slot it in as an
    extra row above .1/.2/.3 without a separate rendering path."""
    sections = parse_kor_summary()
    seen = {}
    for s in sections:
        key = (s["book"], s["chapter"])
        if key not in seen:
            seen[key] = s["chapter_title"]
    running = running_task_ids()
    out = []
    for (book, ch), title in sorted(seen.items()):
        path = f"kor/src/{book}/chapter{ch}.md"
        old_wc = git_word_count(BASE_REF, path)
        new_wc = current_word_count(path)
        task_id = f"{book}_ch{ch}_opener"
        summary = None
        # no notebook gate for openers -- word count alone tells us it's
        # done (also correctly recovers the 2 openers that hit `failed`
        # from a DONE-marker race against their timeout kill but had
        # already written a complete opener).
        if task_id in running:
            status = "in_progress"
        elif new_wc >= 400:
            status = "done"
        else:
            term, summary = log_info(task_id)
            status = "failed" if term == "failed" else "not_started"
        item = {
            "book": book, "chapter": ch, "section": "0",
            "title": "개요", "chapter_title": title,
            "task_id": task_id, "status": status,
            "word_count_before": old_wc, "word_count_after": new_wc,
            "notebook": False, "summary": summary, "is_capstone": False,
            "kind": "opener",
        }
        if status in ("done", "partial"):
            item["html_content"] = built_chapter_html_content(book, ch)
        out.append(item)
    return out


def build_kor():
    sections = parse_kor_summary()
    running = running_task_ids()
    out, counts = [], {"not_started": 0, "in_progress": 0, "done": 0, "partial": 0, "failed": 0}
    for s in sections:
        book, ch, sec = s["book"], s["chapter"], s["section"]
        path = f"kor/src/{book}/chapter{ch}/{sec}.md"
        old_wc = git_word_count(BASE_REF, path)
        new_wc = current_word_count(path)
        task_id = f"{book}_ch{ch}_{sec}"
        summary = None
        has_notebook = notebook_exists_for(book, ch, sec)
        substantial_growth = new_wc > old_wc * 1.3 and new_wc - old_wc > 100

        if task_id in running:
            status = "in_progress"
        else:
            term, summary = log_info(task_id)
            # "done" = cleanly finished (DONE marker) *and* has its notebook.
            # "partial" = real content was written (timed out / killed / no
            # notebook yet) -- still worth reading, just needs a follow-up
            # pass rather than a full redo.
            if term == "done" and has_notebook:
                status = "done"
            elif substantial_growth:
                status = "partial"
            elif term == "failed":
                status = "failed"
            else:
                status = "not_started"

        counts[status] += 1
        item = {
            "book": book, "chapter": ch, "section": sec,
            "title": s["title"], "chapter_title": s["chapter_title"],
            "task_id": task_id, "status": status,
            "word_count_before": old_wc, "word_count_after": new_wc,
            "notebook": notebook_exists_for(book, ch, sec),
            "summary": summary,
            "is_capstone": ch in ("08", "16"),
            "kind": "section",
        }
        if status in ("done", "partial"):
            item["html_content"] = built_html_content(book, ch, sec)
        out.append(item)

    for item in build_chapter_opener_items():
        counts[item["status"]] += 1
        out.append(item)
    return out, counts


def build_eng():
    chapters = parse_eng_chapters()
    out, counts = [], {"not_started": 0, "in_progress": 0, "done": 0, "failed": 0}
    for c in chapters:
        book, ch = c["book"], c["chapter"]
        path = f"eng/src/{book}/chapter{ch}.md"
        old_wc = git_word_count(BASE_REF, path)
        new_wc = current_word_count(path)
        status = "done" if (new_wc > old_wc * 1.3 and new_wc - old_wc > 100) else "not_started"
        counts[status] += 1
        out.append({
            "book": book, "chapter": ch, "title": c["title"],
            "status": status,
            "word_count_before": old_wc, "word_count_after": new_wc,
            "notebook": notebook_exists_for(book, ch),
        })
    return out, counts


FOOTNOTE_REF_RE = re.compile(r'\[\^([a-zA-Z0-9_-]+)\](?!:)')
ARXIV_ID_RE = re.compile(r'arXiv:\s*(\d{4}\.\d{4,5})', re.IGNORECASE)


def paper_url(venue):
    """Best-effort link for the reference list: an arXiv abstract-page URL
    if the venue string names one, else None (list just shows venue text)."""
    m = ARXIV_ID_RE.search(venue or "")
    return f"https://arxiv.org/abs/{m.group(1)}" if m else None


def find_citations(slug):
    """Every kor/src/<book>/chapterNN[/S].md that actually references
    [^slug] (a footnote *use*, not its `[^slug]: ...` definition line) --
    scanned fresh each build so this reflects whatever citation_orchestrator.py
    has inserted so far, no separate bookkeeping file needed."""
    hits = []
    for md_path in sorted((REPO / "kor" / "src").glob("ml*/chapter*/*.md")) + \
                    sorted((REPO / "kor" / "src").glob("ml*/chapter*.md")):
        text = md_path.read_text(encoding="utf-8", errors="replace")
        # a *reference* is any [^slug] occurrence on a line that isn't
        # itself the "[^slug]: ..." definition line
        refs = set()
        for line in text.splitlines():
            for m in FOOTNOTE_REF_RE.finditer(line):
                if line.lstrip().startswith(f"[^{m.group(1)}]:"):
                    continue
                refs.add(m.group(1))
        if slug not in refs:
            continue
        rel = md_path.relative_to(REPO / "kor" / "src")
        parts = rel.with_suffix("").parts  # e.g. ('ml2','chapter03','1') or ('ml2','chapter03')
        book = parts[0]
        chapter = parts[1].replace("chapter", "")
        section = parts[2] if len(parts) > 2 else None
        label = f"{book} {int(chapter)}.{section}" if section else f"{book} {int(chapter)} 개요"
        hits.append({"book": book, "chapter": chapter, "section": section, "label": label})
    return hits


def build_reference_library():
    """Gallery data for the reference-figure library (tools/reference_library_
    orchestrator.py's output: kor/src/images/ref_<slug>.png + .json). Images
    are small crops (tens to low hundreds of KB each), so unlike the
    section content this embeds them as base64 data URIs directly -- no
    need for the lightweight-caption-only trick that section figures use."""
    images_dir = REPO / "kor" / "src" / "images"
    items = []
    for png_path in sorted(images_dir.glob("ref_*.png")):
        slug = png_path.stem[len("ref_"):]
        json_path = images_dir / f"ref_{slug}.json"
        meta = {}
        if json_path.exists():
            try:
                meta = json.loads(json_path.read_text())
            except json.JSONDecodeError:
                meta = {}
        mime = MIME_BY_EXT.get(png_path.suffix, "image/png")
        b64 = base64.b64encode(png_path.read_bytes()).decode("ascii")
        items.append({
            "slug": slug,
            "title": meta.get("title", slug),
            "authors": meta.get("authors", ""),
            "year": meta.get("year"),
            "venue": meta.get("venue", ""),
            "footnote": meta.get("footnote", ""),
            "figure_caption": meta.get("figure_caption", ""),
            "has_metadata": bool(meta),
            "image_data_uri": f"data:{mime};base64,{b64}",
            "bytes": png_path.stat().st_size,
            "cited_in": find_citations(slug),
            "url": paper_url(meta.get("venue", "")),
        })
    return items


# Official Stanford URLs for the slug list in stanford_materials_orchestrator.py
# (from the user's own link-verification pass) -- linking out to these
# instead of embedding the PDFs: same "cite, don't redistribute" policy as
# the Sutton&Barto textbook, and these are copyrighted course slides (a
# single paper *figure* is a defensible fair-use quote; a whole deck isn't).
# status: "confirmed" (fetched live), "uncertain" (site reorganized since,
# probably works but not guaranteed identical), "missing" (no live link found).
_STANFORD_URLS = {
    "cs230_lecture02": ("https://cs230.stanford.edu/fall_2025/2/lecture_2.pdf", "uncertain"),
    "cs230_lecture05": ("https://cs230.stanford.edu/fall_2025/5/lecture_5.pdf", "uncertain"),
    "cs230_lecture06_main": ("https://cs230.stanford.edu/fall_2024/lecture_5.pdf", "uncertain"),
    "cs230_lecture06_guest": ("https://cs230.stanford.edu/fall_2024/lecture5_guest.pdf", "uncertain"),
    "cs230_lecture08": ("https://cs230.stanford.edu/fall_2025/8/lecture_8.pdf", "uncertain"),
    "cs230_lecture09": (None, "missing"),
    "cs230_lecture09_guest": (None, "missing"),
    "cs230_lecture10": ("https://cs230.stanford.edu/fall_2025/10/lecture_10.pdf", "uncertain"),
    "cs230_c5m4_transformer": ("https://cs230.stanford.edu/fall_2025/10/C5_W4.pdf", "uncertain"),
    "cs224n_derivatives": (None, "missing"),
    "cs224n_python_review": (
        "https://web.stanford.edu/class/cs224n/slides_w25/2024%20CS224N%20Python%20Review%20Session%20Slides.pptx.pdf",
        "uncertain",
    ),
}


def build_stanford_catalog():
    """External-reference catalog (tools/stanford_materials_orchestrator.py's
    output): local Stanford course PDFs the user already has, matched to
    book-ml chapters. No files embedded (the PDFs stay outside the repo,
    same policy as the Sutton&Barto textbook) -- just the metadata, plus a
    link to the official host so "view it" goes to Stanford's own site."""
    catalog_path = REPO / "tools" / "stanford_catalog.json"
    if not catalog_path.exists():
        return []
    try:
        items = json.loads(catalog_path.read_text())
    except json.JSONDecodeError:
        return []
    for it in items:
        url, status = _STANFORD_URLS.get(it.get("slug"), (None, "missing"))
        it["url"] = url
        it["url_status"] = status
    return items


def main():
    kor_sections, kor_counts = build_kor()
    eng_chapters, eng_counts = build_eng()
    reference_library = build_reference_library()

    data = {
        "generated_at": subprocess.run(["date", "-Iseconds"], capture_output=True, text=True).stdout.strip(),
        "base_ref": BASE_REF,
        "kor": {"granularity": "section", "counts": kor_counts, "total": len(kor_sections), "items": kor_sections},
        "eng": {"granularity": "chapter", "counts": eng_counts, "total": len(eng_chapters), "items": eng_chapters,
                "note": "영어판은 아직 kor처럼 N.1/N.2/N.3 절로 안 나뉘어 있어 챕터 단위로만 추적됨"},
        "reference_library": reference_library,
        "stanford_catalog": build_stanford_catalog(),
        "running_tasks": running_tasks_detail(),
    }
    print(json.dumps(data, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
