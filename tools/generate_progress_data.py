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


def running_task_ids():
    try:
        out = subprocess.run(["pgrep", "-af", "cq_run_once.py"],
                              capture_output=True, text=True).stdout
    except Exception:
        return set()
    ids = set()
    for line in out.splitlines():
        for m in re.finditer(r'batch_(?:prompts|logs)/([\w]+)\.(?:txt|log)', line):
            ids.add(m.group(1))
    return ids


def _search_done_failed(text):
    m = DONE_RE.search(text)
    if m:
        return "done", unescape(m.group(1))
    m = FAILED_RE.search(text)
    if m:
        return "failed", unescape(m.group(1))
    return None, None


def log_info(task_id):
    """Returns (terminal_status_or_None, summary_text_or_None).

    The DONE/FAILED marker can end up in either file: the main .log (if
    fmt_event's per-line truncation didn't cut it off) or only in the
    .wrapper.log's raw final `result` JSON (untruncated, but only written
    there) -- so check both, preferring whichever actually has it."""
    log = BATCH_LOGS / f"{task_id}.log"
    wrapper = BATCH_LOGS / f"{task_id}.wrapper.log"

    if log.exists():
        # subprocess output occasionally has a stray invalid byte (e.g. a
        # multi-byte UTF-8 sequence chopped in half by a mid-write kill) --
        # these are diagnostic logs, not final content, so decode leniently
        # rather than crashing the whole status pass over one bad log.
        log_text = log.read_text(encoding="utf-8", errors="replace")
        status, summary = _search_done_failed(log_text)
        if status:
            return status, summary
        if "=== TIMEOUT" in log_text:
            timeout_hit = True
        else:
            timeout_hit = False
    else:
        timeout_hit = False

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
                    return status, summary

    if timeout_hit:
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


def main():
    kor_sections, kor_counts = build_kor()
    eng_chapters, eng_counts = build_eng()

    data = {
        "generated_at": subprocess.run(["date", "-Iseconds"], capture_output=True, text=True).stdout.strip(),
        "base_ref": BASE_REF,
        "kor": {"granularity": "section", "counts": kor_counts, "total": len(kor_sections), "items": kor_sections},
        "eng": {"granularity": "chapter", "counts": eng_counts, "total": len(eng_chapters), "items": eng_chapters,
                "note": "영어판은 아직 kor처럼 N.1/N.2/N.3 절로 안 나뉘어 있어 챕터 단위로만 추적됨"},
    }
    print(json.dumps(data, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
