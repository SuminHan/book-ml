#!/usr/bin/env python3
"""Mark newly-added prose (vs. a baseline git ref) blue in the built mdBook HTML.

Workflow:
  1. cq (or a human) edits kor/src/**/*.md, adding to existing sections.
  2. `mdbook build` (in kor/) regenerates docs/kor/**/*.html.
  3. This script diffs each changed .md file against a baseline git ref
     (default: the `pre-cq-expansion` tag) at *block* granularity (paragraphs,
     list items, headings, table rows — fenced code blocks are treated as a
     single atomic block and are always SKIPPED, per house style: code keeps
     the normal theme, only new prose/text gets colored), finds the
     corresponding elements in the already-built HTML, and tags them with
     class="cq-new" (colored via kor/theme/custom.css).

Usage:
  python3 tools/highlight_new_content.py [--base-ref pre-cq-expansion] [--dry-run]

Must be run from the book-ml repo root (or anywhere; paths are repo-relative
internally). Re-run any time after `mdbook build` to refresh the markers —
it always re-derives everything from git + the freshly built HTML, so it's
safe to run repeatedly.
"""
import argparse
import re
import subprocess
import sys
from difflib import SequenceMatcher
from pathlib import Path

try:
    from bs4 import BeautifulSoup
except ImportError:
    sys.exit(
        "beautifulsoup4 not installed. "
        "Try: /home/smhan/miniconda3/bin/pip install beautifulsoup4 "
        "(or `conda install -n base -c conda-forge beautifulsoup4`)"
    )

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "kor" / "src"
BUILD_DIR = REPO_ROOT / "docs" / "kor"

# Tags we're willing to mark. Deliberately excludes anything code-related
# (pre/code) — those must keep the normal theme per house style.
CANDIDATE_TAGS = ["p", "li", "h1", "h2", "h3", "h4", "h5", "h6",
                  "td", "th", "blockquote", "dt", "dd"]

MD_SYNTAX_RE = re.compile(r"[#*_`\[\]()\\|>~-]")
WS_RE = re.compile(r"\s+")


def normalize(text: str) -> str:
    text = MD_SYNTAX_RE.sub(" ", text)
    text = WS_RE.sub(" ", text)
    return text.strip().lower()


def split_blocks(md_text: str):
    """Split markdown into blank-line-delimited blocks, keeping fenced code
    blocks (```...```) intact as one block even though they may contain
    blank lines internally."""
    lines = md_text.split("\n")
    blocks, cur, in_fence = [], [], False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            cur.append(line)
            continue
        if not in_fence and stripped == "":
            if cur:
                blocks.append("\n".join(cur))
                cur = []
        else:
            cur.append(line)
    if cur:
        blocks.append("\n".join(cur))
    return [b for b in blocks if b.strip()]


def is_code_block(block: str) -> bool:
    return block.strip().startswith("```")


def git_show(ref: str, path: str) -> str:
    try:
        out = subprocess.run(
            ["git", "show", f"{ref}:{path}"], cwd=REPO_ROOT,
            capture_output=True, text=True, check=True,
        )
        return out.stdout
    except subprocess.CalledProcessError:
        return ""  # file didn't exist at that ref (new file)


def changed_md_files(base_ref: str):
    out = subprocess.run(
        ["git", "diff", "--name-only", base_ref, "--", "kor/src"],
        cwd=REPO_ROOT, capture_output=True, text=True, check=True,
    )
    return [REPO_ROOT / p for p in out.stdout.splitlines() if p.endswith(".md")]


def new_blocks_for(md_path: Path, base_ref: str):
    rel = md_path.relative_to(REPO_ROOT).as_posix()
    old_text = git_show(base_ref, rel)
    new_text = md_path.read_text() if md_path.exists() else ""
    old_blocks = split_blocks(old_text)
    new_blocks = split_blocks(new_text)
    sm = SequenceMatcher(a=old_blocks, b=new_blocks, autojunk=False)
    result = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag in ("insert", "replace"):
            result.extend(new_blocks[j1:j2])
    return result


def block_needle(block: str, min_len=12, max_len=60):
    norm = normalize(block)
    if len(norm) < min_len:
        return None
    return norm[:max_len]


def html_path_for(md_path: Path) -> Path:
    rel = md_path.relative_to(SRC_DIR)
    return BUILD_DIR / rel.with_suffix(".html")


def mark_html(html_path: Path, needles, dry_run=False) -> int:
    if not html_path.exists():
        print(f"  ! no built HTML at {html_path} (run `mdbook build` in kor/ first) — skipped")
        return 0
    soup = BeautifulSoup(html_path.read_text(), "html.parser")
    marked = 0
    used = set()
    for tag in soup.find_all(CANDIDATE_TAGS):
        # skip anything living inside <pre>/<code> (shouldn't normally match
        # candidate tags anyway, but be defensive)
        if tag.find_parent(["pre", "code"]):
            continue
        norm = normalize(tag.get_text())
        for idx, needle in enumerate(needles):
            if idx in used or needle is None:
                continue
            if needle in norm:
                classes = tag.get("class", [])
                if "cq-new" not in classes:
                    classes.append("cq-new")
                    tag["class"] = classes
                    marked += 1
                used.add(idx)
                break
    if marked and not dry_run:
        html_path.write_text(str(soup))
    return marked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-ref", default="pre-cq-expansion")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    files = changed_md_files(args.base_ref)
    if not files:
        print(f"No changed .md files vs {args.base_ref} under kor/src/.")
        return

    total_marked = 0
    for md_path in files:
        blocks = new_blocks_for(md_path, args.base_ref)
        code_skipped = sum(1 for b in blocks if is_code_block(b))
        needles = [block_needle(b) for b in blocks if not is_code_block(b)]
        needles = [n for n in needles if n]
        html_path = html_path_for(md_path)
        marked = mark_html(html_path, needles, dry_run=args.dry_run)
        total_marked += marked
        rel = md_path.relative_to(REPO_ROOT)
        print(f"{rel}: {len(blocks)} new block(s) ({code_skipped} code, skipped), "
              f"{len(needles)} needle(s), {marked} HTML element(s) marked "
              f"-> {html_path.relative_to(REPO_ROOT)}")

    print(f"\nTotal: {total_marked} element(s) marked cq-new"
          f"{' (dry-run, nothing written)' if args.dry_run else ''}.")


if __name__ == "__main__":
    main()
