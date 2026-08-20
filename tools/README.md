# book-ml restructuring tools

Utility scripts used while restructuring each chapter into the 18-week plan's
N.1/N.2/N.3 block format (one file per class block, real clickable sidebar
sub-links). See the `book-ml-3block-restructure` memory for full context and
current progress.

## split_chapter.py

Splits a chapter overview file that already has exactly 3 `## N.M Title`
headings into a short overview page + `chapterNN/1.md`, `2.md`, `3.md`.
Fixes `../images/` relative paths one level deeper. The `## 연습문제`
trailer (if present) gets appended to block 3.

```bash
python tools/split_chapter.py kor/src/ml1/chapterNN.md
```

Prerequisite: the source file must already be restructured into exactly
3 `## N.M Title` blocks (with `###` narrative subheadings inside each,
matching the plan's block1/block2/block3 content) before running this.
After running, you still need to:
1. Add the 3 sub-item links to `SUMMARY.md` (no `N.M ` prefix in the link
   text — mdBook auto-numbers via the fold structure; see memory for the
   duplicate-numbering bug this avoids).
2. `mdbook build` in `kor/` (or `eng/`).
3. Run `verify_html.py` on the built output.
4. Commit + push (source `.md` + rebuilt `docs/` together).

## verify_html.py

Scans built HTML for two recurring corruption bugs (see the
`book-ml-wikibook` memory for full bug-pattern history):
- Mismatched `\(`/`\)` or `\[`/`\]` MathJax delimiter counts.
- `<em>`/`<strong>` tags bleeding into `\underbrace{...}_{...}`-style math
  (CommonMark underscore-flanking bug).

```bash
python tools/verify_html.py "docs/kor/ml1/chapter01.html" "docs/kor/ml1/chapter01/*.html"
```

Exits 0 either way; read the printed "issues found" count.
