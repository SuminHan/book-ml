#!/usr/bin/env python3
"""Build gen/manifest.json from the kor/src chapter tree. Idempotent:
existing per-deck status/timestamps are preserved; only new decks are added."""
import json, os, re, glob, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # slides/
BOOK = os.path.dirname(ROOT)                                         # book-ml/
SRC  = os.path.join(BOOK, "kor", "src")
MANIFEST = os.path.join(ROOT, "gen", "manifest.json")

COURSES = [("ml1", "Machine Learning 1"), ("ml2", "Machine Learning 2")]

def title_of(md):
    with open(md, encoding="utf-8") as f:
        first = f.readline().strip()
    t = re.sub(r"^#\s*", "", first)
    t = re.sub(r"\[\^[^\]]+\]", "", t)          # strip footnote marks
    return t.strip()

def build():
    old = {}
    if os.path.exists(MANIFEST):
        for d in json.load(open(MANIFEST, encoding="utf-8"))["decks"]:
            old[d["id"]] = d

    decks = []
    for course, course_name in COURSES:
        chaps = sorted(glob.glob(os.path.join(SRC, course, "chapter*.md")))
        for i, ch in enumerate(chaps, start=1):
            stem = os.path.splitext(os.path.basename(ch))[0]        # chapterNN
            secdir = os.path.join(SRC, course, stem)
            secs = sorted(glob.glob(os.path.join(secdir, "*.md")))
            did = f"{course}-week{i:02d}"
            entry = {
                "id": did,
                "course": course,
                "course_name": course_name,
                "week": i,
                "chapter": stem,
                "title": title_of(ch),
                "n_sections": len(secs),
                "src_overview": os.path.relpath(ch, ROOT),
                "src_sections": [os.path.relpath(s, ROOT) for s in secs],
                "out_tex": f"kor/{did}.tex",
                "status": "pending",       # pending|running|done|failed|needs_review
                "frames": None,
                "pages": None,
                "attempts": 0,
                "started": None,
                "finished": None,
                "duration_s": None,
                "notes": "",
            }
            if did in old:
                for k in ("status","frames","pages","attempts","started",
                          "finished","duration_s","notes"):
                    if k in old[did]:
                        entry[k] = old[did][k]
            decks.append(entry)

    data = {"generated_by": "init_manifest.py", "n_decks": len(decks), "decks": decks}
    json.dump(data, open(MANIFEST, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"wrote {MANIFEST}  ({len(decks)} decks)")

if __name__ == "__main__":
    build()
