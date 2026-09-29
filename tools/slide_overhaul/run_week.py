import os, sys, json, time, pathlib, collections
S = os.environ["S"]; sys.path.insert(0, S)
from split_frames import split_frame
from qwen_client import call, build_user
from concurrent.futures import ThreadPoolExecutor
pages = sorted(pathlib.Path(sys.argv[1]).glob("*.tex")); out_path = sys.argv[2]

def overlap(h, src):
    hc = [c for c in h if not c.isspace()]; pool = collections.Counter(c for c in src if not c.isspace())
    hit = 0
    for c in hc:
        if pool[c] > 0: pool[c] -= 1; hit += 1
    return hit / max(1, len(hc))

import re
EXEMPT = re.compile(os.environ.get("EXEMPT", r"$^"))
def job(p):
    f = split_frame(p.read_text(encoding="utf-8"))
    if not f or not f["units"]: return {"page": p.stem, "skip": True, "title": f and f["title"]}
    if EXEMPT.search(f["title"]): return {"page": p.stem, "skip": True, "exempt": True, "title": f["title"], "units": f["units"], "before": sum(len(u["text"]) for u in f["units"])}
    items = [u["text"] for u in f["units"]]
    raw, usage = call(build_user(f["title"], items)); r = json.loads(raw)
    lab = {l["id"]: l for l in r["labels"]}
    chk = {"id_set": sorted(lab) == list(range(1, len(items)+1)) and len(r["labels"]) == len(items)}
    screen = 0; hl_bad = []
    for i, u in enumerate(f["units"], 1):
        l = lab.get(i, {"label": "NOTE", "headline": None})
        u["label"], u["headline"] = l["label"], l["headline"]
        if u["headline"] is not None:
            ok = len(u["headline"]) <= 40 and "\\" not in u["headline"] and overlap(u["headline"], u["text"]) >= 0.6
            if not ok or u["label"] != "KEEP": hl_bad.append(i); u["headline_dropped"] = u["headline"]; u["headline"] = None
        if u["label"] == "KEEP": screen += len(u["headline"] or u["text"])
    chk["keep_ge1"] = any(u["label"] == "KEEP" for u in f["units"])
    chk["headline_ok"] = not hl_bad
    chk["screen_le180"] = screen <= 180
    chk["screen_ge80"] = screen >= 80 or screen >= sum(len(x) for x in items) or f["has_visual"]
    nf = r["needs_figure"] and not f["has_visual"]
    return {"page": p.stem, "title": f["title"], "units": f["units"], "has_visual": f["has_visual"],
            "n_fixed": f["n_fixed"], "before": sum(len(x) for x in items), "screen": screen,
            "needs_figure": nf, "figure_query": r["figure_query"] if nf else None,
            "fig_overridden": bool(r["needs_figure"] and f["has_visual"]), "checks": chk, "hl_bad": hl_bad}

t = time.time()
with ThreadPoolExecutor(16) as ex: res = list(ex.map(job, pages))
dt = time.time() - t
json.dump(res, open(out_path, "w"), ensure_ascii=False, indent=1)
done = [r for r in res if not r.get("skip")]
print(f"{len(done)} frames in {dt:.1f}s -> {len(done)/dt:.2f} frames/s (skipped {len(res)-len(done)} no-text)")
print(f"  exempt (type rule): {sum(1 for r in res if r.get('exempt'))}")
for k in ["id_set", "keep_ge1", "headline_ok", "screen_le180", "screen_ge80"]:
    print(f"  {k:13s} pass {sum(r['checks'][k] for r in done)}/{len(done)}")
import statistics as st
print("  chars before median", st.median(r["before"] for r in done), "-> screen median", st.median(r["screen"] for r in done))
nu = sum(len(r["units"]) for r in done); nk = sum(u["label"] == "KEEP" for r in done for u in r["units"])
print(f"  units {nu}: KEEP {nk} / NOTE {nu-nk} | headlines kept {sum(1 for r in done for u in r['units'] if u['headline'])} dropped {sum(len(r['hl_bad']) for r in done)}")
print(f"  needs_figure {sum(r['needs_figure'] for r in done)} | fig flag overridden (already visual) {sum(r['fig_overridden'] for r in done)}")
