#!/usr/bin/env python3
"""Render gen/manifest.json ->
   - gen/dashboard.html      (local view, 30s auto-refresh)
   - gen/dashboard.art.html  (artifact snapshot: designed, no auto-refresh)
   - gen/STATUS.txt          (plain text)
"""
import json, os, time, datetime, html

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # slides/
MAN  = os.path.join(ROOT, "gen", "manifest.json")
OUT_LOCAL = os.path.join(ROOT, "gen", "dashboard.html")
OUT_ART   = os.path.join(ROOT, "gen", "dashboard.art.html")
TXT       = os.path.join(ROOT, "gen", "STATUS.txt")

# status -> (label, dot colour token)
ST = {
    "pending":      ("대기",  "muted"),
    "running":      ("생성 중", "accent"),
    "done":         ("완료",  "ok"),
    "failed":       ("실패",  "bad"),
    "needs_review": ("검토 필요", "warn"),
}
ORDER = ["running", "needs_review", "failed", "done", "pending"]


def fmt_dur(s):
    if not s:
        return "—"
    s = int(s)
    return f"{s//60}분 {s%60}초" if s >= 60 else f"{s}초"


def fmt_ts(t):
    return datetime.datetime.fromtimestamp(t).strftime("%m/%d %H:%M") if t else "—"


def load():
    d = json.load(open(MAN, encoding="utf-8"))
    decks = d["decks"]
    n = len(decks)
    by = {}
    for x in decks:
        by[x["status"]] = by.get(x["status"], 0) + 1
    done = by.get("done", 0)
    pct = round(100 * done / n) if n else 0
    durs = [x["duration_s"] for x in decks if x["status"] == "done" and x["duration_s"]]
    avg = sum(durs) / len(durs) if durs else None
    remaining = n - done - by.get("failed", 0) - by.get("needs_review", 0)
    eta = fmt_dur(avg * remaining) if (avg and remaining) else "—"
    return decks, n, by, done, pct, (fmt_dur(avg) if avg else "—"), eta


# ---------------------------------------------------------------- STATUS.txt
def write_txt():
    decks, n, by, done, pct, avg_txt, eta = load()
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(TXT, "w", encoding="utf-8") as f:
        f.write(f"KSA 슬라이드 생성 현황  ({now})\n")
        f.write(f"진행: {done}/{n} ({pct}%)   " +
                "  ".join(f"{k}={v}" for k, v in sorted(by.items())) + "\n")
        f.write(f"평균 소요: {avg_txt}   남은 예상: {eta}\n\n")
        for x in decks:
            f.write(f"  {x['status']:12} {x['id']:12} "
                    f"{('%2sf' % x['frames']) if x['frames'] else '   '} "
                    f"{('%2sp' % x['pages']) if x['pages'] else '   '} "
                    f"{fmt_dur(x['duration_s']):>9}  {x['title'][:50]}\n")


# ---------------------------------------------------------------- local view
def write_local():
    decks, n, by, done, pct, avg_txt, eta = load()
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    color = {"pending": "#6b7280", "running": "#1d4ed8", "done": "#15803d",
             "failed": "#b91c1c", "needs_review": "#b45309"}
    rows = []
    for x in decks:
        c = color.get(x["status"], "#333")
        rows.append(
            f"<tr><td class=m>{x['id']}</td><td>{x['week']}</td>"
            f"<td>{html.escape(x['title'])}</td><td style='text-align:center'>{x['n_sections']}</td>"
            f"<td style='color:{c};font-weight:700'>{x['status']}</td>"
            f"<td style='text-align:center'>{x['frames'] or '—'}</td>"
            f"<td style='text-align:center'>{x['pages'] or '—'}</td>"
            f"<td style='text-align:center'>{x['attempts']}</td>"
            f"<td class=m>{fmt_ts(x['started'])}</td><td>{fmt_dur(x['duration_s'])}</td>"
            f"<td style='color:#b45309'>{html.escape(x['notes'] or '')}</td></tr>")
    doc = f"""<!doctype html><html lang=ko><head><meta charset=utf-8>
<meta http-equiv=refresh content=30><title>KSA 슬라이드 생성 현황</title>
<style>body{{font:14px/1.5 system-ui,'Noto Sans KR',sans-serif;margin:0;background:#fafafa;color:#1a1a1a}}
.w{{max-width:1120px;margin:0 auto;padding:26px 20px 60px}}h1{{font-size:19px;margin:0 0 3px}}
.s{{color:#666;font-size:12px;margin-bottom:16px}}.bar{{height:13px;border-radius:7px;background:#e5e7eb;overflow:hidden;margin:9px 0}}
.bar>i{{display:block;height:100%;background:#2E3192;width:{pct}%}}table{{border-collapse:collapse;width:100%;background:#fff;font-size:13px;
box-shadow:0 1px 3px rgba(0,0,0,.08);border-radius:8px;overflow:hidden}}th,td{{padding:6px 8px;border-bottom:1px solid #eee;text-align:left;vertical-align:top}}
th{{background:#2E3192;color:#fff;font-size:11px}}.m{{font-family:ui-monospace,monospace;font-size:12px}}</style></head><body><div class=w>
<h1>KSA 강의 슬라이드 — 생성 현황</h1><div class=s>자동 갱신 30초 · {now}</div>
<div class=bar><i></i></div><div><b>{done} / {n}</b> 완료 ({pct}%) &nbsp; 평균 {avg_txt} &nbsp; 남은 예상 {eta}</div>
<p>{'  '.join(f'{k}={v}' for k,v in sorted(by.items()))}</p>
<table><tr><th>ID<th>주<th>제목<th>차시<th>상태<th>프레임<th>쪽<th>시도<th>시작<th>소요<th>비고</tr>
{''.join(rows)}</table></div></body></html>"""
    open(OUT_LOCAL, "w", encoding="utf-8").write(doc)


# ---------------------------------------------------------------- artifact
def write_art():
    decks, n, by, done, pct, avg_txt, eta = load()
    running   = by.get("running", 0)
    review    = by.get("needs_review", 0)
    failed    = by.get("failed", 0)
    pending   = by.get("pending", 0)
    snap = datetime.datetime.now().strftime("%Y년 %m월 %d일 %H:%M")

    # segmented progress bar
    seg = ""
    for st in ("done", "needs_review", "failed", "running"):
        v = by.get(st, 0)
        if v:
            seg += f'<span style="flex:{v};background:var(--{ST[st][1]})"></span>'
    if pending:
        seg += f'<span style="flex:{pending};background:var(--track)"></span>'

    def chip(label, value, tok):
        return (f'<div class="stat"><span class="dot" style="background:var(--{tok})"></span>'
                f'<b>{value}</b><span>{label}</span></div>')

    stats = "".join([
        chip("완료", done, "ok"),
        chip("생성 중", running, "accent"),
        chip("검토 필요", review, "warn"),
        chip("실패", failed, "bad"),
        chip("대기", pending, "muted"),
    ])

    def course_rows(course):
        out = []
        for x in [d for d in decks if d["course"] == course]:
            lbl, tok = ST.get(x["status"], ("?", "muted"))
            note = html.escape(x["notes"] or "")
            out.append(f"""<tr>
  <td class="mono">{x['week']:02d}</td>
  <td class="title">{html.escape(x['title'])}</td>
  <td class="num">{x['n_sections']}</td>
  <td><span class="pill pill-{tok}"><span class="pd"></span>{lbl}</span></td>
  <td class="num">{x['frames'] or '·'}</td>
  <td class="num">{x['pages'] or '·'}</td>
  <td class="num">{x['attempts'] or '·'}</td>
  <td class="num dim">{fmt_dur(x['duration_s'])}</td>
  <td class="note">{note}</td>
</tr>""")
        return "".join(out)

    ml1 = [d for d in decks if d["course"] == "ml1"]
    ml2 = [d for d in decks if d["course"] == "ml2"]
    ml1_done = sum(1 for d in ml1 if d["status"] == "done")
    ml2_done = sum(1 for d in ml2 if d["status"] == "done")

    doc = f"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>KSA 슬라이드 생성 현황</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+KR:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<style>
  :root{{
    --ground:#f7f5f0; --surface:#ffffff; --surface-2:#f2efe7;
    --ink:#1e1b33; --ink-dim:#6a6580; --line:#e4e0d4;
    --accent:#2E3192; --accent-soft:#e7e7f5;
    --gold:#a9976b; --track:#e6e2d6;
    --ok:#1f7a43; --warn:#b26a12; --bad:#b3261e; --muted:#8a8698;
    --shadow:0 1px 2px rgba(30,27,51,.06),0 8px 24px -12px rgba(30,27,51,.18);
  }}
  @media (prefers-color-scheme:dark){{
    :root:not([data-theme="light"]){{
      --ground:#141225; --surface:#1d1a30; --surface-2:#242138;
      --ink:#e9e7f2; --ink-dim:#9d99b4; --line:#2f2b45;
      --accent:#8f92e6; --accent-soft:#26264a;
      --gold:#c9b78a; --track:#2c2942;
      --ok:#4ec27d; --warn:#e0a24c; --bad:#ef6f66; --muted:#7d798f;
      --shadow:0 1px 2px rgba(0,0,0,.4),0 10px 30px -14px rgba(0,0,0,.6);
    }}
  }}
  :root[data-theme="dark"]{{
    --ground:#141225; --surface:#1d1a30; --surface-2:#242138;
    --ink:#e9e7f2; --ink-dim:#9d99b4; --line:#2f2b45;
    --accent:#8f92e6; --accent-soft:#26264a;
    --gold:#c9b78a; --track:#2c2942;
    --ok:#4ec27d; --warn:#e0a24c; --bad:#ef6f66; --muted:#7d798f;
    --shadow:0 1px 2px rgba(0,0,0,.4),0 10px 30px -14px rgba(0,0,0,.6);
  }}
  *{{box-sizing:border-box}}
  body{{margin:0;background:var(--ground);color:var(--ink);
    font-family:"IBM Plex Sans KR",system-ui,-apple-system,"Noto Sans KR",sans-serif;
    -webkit-font-smoothing:antialiased;line-height:1.5}}
  .wrap{{max-width:1080px;margin:0 auto;padding:44px 24px 72px}}
  .eyebrow{{font-family:"IBM Plex Mono",monospace;font-size:11px;letter-spacing:.16em;
    text-transform:uppercase;color:var(--gold);margin:0 0 10px}}
  h1{{font-size:26px;font-weight:700;margin:0;letter-spacing:-.01em;text-wrap:balance}}
  .snap{{color:var(--ink-dim);font-size:13px;margin-top:6px;
    font-family:"IBM Plex Mono",monospace}}

  .panel{{background:var(--surface);border:1px solid var(--line);border-radius:14px;
    box-shadow:var(--shadow);padding:22px 22px 20px;margin:28px 0}}
  .headline{{display:flex;align-items:baseline;gap:14px;flex-wrap:wrap}}
  .ratio{{font-family:"IBM Plex Mono",monospace;font-weight:600;font-size:30px;
    color:var(--accent);font-variant-numeric:tabular-nums}}
  .ratio small{{font-size:15px;color:var(--ink-dim);font-weight:500}}
  .meta{{margin-left:auto;font-size:13px;color:var(--ink-dim);
    font-family:"IBM Plex Mono",monospace;text-align:right}}
  .seg{{display:flex;height:12px;border-radius:6px;overflow:hidden;background:var(--track);
    margin:16px 0 4px}}
  .seg>span{{display:block}}
  .stats{{display:flex;gap:8px;flex-wrap:wrap;margin-top:16px}}
  .stat{{display:flex;align-items:center;gap:7px;background:var(--surface-2);
    border:1px solid var(--line);border-radius:9px;padding:7px 12px;font-size:13px}}
  .stat b{{font-family:"IBM Plex Mono",monospace;font-variant-numeric:tabular-nums;
    font-size:15px}}
  .stat span:last-child{{color:var(--ink-dim)}}
  .dot{{width:8px;height:8px;border-radius:50%;flex:0 0 auto}}

  h2{{font-size:14px;font-weight:600;margin:34px 0 10px;display:flex;
    align-items:baseline;gap:10px}}
  h2 .c{{font-family:"IBM Plex Mono",monospace;font-size:12px;color:var(--ink-dim);
    font-weight:500}}
  .tablewrap{{overflow-x:auto;border:1px solid var(--line);border-radius:12px;
    background:var(--surface);box-shadow:var(--shadow)}}
  table{{border-collapse:collapse;width:100%;font-size:13px;min-width:640px}}
  thead th{{font-family:"IBM Plex Mono",monospace;font-size:10.5px;letter-spacing:.1em;
    text-transform:uppercase;color:var(--ink-dim);font-weight:500;text-align:left;
    padding:11px 12px;border-bottom:1px solid var(--line);background:var(--surface-2)}}
  tbody td{{padding:10px 12px;border-bottom:1px solid var(--line);vertical-align:top}}
  tbody tr:last-child td{{border-bottom:none}}
  .mono{{font-family:"IBM Plex Mono",monospace;color:var(--ink-dim);
    font-variant-numeric:tabular-nums}}
  .num{{font-family:"IBM Plex Mono",monospace;text-align:right;
    font-variant-numeric:tabular-nums;white-space:nowrap}}
  .dim{{color:var(--ink-dim)}}
  .title{{max-width:340px}}
  .note{{max-width:230px;color:var(--warn);font-size:12px;line-height:1.4}}
  .pill{{display:inline-flex;align-items:center;gap:6px;padding:3px 10px;border-radius:999px;
    font-size:12px;font-weight:600;white-space:nowrap;border:1px solid transparent}}
  .pill .pd{{width:6px;height:6px;border-radius:50%;background:currentColor}}
  .pill-ok{{color:var(--ok);background:color-mix(in srgb,var(--ok) 12%,transparent);
    border-color:color-mix(in srgb,var(--ok) 30%,transparent)}}
  .pill-accent{{color:var(--accent);background:var(--accent-soft);
    border-color:color-mix(in srgb,var(--accent) 30%,transparent)}}
  .pill-warn{{color:var(--warn);background:color-mix(in srgb,var(--warn) 12%,transparent);
    border-color:color-mix(in srgb,var(--warn) 30%,transparent)}}
  .pill-bad{{color:var(--bad);background:color-mix(in srgb,var(--bad) 12%,transparent);
    border-color:color-mix(in srgb,var(--bad) 30%,transparent)}}
  .pill-muted{{color:var(--muted);background:color-mix(in srgb,var(--muted) 12%,transparent);
    border-color:color-mix(in srgb,var(--muted) 26%,transparent)}}
  footer{{margin-top:40px;font-size:12px;color:var(--ink-dim);line-height:1.7}}
  footer code{{font-family:"IBM Plex Mono",monospace;background:var(--surface-2);
    padding:1px 5px;border-radius:4px}}
</style>
</head>
<body>
<div class="wrap">
  <p class="eyebrow">KSA · Machine Learning 1 &amp; 2</p>
  <h1>강의 슬라이드 자동 생성 현황</h1>
  <p class="snap">스냅샷 · {snap} &nbsp;|&nbsp; 로컬 Qwen3.8-27B(cq) 파이프라인</p>

  <div class="panel">
    <div class="headline">
      <span class="ratio">{done}<small> / {n} 주차</small></span>
      <span class="ratio" style="font-size:18px">{pct}%</span>
      <span class="meta">평균 소요 {avg_txt}<br>남은 예상 {eta}</span>
    </div>
    <div class="seg">{seg}</div>
    <div class="stats">{stats}</div>
  </div>

  <h2>Machine Learning 1 <span class="c">— {ml1_done}/{len(ml1)} · 회귀·SVM·트리·CNN·트랜스포머·생성모델</span></h2>
  <div class="tablewrap"><table>
    <thead><tr><th>주</th><th>제목</th><th>차시</th><th>상태</th><th>프레임</th><th>쪽</th><th>시도</th><th>소요</th><th>비고</th></tr></thead>
    <tbody>{course_rows("ml1")}</tbody>
  </table></div>

  <h2>Machine Learning 2 <span class="c">— {ml2_done}/{len(ml2)} · 밴딧·MDP·DP·MC·TD·정책경사·PPO·MCTS</span></h2>
  <div class="tablewrap"><table>
    <thead><tr><th>주</th><th>제목</th><th>차시</th><th>상태</th><th>프레임</th><th>쪽</th><th>시도</th><th>소요</th><th>비고</th></tr></thead>
    <tbody>{course_rows("ml2")}</tbody>
  </table></div>

  <footer>
    한 주차 = 한 챕터 = 3차시(3시간). 목표 분량 주차당 50–66 프레임(4차시 챕터 65–85).<br>
    파이프라인: <code>cq</code>가 교재 <code>kor/src/{{ml1,ml2}}/*.md</code> → <code>.tex</code> 작성 →
    스케줄러가 그림 변환 + <code>xelatex</code> ×2 + 쪽수·오류 검사 → 상태 갱신.<br>
    <b>완료</b> = PDF ≥ {28}쪽 &amp; LaTeX 오류 0. <b>검토 필요</b> = 쪽수 부족하거나 오류 있음.
    이 페이지는 스냅샷이며 진행에 따라 다시 게시된다.
  </footer>
</div>
</body>
</html>"""
    open(OUT_ART, "w", encoding="utf-8").write(doc)


def write_pages_index():
    """Public index served by GitHub Pages at /book-ml/slides/ ."""
    decks, n, by, done, pct, avg_txt, eta = load()
    out = os.path.join(ROOT, "..", "docs", "slides", "index.html")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    upd = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    def rows(course):
        r = []
        for x in [d for d in decks if d["course"] == course]:
            ready = x["status"] in ("done", "needs_review") and x["pages"]
            title = html.escape(x["title"])
            if ready:
                r.append(f'<li><a href="kor/{x["id"]}.pdf">{x["week"]:02d} · {title}</a>'
                         f'<span class="m">{x["pages"]}쪽</span></li>')
            else:
                r.append(f'<li class="pend">{x["week"]:02d} · {title}'
                         f'<span class="m">준비 중</span></li>')
        return "\n".join(r)

    doc = f"""<!doctype html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>KSA ML1·2 강의 슬라이드</title>
<style>
 :root{{--ink:#1e1b33;--dim:#6a6580;--acc:#2E3192;--line:#e4e0d4;--bg:#f7f5f0;--card:#fff}}
 @media(prefers-color-scheme:dark){{:root{{--ink:#e9e7f2;--dim:#9d99b4;--acc:#8f92e6;
   --line:#2f2b45;--bg:#141225;--card:#1d1a30}}}}
 *{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);
   font:15px/1.6 "IBM Plex Sans KR",system-ui,"Noto Sans KR",sans-serif}}
 .w{{max-width:760px;margin:0 auto;padding:48px 22px 80px}}
 h1{{font-size:24px;margin:0 0 4px}}.sub{{color:var(--dim);font-size:13px;margin-bottom:28px}}
 h2{{font-size:14px;text-transform:uppercase;letter-spacing:.08em;color:var(--acc);
   margin:32px 0 8px}}
 ul{{list-style:none;margin:0;padding:0;background:var(--card);border:1px solid var(--line);
   border-radius:12px;overflow:hidden}}
 li{{padding:11px 16px;border-bottom:1px solid var(--line);display:flex;
   justify-content:space-between;gap:12px}}
 li:last-child{{border-bottom:none}}
 a{{color:var(--ink);text-decoration:none;font-weight:600}}a:hover{{color:var(--acc)}}
 .pend{{color:var(--dim)}}.m{{color:var(--dim);font-size:12px;font-family:ui-monospace,monospace;
   white-space:nowrap}}
</style></head><body><div class="w">
<h1>KSA · Machine Learning 1·2 강의 슬라이드 (한국어)</h1>
<div class="sub">한 주차 = 한 챕터 = 3차시(3시간). 진행 {done}/{n}. 갱신 {upd}.</div>
<h2>Machine Learning 1</h2><ul>
{rows("ml1")}
</ul>
<h2>Machine Learning 2</h2><ul>
{rows("ml2")}
</ul>
</div></body></html>"""
    open(out, "w", encoding="utf-8").write(doc)


if __name__ == "__main__":
    write_txt()
    write_local()
    write_art()
    try:
        write_pages_index()
    except Exception as e:
        print("pages index skipped:", e)
    _, nn, _, dn, pc, _, et = load()
    print(f"dashboard: {dn}/{nn} done ({pc}%)  eta {et}")
