#!/usr/bin/env python3
"""ML1 수업 계획 도구 (localhost). 주차 x 3차시 칸에 덱(수업 · 실험)을 끌어다 놓고,
오른쪽 칸에는 프로젝트 일정(공지 · 제출)을 끌어다 놓는다. 저장하면 slides/ml1/course_plan.json,
'배포'하면 덱 폴더를 weekNN/K_topic 으로 옮기고 경로를 고치고, 사이트 목록(docs/.../index.html)을 계획대로 다시 쓰고,
옮긴 덱을 다시 빌드해 사이트 PDF 를 바꾼다. '커밋 · 푸시'는 따로 누른다.

    python3 tools/course_planner.py            # http://localhost:8771
    python3 tools/course_planner.py --check    # 사이트를 계획에서 다시 만들어도 지금과 같은지 (왕복 시험)

삭제는 하지 않는다: 계획을 저장할 때마다 이전 계획은 slides/ml1/_plan_history/ 에 남는다.
"""
import html, json, os, re, shutil, subprocess, sys, threading, time, urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ML1 = ROOT / "slides" / "ml1"
SITE = ROOT / "docs" / "slides" / "ml1" / "2027-spring"
INDEX = SITE / "index.html"
PLAN = ML1 / "course_plan.json"
HIST = ML1 / "_plan_history"
DECK = ROOT / "slides" / "tools" / "deck.py"
UI = Path(__file__).with_suffix(".html")
COLAB = "https://colab.research.google.com/github/SuminHan/book-ml/blob/main/slides/ml1/{rel}"

LOG = []          # deploy log lines (shown in the UI)
BUSY = {"on": False}


def log(*a):
    s = " ".join(str(x) for x in a)
    LOG.append(time.strftime("%H:%M:%S ") + s)
    print(s, flush=True)


# ---------------------------------------------------------------- assets
def tex_title(t):
    m = re.search(r"\\title\{(.*)\}", t)
    s = m.group(1) if m else ""
    s = re.sub(r"\$\\sim\$", "~", s).replace("\\&", "&").replace("\\,", " ")
    s = re.sub(r"\\[a-zA-Z]+\s*", "", s).replace("{", "").replace("}", "")
    return s.strip()


_pages = {}
def pdf_pages(pdf):
    if not pdf.exists():
        return 0
    k = (str(pdf), pdf.stat().st_mtime)
    if k not in _pages:
        try:
            out = subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True).stdout
            _pages[k] = int(re.search(r"Pages:\s+(\d+)", out).group(1))
        except Exception:
            _pages[k] = 0
    return _pages[k]


def scan_assets():
    """every deck folder under slides/ml1/week*/ and slides/ml1/project/ -> {id: info}"""
    out = {}
    for d in sorted(list(ML1.glob("week*/*/")) + list(ML1.glob("project/*/"))):
        topic = re.sub(r"^\d+_", "", d.name)
        tex = d / f"{topic}.tex"
        if not tex.exists():
            continue
        t = tex.read_text()
        kind = "lab" if "ksa-cs-v2-lab" in t else "assignment" if "ksa-cs-v2-assignment" in t else "lecture"
        nbs = sorted(p for p in (d / "lab").glob("*.ipynb") if "solution" not in p.name) if (d / "lab").exists() else []
        rel = d.relative_to(ML1).as_posix()
        out[topic] = dict(id=topic, path=rel, title=tex_title(t), kind=kind, pages=pdf_pages(d / f"{topic}.pdf"),
                          notebook=nbs[0].relative_to(ML1).as_posix() if nbs else "", project=rel.startswith("project/"))
    return out


# ---------------------------------------------------------------- site <-> plan
BLOCK = re.compile(r'<div class="wr">.*?</div></div>(?:</div>)?(?=\n|$)', re.S)

def parse_site(assets):
    t = INDEX.read_text()
    weeks, milestones = [], []
    body = t[t.index('<div class="wr">'):t.rindex("</div></body>")]
    blocks = re.split(r'(?=<div class="wr">)', body)
    for b in blocks:
        if not b.strip():
            continue
        n = int(re.search(r'<span class="wk">(\d+)주차</span>', b).group(1))
        cls = re.search(r'<details class="(\w+)"', b).group(1)
        title = html.unescape(re.search(r'<span class="t">(.*?)</span>', b).group(1))
        dt = re.search(r'<span class="dt">(.*?)</span>', b).group(1)
        summ = re.search(r"<summary>(.*?)</summary>", b, re.S).group(1)
        hol = [html.unescape(x) for x in re.findall(r'<span class="tag hol">(.*?)</span>', summ)]
        dues = [html.unescape(x) for x in re.findall(r'<span class="tag due">(.*?)</span>', summ)]
        ul = re.search(r"<ul>(.*)</ul>", b, re.S).group(1)
        lis = re.findall(r"<li[^>]*>.*?</li>", ul, re.S)
        decks, ms_rows, extra = [], [], []
        for li in lis:
            m = re.search(r'href="week\d+/\d+_(\w+)\.pdf">(?:<b[^>]*>[^<]*</b> )?\d+-\d+ · (.*?)</a>', li)
            if m and m.group(1) in assets:
                decks.append(dict(id=m.group(1), label=html.unescape(m.group(2))))
            elif 'class="ms"' in li:
                ms_rows.append(li)
            else:
                extra.append(li)
        lane = re.search(r'<div class="lane">(.*)$', b, re.S).group(1)
        for proj, href, label in re.findall(r'<a class="lb (mid|fin)" href="([^"]*)">(.*?)<b class="pf">', lane):
            milestones.append(dict(id=f"{proj}-{len([x for x in milestones if x['proj'] == proj])}", proj=proj, week=n,
                                   label=html.unescape(label), file=href, tag="", rows=[]))
        mine = [x for x in milestones if x["week"] == n]
        for i, x in enumerate(mine):            # due tags and 'ms' rows belong to the week's milestone(s)
            if i < len(dues):
                x["tag"] = dues[i]
        if mine:
            mine[0]["rows"] = ms_rows
        else:
            extra = ms_rows + extra
        weeks.append(dict(n=n, cls=cls, title=title, dates=dt, holidays=hol, decks=decks, extra=extra))
    return weeks, milestones


def initial_slots(week, assets):
    items = [d["id"] for d in week["decks"]]
    labs = [i for i in items if assets[i]["kind"] == "lab"]
    lecs = [i for i in items if i not in labs]
    slots = [[], [], []]
    if labs:
        slots[2] = labs
        for k, i in enumerate(lecs):
            slots[0 if k == 0 else 1].append(i)
    else:
        for k, i in enumerate(lecs):
            slots[min(k, 2)].append(i)
    kind = {"exam": "exam", "mk": "makeup"}.get(week["cls"], "class")
    return [dict(kind=kind if not s else "class", items=s) for s in slots]


def bootstrap():
    assets = scan_assets()
    weeks, milestones = parse_site(assets)
    for w in weeks:
        w["slots"] = initial_slots(w, assets)
        w["labels"] = {d["id"]: d["label"] for d in w.pop("decks")}
    labels = {}
    for w in weeks:
        labels.update(w.pop("labels"))
    return dict(version=1, weeks=weeks, milestones=milestones, labels=labels, pool=[])


def load_plan():
    if PLAN.exists():
        return json.loads(PLAN.read_text())
    return bootstrap()


def save_plan(plan):
    if PLAN.exists():
        HIST.mkdir(exist_ok=True)
        shutil.copy2(PLAN, HIST / time.strftime("course_plan_%Y%m%d_%H%M%S.json"))
    PLAN.write_text(json.dumps(plan, ensure_ascii=False, indent=1))


def week_order(w):
    return [i for s in w["slots"] for i in s["items"]]


def target_paths(plan, assets):
    """id -> new folder (weekNN/K_topic) from the plan's slot order."""
    out = {}
    for w in plan["weeks"]:
        for k, i in enumerate(week_order(w), 1):
            if i in assets and not assets[i]["project"]:
                out[i] = f"week{w['n']:02d}/{k}_{i}"
    return out


def render_site(plan, assets, paths=None):
    """week list HTML from the plan (paths: id -> folder, defaults to where the deck is now)."""
    paths = paths or {i: a["path"] for i, a in assets.items()}
    ms = plan["milestones"]
    span = {}
    for proj in ("mid", "fin"):
        ws = sorted(x["week"] for x in ms if x["proj"] == proj)
        if ws:
            span[proj] = (ws[0], ws[-1])
    out = []
    for w in plan["weeks"]:
        n = w["n"]
        mine = [x for x in ms if x["week"] == n]
        tags = "".join(f'<span class="tag hol">{html.escape(h)}</span>' for h in w["holidays"])
        if w["cls"] == "exam":
            tags += '<span class="tag ex">시험</span>'
        if w["cls"] == "mk":
            tags += '<span class="tag mk">보강</span>'
        tags += "".join(f'<span class="tag due">{html.escape(x["tag"])}</span>' for x in mine if x["tag"])
        head = (f'<div class="wr"><details class="{w["cls"]}"><summary><span class="wk">{n}주차</span>'
                f'<span class="t">{html.escape(w["title"], quote=False)}</span>{tags}<span class="dt">{w["dates"]}</span></summary><ul>')
        rows = []
        for x in mine:
            rows += x["rows"]
        for k, i in enumerate(week_order(w), 1):
            if i not in assets:
                continue
            a, p = assets[i], paths[i]
            label = html.escape(plan["labels"].get(i) or a["title"], quote=False)
            href = f"{p}.pdf"
            meta = f'{a["pages"]}쪽'
            if a["kind"] == "lab":
                nb = a["notebook"].replace(a["path"], p, 1) if a["notebook"] else ""
                colab = f' · <a class="m" href="{COLAB.format(rel=nb)}">Colab</a>' if nb else ""
                rows.append(f'<li class="lab"><a href="{href}">{n:02d}-{k} · {label}</a><span class="m"><b class="lb2">실험</b> {meta}{colab}</span></li>')
            else:
                rows.append(f'<li><a href="{href}">{n:02d}-{k} · {label}</a><span class="m">{meta}</span></li>')
        rows += w["extra"]
        lane = ""
        for proj, (a0, a1) in span.items():
            if not (a0 <= n <= a1):
                continue
            here = [x for x in mine if x["proj"] == proj]
            if n == a0:
                lane += f'<i class="ln {proj}" style="top:23px;bottom:-8px"></i>' + (f'<i class="dot {proj} start"></i>' if here else "")
            elif n == a1:
                lane += f'<i class="ln {proj}" style="top:-8px;height:31px"></i>' + (f'<i class="dot {proj} end"></i>' if here else "")
            else:
                lane += f'<i class="ln {proj}" style="top:-8px;bottom:-8px"></i>' + (f'<i class="dot {proj} mid"></i>' if here else "")
            for x in here:
                lane += (f'<div class="lbs"><a class="lb {proj}" href="{x["file"]}">{html.escape(x["label"], quote=False)}'
                         f'<b class="pf">PDF</b><span>{n}주차</span></a></div>')
        out.append(head + "\n" + "".join(r + "\n" for r in rows) + "</ul></details>"
                   + f'<div class="lane">{lane}</div></div>')
    return "\n".join(out)


def write_site(plan, assets, paths=None):
    t = INDEX.read_text()
    i, j = t.index('<div class="wr">'), t.rindex("</div></body>")
    INDEX.write_text(t[:i] + render_site(plan, assets, paths) + "\n" + t[j:])


# ---------------------------------------------------------------- deploy
def preview(plan):
    assets = scan_assets()
    tgt = target_paths(plan, assets)
    moves = [(assets[i]["path"], p) for i, p in tgt.items() if assets[i]["path"] != p]
    unplaced = [i for i, a in assets.items() if not a["project"] and i not in tgt]
    return dict(moves=moves, unplaced=unplaced)


TEXT_SUFFIX = {".tex", ".py", ".ipynb", ".md", ".html"}

def rewrite_paths(pairs):
    """old 'weekAA/I_topic' -> new 'weekBB/J_topic' in text files (two-phase with placeholders)."""
    ph = {old: f"@@MOVE{k}@@" for k, (old, _) in enumerate(pairs)}
    roots = [ML1, SITE, ROOT / "tools"]
    n = 0
    for r in roots:
        for p in r.rglob("*"):
            if not p.is_file() or p.suffix not in TEXT_SUFFIX or "_이전" in p.parts or "_plan_history" in p.parts:
                continue
            try:
                t = p.read_text()
            except Exception:
                continue
            o = t
            for old, _ in pairs:
                t = re.sub(re.escape(old) + r"(?![0-9A-Za-z_])", ph[old], t)
            for old, new in pairs:
                t = t.replace(ph[old], new)
            if t != o:
                p.write_text(t)
                n += 1
    return n


def deploy(plan):
    BUSY["on"] = True
    try:
        LOG.clear()
        assets = scan_assets()
        tgt = target_paths(plan, assets)
        pairs = [(assets[i]["path"], p) for i, p in tgt.items() if assets[i]["path"] != p]
        log(f"옮길 덱 {len(pairs)}개")
        # 1. move folders (and site PDFs) via temporary names so swaps never collide
        tmp = []
        for k, (old, new) in enumerate(pairs):
            t = ML1 / f"_moving_{k}"
            (ML1 / old).rename(t)
            sp = SITE / f"{old}.pdf"
            st = SITE / f"_moving_{k}.pdf"
            if sp.exists():
                sp.rename(st)
            tmp.append((t, st, new))
        for t, st, new in tmp:
            (ML1 / new).parent.mkdir(parents=True, exist_ok=True)
            t.rename(ML1 / new)
            if st.exists():
                (SITE / new).parent.mkdir(parents=True, exist_ok=True)
                st.rename(SITE / f"{new}.pdf")
            log("  ", new)
        # 2. rewrite path strings
        if pairs:
            log("경로를 고친 파일", rewrite_paths(pairs))
        # 3. site list from the plan
        assets = scan_assets()
        write_site(plan, assets)
        log("사이트 목록을 계획대로 다시 씀")
        # 4. rebuild moved decks, copy PDFs
        for _, new in pairs:
            topic = re.sub(r"^\d+_", "", Path(new).name)
            r = subprocess.run([sys.executable, str(DECK), "build", str(ML1 / new)], capture_output=True, text=True)
            last = (r.stdout.strip().splitlines() or ["?"])[-1]
            pdf = ML1 / new / f"{topic}.pdf"
            if pdf.exists():
                (SITE / new).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(pdf, SITE / f"{new}.pdf")
            log("  빌드", new, last.strip())
        assets = scan_assets()
        write_site(plan, assets)              # page counts after rebuild
        save_plan(plan)
        log("배포 끝. 확인 후 '커밋 · 푸시'를 누르세요.")
    except Exception as e:
        log("오류:", repr(e))
    finally:
        BUSY["on"] = False


def publish(msg):
    paths = ["slides/ml1", "docs/slides/ml1/2027-spring"]
    r = subprocess.run(["git", "status", "--porcelain", "--"] + paths, cwd=ROOT, capture_output=True, text=True).stdout
    if not r.strip():
        return "바뀐 것이 없습니다"
    subprocess.run(["git", "add", "-A", "--"] + [p for p in ["slides/ml1/course_plan.json", "docs/slides/ml1/2027-spring"]], cwd=ROOT)
    moved = sorted({"/".join(l[3:].split("/")[:4]) for l in r.splitlines() if l[3:].startswith("slides/ml1/week")})
    if moved:
        subprocess.run(["git", "add", "-A", "--"] + moved, cwd=ROOT)
    c = subprocess.run(["git", "commit", "-q", "-m", msg + "\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"],
                       cwd=ROOT, capture_output=True, text=True)
    if c.returncode:
        return "커밋 실패: " + (c.stderr or c.stdout)[-300:]
    p = subprocess.run(["git", "push", "-q"], cwd=ROOT, capture_output=True, text=True)
    return "커밋 · 푸시 완료" if p.returncode == 0 else "푸시 실패: " + p.stderr[-300:]


# ---------------------------------------------------------------- http
class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send(self, body, ctype="application/json; charset=utf-8", code=200):
        if isinstance(body, (dict, list)):
            body = json.dumps(body, ensure_ascii=False)
        if isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urllib.parse.urlparse(self.path)
        if u.path == "/":
            return self.send(UI.read_text(), "text/html; charset=utf-8")
        if u.path == "/api/state":
            return self.send(dict(assets=scan_assets(), plan=load_plan(), busy=BUSY["on"], log=LOG[-200:]))
        if u.path == "/api/log":
            return self.send(dict(busy=BUSY["on"], log=LOG[-200:]))
        if u.path.startswith("/files/"):
            rel = urllib.parse.unquote(u.path[len("/files/"):])
            p = (ROOT / rel).resolve()
            if not str(p).startswith(str(ROOT / "slides")) and not str(p).startswith(str(ROOT / "docs")) or not p.is_file():
                return self.send("not found", "text/plain", 404)
            ctype = "application/pdf" if p.suffix == ".pdf" else "text/html; charset=utf-8" if p.suffix == ".html" else "application/octet-stream"
            return self.send(p.read_bytes(), ctype)
        self.send("not found", "text/plain", 404)

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        data = json.loads(self.rfile.read(n) or b"{}")
        if self.path == "/api/plan":
            save_plan(data)
            return self.send(dict(ok=True))
        if self.path == "/api/preview":
            return self.send(preview(data))
        if self.path == "/api/deploy":
            if BUSY["on"]:
                return self.send(dict(ok=False, msg="이미 배포 중"))
            threading.Thread(target=deploy, args=(data,), daemon=True).start()
            return self.send(dict(ok=True))
        if self.path == "/api/publish":
            return self.send(dict(msg=publish(data.get("msg") or "ML1 수업 계획 반영 (course planner)")))
        if self.path == "/api/reset":
            return self.send(bootstrap())
        self.send("not found", "text/plain", 404)


def main():
    if "--check" in sys.argv:
        assets = scan_assets()
        plan = bootstrap()
        t = INDEX.read_text()
        cur = t[t.index('<div class="wr">'):t.rindex("</div></body>")].rstrip("\n")
        new = render_site(plan, assets)
        norm = lambda s: re.sub(r">\s*\n\s*<", "><", s)
        print("same" if norm(cur) == norm(new) else "DIFF")
        if norm(cur) != norm(new):
            cur, new = norm(cur).replace("<div class=\"wr\">", "\n<div class=\"wr\">"), norm(new).replace("<div class=\"wr\">", "\n<div class=\"wr\">")
            import difflib
            for l in list(difflib.unified_diff(cur.split("\n"), new.split("\n"), lineterm="", n=0))[:40]:
                print(l[:220])
        return
    port = int(sys.argv[sys.argv.index("--port") + 1]) if "--port" in sys.argv else 8771
    print(f"http://localhost:{port}")
    ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()


if __name__ == "__main__":
    main()
