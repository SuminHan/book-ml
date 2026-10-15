"""데이터 가게 웹 페이지 (docs/datashop/): 목록 index.html + 데이터마다 <slug>/index.html + 공용 shop.css.
make_shop.py 가 불러 쓴다. 그래프는 Plotly (jsdelivr plotly.js-dist-min 2.35.2), 색은 CSS 변수 (dataviz 기본 팔레트 1~3번)."""
import html, json

PLOTLY = "https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js"
KATEX = """<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css">
<script src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"></script>
<script>document.querySelectorAll("[data-tex]").forEach((el) => { try { katex.render(el.dataset.tex, el, { displayMode: true, throwOnError: false, strict: false }); } catch (e) { el.textContent = el.dataset.tex; } });</script>"""
FONT = '<link rel="preconnect" href="https://fonts.googleapis.com"><link href="https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;500;700;900&display=swap" rel="stylesheet">'
E = html.escape

CSS = """:root {
  --bg: #f6f4ef; --paper: #ffffff; --ink: #1d1d22; --muted-ink: #6b6a66; --line: #e3dfd5;
  --navy: #2e3192; --gold: #a9976b; --gold-soft: #efe9da; --shadow: 0 1px 2px rgba(0,0,0,.05), 0 6px 18px rgba(0,0,0,.06);
  --series-1: #2a78d6; --series-2: #eb6834; --series-3: #1baf7a; --muted: #b9b7b0; --grid: #ebe8e0;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #151517; --paper: #1f1f23; --ink: #ececea; --muted-ink: #a2a09a; --line: #34343a;
    --navy: #9fa3ff; --gold: #cdb98a; --gold-soft: #2c2a24; --shadow: 0 1px 2px rgba(0,0,0,.4);
    --series-1: #3987e5; --series-2: #d95926; --series-3: #199e70; --muted: #5d5c58; --grid: #2c2c31;
  }
}
:root[data-theme="dark"] {
  --bg: #151517; --paper: #1f1f23; --ink: #ececea; --muted-ink: #a2a09a; --line: #34343a;
  --navy: #9fa3ff; --gold: #cdb98a; --gold-soft: #2c2a24; --shadow: 0 1px 2px rgba(0,0,0,.4);
  --series-1: #3987e5; --series-2: #d95926; --series-3: #199e70; --muted: #5d5c58; --grid: #2c2c31;
}
* { box-sizing: border-box; }
html, body { margin: 0; background: var(--bg); color: var(--ink); }
body { font-family: "Noto Sans KR", -apple-system, "Segoe UI", sans-serif; line-height: 1.6; }
a { color: var(--navy); }
.wrap { max-width: 1080px; margin: 0 auto; padding: 0 16px; }
header.top { padding: 52px 0 24px; }
.eyebrow { color: var(--gold); font-weight: 700; letter-spacing: .08em; font-size: .8rem; text-transform: uppercase; }
h1 { font-size: clamp(2rem, 5vw, 3rem); font-weight: 900; margin: .15em 0 .3em; letter-spacing: -.02em; line-height: 1.2; }
.lead { color: var(--muted-ink); max-width: 680px; margin: 0; }
.five-strip { display: flex; flex-wrap: wrap; gap: 8px; margin: 20px 0 0; padding: 0; list-style: none; }
.five-strip li { background: var(--paper); border: 1px solid var(--line); border-radius: 999px; padding: 4px 12px; font-size: .85rem; }
.five-strip b { color: var(--navy); margin-right: 4px; }
.section-title { font-size: .95rem; font-weight: 700; color: var(--muted-ink); margin: 30px 0 12px; display: flex; align-items: center; gap: 10px; }
.section-title::after { content: ""; flex: 1; height: 1px; background: var(--line); }
.shelf { display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 16px; }
.item { display: flex; flex-direction: column; background: var(--paper); border: 1px solid var(--line); border-radius: 14px; overflow: hidden;
        box-shadow: var(--shadow); text-decoration: none; color: var(--ink); transition: transform .15s ease, border-color .15s ease; }
.item:hover, .item:focus-visible { transform: translateY(-3px); border-color: var(--gold); }
.item:focus-visible { outline: 2px solid var(--navy); outline-offset: 2px; }
.item img { width: 100%; height: auto; aspect-ratio: 16 / 9; object-fit: cover; display: block; background: #fff; border-bottom: 1px solid var(--line); }
.item .body { padding: 16px 18px 18px; display: flex; flex-direction: column; gap: 6px; flex: 1; }
.tag { display: inline-flex; align-items: center; gap: 8px; font-size: .78rem; color: var(--muted-ink); }
.num { font-weight: 900; color: var(--gold); font-variant-numeric: tabular-nums; }
.item .title { font-size: 1.15rem; font-weight: 700; margin: 0; }
.item .teaser { margin: 0; font-size: .93rem; flex: 1; }
.enter { font-size: .85rem; font-weight: 700; color: var(--navy); margin-top: 6px; }
.enter::after { content: " →"; }
footer { color: var(--muted-ink); font-size: .82rem; padding: 32px 0 40px; }
footer a { color: var(--muted-ink); }
/* item page */
.crumb { padding-top: 24px; font-size: .9rem; }
.crumb a { text-decoration: none; }
.hero { display: grid; grid-template-columns: 1.2fr 1fr; gap: 28px; align-items: center; padding: 18px 0 8px; }
.hero img { width: 100%; border-radius: 12px; border: 1px solid var(--line); background: #fff; }
.chips { display: flex; flex-wrap: wrap; gap: 6px; margin: 10px 0 0; }
.chip { background: var(--gold-soft); border-radius: 999px; padding: 3px 10px; font-size: .82rem; }
.teaser-big { font-size: 1.15rem; margin: 14px 0 0; }
.actions { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 18px; }
.btn { display: inline-block; text-decoration: none; padding: 10px 18px; border-radius: 10px; font-weight: 700; font-size: .95rem; }
.btn.primary { background: var(--navy); color: #fff; }
:root[data-theme="dark"] .btn.primary { color: #111; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) .btn.primary { color: #111; } }
.btn.ghost { border: 1px solid var(--line); color: var(--ink); background: var(--paper); }
section.block { background: var(--paper); border: 1px solid var(--line); border-radius: 14px; padding: 20px 22px; margin: 18px 0; box-shadow: var(--shadow); }
section.block h2 { font-size: .82rem; letter-spacing: .08em; color: var(--gold); text-transform: uppercase; margin: 0 0 10px; }
.data-name { font-weight: 700; font-size: 1.05rem; }
.src { color: var(--muted-ink); font-size: .9rem; margin: 2px 0 0; }
.scroll { overflow-x: auto; }
table { width: 100%; border-collapse: collapse; font-size: .9rem; }
th, td { border-top: 1px solid var(--line); padding: 7px 8px; vertical-align: top; text-align: left; }
th { color: var(--muted-ink); font-weight: 500; font-size: .82rem; border-top: none; white-space: nowrap; }
table.sample td { white-space: nowrap; max-width: 360px; overflow: hidden; text-overflow: ellipsis; font-variant-numeric: tabular-nums; }
table.five td:first-child { width: 120px; color: var(--navy); font-weight: 700; white-space: nowrap; }
.charts { display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 18px; }
.chart h3 { font-size: 1rem; margin: 0; }
.chart .note { color: var(--muted-ink); font-size: .86rem; margin: 2px 0 4px; }
.plot { width: 100%; height: 300px; }
.chart.small .plot { height: 190px; }
pre.result { background: var(--bg); border-radius: 10px; padding: 12px 14px; overflow-x: auto; font-size: .85rem; margin: 0; }
.two { display: grid; grid-template-columns: 1fr 1fr; gap: 18px; }
ul.papers { margin: 0; padding-left: 18px; }
ul.papers li { margin: 4px 0; }
.fit dt { font-weight: 700; color: var(--navy); margin-top: 6px; }
.fit dd { margin: 2px 0 0; }
.pager { display: flex; justify-content: space-between; gap: 12px; padding: 8px 0 0; font-size: .92rem; }
.pager a { text-decoration: none; }
.cover { position: relative; border-radius: 16px; overflow: hidden; margin-top: 14px; aspect-ratio: 5 / 1.4; background: #222; }
.cover img { width: 100%; height: 100%; object-fit: cover; display: block; }
.cover .credit { position: absolute; right: 8px; bottom: 6px; font-size: .7rem; color: #fff; background: rgba(0,0,0,.45); padding: 1px 6px; border-radius: 4px; }
.cover .credit a { color: #fff; }
.item .cov { position: relative; }
.item .cov .badge { position: absolute; left: 10px; top: 10px; font-size: .72rem; background: rgba(0,0,0,.6); color: #fff; padding: 2px 8px; border-radius: 99px; }
.blackbox { display: grid; grid-template-columns: minmax(0, 1fr) auto minmax(0, 1fr); gap: 10px; align-items: center; }
.blackbox .io { background: var(--bg); border-radius: 10px; padding: 10px 12px; }
.blackbox .io b { display: block; font-size: .78rem; color: var(--muted-ink); font-weight: 500; }
.blackbox .model { background: var(--ink); color: var(--paper); border-radius: 10px; padding: 12px 14px; text-align: center; font-weight: 700; white-space: nowrap; }
.blackbox .model small { display: block; font-weight: 400; opacity: .8; }
.stat-head { display: flex; gap: 18px; flex-wrap: wrap; margin-bottom: 8px; }
.stat-head b { font-size: 1.4rem; font-variant-numeric: tabular-nums; }
.stat-head span { color: var(--muted-ink); font-size: .85rem; display: block; }
.metric { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1.2fr); gap: 4px 16px; align-items: center; border-top: 1px solid var(--line); padding: 8px 0; }
.metric:first-of-type { border-top: none; }
.metric .tex { overflow-x: auto; }
.metric .what b { display: block; }
.metric .what span { color: var(--muted-ink); font-size: .86rem; }
ul.related { margin: 0; padding-left: 18px; }
ul.related li { margin: 5px 0; }
ul.related span { color: var(--muted-ink); font-size: .86rem; }
.soon { background: var(--gold-soft); border-radius: 10px; padding: 12px 14px; }
@media (max-width: 760px) { .blackbox { grid-template-columns: 1fr; } .blackbox .model { justify-self: start; } .cover { aspect-ratio: 5 / 2; } }
@media (max-width: 760px) {
  .hero, .two { grid-template-columns: 1fr; }
  .charts { grid-template-columns: 1fr; }
  table.five td:first-child { width: 84px; white-space: normal; }
}
"""

PLOT_JS = """<script src="%s"></script>
<script>
const CHARTS = %s;
function cssVar(n) { return getComputedStyle(document.documentElement).getPropertyValue(n).trim(); }
function resolve(v) {
  if (typeof v === "string" && v.startsWith("var:")) return cssVar("--" + v.slice(4));
  if (Array.isArray(v)) return v.map(resolve);
  if (v && typeof v === "object") { const o = {}; for (const k in v) o[k] = resolve(v[k]); return o; }
  return v;
}
function draw() {
  const ink = cssVar("--ink"), muted = cssVar("--muted-ink"), grid = cssVar("--grid");
  CHARTS.forEach((c, i) => {
    const el = document.getElementById("plot-" + i); if (!el || !window.Plotly) return;
    const axis = { gridcolor: grid, zerolinecolor: grid, linecolor: grid, tickfont: { color: muted }, title: { font: { color: muted, size: 12 } }, automargin: true };
    const lay = resolve(JSON.parse(JSON.stringify(c.layout)));
    const xa = Object.assign({}, axis, lay.xaxis || {}, { title: Object.assign({}, axis.title, { text: (lay.xaxis || {}).title || "" }) });
    const ya = Object.assign({}, axis, lay.yaxis || {}, { title: Object.assign({}, axis.title, { text: (lay.yaxis || {}).title || "" }) });
    const layout = Object.assign({ paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)", font: { family: "Noto Sans KR, sans-serif", color: ink, size: 12 },
      margin: { l: 56, r: 12, t: 8, b: 44 }, showlegend: c.traces.length > 1, legend: { orientation: "h", y: 1.12, x: 0, font: { color: muted } },
      bargap: 0.15, hoverlabel: { font: { family: "Noto Sans KR, sans-serif" } } }, lay, { xaxis: xa, yaxis: ya });
    if (lay.margin) layout.margin = Object.assign({ l: 56, r: 12, t: 8, b: 44 }, lay.margin);
    delete layout.height;
    Plotly.react(el, resolve(c.traces), layout, { displayModeBar: false, responsive: true });
  });
}
draw();
window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", draw);
</script>"""


def head(title, desc, css_href):
    return f"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{E(title)}</title>
<meta name="description" content="{E(desc)}">
{FONT}
<link rel="stylesheet" href="{css_href}">
<link rel="icon" href="{css_href.replace('shop.css', 'favicon.svg')}" type="image/svg+xml">
<!-- notebooks/datashop/make_shop.py 가 만든다. 고칠 때는 site_pages.py · make_shop.py 를 고치고 다시 실행 -->
</head>
<body>
"""


def credit_html(c):
    if not c:
        return ""
    return f'<span class="credit">사진: <a href="{E(c["page"])}" target="_blank" rel="noopener">{E(c["artist"] or "Wikimedia Commons")}</a> · {E(c["license"])}</span>'


def card(i, stub=False):
    badge = '<span class="badge">노트북 준비 중</span>' if stub else ""
    return f"""<a class="item" href="{i['slug']}/">
  <div class="cov"><img src="{i['slug']}/card.jpg" alt="" loading="lazy" width="720" height="405">{badge}</div>
  <div class="body"><div class="tag"><span class="num">{i['num']}</span><span>{E(i['keywords'][0])}</span></div>
  <p class="title">{E(i['title'])}</p><p class="teaser">{E(i['teaser'])}</p><span class="enter">들어가 보기</span></div>
</a>"""


def render_index(items, sections, stubs):
    out = [head("현실의 데이터 · Machine Learning 1", "산업 현장에서 실제로 푸는 문제와 닮은 공개 데이터. 데이터를 열어 보고 문제를 직접 정해 보는 Colab 노트북", "shop.css")]
    out.append(f"""<div class="wrap">
<header class="top">
  <div class="eyebrow">Machine Learning 1 · 데이터 가게</div>
  <h1>현실의 데이터</h1>
  <p class="lead">추천, 이상거래 탐지, 수요예측, 불량 예측, 문서 이해 … 산업 현장에서 실제로 푸는 문제와 닮은 공개 데이터 {len(items) + len(stubs)}가지를 모았습니다. 인턴이나 산학협력 과제에서 처음 받게 될 데이터도 대개 이런 모양입니다. 하나를 골라 들어가 데이터를 보고, 노트북을 받아 문제를 직접 정해 보세요.</p>
  <p class="lead" style="margin-top:10px"><a href="worldcup/">데이터 월드컵 하러 가기 →</a> 32개 중 가장 해 보고 싶은 데이터를 골라 보세요.</p>
</header>
<main>""")
    for sec in sections:
        its = [i for i in items if i["sector"] == sec]
        if its:
            out.append(f'<h2 class="section-title">{E(sec)}</h2>\n<div class="shelf">' + "".join(card(i) for i in its) + "</div>")
    if stubs:
        out.append('<h2 class="section-title">다음에 열 데이터 (노트북 준비 중)</h2>\n<div class="shelf">' + "".join(card(i, True) for i in stubs) + "</div>")
    out.append("""</main>
<footer>데이터마다 논문 · 출처를 함께 적었습니다. 분야별 문제와 관심 업종은 일반적인 예시이고 특정 회사의 과제가 아닙니다. 커버 그림은 AI 로 생성했고 (프롬프트 공개), 각 페이지에 출처를 적었습니다.
 · <a href="https://github.com/SuminHan/book-ml/tree/main/notebooks/datashop">노트북 원본 (GitHub)</a></footer>
</div>
</body>
</html>
""")
    return "\n".join(out)


def table_html(t, cls="sample"):
    rows = "".join("<tr>" + "".join(f"<td title=\"{E(v)}\">{E(v)}</td>" for v in r) + "</tr>" for r in t["rows"])
    return f'<div class="scroll"><table class="{cls}"><thead><tr>{"".join(f"<th>{E(c)}</th>" for c in t["columns"])}</tr></thead><tbody>{rows}</tbody></table></div>'


def blackbox_html(i):
    five = dict(i["five"])
    return f"""<div class="blackbox">
  <div class="io"><b>입력 X</b>{E(five.get("② 입력 X", ""))}</div>
  <div class="model">→ 모델 →<small>{E(i.get("type", ""))}</small></div>
  <div class="io"><b>출력 y (기대하는 답)</b>{E(five.get("③ 출력 y", ""))}</div>
</div>"""


def papers_fit_html(i):
    papers = "".join(f'<li><a href="{E(u)}" target="_blank" rel="noopener">{E(t)}</a></li>' for t, u in i["papers"])
    fit = i["fit"]
    return f"""<div class="two">
  <section class="block"><h2>논문 · 출처</h2><ul class="papers">{papers}</ul>
    <p class="src" style="margin-top:8px">누가, 왜, 어떻게 만든 데이터인지 직접 확인해 보세요.</p></section>
  <section class="block fit"><h2>이 데이터에 관심 있을 곳</h2><dl><dt>업종</dt><dd>{E(fit[0])}</dd><dt>직무</dt><dd>{E(fit[1])}</dd></dl>
    <p class="src" style="margin-top:8px">인턴 · 산학협력 · 프로젝트 주제를 고를 때 참고하세요 (일반적인 예시).</p></section>
</div>"""


def metrics_related_html(i):
    import extras
    ms = extras.METRICS.get(i["slug"], [])
    mhtml = "".join(f'<div class="metric"><div class="what"><b>{E(extras.F[m][0])}</b><span>{E(extras.F[m][2])}</span></div><div class="tex" data-tex="{E(extras.F[m][1])}">{E(extras.F[m][1])}</div></div>' for m in ms)
    if not ms:
        mhtml = '<p class="src">이 데이터는 하나의 정답 지표보다 분석 · 순위로 쓴다. 노트북의 \'다른 문제로 바꿔 보기\' 에서 지표를 직접 정해 보세요.</p>'
    rel = extras.RELATED.get(i["slug"], ([], ""))[0]
    rhtml = "".join(f'<li><a href="{E(u)}" target="_blank" rel="noopener">{E(t)}</a> <span>· {E(d)}</span></li>' for t, u, d in rel)
    rhtml += f'<li><a href="{E(extras.hf_search(i["slug"]))}" target="_blank" rel="noopener">Hugging Face 에서 비슷한 데이터 더 찾기</a> <span>· 데이터셋 검색 결과</span></li>'
    return f"""<div class="two">
  <section class="block"><h2>평가지표 수식</h2>{mhtml}</section>
  <section class="block"><h2>관련 데이터</h2><ul class="related">{rhtml}</ul></section>
</div>"""


def hero_html(i, actions):
    return f"""<nav class="crumb"><a href="../">← 현실의 데이터</a> · <a href="../worldcup/">데이터 월드컵</a></nav>
<div class="cover"><img src="cover.jpg" alt="">{credit_html(i.get("credit"))}</div>
<div class="hero" style="grid-template-columns: 1fr">
  <div>
    <div class="tag"><span class="num">{i['num']}</span><span>{E(i['sector'])}</span></div>
    <h1>{E(i['title'])}</h1>
    <div class="chips">{"".join(f'<span class="chip">{E(k)}</span>' for k in i['keywords'])}</div>
    <p class="teaser-big">{E(i['teaser'])}</p>
    <div class="actions">{actions}</div>
  </div>
</div>"""


def render_item(i, prev, nxt, preview, results):
    charts = preview["charts"]
    chart_html = "".join(
        f'<div class="chart{" small" if c["layout"].get("height", 300) < 250 else ""}"><h3>{E(c["title"])}</h3>'
        + (f'<p class="note">{E(c["note"])}</p>' if c["note"] else "")
        + f'<div class="plot" id="plot-{k}" role="img" aria-label="{E(c["title"])}"></div></div>'
        for k, c in enumerate(charts))
    st = preview.get("stats")
    stats_html = ""
    if st:
        more = f' (앞의 16개 열만, 나머지 {st["more"]}개 열은 노트북에서)' if st["more"] else ""
        stats_html = f"""<section class="block"><h2>데이터 통계</h2>
  <div class="stat-head"><div><b>{st["n_rows"]:,}</b><span>줄 (데이터 파일의 줄 수. 문제의 '한 행' 은 아래 예시에서 따로 정한다)</span></div><div><b>{st["n_cols"]}</b><span>열{E(more)}</span></div></div>
  {table_html(st, "sample")}</section>"""
    five = "".join(f"<tr><td>{E(a)}</td><td>{E(b)}</td></tr>" for a, b in i["five"])
    res = E("\n".join(results)) if results else "(노트북을 실행하면 여기에 결과가 나온다)"
    pager = (f'<a href="../{prev["slug"]}/">← {prev["num"]} {E(prev["title"])}</a>' if prev else "<span></span>") + \
            (f'<a href="../{nxt["slug"]}/">{nxt["num"]} {E(nxt["title"])} →</a>' if nxt else "<span></span>")
    thumb = preview.get("thumb", "thumb.png")
    actions = (f'<a class="btn primary" href="{E(i["colab"])}" target="_blank" rel="noopener">Colab 에서 열기</a>'
               f'<a class="btn ghost" href="{E(i["file"])}.ipynb" download>노트북 받기 (.ipynb)</a>')
    out = head(f"{i['title']} · 현실의 데이터", i["teaser"], "../shop.css")
    out += f"""<div class="wrap">
{hero_html(i, actions)}
<section class="block"><h2>어떤 문제인가</h2>{blackbox_html(i)}</section>
<section class="block"><h2>데이터</h2>
  <div class="two" style="align-items:center"><div><div class="data-name">{E(i['data'])} · {E(i['size'])}</div><p class="src">{E(i['data_src'])}</p></div>
  <img src="{thumb}" alt="{E(i['title'])} 데이터 샘플" style="width:100%;border-radius:10px;border:1px solid var(--line)"></div>
  <h2 style="margin-top:16px">무작위로 뽑은 몇 줄</h2>{table_html(preview["sample"])}
</section>
{stats_html}
<section class="block"><h2>데이터 모양 보기</h2><div class="charts">{chart_html}</div></section>
<div class="two">
  <section class="block"><h2>문제를 정해 본 예시</h2><table class="five"><tbody>{five}</tbody></table>
    <p class="src" style="margin-top:10px">문제를 정할 때 처음 채워 보는 칸들입니다. 여기서 끝이 아니라, 시험 데이터를 어떻게 나눌지 · 틀렸을 때 손해가 얼마인지 · 예측 시점에 정말 알 수 있는 값인지 같은 칸을 더 붙일수록 좋은 문제가 됩니다.</p></section>
  <section class="block"><h2>노트북에서 나온 결과 (기준선 vs 모델)</h2><pre class="result">{res}</pre></section>
</div>
{metrics_related_html(i)}
{papers_fit_html(i)}
<nav class="pager">{pager}</nav>
<footer><a href="../">← 전체 목록</a> · 데이터 출처는 위에 적었습니다.</footer>
</div>
{KATEX}
{PLOT_JS % (PLOTLY, json.dumps(charts, ensure_ascii=False, default=lambda o: o.item() if hasattr(o, "item") else str(o)))}
</body>
</html>
"""
    return out


def render_stub(i):
    """노트북이 아직 없는 데이터: 소개 · 공식 링크 · 논문 · 관심 업종"""
    links = "".join(f'<a class="btn {"primary" if k == 0 else "ghost"}" href="{E(u)}" target="_blank" rel="noopener">{E(t)}</a>' for k, (t, u) in enumerate(i["links"]))
    out = head(f"{i['title']} · 현실의 데이터", i["teaser"], "../shop.css")
    five = "".join(f"<tr><td>{E(a)}</td><td>{E(b)}</td></tr>" for a, b in i["five"])
    out += f"""<div class="wrap">
{hero_html(i, links)}
<section class="block soon"><b>노트북 준비 중</b> · 데이터 월드컵에서 많이 뽑힌 데이터부터 노트북 · 통계 · 그래프를 만듭니다. 지금은 위 공식 페이지에서 데이터를 직접 볼 수 있습니다.</section>
<section class="block"><h2>어떤 문제인가</h2>{blackbox_html(i)}</section>
<section class="block"><h2>데이터</h2><div class="data-name">{E(i['data'])}</div><p class="src">{E(i['data_src'])}</p></section>
<section class="block"><h2>문제를 정해 본 예시</h2><table class="five"><tbody>{five}</tbody></table></section>
{metrics_related_html(i)}
{papers_fit_html(i)}
<footer><a href="../">← 전체 목록</a></footer>
</div>
{KATEX}
</body>
</html>
"""
    return out
