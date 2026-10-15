"""데이터 가게 웹 페이지 (docs/datashop/): 목록 index.html + 데이터마다 <slug>/index.html + 공용 shop.css.
make_shop.py 가 불러 쓴다. 그래프는 Plotly (jsdelivr plotly.js-dist-min 2.35.2), 색은 CSS 변수 (dataviz 기본 팔레트 1~3번)."""
import html, json

PLOTLY = "https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js"
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
.item img { width: 100%; aspect-ratio: 8 / 5; object-fit: cover; display: block; background: #fff; border-bottom: 1px solid var(--line); }
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
<!-- notebooks/datashop/make_shop.py 가 만든다. 고칠 때는 site_pages.py · make_shop.py 를 고치고 다시 실행 -->
</head>
<body>
"""


def render_index(items, sections):
    out = [head("현실의 데이터 · Machine Learning 1", "산업 현장에서 실제로 푸는 문제와 닮은 공개 데이터. 데이터를 열어 보고 문제를 직접 정해 보는 Colab 노트북", "shop.css")]
    out.append(f"""<div class="wrap">
<header class="top">
  <div class="eyebrow">Machine Learning 1 · 데이터 가게</div>
  <h1>현실의 데이터</h1>
  <p class="lead">추천, 이상거래 탐지, 수요예측, 불량 예측, 문서 이해 … 산업 현장에서 실제로 푸는 문제와 닮은 공개 데이터 {len(items)}가지를 모았습니다. 인턴이나 산학협력 과제에서 처음 받게 될 데이터도 대개 이런 모양입니다. 하나를 골라 들어가 데이터를 보고, Colab 노트북으로 열어 문제를 직접 정해 보세요.</p>
  <ul class="five-strip" aria-label="문제를 정하는 다섯 칸"><li><b>①</b>한 행</li><li><b>②</b>입력 X</li><li><b>③</b>출력 y</li><li><b>④</b>평가지표</li><li><b>⑤</b>기준선</li></ul>
</header>
<main>""")
    for sec in sections:
        its = [i for i in items if i["sector"] == sec]
        if not its:
            continue
        out.append(f'<h2 class="section-title">{E(sec)}</h2>\n<div class="shelf">')
        for i in its:
            out.append(f"""<a class="item" href="{i['slug']}/">
  <img src="{i['slug']}/thumb.png" alt="{E(i['title'])} 데이터 샘플" loading="lazy" width="480" height="300">
  <div class="body"><div class="tag"><span class="num">{i['num']}</span><span>{E(i['keywords'][0])}</span></div>
  <p class="title">{E(i['title'])}</p><p class="teaser">{E(i['teaser'])}</p><span class="enter">들어가 보기</span></div>
</a>""")
        out.append("</div>")
    out.append("""</main>
<footer>데이터는 모두 로그인 없이 받을 수 있는 공개 데이터이며, 데이터마다 논문 · 출처를 함께 적었습니다. 분야별 문제와 관심 업종은 일반적인 예시이고 특정 회사의 과제가 아닙니다.
 · <a href="https://github.com/SuminHan/book-ml/tree/main/notebooks/datashop">노트북 원본 (GitHub)</a></footer>
</div>
</body>
</html>
""")
    return "\n".join(out)


def render_item(i, prev, nxt, preview, results):
    sample = preview["sample"]
    rows = "".join("<tr>" + "".join(f"<td title=\"{E(v)}\">{E(v)}</td>" for v in r) + "</tr>" for r in sample["rows"])
    sample_html = f'<div class="scroll"><table class="sample"><thead><tr>{"".join(f"<th>{E(c)}</th>" for c in sample["columns"])}</tr></thead><tbody>{rows}</tbody></table></div>'
    charts = preview["charts"]
    chart_html = "".join(
        f'<div class="chart{" small" if c["layout"].get("height", 300) < 250 else ""}"><h3>{E(c["title"])}</h3>'
        + (f'<p class="note">{E(c["note"])}</p>' if c["note"] else "")
        + f'<div class="plot" id="plot-{k}" role="img" aria-label="{E(c["title"])}"></div></div>'
        for k, c in enumerate(charts))
    five = "".join(f"<tr><td>{E(a)}</td><td>{E(b)}</td></tr>" for a, b in i["five"])
    papers = "".join(f'<li><a href="{E(u)}" target="_blank" rel="noopener">{E(t)}</a></li>' for t, u in i["papers"])
    fit = i["fit"]
    res = E("\n".join(results)) if results else "(노트북을 실행하면 여기에 결과가 나온다)"
    pager = (f'<a href="../{prev["slug"]}/">← {prev["num"]} {E(prev["title"])}</a>' if prev else "<span></span>") + \
            (f'<a href="../{nxt["slug"]}/">{nxt["num"]} {E(nxt["title"])} →</a>' if nxt else "<span></span>")
    out = head(f"{i['title']} · 현실의 데이터", i["teaser"], "../shop.css")
    out += f"""<div class="wrap">
<nav class="crumb"><a href="../">← 현실의 데이터 (전체 {i['total']}개)</a></nav>
<div class="hero">
  <div>
    <div class="tag"><span class="num">{i['num']}</span><span>{E(i['sector'])}</span></div>
    <h1>{E(i['title'])}</h1>
    <div class="chips">{"".join(f'<span class="chip">{E(k)}</span>' for k in i['keywords'])}</div>
    <p class="teaser-big">{E(i['teaser'])}</p>
    <div class="actions"><a class="btn primary" href="{E(i['colab'])}" target="_blank" rel="noopener">Colab 에서 열기</a>
      <a class="btn ghost" href="{E(i['github'])}" target="_blank" rel="noopener">GitHub 에서 보기</a></div>
  </div>
  <img src="thumb.png" alt="{E(i['title'])} 데이터 샘플" width="480" height="300">
</div>
<section class="block"><h2>데이터</h2>
  <div class="data-name">{E(i['data'])} · {E(i['size'])}</div><p class="src">{E(i['data_src'])}</p>
  <h2 style="margin-top:16px">처음 몇 줄</h2>{sample_html}
</section>
<section class="block"><h2>데이터 모양 보기</h2><div class="charts">{chart_html}</div></section>
<div class="two">
  <section class="block"><h2>문제를 정해 본 예시 (다섯 칸)</h2><table class="five"><tbody>{five}</tbody></table>
    <p class="src" style="margin-top:10px">예시와 다르게 정해도 됩니다. 노트북 맨 끝의 '나의 다섯 칸' 을 채워 보세요.</p></section>
  <section class="block"><h2>노트북에서 나온 결과 (기준선 vs 모델)</h2><pre class="result">{res}</pre></section>
</div>
<div class="two">
  <section class="block"><h2>논문 · 출처</h2><ul class="papers">{papers}</ul>
    <p class="src" style="margin-top:8px">누가, 왜, 어떻게 만든 데이터인지 직접 확인해 보세요.</p></section>
  <section class="block fit"><h2>이 데이터에 관심 있을 곳</h2><dl><dt>업종</dt><dd>{E(fit[0])}</dd><dt>직무</dt><dd>{E(fit[1])}</dd></dl>
    <p class="src" style="margin-top:8px">인턴 · 산학협력 · 프로젝트 주제를 고를 때 참고하세요 (일반적인 예시).</p></section>
</div>
<nav class="pager">{pager}</nav>
<footer><a href="../">← 전체 목록</a> · 데이터는 모두 공개 데이터이며 출처는 위에 적었습니다.</footer>
</div>
{PLOT_JS % (PLOTLY, json.dumps(charts, ensure_ascii=False, default=lambda o: o.item() if hasattr(o, "item") else str(o)))}
</body>
</html>
"""
    return out
