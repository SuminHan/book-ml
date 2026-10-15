"""월드컵 카드 그림: Wikimedia Commons 사진 (covers.json + 새 후보) 을 480x300 으로 잘라 base64 로. 자금세탁은 패턴 그림을 직접 그린다."""
import base64, io, json, os, urllib.request
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
HERE = os.path.dirname(os.path.abspath(__file__))
UA = {"User-Agent": "ML course datashop (ksa.stlab@gmail.com)"}
covers = json.load(open(os.path.join(HERE, "..", "covers.json")))
cands = json.load(open("/private/tmp/claude-501/-Users-hansumin-book-ml/7e0a38fc-63e4-4738-aaf4-a8251b1a8d5b/scratchpad/covers/cands.json"))
pick = {"point_cloud": 0, "driving": 0, "robot": 0, "medical": 1, "satellite": 0, "air_quality": 0, "electricity": 1, "football": 5,
        "chess": 1, "speech": 1, "drug": 2, "protein": 1, "crop": 2}
for k, v in pick.items():
    covers[k] = cands[k][v]
covers.pop("stock_index_unused", None)
json.dump(covers, open(os.path.join(HERE, "..", "covers.json"), "w"), ensure_ascii=False, indent=1)

def crop(im, w=480, h=300):
    r = max(w / im.width, h / im.height); im = im.resize((round(im.width * r), round(im.height * r)))
    l, t = (im.width - w) // 2, (im.height - h) // 2
    return im.crop((l, t, l + w, t + h))

def b64(im):
    buf = io.BytesIO(); im.convert("RGB").save(buf, "JPEG", quality=72); return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

def aml_picture():
    plt.rcParams.update({"font.family": "Noto Sans CJK KR"})
    fig, axs = plt.subplots(1, 3, figsize=(4.8, 3), dpi=100); fig.patch.set_facecolor("#1f2433")
    RED, BLUE, WHITE = "#ff6b5b", "#7fb2ff", "#f2f2f2"
    def node(ax, x, y, c, s=180): ax.scatter([x], [y], s=s, color=c, zorder=3, edgecolor="#1f2433", lw=1.5)
    def arrow(ax, a, b, c): ax.annotate("", b, a, arrowprops=dict(arrowstyle="-|>", color=c, lw=1.6, shrinkA=8, shrinkB=8))
    # circular
    ax = axs[0]; th = np.linspace(0, 2 * np.pi, 5)[:-1] + np.pi / 2; P = np.c_[np.cos(th), np.sin(th)]
    for i in range(4): arrow(ax, P[i], P[(i + 1) % 4], RED)
    for i, p in enumerate(P): node(ax, *p, RED if i == 0 else BLUE)
    ax.set_title("원형", color=WHITE, fontsize=12)
    # fan-in fan-out (scatter-gather)
    ax = axs[1]; src, dst = (-1, 0), (1, 0); mids = [(0, y) for y in (-1, -0.33, 0.33, 1)]
    for m in mids: arrow(ax, src, m, RED); arrow(ax, m, dst, RED)
    node(ax, *src, RED); node(ax, *dst, RED); [node(ax, *m, BLUE, 120) for m in mids]
    ax.set_title("쪼개기 · 모으기", color=WHITE, fontsize=12)
    # fan-in (pump)
    ax = axs[2]; c = (0.6, 0); ins = [(-0.9, y) for y in (-1, -0.33, 0.33, 1)]
    for p in ins: arrow(ax, p, c, RED)
    [node(ax, *p, BLUE, 120) for p in ins]; node(ax, *c, RED, 260)
    ax.set_title("한곳에 모으기", color=WHITE, fontsize=12)
    for ax in axs: ax.set_xlim(-1.4, 1.4); ax.set_ylim(-1.4, 1.5); ax.set_aspect("equal"); ax.axis("off")
    fig.subplots_adjust(0.01, 0.02, 0.99, 0.9, 0.05)
    buf = io.BytesIO(); fig.savefig(buf, format="png", facecolor=fig.get_facecolor()); plt.close(fig)
    return Image.open(buf)

ids = [c["id"] for c in json.load(open(os.path.join(HERE, "candidates.json")))]
images, credits = {}, {}
for i in ids:
    if i == "aml_transactions":
        images[i] = b64(crop(aml_picture())); credits[i] = None; continue
    c = covers[i]
    raw = urllib.request.urlopen(urllib.request.Request(c["thumb"], headers=UA), timeout=120).read()
    images[i] = b64(crop(Image.open(io.BytesIO(raw))))
    credits[i] = dict(title=c["title"].replace("File:", ""), artist=c["artist"] or "작자 미상", license=c["license"], page=c["page"])
    print(i, len(images[i]) // 1024, "KB")
json.dump(dict(images=images, credits=credits), open(os.path.join(HERE, "images.json"), "w"))
print("total", sum(len(v) for v in images.values()) // 1024, "KB")
