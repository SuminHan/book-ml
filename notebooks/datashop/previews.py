"""데이터 가게 웹 페이지용 미리보기: 썸네일 그림 (png) · 샘플 표 · 그래프 (Plotly 트레이스) · 노트북 결과 줄.
make_shop.py 가 불러 쓴다. 데이터는 data/ (노트북을 한 번 실행하면 생긴다).
그래프 색은 'var:series-1' 처럼 역할 이름으로 적고, 페이지의 JS 가 CSS 변수 값으로 바꾼다 (밝은 · 어두운 모드)."""
import io, json, os, re, zipfile
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"          # dataviz 기본 팔레트 1~3번 (밝은 모드)
S1, S2, S3 = "var:series-1", "var:series-2", "var:series-3"
plt.rcParams.update({"font.family": "Noto Sans CJK KR", "axes.unicode_minus": False, "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False})


def bar(x, y, color=S1, horizontal=False, name=None):
    t = dict(type="bar", marker=dict(color=color), name=name or "")
    if horizontal:
        t.update(x=list(y), y=list(x), orientation="h")
    else:
        t.update(x=list(x), y=list(y))
    return t


def line(x, y, color=S1, name=None):
    return dict(type="scatter", mode="lines+markers", x=list(x), y=list(y), name=name or "", line=dict(color=color, width=2), marker=dict(size=6))


def chart(title, traces, xlab="", ylab="", note="", **layout):
    lay = dict(xaxis=dict(title=xlab), yaxis=dict(title=ylab)); lay.update(layout)
    return dict(title=title, note=note, traces=traces, layout=lay)


def hist_bins(values, bins, density=False):
    h, e = np.histogram(values, bins=bins, density=density)
    return ((e[:-1] + e[1:]) / 2).round(3).tolist(), h.round(5).tolist()


def save_thumb(fig, path):
    fig.savefig(path, dpi=100, facecolor="white"); plt.close(fig)


def fmt(v):
    """보기 좋은 숫자: 정수면 정수로, 큰 수는 쉼표, 작은 수는 유효숫자 4개 (지수 표기 없이)"""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    if isinstance(v, (bool, np.bool_)):
        return str(v)
    if isinstance(v, (int, np.integer)):
        return f"{int(v):,}" if abs(int(v)) >= 10000 else str(int(v))
    if isinstance(v, (float, np.floating)):
        v = float(v)
        if v.is_integer():
            return f"{int(v):,}" if abs(v) >= 10000 else str(int(v))
        if abs(v) >= 1000:
            return f"{v:,.1f}"
        return f"{v:.4g}" if abs(v) >= 1e-3 else f"{v:.6f}".rstrip("0")
    return str(v)


def table(df, n=6, seed=0):
    """무작위로 섞어 n 줄 (처음 몇 줄만 보면 데이터가 정렬된 모습만 보인다)"""
    df = df.sample(min(n, len(df)), random_state=seed)
    return dict(columns=[str(c) for c in df.columns], rows=[[fmt(v) for v in r] for r in df.itertuples(index=False)])


def stats(df, max_cols=16):
    """데이터 통계: 행 · 열 수 + 열마다 종류 · 빈칸 · 고유값 · 범위 (숫자) 또는 가장 흔한 값 (글자)"""
    rows = []
    for c in list(df.columns)[:max_cols]:
        s = df[c]
        miss = f"{s.isna().mean():.1%}" if s.isna().any() else "0"
        if pd.api.types.is_bool_dtype(s):
            kind, rng = "참/거짓", f"참 {s.mean():.1%}"
        elif pd.api.types.is_numeric_dtype(s):
            kind = "숫자"
            rng = f"{fmt(s.min())} ~ {fmt(s.max())} (평균 {fmt(round(float(s.mean()), 4))})"
        elif pd.api.types.is_datetime64_any_dtype(s):
            kind, rng = "날짜", f"{s.min()} ~ {s.max()}"
        else:
            kind = "글자 · 범주"; top = s.astype(str).value_counts()
            rng = f"가장 흔한 값: {top.index[0][:30]} ({top.iloc[0] / len(s):.1%})"
        rows.append([str(c), kind, miss, fmt(int(s.nunique())), rng])
    return dict(n_rows=len(df), n_cols=df.shape[1], columns=["열", "종류", "빈칸", "고유값 수", "범위 · 대표값"], rows=rows,
                more=max(0, df.shape[1] - max_cols))


# ------------------------------------------------------------------ 데이터별
def fashion_commerce(thumb):
    df = pd.read_parquet(os.path.join(D, "fashion_0.parquet"))
    s = df.sample(8, random_state=3)
    fig, axs = plt.subplots(2, 4, figsize=(4.8, 3))
    for ax, (_, r) in zip(axs.ravel(), s.iterrows()):
        ax.imshow(Image.open(io.BytesIO(r["image"]["bytes"]))); ax.axis("off")
    fig.subplots_adjust(0, 0, 1, 1, 0.02, 0.02); save_thumb(fig, thumb)
    top = df["articleType"].value_counts().head(12)[::-1]
    return dict(stats=stats(df.drop(columns="image")), sample=table(df.drop(columns="image")[["productDisplayName", "gender", "articleType", "baseColour", "season", "year"]]),
                charts=[chart("상품 종류 (articleType) 상위 12개", [bar(top.index, top.values, horizontal=True)], "상품 수", "", "22,036개 중. 종류가 143가지라 한쪽으로 몰려 있다"),
                        chart("큰 분류 (masterCategory)", [bar(*zip(*df["masterCategory"].value_counts().items()))], "", "상품 수")])


def shop_conversion(thumb):
    with zipfile.ZipFile(os.path.join(D, "shoppers.zip")) as z:
        df = pd.read_csv(z.open("online_shoppers_intention.csv"))
    months = ["Feb", "Mar", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    g = df.groupby("Month").agg(rate=("Revenue", "mean"), n=("Revenue", "size")).reindex(months)
    fig, ax = plt.subplots(figsize=(4.8, 3))
    ax.bar(range(len(months)), g["rate"] * 100, color=C1); ax.set_xticks(range(len(months)), [m[:3] for m in months], fontsize=8)
    ax.set_ylabel("구매 전환율 (%)"); fig.tight_layout(); save_thumb(fig, thumb)
    pv = np.log10(df["PageValues"] + 1)
    bins = np.linspace(0, pv.max(), 30)
    x0, y0 = hist_bins(pv[~df["Revenue"]], bins, True); x1, y1 = hist_bins(pv[df["Revenue"]], bins, True)
    return dict(stats=stats(df), sample=table(df[["Administrative", "ProductRelated", "ProductRelated_Duration", "BounceRates", "PageValues", "Month", "VisitorType", "Revenue"]]),
                charts=[chart("달마다 구매 전환율", [bar(months, (g["rate"] * 100).round(1))], "", "전환율 (%)", "11월이 가장 높다 (연말 쇼핑)"),
                        chart("PageValues: 산 방문 vs 안 산 방문", [bar(x0, y0, S1, name="안 샀다"), bar(x1, y1, S2, name="샀다")], "log10(PageValues + 1)", "비율 (밀도)",
                              "산 방문은 PageValues 가 0 이 아닌 경우가 대부분. 이 값을 방문 도중에 알 수 있을까?", barmode="overlay")])


def movie_ratings(thumb):
    z = zipfile.ZipFile(os.path.join(D, "ml-latest-small.zip"))
    r = pd.read_csv(z.open("ml-latest-small/ratings.csv")); mv = pd.read_csv(z.open("ml-latest-small/movies.csv"))
    top_m = r["movieId"].value_counts().index[:5]; users = r[r["movieId"].isin(top_m)]["userId"].value_counts().index[:6]
    sub = r[r["userId"].isin(users) & r["movieId"].isin(top_m)].pivot_table(index="userId", columns="movieId", values="rating").reindex(index=users, columns=top_m)
    names = mv.set_index("movieId")["title"].str.replace(r" \(\d{4}\)$", "", regex=True).str[:14]
    fig, ax = plt.subplots(figsize=(4.8, 3)); ax.axis("off"); ax.set_xlim(0, 6); ax.set_ylim(0, 7.2)
    for j, m in enumerate(top_m): ax.text(j + 1.05, 6.6, names[m], ha="center", fontsize=6.5, rotation=0, wrap=True)
    for i, u in enumerate(users):
        y = 5.6 - i * 0.95; ax.text(0.45, y, f"사람 {i + 1}", ha="center", va="center", fontsize=8)
        for j, m in enumerate(top_m):
            v = sub.loc[u, m]; ax.add_patch(plt.Rectangle((j + 0.6, y - 0.4), 0.9, 0.8, fc="#eef3fb" if v == v else "#fff3ec", ec="#cccccc"))
            ax.text(j + 1.05, y, f"★{v:g}" if v == v else "?", ha="center", va="center", fontsize=9, color=C1 if v == v else C2, fontweight="bold")
    save_thumb(fig, thumb)
    cnt = r["rating"].value_counts().sort_index()
    per = r.groupby("movieId").size()
    x, y = hist_bins(np.log10(per), 30)
    return dict(stats=stats(r.merge(mv, on="movieId")), sample=table(r.merge(mv, on="movieId")[["userId", "title", "genres", "rating"]]),
                charts=[chart("별점 분포", [bar(cnt.index.astype(str), cnt.values)], "별점", "개수", "4점이 가장 많다. 0.5점 단위"),
                        chart("영화마다 받은 별점 수", [bar(x, y)], "log10(별점 수)", "영화 수", "대부분의 영화는 별점이 몇 개뿐이다 (긴 꼬리)")])


def korean_reviews(thumb):
    df = pd.read_csv(os.path.join(D, "nsmc_train.txt"), sep="\t").dropna()
    s = df.sample(6, random_state=7)
    fig, ax = plt.subplots(figsize=(4.8, 3)); ax.axis("off")
    for k, (_, row) in enumerate(s.iterrows()):
        t = row["document"][:26] + ("…" if len(row["document"]) > 26 else "")
        ax.text(0.03, 0.9 - k * 0.16, ("😀 " if False else "") + t, fontsize=11, color=C1 if row["label"] == 1 else C2, transform=ax.transAxes)
    ax.text(0.97, 0.02, "파랑 = 긍정, 주황 = 부정", ha="right", fontsize=8, color="gray", transform=ax.transAxes)
    save_thumb(fig, thumb)
    L = df["document"].str.len(); bins = np.arange(0, 150, 5)
    x0, y0 = hist_bins(L[df["label"] == 0], bins); x1, y1 = hist_bins(L[df["label"] == 1], bins)
    return dict(stats=stats(df), sample=table(df.sample(6, random_state=1)[["document", "label"]], 6),
                charts=[chart("리뷰 길이 (글자 수)", [bar(x1, y1, S1, name="긍정"), bar(x0, y0, S2, name="부정")], "글자 수", "리뷰 수",
                              "140자 제한 때문에 끝이 잘린다", barmode="overlay")])


def card_fraud(thumb):
    from sklearn.datasets import fetch_openml
    df = fetch_openml(data_id=1597, as_frame=True, parser="auto", data_home=D).frame
    df["Class"] = df["Class"].astype(int)
    s = df.sample(4000, random_state=0); f = df[df["Class"] == 1]
    fig, ax = plt.subplots(figsize=(4.8, 3))
    ax.scatter(s["V14"], s["V17"], s=3, color="#bbbbbb", label="정상 (일부)"); ax.scatter(f["V14"], f["V17"], s=6, color=C2, label="사기")
    ax.set_xlabel("V14"); ax.set_ylabel("V17"); ax.legend(fontsize=8, frameon=False); fig.tight_layout(); save_thumb(fig, thumb)
    bins = np.linspace(0, 4.5, 40)
    x0, y0 = hist_bins(np.log10(df.loc[df["Class"] == 0, "Amount"] + 1), bins, True)
    x1, y1 = hist_bins(np.log10(df.loc[df["Class"] == 1, "Amount"] + 1), bins, True)
    return dict(stats=stats(df), sample=table(df[["V1", "V2", "V3", "V14", "V17", "Amount", "Class"]]),
                charts=[chart("정상 vs 사기 거래 수", [bar(["정상", "사기"], [int((df["Class"] == 0).sum()), int((df["Class"] == 1).sum())])], "", "거래 수 (로그 눈금)",
                              "284,315 대 492. 로그 눈금이라 막대 높이가 비슷해 보인다", yaxis=dict(type="log", title="거래 수 (로그 눈금)")),
                        chart("거래 금액: 정상 vs 사기", [bar(x0, y0, S1, name="정상"), bar(x1, y1, S2, name="사기")], "log10(금액 + 1)", "비율 (밀도)",
                              "사기는 아주 작은 금액 (시험 결제) 이 많다", barmode="overlay")])


def support_routing(thumb):
    df = pd.read_csv(os.path.join(D, "banking77_train.csv"))
    s = df.sample(5, random_state=4)
    fig, ax = plt.subplots(figsize=(4.8, 3)); ax.axis("off")
    for k, (_, row) in enumerate(s.iterrows()):
        ax.text(0.03, 0.88 - k * 0.19, row["text"][:44] + ("…" if len(row["text"]) > 44 else ""), fontsize=9.5, transform=ax.transAxes)
        ax.text(0.06, 0.80 - k * 0.19, "→ " + row["category"], fontsize=8.5, color=C1, transform=ax.transAxes)
    save_thumb(fig, thumb)
    c = df["category"].value_counts()
    sel = pd.concat([c.head(8), c.tail(7)])[::-1]
    return dict(stats=stats(df), sample=table(df.sample(6, random_state=2), 6),
                charts=[chart("의도(팀)별 문의 수: 많은 8개 · 적은 7개", [bar(sel.index, sel.values, horizontal=True)], "학습 문장 수", "", "77가지, 가장 많은 것과 적은 것이 약 5배 차이", margin=dict(l=230))])


def hotel_booking(thumb):
    df = pd.read_csv(os.path.join(D, "hotels.csv"))
    months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
    p = df[df["is_canceled"] == 0].groupby(["arrival_date_month", "hotel"])["adr"].mean().unstack().reindex(months)
    fig, ax = plt.subplots(figsize=(4.8, 3))
    ax.plot(range(12), p["Resort Hotel"], color=C1, marker="o", label="리조트"); ax.plot(range(12), p["City Hotel"], color=C2, marker="o", label="도시")
    ax.set_xticks(range(12), [m[:1] for m in months]); ax.set_ylabel("평균 1박 가격 (EUR)"); ax.legend(frameon=False, fontsize=8); fig.tight_layout(); save_thumb(fig, thumb)
    bins = [0, 7, 30, 90, 180, 365, 800]; lab = ["0~7일", "8~30일", "31~90일", "91~180일", "181~365일", "366일~"]
    cr = df.groupby(pd.cut(df["lead_time"], bins, right=True, include_lowest=True, labels=lab), observed=False)["is_canceled"].mean()
    return dict(stats=stats(df), sample=table(df[["hotel", "is_canceled", "lead_time", "arrival_date_month", "adults", "customer_type", "adr", "reservation_status"]]),
                charts=[chart("달마다 평균 1박 가격 (취소 안 된 예약)", [line([m[:3] for m in months], p["Resort Hotel"].round(1), S1, "리조트 호텔"), line([m[:3] for m in months], p["City Hotel"].round(1), S2, "도시 호텔")], "", "EUR",
                              "리조트는 여름에 크게 오른다"),
                        chart("며칠 전에 예약했나 → 취소율", [bar(lab, (cr * 100).round(1))], "체크인까지 남은 날", "취소율 (%)", "일찍 예약할수록 많이 취소한다")])


def link_prediction(thumb):
    e = pd.read_csv(os.path.join(D, "email-Eu-core.txt.gz"), sep=" ", header=None, names=["a", "b"])
    dept = pd.read_csv(os.path.join(D, "email-dept.txt.gz"), sep=" ", header=None, names=["p", "d"]).set_index("p")["d"]
    import networkx as nx
    G = nx.Graph(); G.add_edges_from((a, b) for a, b in e[["a", "b"]].itertuples(index=False) if a != b)
    hub = sorted(G.degree, key=lambda t: -t[1])[40][0]
    sub = G.subgraph(list(G.neighbors(hub))[:28] + [hub]); pos = nx.spring_layout(sub, seed=1)
    fig, ax = plt.subplots(figsize=(4.8, 3)); ax.axis("off")
    nx.draw_networkx_edges(sub, pos, ax=ax, edge_color="#bbbbbb", width=0.6)
    nx.draw_networkx_nodes(sub, pos, ax=ax, node_size=60, node_color=[C2 if n == hub else C1 for n in sub.nodes])
    fig.subplots_adjust(0, 0, 1, 1); save_thumb(fig, thumb)
    pairs = {tuple(sorted(p)) for p in e[["a", "b"]].itertuples(index=False) if p[0] != p[1]}
    deg = np.zeros(len(dept), int)
    for a, b in pairs: deg[a] += 1; deg[b] += 1
    x, y = hist_bins(deg, np.arange(0, 360, 10))
    same = np.mean([dept[a] == dept[b] for a, b in pairs])
    return dict(stats=stats(e.assign(dept_a=e["a"].map(dept), dept_b=e["b"].map(dept))), sample=dict(columns=["사람 a", "사람 b", "a 의 부서", "b 의 부서"], rows=[[str(a), str(b), str(dept[a]), str(dept[b])] for a, b in list(sorted(pairs))[:5]]),
                charts=[chart("사람마다 연결 수", [bar(x, y)], "연결 수 (10개 단위)", "사람 수", "몇 명에게 연결이 몰려 있다"),
                        chart("연결된 쌍 중 같은 부서 비율", [bar(["같은 부서", "다른 부서"], [round(same * 100, 1), round((1 - same) * 100, 1)])], "", "%",
                              f"연결의 {same:.0%} 가 같은 부서 안. 부서가 42개라 무작위라면 훨씬 낮을 것")])


def bike_demand(thumb):
    with zipfile.ZipFile(os.path.join(D, "seoul_bike.zip")) as z:
        df = pd.read_csv(z.open("SeoulBikeData.csv"), encoding="latin1")
    df.columns = ["date", "count", "hour", "temp", "humidity", "wind", "visibility", "dew", "solar", "rain", "snow", "season", "holiday", "functioning"]
    df["date"] = pd.to_datetime(df["date"], format="%d/%m/%Y"); df["weekend"] = df["date"].dt.dayofweek >= 5
    h = df[df["functioning"] == "Yes"].groupby(["hour", "weekend"])["count"].mean().unstack()
    fig, ax = plt.subplots(figsize=(4.8, 3))
    ax.plot(h.index, h[False], color=C1, label="평일"); ax.plot(h.index, h[True], color=C2, label="주말")
    ax.set_xlabel("시각"); ax.set_ylabel("시간당 대여"); ax.legend(frameon=False, fontsize=8); fig.tight_layout(); save_thumb(fig, thumb)
    d = df.groupby("date")["count"].sum()
    return dict(stats=stats(df), sample=table(df[["date", "hour", "count", "temp", "humidity", "rain", "season", "holiday"]].assign(date=lambda x: x["date"].dt.date)),
                charts=[chart("시각별 평균 대여 수", [line(h.index, h[False].round(0), S1, "평일"), line(h.index, h[True].round(0), S2, "주말")], "시각", "대여 수 / 시간", "평일은 출근 8시 · 퇴근 18시에 두 봉우리"),
                        chart("하루 대여 수 (2017.12 ~ 2018.11)", [dict(type="scatter", mode="lines", x=[str(t.date()) for t in d.index], y=d.values.tolist(), line=dict(color=S1, width=1.5), name="")], "", "대여 수 / 일",
                              "겨울엔 적고 여름 · 가을에 많다. 0 인 날은 서비스를 쉰 날")])


def semiconductor_yield(thumb):
    with zipfile.ZipFile(os.path.join(D, "secom.zip")) as z:
        X = pd.read_csv(z.open("secom.data"), sep=" ", header=None)
        L = pd.read_csv(z.open("secom_labels.data"), sep=" ", header=None, names=["label", "time"])
    fig, ax = plt.subplots(figsize=(4.8, 3))
    Z = (X - X.mean()) / X.std(); ax.imshow(Z.iloc[:120, :200].clip(-3, 3).fillna(-9).values, cmap="Blues", aspect="auto", vmin=-3, vmax=3)
    ax.axis("off"); ax.text(0.99, 0.02, "제품 120개 × 센서 200개 (흰 칸 = 빈칸)", ha="right", fontsize=8, color="gray", transform=ax.transAxes)
    fig.subplots_adjust(0, 0, 1, 1); save_thumb(fig, thumb)
    miss = X.isna().mean() * 100
    x, y = hist_bins(miss, np.arange(0, 105, 5))
    t = pd.to_datetime(L["time"], format="%d/%m/%Y %H:%M:%S")
    m = (L["label"] == 1).groupby(t.dt.to_period("M")).agg(["mean", "size"])
    return dict(stats=stats(X.assign(label=L["label"].map({-1: "pass", 1: "fail"}))), sample=table(X.iloc[:, :8].assign(label=L["label"].map({-1: "pass", 1: "fail"}))),
                charts=[chart("센서마다 빈칸 비율", [bar(x, y)], "빈칸 비율 (%)", "센서 수", "대부분 거의 다 차 있지만, 반 이상 빈 센서도 있다"),
                        chart("달마다 불량률", [bar([str(p) for p in m.index], (m["mean"] * 100).round(1))], "", "불량률 (%)", "달마다 다르다: 시간 순서로 나눠야 하는 이유")])


def predictive_maintenance(thumb):
    with zipfile.ZipFile(os.path.join(D, "ai4i2020.zip")) as z:
        df = pd.read_csv(z.open("ai4i2020.csv"))
    ok = df[df["Machine failure"] == 0].sample(2000, random_state=0); f = df[df["Machine failure"] == 1]
    fig, ax = plt.subplots(figsize=(4.8, 3))
    ax.scatter(ok["Rotational speed [rpm]"], ok["Torque [Nm]"], s=3, color="#bbbbbb", label="정상 (일부)")
    ax.scatter(f["Rotational speed [rpm]"], f["Torque [Nm]"], s=7, color=C2, label="고장")
    ax.set_xlabel("회전 속도 (rpm)"); ax.set_ylabel("토크 (Nm)"); ax.legend(frameon=False, fontsize=8); fig.tight_layout(); save_thumb(fig, thumb)
    sc = lambda d, c, n: dict(type="scattergl", mode="markers", x=d["Rotational speed [rpm]"].tolist(), y=d["Torque [Nm]"].tolist(), marker=dict(color=c, size=5, opacity=0.6), name=n)
    modes = df[["TWF", "HDF", "PWF", "OSF", "RNF"]].sum()
    names = {"TWF": "공구 마모", "HDF": "열 방출", "PWF": "전력", "OSF": "과부하", "RNF": "무작위"}
    return dict(stats=stats(df), sample=table(df[["Type", "Air temperature [K]", "Process temperature [K]", "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]", "Machine failure"]]),
                charts=[chart("회전 속도 vs 토크", [sc(ok, "var:muted", "정상 (2,000개만)"), sc(f, S2, "고장")], "회전 속도 (rpm)", "토크 (Nm)", "고장은 양 끝 (느리고 센 힘, 빠르고 약한 힘) 에 몰린다"),
                        chart("고장 종류별 건수", [bar([names[k] for k in modes.index], modes.values)], "", "건수", "한 기록에 고장 종류가 둘 이상일 수도 있다")])


def steel_defects(thumb):
    with zipfile.ZipFile(os.path.join(D, "steel_faults.zip")) as z:
        names = z.read("Faults27x7_var").decode().split()
        df = pd.read_csv(z.open("Faults.NNA"), sep=r"\s+", header=None, names=names)
    faults = names[-7:]; df["fault"] = df[faults].idxmax(axis=1)
    c = df["fault"].value_counts()
    fig, ax = plt.subplots(figsize=(4.8, 3)); ax.barh(c.index[::-1], c.values[::-1], color=C1); ax.set_xlabel("결함 수"); fig.tight_layout(); save_thumb(fig, thumb)
    med = df.groupby("fault")["Pixels_Areas"].median().sort_values()
    return dict(stats=stats(df[names[:-7] + ["fault"]]), sample=table(df[["X_Minimum", "Y_Minimum", "Pixels_Areas", "Sum_of_Luminosity", "Steel_Plate_Thickness", "TypeOfSteel_A300", "fault"]]),
                charts=[chart("결함 종류별 개수", [bar(c.index, c.values)], "", "개수", "'Other_Faults' (기타) 가 가장 많다"),
                        chart("결함 종류별 결함 넓이 (중앙값)", [bar(med.index, med.values)], "", "픽셀 수 (로그 눈금)", "종류마다 크기가 수십 배 다르다", yaxis=dict(type="log", title="픽셀 수 (로그 눈금)"))])


def bearing_vibration(thumb):
    from scipy.io import loadmat
    files = {"정상": 97, "내륜 결함": 105, "볼 결함": 118, "외륜 결함": 130}
    sig = {}
    for k, n in files.items():
        m = loadmat(os.path.join(D, f"cwru_{n}.mat")); sig[k] = m[[x for x in m if x.endswith("DE_time")][0]].ravel()
    fig, axs = plt.subplots(4, 1, figsize=(4.8, 3), sharex=True, sharey=True)
    for ax, (k, s) in zip(axs, sig.items()):
        ax.plot(s[:400], color=C1, lw=0.7); ax.axis("off"); ax.text(0, 1.5, k, fontsize=8)
    fig.subplots_adjust(0.01, 0.01, 0.99, 0.97, 0, 0.1); save_thumb(fig, thumb)
    t = (np.arange(600) / 12000 * 1000).round(2).tolist()
    small = [chart(f"{k}: 0.05초", [dict(type="scatter", mode="lines", x=t, y=s[:600].round(4).tolist(), line=dict(color=S1, width=1), name=k)], "밀리초", "가속도", "",
                   yaxis=dict(range=[-3, 3], title="가속도"), height=200) for k, s in sig.items()]
    from scipy.stats import kurtosis
    kur = {k: float(np.mean([kurtosis(s[i:i + 1024]) for i in range(0, len(s) - 1024, 1024)])) for k, s in sig.items()}
    return dict(stats=None, sample=dict(columns=["상태", "파일", "샘플 수", "길이 (초)"], rows=[[k, f"{files[k]}.mat", f"{len(s):,}", f"{len(s) / 12000:.1f}"] for k, s in sig.items()]),
                charts=small + [chart("조각(1,024 샘플)의 평균 첨도", [bar(list(kur), [round(v, 2) for v in kur.values()])], "", "첨도", "결함이 있으면 뾰족한 충격 → 첨도가 커진다")])


def document_types(thumb):
    names = ["letter", "form", "email", "handwritten", "advertisement", "scientific report", "scientific publication", "specification",
             "file folder", "news article", "budget", "invoice", "presentation", "questionnaire", "resume", "memo"]
    df = pd.read_parquet(os.path.join(D, "rvlcdip100_train.parquet"))
    fig, axs = plt.subplots(2, 4, figsize=(4.8, 3))
    for ax, k in zip(axs.ravel(), [0, 2, 3, 4, 8, 11, 12, 14]):
        i = np.where(df["label"].values == k)[0][0]
        ax.imshow(Image.open(io.BytesIO(df["image"][i]["bytes"])), cmap="gray"); ax.axis("off"); ax.set_title(names[k], fontsize=7)
    fig.subplots_adjust(0, 0, 1, 0.93, 0.03, 0.18); save_thumb(fig, thumb)
    rows = []
    for b, l in zip(df["image"], df["label"]):
        im = Image.open(io.BytesIO(b["bytes"])).convert("L"); g = np.asarray(im.resize((96, 128)), dtype=float) / 255
        rows.append(dict(label=names[l], width=im.size[0], height=im.size[1], ink=round(float((1 - g).mean()), 3)))
    meta = pd.DataFrame(rows)
    ink = meta.groupby("label")["ink"].mean().sort_values()
    return dict(stats=stats(meta), sample=table(meta),
                charts=[chart("종류별 평균 잉크 양 (검은 부분의 비율)", [bar(ink.index, ink.values.round(3))], "", "잉크 비율", "광고 · 발표 자료는 진하고, 편지 · 이메일은 연하다: 모양만으로도 단서가 된다", xaxis=dict(tickangle=-45, title=""))])


def ai_usecases(thumb):
    usage = pd.read_csv(os.path.join(D, "onet_task_mappings.csv")); tasks = pd.read_csv(os.path.join(D, "onet_task_statements.csv"))
    tasks["task_name"] = tasks["Task"].str.lower().str.strip()
    m = tasks.merge(usage, on="task_name", how="left").fillna({"pct": 0})
    m["pct_split"] = m["pct"] / m.groupby("task_name")["Title"].transform("count")
    job = m.groupby("Title")["pct_split"].sum().sort_values(ascending=False)
    top = job.head(8)[::-1]
    fig, ax = plt.subplots(figsize=(4.8, 3)); ax.barh([t[:28] for t in top.index], top.values, color=C1); ax.tick_params(labelsize=7); ax.set_xlabel("AI 대화 중 비율 (%)")
    fig.tight_layout(); save_thumb(fig, thumb)
    t15 = job.head(15)[::-1]
    return dict(stats=stats(usage), sample=table(usage.sort_values("pct", ascending=False).assign(task_name=lambda d: d["task_name"].str[:80]).round(2)),
                charts=[chart("AI 대화가 가장 많이 닿은 직업 15개", [bar(t15.index, t15.values.round(2), horizontal=True)], "AI 대화 중 비율 (%)", "",
                              "소프트웨어 · 글쓰기 · 분석 쪽에 몰려 있다 (2025년 2월 공개분)", margin=dict(l=320))])


def result_lines(nb_path):
    """노트북에 저장된 출력에서 기준선 · 모델 비교 줄만 뽑는다"""
    import nbformat as nbf
    nb = nbf.read(nb_path, as_version=4); out = []
    pat = re.compile(r"baseline|always|nobody|accuracy|MAE|RMSE|AUC|same type|re-inspect|random score|common friends|same department|tasks with any AI use")
    for c in nb.cells:
        if c.cell_type == "code":
            for o in c.get("outputs", []):
                if o.get("output_type") == "stream":
                    out += [l.rstrip() for l in o["text"].split("\n") if pat.search(l) and not l.strip().startswith("print")]
    return out


# ------------------------------------------------------------------ 새 데이터 4개 (+ GIF 썸네일)
def save_gif(frames_fn, n, path, fps=6):
    """frames_fn(k) 로 그린 matplotlib 그림 n장을 GIF 로 (480x300)"""
    ims = []
    for k in range(n):
        fig = frames_fn(k); buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=100, facecolor="white"); plt.close(fig)
        ims.append(Image.open(buf).convert("P", palette=Image.ADAPTIVE, colors=64))
    ims[0].save(path, save_all=True, append_images=ims[1:], duration=int(1000 / fps), loop=0, optimize=True)


def aml_transactions(thumb):
    with zipfile.ZipFile(os.path.join(D, "HI-Small_Trans.csv.zip")) as z:
        df = pd.read_csv(z.open("HI-Small_Trans.csv"), nrows=None)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    # 썸네일: 세탁 패턴 세 가지 (직접 그림)
    fig, axs = plt.subplots(1, 3, figsize=(4.8, 3))
    def node(ax, x, y, c, s=150): ax.scatter([x], [y], s=s, color=c, zorder=3, edgecolor="white", lw=1.2)
    def arrow(ax, a, b, c): ax.annotate("", b, a, arrowprops=dict(arrowstyle="-|>", color=c, lw=1.5, shrinkA=8, shrinkB=8))
    th = np.linspace(0, 2 * np.pi, 5)[:-1] + np.pi / 2; P = np.c_[np.cos(th), np.sin(th)]
    for i in range(4): arrow(axs[0], P[i], P[(i + 1) % 4], C2)
    for i, p in enumerate(P): node(axs[0], *p, C2 if i == 0 else C1)
    mids = [(0, y) for y in (-1, -0.33, 0.33, 1)]
    for m in mids: arrow(axs[1], (-1, 0), m, C2); arrow(axs[1], m, (1, 0), C2)
    node(axs[1], -1, 0, C2); node(axs[1], 1, 0, C2); [node(axs[1], *m, C1, 100) for m in mids]
    ins = [(-0.9, y) for y in (-1, -0.33, 0.33, 1)]
    for q in ins: arrow(axs[2], q, (0.6, 0), C2)
    [node(axs[2], *q, C1, 100) for q in ins]; node(axs[2], 0.6, 0, C2, 220)
    for ax, t in zip(axs, ["원형", "쪼개기 · 모으기", "한곳에 모으기"]):
        ax.set_title(t, fontsize=10); ax.set_xlim(-1.4, 1.4); ax.set_ylim(-1.4, 1.4); ax.set_aspect("equal"); ax.axis("off")
    fig.subplots_adjust(0.01, 0.02, 0.99, 0.9, 0.05); save_thumb(fig, thumb)
    fmt_rate = df.groupby("Payment Format")["Is Laundering"].mean().sort_values(ascending=False) * 100
    daily = df.groupby(df["Timestamp"].dt.date).agg(n=("Is Laundering", "size"), pos=("Is Laundering", "sum"))
    return dict(stats=stats(df), sample=table(df),
                charts=[chart("결제 형식별 세탁 비율", [bar(fmt_rate.index, fmt_rate.values.round(3))], "", "세탁 비율 (%)", "ACH (자동 이체) 에서 가장 높다"),
                        chart("날짜별 세탁 거래 수", [dict(type="scatter", mode="lines+markers", x=[str(d) for d in daily.index], y=daily["pos"].tolist(), line=dict(color=S2, width=2), name="")],
                              "", "세탁 거래 수", "처음 열흘 남짓에 몰려 있고 뒤로 갈수록 적다: 시간 순서로 나누면 시험 기간의 세탁 비율이 달라진다")])


def stock_index(thumb):
    raw = json.load(open(os.path.join(D, "yahoo_KS11.json")))["chart"]["result"][0]
    q = raw["indicators"]["quote"][0]
    df = pd.DataFrame({"date": pd.to_datetime(raw["timestamp"], unit="s").normalize(), "close": q["close"], "volume": q["volume"]}).dropna().drop_duplicates("date")
    df["ret_pct"] = df["close"].pct_change() * 100
    x = np.arange(len(df)); y = df["close"].values; years = df["date"].dt.year.values
    def frame(k):
        n = max(2, int(len(df) * (k + 1) / 30))
        fig, ax = plt.subplots(figsize=(4.8, 3)); ax.plot(x[:n], y[:n], color=C1, lw=1.4); ax.scatter([x[n - 1]], [y[n - 1]], color=C2, s=25, zorder=3)
        ax.set_xlim(0, len(df)); ax.set_ylim(y.min() * 0.95, y.max() * 1.03); ax.set_xticks([]); ax.set_ylabel("KOSPI")
        ax.text(0.03, 0.9, str(years[n - 1]), transform=ax.transAxes, fontsize=14, fontweight="bold", color=C1); fig.tight_layout(); return fig
    save_gif(frame, 30, thumb.replace(".png", ".gif"), fps=8)
    r = df["ret_pct"].dropna(); xs, ys = hist_bins(r.clip(-6, 6), np.arange(-6, 6.25, 0.25))
    up = df.assign(year=df["date"].dt.year).groupby("year")["ret_pct"].apply(lambda v: (v > 0).mean() * 100)
    return dict(thumb="thumb.gif", stats=stats(df), sample=table(df.assign(date=df["date"].dt.date)),
                charts=[chart("코스피 종가 (10년)", [dict(type="scatter", mode="lines", x=[str(d.date()) for d in df["date"]], y=df["close"].round(1).tolist(), line=dict(color=S1, width=1.5), name="")], "", "지수"),
                        chart("하루 수익률 분포", [bar(xs, ys)], "하루 수익률 (%)", "날 수", "0 근처에 몰려 있고, 가끔 크게 움직인다 (꼬리가 두껍다)"),
                        chart("해마다 '오른 날' 비율", [bar([str(v) for v in up.index], up.values.round(1))], "", "%", "대부분 50% 근처: 내일 오를지 맞히기가 어려운 이유")])


def real_estate(thumb):
    from sklearn.datasets import fetch_california_housing
    df = fetch_california_housing(as_frame=True, data_home=D).frame
    fig, ax = plt.subplots(figsize=(4.8, 3)); ax.scatter(df["Longitude"], df["Latitude"], c=df["MedHouseVal"], s=0.6, cmap="viridis")
    ax.set_aspect("equal"); ax.axis("off"); ax.text(0.98, 0.95, "색 = 집값", ha="right", transform=ax.transAxes, fontsize=9, color="gray")
    fig.subplots_adjust(0, 0, 1, 1); save_thumb(fig, thumb)
    s = df.sample(4000, random_state=0)
    mp = dict(type="scattergl", mode="markers", x=s["Longitude"].tolist(), y=s["Latitude"].tolist(),
              marker=dict(color=s["MedHouseVal"].round(2).tolist(), colorscale="Viridis", size=4, colorbar=dict(title="10만 $")), name="",
              hovertemplate="경도 %{x}<br>위도 %{y}<br>집값 %{marker.color} (10만 $)<extra></extra>")
    xs, ys = hist_bins(df["MedHouseVal"], np.arange(0, 5.2, 0.1))
    inc = df.groupby(pd.cut(df["MedInc"], [0, 2, 3, 4, 5, 6, 8, 16]), observed=False)["MedHouseVal"].median()
    return dict(stats=stats(df), sample=table(df),
                charts=[chart("지도: 동네 4,000곳 (색 = 집값)", [mp], "경도", "위도", "해안 (샌프란시스코 · 로스앤젤레스) 이 비싸다"),
                        chart("집값 분포", [bar(xs, ys)], "집값 중앙값 (10만 달러)", "동네 수", "5.0 에 몰린 막대: 5.0 이상을 모두 5.0 으로 적었다 (잘린 값)"),
                        chart("소득 구간별 집값 (중앙값)", [bar([str(i) for i in inc.index], inc.values.round(2))], "소득 중앙값 (만 달러)", "집값 (10만 달러)", "소득이 가장 강한 단서")])


def earthquakes(thumb):
    df = pd.read_csv(os.path.join(D, "usgs_2023_2025.csv"), parse_dates=["time"])
    df["month"] = df["time"].dt.to_period("M")
    months = sorted(df["month"].unique())
    def frame(k):
        fig, ax = plt.subplots(figsize=(4.8, 3))
        past = df[df["month"] < months[k]]; now = df[df["month"] == months[k]]
        ax.scatter(past["longitude"], past["latitude"], s=1, color="#cccccc")
        ax.scatter(now["longitude"], now["latitude"], s=(now["mag"] - 4) ** 2 * 6, color=C2, alpha=0.8, edgecolor="none")
        ax.set_xlim(-180, 180); ax.set_ylim(-70, 80); ax.set_xticks([]); ax.set_yticks([])
        ax.text(0.02, 0.92, str(months[k]), transform=ax.transAxes, fontsize=13, fontweight="bold", color=C1)
        for sp in ax.spines.values(): sp.set_visible(False)
        fig.subplots_adjust(0, 0, 1, 1); return fig
    save_gif(frame, len(months), thumb.replace(".png", ".gif"), fps=3)
    s = df
    mp = dict(type="scattergl", mode="markers", x=s["longitude"].tolist(), y=s["latitude"].tolist(),
              marker=dict(color=s["mag"].tolist(), colorscale="YlOrRd", size=(np.clip(s["mag"] - 4, 0.5, 4) * 3).round(1).tolist(), colorbar=dict(title="규모")),
              text=s["time"].dt.strftime("%Y-%m-%d").tolist(), hovertemplate="%{text}<br>규모 %{marker.color}<extra></extra>", name="")
    xs, ys = hist_bins(df["mag"], np.arange(4.5, 8.3, 0.1).round(1))
    per = df.groupby("month").size()
    return dict(thumb="thumb.gif", stats=stats(df[["time", "latitude", "longitude", "depth", "mag", "magType", "place"]]),
                sample=table(df[["time", "latitude", "longitude", "depth", "mag", "place"]].assign(time=lambda d: d["time"].dt.strftime("%Y-%m-%d %H:%M"))),
                charts=[chart("지도: 2년 동안의 지진 (색 · 크기 = 규모)", [mp], "경도", "위도", "판의 경계 (불의 고리) 를 따라 늘어선다"),
                        chart("규모 분포", [bar(xs, ys)], "규모", "개수 (로그 눈금)", "규모가 0.1 커질 때마다 개수가 일정한 비율로 준다 (구텐베르크-리히터 법칙)", yaxis=dict(type="log", title="개수 (로그 눈금)")),
                        chart("달마다 지진 수", [bar([str(m) for m in per.index], per.values)], "", "개수")])


# ------------------------------------------------------------------ 2차 추가 16개
def point_cloud(thumb):
    df = pd.read_parquet(os.path.join(D, "modelnet40_test.parquet"))
    names = ["airplane", "bathtub", "bed", "bench", "bookshelf", "bottle", "bowl", "car", "chair", "cone", "cup", "curtain", "desk", "door", "dresser",
             "flower_pot", "glass_box", "guitar", "keyboard", "lamp", "laptop", "mantel", "monitor", "night_stand", "person", "piano", "plant", "radio",
             "range_hood", "sink", "sofa", "stairs", "stool", "table", "tent", "toilet", "tv_stand", "vase", "wardrobe", "xbox"]
    fig = plt.figure(figsize=(4.8, 3))
    for k, i in enumerate([0, 700, 1500, 2000, 1100, 300]):
        ax = fig.add_subplot(2, 3, k + 1, projection="3d"); p = np.stack(df["inputs"][i])
        ax.scatter(p[:, 0], p[:, 2], p[:, 1], s=0.3, c=p[:, 1], cmap="viridis"); ax.set_axis_off(); ax.set_title(names[df["label"][i]], fontsize=8)
    fig.subplots_adjust(0, 0, 1, 0.95, 0, 0.1); save_thumb(fig, thumb)
    p = np.stack(df["inputs"][0]); s = p[np.random.default_rng(0).choice(len(p), 800, replace=False)]
    sc = dict(type="scatter3d", mode="markers", x=s[:, 0].round(3).tolist(), y=s[:, 2].round(3).tolist(), z=s[:, 1].round(3).tolist(), marker=dict(size=2, color=S1), name="")
    cnt = pd.Series([names[i] for i in df["label"]]).value_counts()
    meta = pd.DataFrame({"label": [names[i] for i in df["label"]], "n_points": 2048,
                         "x_range": [float(np.ptp(np.stack(v)[:, 0])) for v in df["inputs"][:300]] + [np.nan] * (len(df) - 300)})
    return dict(stats=stats(meta.drop(columns="x_range")), sample=dict(columns=["점 번호", "x", "y", "z"], rows=[[str(i), fmt(round(float(a), 4)), fmt(round(float(b), 4)), fmt(round(float(c), 4))] for i, (a, b, c) in enumerate(p[:6])]),
                charts=[chart(f"점 구름 하나 ({names[df['label'][0]]}, 점 800개만) — 마우스로 돌려 보기", [sc], "", "", "", height=360, margin=dict(l=0, r=0, t=0, b=0)),
                        chart("종류별 물체 수 (시험 묶음)", [bar(cnt.index, cnt.values)], "", "개수", "100개짜리와 20개짜리가 섞여 있다", xaxis=dict(tickangle=-60, title=""))])


def medical(thumb):
    z = np.load(os.path.join(D, "pneumoniamnist.npz")); X, y = z["train_images"], z["train_labels"].ravel()
    fig, axs = plt.subplots(2, 4, figsize=(4.8, 3))
    for r, lab in enumerate([0, 1]):
        for c, i in enumerate(np.where(y == lab)[0][:4]): axs[r, c].imshow(X[i], cmap="gray"); axs[r, c].axis("off")
        axs[r, 0].set_title(["정상", "폐렴"][lab], fontsize=9, loc="left")
    fig.subplots_adjust(0, 0, 1, 0.92, 0.03, 0.15); save_thumb(fig, thumb)
    meta = pd.DataFrame({"label": np.where(y == 1, "pneumonia", "normal"), "mean_pixel": X.reshape(len(X), -1).mean(1)})
    x0, y0 = hist_bins(meta.loc[y == 0, "mean_pixel"], np.arange(40, 200, 5), True); x1, y1 = hist_bins(meta.loc[y == 1, "mean_pixel"], np.arange(40, 200, 5), True)
    return dict(stats=stats(meta), sample=table(meta),
                charts=[chart("정상 vs 폐렴 수 (학습)", [bar(["정상", "폐렴"], [int((y == 0).sum()), int((y == 1).sum())])], "", "장 수", "폐렴이 3배쯤 많다 (병원 데이터는 아픈 사람이 많다)"),
                        chart("사진 평균 밝기", [bar(x0, y0, S1, name="정상"), bar(x1, y1, S2, name="폐렴")], "픽셀 평균 (0~255)", "비율", "폐렴 사진이 조금 더 하얗다 (폐에 하얀 그림자)", barmode="overlay")])


def satellite(thumb):
    df = pd.read_parquet(os.path.join(D, "eurosat_test.parquet"))
    names = ["AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial", "Pasture", "PermanentCrop", "Residential", "River", "SeaLake"]
    fig, axs = plt.subplots(2, 5, figsize=(4.8, 3))
    for c in range(10):
        i = np.where(df["label"].values == c)[0][0]; ax = axs.ravel()[c]
        ax.imshow(Image.open(io.BytesIO(df["image"][i]["bytes"]))); ax.axis("off"); ax.set_title(names[c][:10], fontsize=6.5)
    fig.subplots_adjust(0, 0, 1, 0.93, 0.03, 0.2); save_thumb(fig, thumb)
    imgs = [np.asarray(Image.open(io.BytesIO(b["bytes"])).convert("RGB"), dtype=float) for b in df["image"][:2000]]
    meta = pd.DataFrame({"label": [names[i] for i in df["label"][:2000]], "R": [im[..., 0].mean() for im in imgs], "G": [im[..., 1].mean() for im in imgs], "B": [im[..., 2].mean() for im in imgs]})
    g = meta.groupby("label")[["R", "G", "B"]].mean()
    return dict(stats=stats(meta), sample=table(meta.round(1)),
                charts=[chart("종류별 평균 색 (R · G · B)", [bar(g.index, g["R"].round(1), "#d62728", name="R"), bar(g.index, g["G"].round(1), "#2ca02c", name="G"), bar(g.index, g["B"].round(1), "#1f77b4", name="B")],
                              "", "평균 값 (0~255)", "숲은 어둡고 초록, 바다는 파랑, 고속도로 · 주거지는 회색", barmode="group", xaxis=dict(tickangle=-40, title=""))])


def crop(thumb):
    df = pd.read_parquet(os.path.join(D, "plantvillage_tiny.parquet"))
    s = df.sample(8, random_state=2)
    fig, axs = plt.subplots(2, 4, figsize=(4.8, 3))
    for ax, (_, r) in zip(axs.ravel(), s.iterrows()):
        ax.imshow(Image.open(io.BytesIO(r["image"]["bytes"]))); ax.axis("off"); ax.set_title(r["disease"][:16], fontsize=6.5)
    fig.subplots_adjust(0, 0, 1, 0.94, 0.03, 0.2); save_thumb(fig, thumb)
    meta = df.drop(columns="image").assign(healthy=lambda d: d["disease"].str.contains("healthy"))
    h = meta.groupby("host")["healthy"].agg(["sum", "size"])
    return dict(stats=stats(meta.drop(columns=["leaf_id"])), sample=table(meta[["host", "disease", "class_label", "split", "leaf_grouped"]]),
                charts=[chart("작물별 사진 수 (건강 / 병)", [bar(h.index, h["sum"], S3, name="건강"), bar(h.index, h["size"] - h["sum"], S2, name="병")], "", "장 수", "", barmode="stack", xaxis=dict(tickangle=-40, title=""))])


def speech(thumb):
    import tarfile
    from scipy.io import wavfile
    tar = tarfile.open(os.path.join(D, "speech_commands_test.tar.gz"))
    mems = [m for m in tar.getmembers() if m.name.endswith(".wav")]
    labels = [m.name.split("/")[-2] for m in mems]; spk = [m.name.split("/")[-1].split("_")[0] for m in mems]
    def wave(word):
        m = mems[labels.index(word)]; r, w = wavfile.read(io.BytesIO(tar.extractfile(m).read())); return w.astype(float) / 32768
    fig, axs = plt.subplots(2, 2, figsize=(4.8, 3))
    for ax, wd in zip(axs.ravel(), ["yes", "no", "up", "stop"]):
        from scipy.signal import spectrogram
        f, t, S = spectrogram(wave(wd), fs=16000, nperseg=400, noverlap=240)
        ax.pcolormesh(t, f, 10 * np.log10(S + 1e-10), cmap="magma", shading="auto"); ax.set_ylim(0, 6000); ax.axis("off"); ax.set_title(f'"{wd}"', fontsize=9)
    fig.subplots_adjust(0, 0, 1, 0.92, 0.03, 0.22); save_thumb(fig, thumb)
    meta = pd.DataFrame({"label": labels, "speaker": spk})
    cnt = meta["label"].value_counts(); per = meta.groupby("speaker").size()
    w = wave("yes"); t = (np.arange(0, 16000, 8) / 16000).round(4).tolist()
    return dict(stats=stats(meta), sample=table(meta),
                charts=[chart('"yes" 의 파형 (1초)', [dict(type="scatter", mode="lines", x=t, y=w[::8].round(4).tolist(), line=dict(color=S1, width=1), name="")], "초", "진폭"),
                        chart("단어별 녹음 수", [bar(cnt.index, cnt.values)], "", "개수", f"녹음한 사람 {len(per)}명: 한 사람이 여러 단어를 녹음했다 → 사람 단위로 나눠야 한다")])


def air_quality(thumb):
    with zipfile.ZipFile(os.path.join(D, "beijing_pm25.zip")) as z:
        df = pd.read_csv(z.open(z.namelist()[0]))
    df["time"] = pd.to_datetime(df[["year", "month", "day", "hour"]]); daily = df.set_index("time")["pm2.5"].resample("D").mean()
    def frame(k):
        n = max(2, int(len(daily) * (k + 1) / 30))
        fig, ax = plt.subplots(figsize=(4.8, 3)); ax.plot(daily.index[:n], daily.values[:n], color=C2, lw=0.6)
        ax.axhline(76, color="gray", ls=":", lw=1); ax.set_xlim(daily.index[0], daily.index[-1]); ax.set_ylim(0, 600); ax.set_ylabel("PM2.5 (일 평균)")
        ax.text(0.02, 0.9, str(daily.index[n - 1].date())[:7], transform=ax.transAxes, fontsize=12, fontweight="bold", color=C1); fig.tight_layout(); return fig
    save_gif(frame, 30, thumb.replace(".png", ".gif"), fps=6)
    bym = df.groupby("month")["pm2.5"].mean(); byh = df.groupby("hour")["pm2.5"].mean()
    return dict(thumb="thumb.gif", stats=stats(df.drop(columns=["No", "time"])), sample=table(df.drop(columns=["No", "time"])),
                charts=[chart("달별 평균 PM2.5", [bar([str(m) for m in bym.index], bym.values.round(1))], "월", "㎍/㎥", "겨울 (난방철) 에 높다"),
                        chart("시각별 평균 PM2.5", [line(byh.index, byh.values.round(1))], "시", "㎍/㎥", "밤에 높고 오후에 낮다")])


def electricity(thumb):
    with zipfile.ZipFile(os.path.join(D, "household_power.zip")) as z:
        raw = pd.read_csv(z.open("household_power_consumption.txt"), sep=";", na_values="?", low_memory=False)
    raw["time"] = pd.to_datetime(raw["Date"] + " " + raw["Time"], format="%d/%m/%Y %H:%M:%S")
    h = raw.set_index("time")["Global_active_power"].resample("h").mean()
    wk = h["2008-01-07":"2008-01-13"]
    fig, ax = plt.subplots(figsize=(4.8, 3)); ax.plot(wk.index, wk.values, color=C1, lw=1); ax.set_ylabel("kW (시간 평균)"); ax.tick_params(axis="x", labelsize=7, rotation=30); fig.tight_layout(); save_thumb(fig, thumb)
    byh = h.groupby(h.index.hour).mean(); weekly = h.resample("W").mean()
    return dict(stats=stats(raw.drop(columns=["Date", "Time"]).sample(200000, random_state=0)), sample=table(raw.drop(columns=["time"])),
                charts=[chart("주별 평균 사용량", [dict(type="scatter", mode="lines", x=[str(t.date()) for t in weekly.index], y=weekly.round(3).tolist(), line=dict(color=S1), name="")], "", "kW", "여름에 적고 겨울에 많다 (8월엔 휴가로 거의 비어 있다)"),
                        chart("시각별 평균", [line(byh.index, byh.values.round(3))], "시", "kW", "아침 · 저녁에 봉우리")])


def football(thumb):
    import json as js
    ms = sorted(js.load(open(os.path.join(D, "statsbomb_matches_43_106.json"))), key=lambda m: m["match_date"])
    rows = []
    for k, m in enumerate(ms):
        for e in js.load(open(os.path.join(D, f"sb_events_{m['match_id']}.json"))):
            if e["type"]["name"] == "Shot":
                s = e["shot"]; rows.append(dict(match=k, x=e["location"][0], y=e["location"][1], body=s["body_part"]["name"], kind=s["type"]["name"],
                                                goal=int(s["outcome"]["name"] == "Goal"), sb_xg=round(s["statsbomb_xg"], 3), team=e["team"]["name"]))
    sh = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(4.8, 3))
    ax.plot([60, 120, 120, 60, 60], [0, 0, 80, 80, 0], color="gray", lw=1); ax.plot([102, 120, 120, 102], [18, 18, 62, 62], color="gray", lw=1)
    ax.scatter(sh.loc[sh.goal == 0, "x"], sh.loc[sh.goal == 0, "y"], s=3, color="#bbbbbb"); ax.scatter(sh.loc[sh.goal == 1, "x"], sh.loc[sh.goal == 1, "y"], s=10, color=C2)
    ax.set_xlim(58, 122); ax.set_aspect("equal"); ax.axis("off"); fig.subplots_adjust(0, 0, 1, 1); save_thumb(fig, thumb)
    sc = lambda d, c, n: dict(type="scatter", mode="markers", x=d["x"].tolist(), y=d["y"].tolist(), marker=dict(color=c, size=5, opacity=0.7), name=n, text=d["sb_xg"].astype(str).tolist(),
                              hovertemplate="xG %{text}<extra></extra>")
    k = sh.groupby("kind")["goal"].agg(["mean", "size"])
    return dict(stats=stats(sh), sample=table(sh),
                charts=[chart("슛 위치 (주황 = 골)", [sc(sh[sh.goal == 0], "var:muted", "골 아님"), sc(sh[sh.goal == 1], S2, "골")], "x (골대 = 120)", "y", "", xaxis=dict(range=[58, 122], title="x (골대 = 120)"), yaxis=dict(range=[0, 80], title="y", scaleanchor="x")),
                        chart("슛 종류별 골 비율", [bar(k.index, (k["mean"] * 100).round(1))], "", "%", "페널티킥은 약 70%, 일반 슛은 10% 남짓")])


def chess(thumb):
    df = pd.read_parquet(os.path.join(D, "lichess_2013_01.parquet"), columns=["Result", "WhiteElo", "BlackElo", "TimeControl", "Termination", "Opening", "ECO"])
    df["WhiteElo"] = pd.to_numeric(df["WhiteElo"], errors="coerce"); df["BlackElo"] = pd.to_numeric(df["BlackElo"], errors="coerce")
    df = df[df["Result"].isin(["1-0", "0-1", "1/2-1/2"])].dropna(subset=["WhiteElo", "BlackElo"]); df["diff"] = df["WhiteElo"] - df["BlackElo"]
    b = pd.cut(df["diff"], np.arange(-600, 601, 100)); win = df.groupby(b, observed=True)["Result"].apply(lambda r: (r == "1-0").mean() * 100)
    fig, ax = plt.subplots(figsize=(4.8, 3)); ax.plot(np.arange(-550, 551, 100)[:len(win)], win.values, "o-", color=C1); ax.axhline(50, color="gray", ls=":")
    ax.set_xlabel("레이팅 차 (백 − 흑)"); ax.set_ylabel("백 승률 (%)"); fig.tight_layout(); save_thumb(fig, thumb)
    op = df["Opening"].str.split(":").str[0].value_counts().head(10)
    return dict(stats=stats(df), sample=table(df),
                charts=[chart("레이팅 차에 따른 백 승률", [line([str(i) for i in win.index], win.values.round(1))], "레이팅 차 (백 − 흑)", "%", "차가 클수록 높은 쪽이 이긴다 (Elo 의 설계대로)"),
                        chart("가장 많이 둔 오프닝 10개", [bar(op.index[::-1], op.values[::-1], horizontal=True)], "판 수", "", "", margin=dict(l=200))])


def cyber_security(thumb):
    cols = ["duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes"] + [f"f{i}" for i in range(6, 41)] + ["attack", "difficulty"]
    tr = pd.read_csv(os.path.join(D, "nslkdd_train.txt"), header=None, names=cols); te = pd.read_csv(os.path.join(D, "nslkdd_test.txt"), header=None, names=cols)
    a = tr["attack"].value_counts().head(8)
    fig, ax = plt.subplots(figsize=(4.8, 3)); ax.barh(a.index[::-1], a.values[::-1], color=[C1 if x == "normal" else C2 for x in a.index[::-1]]); ax.set_xscale("log"); ax.set_xlabel("연결 수 (로그 눈금)"); fig.tight_layout(); save_thumb(fig, thumb)
    new = sorted(set(te["attack"]) - set(tr["attack"]))
    nt = te[te["attack"].isin(new)]["attack"].value_counts().head(10)
    return dict(stats=stats(tr[["duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "attack"]]), sample=table(tr[["duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "attack"]]),
                charts=[chart("학습 데이터의 연결 종류 (상위 8)", [bar(a.index, a.values)], "", "연결 수 (로그 눈금)", "neptune (SYN flood, DoS) 이 정상 다음으로 많다", yaxis=dict(type="log", title="연결 수 (로그 눈금)")),
                        chart(f"시험에만 있는 공격 {len(new)}종 (상위 10)", [bar(nt.index, nt.values, S2)], "", "연결 수", "학습 때 본 적 없는 공격: 진짜 실력을 재는 부분")])


def telco_churn(thumb):
    df = pd.read_csv(os.path.join(D, "telco_churn.csv")); df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    c = df.groupby("Contract")["Churn"].apply(lambda v: (v == "Yes").mean() * 100)
    fig, ax = plt.subplots(figsize=(4.8, 3)); ax.bar(["월 단위", "1년", "2년"], c.reindex(["Month-to-month", "One year", "Two year"]).values, color=[C2, C1, C1]); ax.set_ylabel("해지율 (%)"); fig.tight_layout(); save_thumb(fig, thumb)
    bins = np.arange(0, 75, 3); x0, y0 = hist_bins(df.loc[df.Churn == "No", "tenure"], bins, True); x1, y1 = hist_bins(df.loc[df.Churn == "Yes", "tenure"], bins, True)
    return dict(stats=stats(df.drop(columns="customerID")), sample=table(df.drop(columns="customerID").iloc[:, :12]),
                charts=[chart("계약 종류별 해지율", [bar(c.index, c.values.round(1))], "", "%", "월 단위 계약의 해지율이 압도적으로 높다"),
                        chart("가입 기간: 남은 사람 vs 떠난 사람", [bar(x0, y0, S1, name="남음"), bar(x1, y1, S2, name="떠남")], "가입 기간 (개월)", "비율", "가입 초반에 많이 떠난다", barmode="overlay")])


def credit_default(thumb):
    df = pd.read_csv(os.path.join(D, "uci_credit_card.csv")).rename(columns={"default.payment.next.month": "default", "PAY_0": "PAY_1"})
    g = df.groupby("PAY_1")["default"].agg(["mean", "size"]).query("size > 100")
    fig, ax = plt.subplots(figsize=(4.8, 3)); ax.bar([str(i) for i in g.index], g["mean"] * 100, color=C1); ax.set_xlabel("지난달 상환 상태 (−1 = 제때, 1 = 한 달 밀림 …)"); ax.set_ylabel("다음 달 연체율 (%)"); fig.tight_layout(); save_thumb(fig, thumb)
    a = df.groupby(pd.cut(df["AGE"], [20, 25, 30, 35, 40, 45, 50, 60, 80]), observed=True)["default"].mean() * 100
    return dict(stats=stats(df.drop(columns="ID")), sample=table(df.drop(columns="ID").iloc[:, :12]),
                charts=[chart("지난달 상환 상태별 다음 달 연체율", [bar([str(i) for i in g.index], (g["mean"] * 100).round(1))], "PAY_1", "%", "두 달 이상 밀린 사람은 70% 가까이 다시 연체한다"),
                        chart("나이대별 연체율", [bar([str(i) for i in a.index], a.values.round(1))], "나이", "%", "나이 차이는 상환 기록에 비하면 작다")])


def ad_ctr(thumb):
    df = pd.read_parquet(os.path.join(D, "avazu_x4_0.parquet"), columns=["click", "hour", "banner_pos", "site_category", "app_category", "device_type", "device_conn_type", "C1", "C15", "C16"]).sample(300000, random_state=0)
    df["hod"] = df["hour"] % 100
    h = df.groupby("hod")["click"].mean() * 100
    fig, ax = plt.subplots(figsize=(4.8, 3)); ax.plot(h.index, h.values, "o-", color=C2); ax.set_xlabel("시각"); ax.set_ylabel("클릭률 (%)"); fig.tight_layout(); save_thumb(fig, thumb)
    sz = df.assign(size=df["C15"].astype(str) + "×" + df["C16"].astype(str)).groupby("size")["click"].agg(["mean", "size"]).query("size > 1000")
    return dict(stats=stats(df.drop(columns=["hod"])), sample=table(df.drop(columns=["hod"])),
                charts=[chart("시각별 클릭률", [line(h.index, h.values.round(2))], "시", "%"),
                        chart("광고 크기 (C15 × C16) 별 클릭률", [bar(sz.index, (sz["mean"] * 100).round(2))], "", "%", "같은 광고라도 크기 · 위치에 따라 클릭률이 다르다")])


def hiring_match(thumb):
    tr = pd.read_parquet(os.path.join(D, "resume_fit_test.parquet"))
    fig, ax = plt.subplots(figsize=(4.8, 3)); ax.axis("off")
    r = tr.iloc[3]
    ax.text(0.02, 0.95, "이력서", fontsize=10, fontweight="bold", color=C1, transform=ax.transAxes, va="top")
    ax.text(0.02, 0.86, r["resume_text"][:170].replace("\n", " "), fontsize=7, wrap=True, transform=ax.transAxes, va="top")
    ax.text(0.02, 0.5, "채용 공고", fontsize=10, fontweight="bold", color=C2, transform=ax.transAxes, va="top")
    ax.text(0.02, 0.41, r["job_description_text"][:170].replace("\n", " "), fontsize=7, wrap=True, transform=ax.transAxes, va="top")
    ax.text(0.98, 0.04, "→ " + r["label"], ha="right", fontsize=11, fontweight="bold", transform=ax.transAxes)
    save_thumb(fig, thumb)
    meta = pd.DataFrame({"label": tr["label"], "resume_words": tr["resume_text"].str.split().str.len(), "job_words": tr["job_description_text"].str.split().str.len()})
    cnt = tr["label"].value_counts()
    return dict(stats=stats(meta), sample=table(tr.assign(resume_text=tr["resume_text"].str[:80], job_description_text=tr["job_description_text"].str[:80])),
                charts=[chart("단계별 쌍의 수 (시험)", [bar(cnt.index, cnt.values)], "", "쌍"),
                        chart("이력서 길이 (단어 수)", [bar(*hist_bins(meta["resume_words"].clip(0, 1500), np.arange(0, 1550, 50)))], "단어 수", "이력서 수")])


def traffic_speed(thumb):
    df = pd.read_parquet(os.path.join(D, "metrla_test.parquet"), columns=["node_id", "t0_timestamp", "x_t+0_d0"])
    loc = pd.read_csv(os.path.join(D, "metrla_locations.csv")); df["time"] = pd.to_datetime(df["t0_timestamp"])
    piv = df.pivot(index="time", columns="node_id", values="x_t+0_d0").replace(0, np.nan)
    day = piv.loc["2012-06-05"].iloc[::3]
    lat, lon = loc.iloc[:, -2].values, loc.iloc[:, -1].values
    def frame(k):
        fig, ax = plt.subplots(figsize=(4.8, 3)); row = day.iloc[k].reindex(range(len(loc))).values
        ax.scatter(lon, lat, c=row, cmap="RdYlGn", vmin=20, vmax=70, s=14); ax.set_xticks([]); ax.set_yticks([])
        ax.text(0.02, 0.92, day.index[k].strftime("%H:%M"), transform=ax.transAxes, fontsize=13, fontweight="bold", color=C1)
        for sp in ax.spines.values(): sp.set_visible(False)
        fig.subplots_adjust(0, 0, 1, 1); return fig
    save_gif(frame, len(day), thumb.replace(".png", ".gif"), fps=8)
    one = piv[5].loc["2012-06-04":"2012-06-10"]
    avg = piv.mean(1); byt = avg.groupby(avg.index.strftime("%H:%M")).mean().iloc[::6]
    return dict(thumb="thumb.gif", stats=stats(df[["node_id", "time", "x_t+0_d0"]].rename(columns={"x_t+0_d0": "speed_mph"})),
                sample=table(df[["node_id", "t0_timestamp", "x_t+0_d0"]].rename(columns={"x_t+0_d0": "speed_mph"})),
                charts=[chart("센서 5번, 일주일 속도 (mph)", [dict(type="scatter", mode="lines", x=[str(t) for t in one.index], y=one.round(1).tolist(), line=dict(color=S1, width=1), name="")], "", "mph", "출퇴근 시간에 푹 꺼진다 (빈 곳은 센서 고장)"),
                        chart("하루 중 시각별 전체 평균 속도", [line(byt.index, byt.values.round(1))], "시각", "mph")])


def corporate_card(thumb):
    df = pd.read_csv(os.path.join(D, "ok_pcard_2014.csv"), low_memory=False).drop(columns=["Cardholder Last Name", "Cardholder First Initial"])
    df = df.rename(columns={"Agency Name": "agency", "Description": "memo", "Amount": "amount", "Vendor": "vendor", "Transaction Date": "date", "Merchant Category Code (MCC)": "mcc"})
    df["date"] = pd.to_datetime(df["date"], format="%m/%d/%Y %I:%M:%S %p")
    top = df.groupby("mcc")["amount"].sum().sort_values(ascending=False).head(8) / 1e6
    fig, ax = plt.subplots(figsize=(4.8, 3)); ax.barh([m[:26].title() for m in top.index[::-1]], top.values[::-1], color=C1); ax.set_xlabel("결제 금액 합계 (백만 달러)"); ax.tick_params(labelsize=7)
    fig.tight_layout(); save_thumb(fig, thumb)
    mon = df.groupby(df["date"].dt.to_period("M"))["amount"].sum() / 1e6
    xs, ys = hist_bins(np.log10(df.loc[df["amount"] > 0, "amount"]), np.arange(-1, 6.1, 0.2))
    return dict(stats=stats(df.drop(columns=["Year-Month", "Agency Number", "Posted Date"])), sample=table(df[["agency", "memo", "amount", "vendor", "date", "mcc"]].assign(date=lambda d: d["date"].dt.date)),
                charts=[chart("달마다 법인카드 지출 (백만 달러)", [bar([str(m) for m in mon.index], mon.values.round(2))], "", "백만 달러"),
                        chart("결제 금액 분포", [bar(xs, ys)], "log10(금액, 달러)", "결제 수", "대부분 10~1,000달러, 가끔 아주 큰 결제")])
