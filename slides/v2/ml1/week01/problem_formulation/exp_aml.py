"""IBM AML (Altman et al., 2023, 합성 데이터) HI-Small 거래 → assets/aml.json. 실습 노트북 파트 3 과 같은 계산.
Kaggle 로그인 필요: ~/.kaggle/kaggle.json (또는 KAGGLE_USERNAME · KAGGLE_KEY). lab/data/HI-Small_Trans.csv 로 내려받는다 (약 476MB).
  ① 한 행 = 거래: 세탁 비율, '전부 정상' 의 정확도, 규칙 하나 (세탁 비율이 가장 높은 결제 형식이면 '세탁') 의 정확도 · 정밀도 · 재현율
  ② 한 행 = 계좌 (보낸 쪽): 세탁 거래를 한 번이라도 보낸 계좌의 비율"""
import base64, io, json, os, urllib.request, zipfile
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); D = os.path.join(HERE, "lab", "data")
CSV = os.path.join(D, "HI-Small_Trans.csv")


def kaggle_download(dataset, file_name, out_path):
    """Download one file of a Kaggle dataset with the API key (Basic auth), unzip if needed"""
    path = os.path.expanduser("~/.kaggle/kaggle.json")
    cred = json.load(open(path)) if os.path.exists(path) else {"username": os.environ["KAGGLE_USERNAME"], "key": os.environ["KAGGLE_KEY"]}
    url = f"https://www.kaggle.com/api/v1/datasets/download/{dataset}/{file_name}"
    token = base64.b64encode(f"{cred['username']}:{cred['key']}".encode()).decode()
    req = urllib.request.Request(url, headers={"Authorization": f"Basic {token}", "User-Agent": "ML course lab"})
    data = urllib.request.urlopen(req, timeout=600).read()
    if data[:2] == b"PK":
        data = zipfile.ZipFile(io.BytesIO(data)).read(file_name)
    open(out_path, "wb").write(data)


ZIP = os.path.join(D, "HI-Small_Trans.csv.zip")      # Hugging Face 사본 (로그인 없이): eexzzm/IBM-Transactions-for-Anti-Money-Laundering-HI-Small-Trans
if os.path.exists(ZIP):
    with zipfile.ZipFile(ZIP) as z:
        df = pd.read_csv(z.open("HI-Small_Trans.csv"))
else:
    if not os.path.exists(CSV):
        kaggle_download("ealtman2019/ibm-transactions-for-anti-money-laundering-aml", "HI-Small_Trans.csv", CSV)
    df = pd.read_csv(CSV)
print(df.columns.tolist()); print(df.head(3))
y = df["Is Laundering"].values
n, pos = len(df), int(y.sum())
by_format = df.groupby("Payment Format")["Is Laundering"].agg(["mean", "sum", "size"]).sort_values("mean", ascending=False)
print(by_format)
fmt = by_format.index[0]
pred = (df["Payment Format"] == fmt).values.astype(int)
tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
acc_rule = float((pred == y).mean())
sender = df["From Bank"].astype(str) + "_" + df["Account"].astype(str)
acct = pd.Series(y).groupby(sender.values).max()
ts = pd.to_datetime(df["Timestamp"])
J = dict(n=n, pos=pos, rate=pos / n, acc_zero=1 - pos / n, rule_format=fmt, acc_rule=acc_rule,
         rule_precision=tp / (tp + fp), rule_recall=tp / (tp + fn), tp=tp, fp=fp, fn=fn,
         by_format={k: dict(rate=float(r["mean"]), pos=int(r["sum"]), n=int(r["size"])) for k, r in by_format.iterrows()},
         n_acct=int(len(acct)), pos_acct=int(acct.sum()), rate_acct=float(acct.mean()),
         t_min=str(ts.min()), t_max=str(ts.max()), currencies=int(df["Payment Currency"].nunique()),
         amount_min=float(df["Amount Paid"].min()), amount_max=float(df["Amount Paid"].max()))
json.dump(J, open(os.path.join(HERE, "assets", "aml.json"), "w"), indent=1)
print({k: v for k, v in J.items() if k != "by_format"})
