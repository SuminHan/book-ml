"""체스 대국 데이터 준비 → data/chess_2013_01.csv.gz  (make_figs.py 가 읽는다)
출처: lichess.org 공개 데이터베이스 (database.lichess.org, CC0), 2013년 1월 레이팅 대국 121,332판.
  원본 PGN: https://database.lichess.org/standard/lichess_db_standard_rated_2013-01.pgn.zst
  여기서는 데이터숍이 이미 받아 둔 표 (~/book-ml/notebooks/datashop/data/lichess_2013_01.parquet) 를 읽는다.
남기는 것: 결과가 승 · 패인 판 (무승부 3,982판 제외), 레이팅 차이 ±800 안. 열 = white (백 레이팅), black (흑 레이팅), white_win (백 승 1 / 흑 승 0)."""
import os, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
d = pd.read_parquet(os.path.expanduser("~/book-ml/notebooks/datashop/data/lichess_2013_01.parquet"), columns=["Result", "WhiteElo", "BlackElo"])
d = d.dropna(subset=["WhiteElo", "BlackElo"]); d = d[d.Result.isin(["1-0", "0-1"])]
d = d[(d.WhiteElo - d.BlackElo).abs() <= 800]
out = pd.DataFrame({"white": d.WhiteElo.astype(int), "black": d.BlackElo.astype(int), "white_win": (d.Result == "1-0").astype(int)})
os.makedirs(os.path.join(HERE, "data"), exist_ok=True)
out.to_csv(os.path.join(HERE, "data", "chess_2013_01.csv.gz"), index=False)
print(len(out), "games, white wins", round(out.white_win.mean(), 3))
