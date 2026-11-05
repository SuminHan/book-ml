"""푸아송 혼합 예시 데이터 → data/taxi_gramercy_3am.csv  (make_figs.py 가 읽는다)
출처: NYC Taxi & Limousine Commission, Yellow Taxi Trip Records 2024년 1월 (공개 자료)
  https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet
  여기서는 데이터숍이 받아 둔 ~/book-ml/notebooks/datashop/data/yellow_2024_01.parquet 를 읽는다.
남기는 것: Gramercy 구역 (LocationID 107) 에서 새벽 3:00~3:59 에 10분마다 태운 손님 수, 1월 2일 ~ 31일 (설날 다음 날 1월 1일은 뺌) = 180칸.
weekend = 토 · 일 새벽 (금 · 토 밤에 이어지는 새벽). EM 에는 count 만 넣고, weekend 는 결과 확인에만 쓴다."""
import os, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
d = pd.read_parquet(os.path.expanduser("~/book-ml/notebooks/datashop/data/yellow_2024_01.parquet"), columns=["tpep_pickup_datetime", "PULocationID"])
d = d[(d.tpep_pickup_datetime >= "2024-01-02") & (d.tpep_pickup_datetime < "2024-02-01") & (d.PULocationID == 107)]
s = d.set_index("tpep_pickup_datetime").resample("10min").size()
s = s[s.index.hour == 3]
out = pd.DataFrame({"start": s.index.strftime("%Y-%m-%d %H:%M"), "count": s.values, "weekend": s.index.dayofweek.isin([5, 6]).astype(int)})
out.to_csv(os.path.join(HERE, "data", "taxi_gramercy_3am.csv"), index=False)
print(len(out), "windows, weekend", out.weekend.sum(), "mean", round(out["count"].mean(), 2))
