"""항공편 데이터 준비 → data/flights_2024_01.csv.gz  (make_figs.py 가 읽는다)
출처: 미국 교통통계국 (BTS), On-Time Reporting Carrier On-Time Performance, 2024년 1월 (미국 정부 공개 자료).
  https://transtats.bts.gov/PREZIP/On_Time_Reporting_Carrier_On_Time_Performance_1987_present_2024_1.zip
남기는 것: 결항 · 회항이 아닌 편 중 출발 지연 -30 ~ 90분 (497,164편). 열 = dep_delay (분), taxi_out (분), late (도착 15분 이상 지연 1/0).
실행: ~/Documents/manim_shorts/_env/v/bin/python prep_flights.py  (zip 약 28MB 를 내려받는다)"""
import os, io, zipfile, urllib.request, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
URL = "https://transtats.bts.gov/PREZIP/On_Time_Reporting_Carrier_On_Time_Performance_1987_present_2024_1.zip"
z = zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(URL).read()))
name = [n for n in z.namelist() if n.endswith(".csv")][0]
d = pd.read_csv(z.open(name), usecols=["DepDelay", "TaxiOut", "ArrDel15", "Cancelled", "Diverted"], low_memory=False)
d = d[(d.Cancelled == 0) & (d.Diverted == 0)].dropna(subset=["DepDelay", "TaxiOut", "ArrDel15"])
d = d[(d.DepDelay >= -30) & (d.DepDelay <= 90)]
out = pd.DataFrame({"dep_delay": d.DepDelay.astype(int), "taxi_out": d.TaxiOut.astype(int), "late": d.ArrDel15.astype(int)})
out.to_csv(os.path.join(HERE, "data", "flights_2024_01.csv.gz"), index=False)
print(len(out), "flights,", round(out.late.mean(), 3), "late")
