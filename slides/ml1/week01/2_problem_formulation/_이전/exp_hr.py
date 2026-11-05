"""1쪽 질문용 숫자: 2024 MLB 정규시즌 타석(PA) · 홈런(HR). MLB 공식 Stats API (statsapi.mlb.com).
실행: python3 exp_hr.py  → assets/hr.json"""
import json, os, urllib.request

os.chdir(os.path.dirname(os.path.abspath(__file__)))
API = "https://statsapi.mlb.com/api/v1"
SEASON = 2024
OHTANI = 660271  # MLBAM player id


def get(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "book-ml-course"}), timeout=30) as r:
        return json.load(r)


p = get(f"{API}/people/{OHTANI}/stats?stats=season&season={SEASON}&group=hitting&gameType=R")
s = p["stats"][0]["splits"][0]["stat"]
o_pa, o_hr = s["plateAppearances"], s["homeRuns"]

t = get(f"{API}/teams/stats?season={SEASON}&group=hitting&stats=season&sportIds=1&gameType=R")
teams = t["stats"][0]["splits"]
l_pa = sum(x["stat"]["plateAppearances"] for x in teams)
l_hr = sum(x["stat"]["homeRuns"] for x in teams)

out = {"season": SEASON, "ohtani_pa": o_pa, "ohtani_hr": o_hr, "ohtani_acc_zero": (o_pa - o_hr) / o_pa,
       "mlb_teams": len(teams), "mlb_pa": l_pa, "mlb_hr": l_hr, "mlb_acc_zero": (l_pa - l_hr) / l_pa}
os.makedirs("assets", exist_ok=True)
json.dump(out, open("assets/hr.json", "w"), indent=1)
print(json.dumps(out, indent=1))
