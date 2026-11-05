"""더티 데이터 차시 공유 숫자 (슬라이드 · 그림 · 쇼츠 lessons/ml1/dirty_data 와 같은 예시: 우리 반 10명 키, cm)"""
# 키를 안 적은 두 칸(수아·준호)에 0이 들어간 표
CLASS = [("지우", 158), ("서연", 162), ("수아", 0), ("하준", 165), ("도윤", 170),
         ("예린", 155), ("시우", 168), ("준호", 0), ("민수", 174), ("유나", 160)]
REAL = [(n, v) for n, v in CLASS if v > 0]
DUP = "민수"                       # 중복: 민수 기록이 세 번
TYPO, TYPO_V = "준호", 250         # 이상치(오타)
ATH, ATH_V = "태오", 205           # 이상치(진짜일 수도): 농구 선수
SENTINELS = ["", "0", "-999"]       # 빈칸 vs 숫자인 척하는 '비었음' 표시

def mean(vs): return sum(vs) / len(vs)

MEAN_DIRTY = mean([v for _, v in CLASS])                       # 0 포함
MEAN_CLEAN = mean([v for _, v in REAL])                        # 0 제외
DUP_V = dict(REAL)[DUP]
MEAN_DUP = mean([v for _, v in REAL] + [DUP_V] * 2)           # 민수 ×3

assert len(CLASS) == 10 and len(REAL) == 8
assert round(MEAN_DIRTY, 1) == 131.2 and MEAN_CLEAN == 164 and MEAN_DUP == 166
assert all(v < MEAN_DIRTY for _, v in CLASS if v == 0) and all(v > MEAN_DIRTY for _, v in REAL)  # 평균선이 모든 실제 키보다 낮다
