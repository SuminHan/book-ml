"""실습 노트북 원본: 이 파일을 고치고 실행하면 <주제>.ipynb(학생용) / <주제>_solution.ipynb(정답) 이 다시 만들어진다.
   ~/Documents/manim_shorts/_env/v/bin/python make_lab.py
코드 줄 끝의 '#@ANS 빈칸버전' = solution 노트북엔 앞부분, student 노트북엔 빈칸버전이 들어간다.
데이터: 없음 (뽑기 확률은 설명용 예시). 슬라이드 실험 exp_kl.py 와 같은 설정."""
import os, sys
sys.path.insert(0, os.path.expanduser("~/book-ml/lessons/_tools"))
import lab

HERE = os.path.dirname(os.path.abspath(__file__))
TOPIC = os.path.basename(os.path.dirname(HERE))          # 폴더 이름 (kl_divergence) 이 파일명 앞에 붙는다

CELLS = [
("md", """# KL 발산 실습

놀라움 $-\\log_2 p$, 엔트로피 = 평균 놀라움, 교차 엔트로피 = 틀린 확률로 잰 평균 놀라움, KL = 그 차이.

- **파트 1**: 함수 만들기 → 뽑기 예시 0.24 · 0.40 · 0.16 비트
- **파트 2**: 뽑기 흉내 · 기록에 드는 비트
- **파트 3**: 봉우리 둘을 정규분포 하나로: 앞 방향 vs 뒤 방향 KL

`None` 으로 비워 둔 `TODO` 줄을 채우고 셀을 차례로 실행하세요."""),
("code", """import numpy as np
import matplotlib.pyplot as plt"""),
("md", """## 파트 1. 네 가지 함수

공지 Q = (당첨 0.2, 꽝 0.8), 실제 P = (0.04, 0.96). 설명용 예시."""),
("code", """def surprise(p):
    return -np.log2(p)

def entropy(p):
    p = np.asarray(p)
    return float(np.sum(p * surprise(p)))

def cross_entropy(p, q):
    p, q = np.asarray(p), np.asarray(q)
    return float(np.sum(p * surprise(q)))                #@ANS return None  # TODO: surprise measured with q, averaged with p

def kl(p, q):
    p, q = np.asarray(p), np.asarray(q)
    return float(np.sum(p * np.log2(p / q)))             #@ANS return None  # TODO: KL(p || q) in bits

P, Q = [0.04, 0.96], [0.20, 0.80]
print("surprise of a win  (real / notice):", surprise(0.04).round(2), surprise(0.20).round(2))
print(f"H(P) = {entropy(P):.2f}   H(P,Q) = {cross_entropy(P, Q):.2f}   KL(P||Q) = {kl(P, Q):.2f}   KL(Q||P) = {kl(Q, P):.2f}")
print("classification, true (1, 0):", [round(cross_entropy([1, 1e-12], [q, 1 - q]), 2) for q in (0.7, 0.95, 0.5)])"""),
("md", """## 파트 2. 뽑기 흉내"""),
("code", """rng = np.random.default_rng(0)
draws = rng.geometric(0.04, 100_000)                     # draws until the first win, real 4%
draws_notice = rng.geometric(0.20, 100_000)
print(f"real 4%: mean {draws.mean():.1f}, median {np.median(draws):.0f}, more than 50 draws {np.mean(draws > 50):.0%}")
print(f"notice 20%: mean {draws_notice.mean():.1f}")
plt.hist(draws, bins=np.arange(1, 151, 3)); plt.axvline(5, color="orange", ls="--"); plt.axvline(draws.mean(), color="red", ls="--")
plt.xlabel("draws until the first win"); plt.show()

# Recording 1,000 results: average bits with a code built for P vs for Q
results = rng.random(1000) < 0.04                        # True = win
bits_real = sum(surprise(0.04) if r else surprise(0.96) for r in results)
bits_notice = sum(surprise(0.20) if r else surprise(0.80) for r in results)
print(f"bits for 1,000 records: real code {bits_real:.0f}, notice code {bits_notice:.0f}, wasted {bits_notice - bits_real:.0f}")"""),
("md", """## 파트 3. 다 덮기 vs 하나 고르기

$P$: −2 와 +2 에 봉우리 (표준편차 0.5). 정규분포 $Q = N(m, s^2)$ 의 $m, s$ 를 격자로 찾는다."""),
("code", """x = np.linspace(-8, 8, 4001); dx = x[1] - x[0]

def normal_pdf(x, m, s):
    return np.exp(-(x - m) ** 2 / (2 * s * s)) / (s * np.sqrt(2 * np.pi))

def kl_continuous(p, q, dx):
    p, q = p + 1e-300, q + 1e-300
    return float(np.sum(p * np.log(p / q)) * dx)         # natural log here

P_two = 0.5 * normal_pdf(x, -2, 0.5) + 0.5 * normal_pdf(x, 2, 0.5)
best = {"KL(P||Q)": (np.inf, None), "KL(Q||P)": (np.inf, None)}
for m in np.linspace(-3, 3, 121):
    for s in np.linspace(0.2, 3.5, 166):
        Qx = normal_pdf(x, m, s)
        for name, value in [("KL(P||Q)", kl_continuous(P_two, Qx, dx)), ("KL(Q||P)", kl_continuous(Qx, P_two, dx))]:
            if value < best[name][0]:
                best[name] = (value, (round(m, 2), round(s, 2)))
for name, (value, (m, s)) in best.items():
    print(f"min {name}: m = {m}, s = {s}, value {value:.3f}")
    plt.plot(x, normal_pdf(x, m, s), label=name)
plt.fill_between(x, P_two, alpha=0.3, label="P"); plt.xlim(-6, 6); plt.legend(); plt.show()"""),
("md", """**확인 질문**
1. 파트 1 에서 KL(P‖Q) 와 KL(Q‖P) 가 다른 이유를 식으로 설명해 보라.
2. 파트 2 에서 낭비된 비트 수는 1,000 × KL 과 얼마나 비슷한가? 왜 정확히 같지는 않을까?
3. 파트 3 의 뒤 방향 최솟값 0.693 은 ln 2 와 같다. 왜 그럴까? (힌트: 봉우리 하나에만 붙으면 P 의 절반만 설명)
4. (도전) 실제 확률을 4% → 10% 로 바꾸면 KL(P‖Q) 는 어떻게 되나? 20% 면?"""),
]

if __name__ == "__main__":
    REL = os.path.relpath(HERE, os.path.expanduser("~/book-ml"))
    COLAB = f"https://colab.research.google.com/github/SuminHan/book-ml/blob/main/{REL}/{TOPIC}.ipynb"
    kind, first = CELLS[0]                                   # 첫 칸 제목 아래에 Colab 배지
    CELLS[0] = (kind, first.replace("\n", f"\n\n[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({COLAB})\n", 1))
    lab.build(CELLS, f"{HERE}/{TOPIC}.ipynb", f"{HERE}/{TOPIC}_solution.ipynb")
    sys.path.insert(0, os.path.join(HERE, "../../../../tools"))
    from expected import add_expected                         # 학생용 코드 칸 아래에 '예상 결과' (정답을 실제 실행한 출력)
    print(add_expected(f"{HERE}/{TOPIC}.ipynb", f"{HERE}/{TOPIC}_solution.ipynb"))
