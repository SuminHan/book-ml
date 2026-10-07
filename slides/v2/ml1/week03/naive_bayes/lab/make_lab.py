"""실습 노트북 원본: 이 파일을 고치고 실행하면 <주제>.ipynb(학생용) / <주제>_solution.ipynb(정답) 이 다시 만들어진다.
   ~/Documents/manim_shorts/_env/v/bin/python make_lab.py
코드 줄 끝의 '#@ANS 빈칸버전' = solution 노트북엔 앞부분, student 노트북엔 빈칸버전이 들어간다.
데이터: UCI SMS Spam Collection (영어 문자메시지 5,574건, CC BY 4.0)."""
import os, sys
sys.path.insert(0, os.path.expanduser("~/book-ml/lessons/_tools"))
import lab

HERE = os.path.dirname(os.path.abspath(__file__))
TOPIC = os.path.basename(os.path.dirname(HERE))          # 폴더 이름 (예: naive_bayes) 이 파일명 앞에 붙는다

CELLS = [
("md", """# 나이브 베이즈 실습: 진짜 스팸 문자 분류 (25분)

영어 문자메시지 5,574건(스팸 747 · 정상 4,827)으로 스팸 필터를 만든다. 슬라이드의 숫자를 진짜 데이터에서 직접 세어 본다.

| 파트 | 내용 | 시간 |
|---|---|---|
| 1 | 베이즈 정리로 'free' 메시지가 스팸일 확률 구하기 · 직접 세어 확인 · 사전확률 바꿔 보기 | 9분 |
| 2 | 나이브 베이즈(베르누이) 직접 구현: 스무딩 · 로그 합 후 테스트 정확도 | 10분 |
| 3 | scikit-learn `BernoulliNB` 와 비교 · alpha 바꿔 보기 | 6분 |

`None` 으로 비워 둔 `TODO` 줄을 채우고 셀을 차례로 실행하세요. 데이터는 처음 실행할 때 인터넷에서 내려받는다."""),
("code", """import io, re, urllib.request, zipfile
import numpy as np
from collections import Counter

URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
raw = zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(URL, timeout=60).read())).read("SMSSpamCollection").decode("utf-8")
rows = [line.split("\\t", 1) for line in raw.strip().split("\\n")]
is_spam = np.array([label == "spam" for label, _ in rows])      # True = 스팸
texts = [text for _, text in rows]

def tokenize(text):
    return re.findall(r"[a-z0-9']+", text.lower())              # 소문자 단어 목록

# 학습 80% / 테스트 20% (섞는 순서 고정)
order = np.random.default_rng(0).permutation(len(texts))
cut = int(0.8 * len(texts))
train, test = order[:cut], order[cut:]

print(f"전체 {len(texts)}건, 스팸 {is_spam.sum()}건 ({is_spam.mean():.1%})")
print(f"학습 {len(train)}건 / 테스트 {len(test)}건")
for i in np.where(is_spam)[0][:2]: print("스팸 예:", texts[i][:90])
for i in np.where(~is_spam)[0][:2]: print("정상 예:", texts[i][:90])"""),
("md", """## 파트 1. 베이즈 정리: 'free' 가 든 메시지는 스팸일까?

데이터로 셀 수 있는 것은 $P(\\text{‘free’} \\mid \\text{spam})$ 이다. 알고 싶은 것은 $P(\\text{spam} \\mid \\text{‘free’})$ 이다."""),
("code", """words = [set(tokenize(t)) for t in texts]                  # 메시지마다 들어 있는 단어 집합
spam_tr = [i for i in train if is_spam[i]]
ham_tr  = [i for i in train if not is_spam[i]]

prior = len(spam_tr) / len(train)                         # 사전확률 P(spam)
w = "free"
p_w_spam = np.mean([w in words[i] for i in spam_tr])      # P('free' | spam)
p_w_ham  = np.mean([w in words[i] for i in ham_tr])       #@ANS p_w_ham = None          # TODO: 정상 메시지 중 'free' 가 든 비율 (위 줄과 같은 방식으로)
print(f"P(spam) = {prior:.3f}")
print(f"P('free' | spam) = {p_w_spam:.3f},  P('free' | ham) = {p_w_ham:.4f}")"""),
("code", """def posterior(prior, p_pos, p_neg):
    \"\"\"prior: 사전확률, p_pos: 스팸에서 단어가 나올 확률, p_neg: 정상에서 나올 확률\"\"\"
    num = prior * p_pos
    return num / (num + (1 - prior) * p_neg)               #@ANS return None               # TODO: 베이즈 정리 (분모 = 두 경우의 합)

print(f"베이즈 정리:  P(spam | 'free') = {posterior(prior, p_w_spam, p_w_ham):.1%}")

# 직접 세기: 'free' 가 든 학습 메시지 중 스팸의 비율
has_w = [i for i in train if w in words[i]]
print(f"직접 세기:    'free' 든 {len(has_w)}건 중 스팸 {sum(is_spam[i] for i in has_w)}건 = {np.mean([is_spam[i] for i in has_w]):.1%}")"""),
("md", """같은 단어라도 **사전확률**이 다르면 결론이 달라진다. 스팸이 훨씬 드문 받은편지함을 가정해 보자."""),
("code", """for pr in [0.50, prior, 0.01]:
    print(f"스팸 비율 {pr:5.1%}  ->  'free' 메시지가 스팸일 확률 {posterior(pr, p_w_spam, p_w_ham):.1%}")

# 단어를 바꿔 보자 (다른 단어도 확인)
for word in ["win", "txt", "call", "ok"]:
    s = np.mean([word in words[i] for i in spam_tr]); h = np.mean([word in words[i] for i in ham_tr])
    print(f"{word:5s} P(word|spam)={s:.3f}  P(word|ham)={h:.4f}  ->  스팸일 확률 {posterior(prior, s, h):.1%}")"""),
("md", """## 파트 2. 나이브 베이즈 직접 구현 (베르누이)

메시지를 "어휘의 각 단어가 있다(1) / 없다(0)" 로 보고, 단어끼리 독립이라고 가정해 점수로 비교한다. 파트 1의 $P(\\text{word present} \\mid \\text{spam})$ 이 바로 그 단어 확률이다.

- 단어 확률(라플라스 스무딩): $P(x_i=1\\mid y) = \\dfrac{n_{i,y}+1}{M_y + 2}$ ($n_{i,y}$: 단어 $i$ 가 든 $y$ 메시지 수, $M_y$: $y$ 메시지 수)
- 점수(로그 합): $\\log P(y) + \\sum_i \\big[x_i\\log p_i + (1-x_i)\\log(1-p_i)\\big]$ ($p_i = P(x_i=1\\mid y)$)"""),
("code", """vocab = sorted({w for i in train for w in words[i]}); V = len(vocab)     # 어휘 (학습 데이터에 나온 단어)
index = {w: j for j, w in enumerate(vocab)}

def to_binary(ids):
    \"\"\"메시지 x 어휘 행렬: 단어가 있으면 1, 없으면 0 (처음 보는 단어는 버린다)\"\"\"
    X = np.zeros((len(ids), V))
    for r, i in enumerate(ids):
        for w in words[i]:
            if w in index: X[r, index[w]] = 1
    return X

X_tr, X_te = to_binary(train), to_binary(test)
y_tr, truth = is_spam[train], is_spam[test]

M = {"spam": int(y_tr.sum()), "ham": int((~y_tr).sum())}                     # 클래스별 메시지 수
n = {"spam": X_tr[y_tr].sum(axis=0), "ham": X_tr[~y_tr].sum(axis=0)}         # 단어가 든 메시지 수
prior_c = {c: M[c] / len(train) for c in M}
print("M =", M, " V =", V, " 사전확률 =", {c: round(p, 3) for c, p in prior_c.items()})

# 스무딩이 없으면? 'claim' 은 정상 메시지에 한 번도 없다
j = index["claim"]
print("'claim' 이 든 정상 메시지:", int(n["ham"][j]), "건,  스팸 메시지:", int(n["spam"][j]), "건")
print("스무딩 없이 P(claim present | ham) =", n["ham"][j] / M["ham"], " -> 곱하면 전체가 0")"""),
("code", """def p_word(c):
    \"\"\"클래스 c 에서 각 단어가 있을 확률 (길이 V 의 배열)\"\"\"
    return (n[c] + 1) / (M[c] + 2)                                             #@ANS return None                  # TODO: 라플라스 스무딩 (가상 메일 1통씩)

def log_score(X, c):
    \"\"\"X: 메시지 x 어휘 행렬. 메시지마다 log 사전 + Σ [있으면 log p, 없으면 log(1-p)]\"\"\"
    p = p_word(c)
    return np.log(prior_c[c]) + X @ np.log(p) + (1 - X) @ np.log(1 - p)         #@ANS return None                  # TODO: log 사전 + 있는 단어 log p + 없는 단어 log(1-p)

def spam_prob(X):
    s_spam, s_ham = log_score(X, "spam"), log_score(X, "ham")
    return 1 / (1 + np.exp(s_ham - s_spam))                                    # 로그 점수 -> 스팸 확률

print("스무딩 후 P(claim present | ham) =", round(p_word("ham")[index["claim"]], 4))
demo = ["Free entry! Claim your prize now, text WIN to 80082", "Ok, see you at the cafe tomorrow", "Can you call me when you are free?"]
for msg in demo:
    x = np.zeros((1, V))
    for w in set(tokenize(msg)):
        if w in index: x[0, index[w]] = 1
    print(f"{spam_prob(x)[0]:6.1%}  {msg}")"""),
("code", """# 테스트 20% 로 평가
pred = spam_prob(X_te) > 0.5
tp, fp = (pred & truth).sum(), (pred & ~truth).sum()
fn, tn = (~pred & truth).sum(), (~pred & ~truth).sum()
print(f"정확도 {(pred == truth).mean():.1%}")
print(f"스팸을 스팸으로 {tp}건 · 스팸을 놓침 {fn}건 · 정상을 스팸으로 오인 {fp}건 · 정상을 정상으로 {tn}건")
print(f"스팸 정밀도 {tp / (tp + fp):.1%}  재현율 {tp / (tp + fn):.1%}")

# 어떤 단어가 스팸 판정에 가장 크게 기여할까? (로그 가능도비)
ratio = np.log(p_word("spam") / p_word("ham"))
common = np.where(n["spam"] + n["ham"] >= 20)[0]
top = common[np.argsort(-ratio[common])]
print("스팸을 가리키는 단어:", [vocab[j] for j in top[:12]])
print("정상을 가리키는 단어:", [vocab[j] for j in top[-12:]])"""),
("md", """## 파트 3. scikit-learn과 비교
`BernoulliNB(alpha=1.0)` 이 우리가 구현한 라플라스 스무딩과 같다."""),
("code", """from sklearn.naive_bayes import BernoulliNB

clf = BernoulliNB(alpha=1.0).fit(X_tr, y_tr)
sk_pred = clf.predict(X_te).astype(bool)
print(f"sklearn 정확도 {(sk_pred == truth).mean():.1%}   직접 구현 {(pred == truth).mean():.1%}")
print(f"두 방법의 예측이 같은 비율: {(sk_pred == pred).mean():.1%}")"""),
("code", """# alpha(스무딩 세기)를 바꾸면?
for a in [0.01, 0.1, 1.0, 10.0]:
    c = BernoulliNB(alpha=a).fit(X_tr, y_tr)
    print(f"alpha={a:5}: 테스트 정확도 {(c.predict(X_te).astype(bool) == truth).mean():.1%}")"""),
("md", """**확인 질문**
1. 'free' 가 든 메시지의 스팸 확률이 직접 세기와 베이즈 정리에서 같게 나온 이유는?
2. 스팸 비율이 1% 인 받은편지함에서는 같은 'free' 도 스팸일 확률이 얼마로 떨어지나? 왜 그럴까?
3. 'claim' 이 정상 메시지에 한 번도 없는데 스무딩을 하면 정상에서의 확률이 0 이 되지 않는 이유는? 분모가 `+2` 인 이유는?
4. 정상 메시지를 스팸으로 잘못 분류하면 어떤 문제가 생길까? 정밀도와 재현율 중 무엇이 더 중요할까?
5. `alpha` 를 10 으로 키우면 정확도가 떨어지는 이유는?

데이터 출처: Almeida, Hidalgo, Yamakami (2011), SMS Spam Collection, UCI Machine Learning Repository (CC BY 4.0)."""),
]

if __name__ == "__main__":
    lab.build(CELLS, f"{HERE}/{TOPIC}.ipynb", f"{HERE}/{TOPIC}_solution.ipynb")
    print(lab.verify(f"{HERE}/{TOPIC}_solution.ipynb"))
