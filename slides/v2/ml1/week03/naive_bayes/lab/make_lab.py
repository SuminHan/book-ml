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

- **파트 1** (9분): 베이즈 정리로 'free' 메시지가 스팸일 확률 구하기 · 직접 세어 확인 · 사전확률 바꿔 보기
- **파트 2** (10분): 나이브 베이즈(베르누이) 직접 구현: 스무딩 · 로그 합 후 테스트 정확도
- **파트 3** (6분): scikit-learn `BernoulliNB` 와 비교 · alpha 바꿔 보기

`None` 으로 비워 둔 `TODO` 줄을 채우고 셀을 차례로 실행하세요. 데이터는 처음 실행할 때 인터넷에서 내려받는다."""),
("code", """import io, re, urllib.request, zipfile
import numpy as np
from collections import Counter

URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
raw = zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(URL, timeout=60).read())).read("SMSSpamCollection").decode("utf-8")
rows = [line.split("\\t", 1) for line in raw.strip().split("\\n")]
is_spam = np.array([label == "spam" for label, _ in rows])      # True = spam
texts = [text for _, text in rows]

def tokenize(text):
    return re.findall(r"[a-z0-9']+", text.lower())              # list of lowercase words

# 80% train / 20% test (fixed shuffle)
order = np.random.default_rng(0).permutation(len(texts))
cut = int(0.8 * len(texts))
train, test = order[:cut], order[cut:]

print(f"{len(texts)} messages, {is_spam.sum()} spam ({is_spam.mean():.1%})")
print(f"{len(train)} train / {len(test)} test")
for i in np.where(is_spam)[0][:2]: print("spam example:", texts[i][:90])
for i in np.where(~is_spam)[0][:2]: print("ham example: ", texts[i][:90])"""),
("md", """## 파트 1. 베이즈 정리: 'free' 가 든 메시지는 스팸일까?

데이터로 셀 수 있는 것은 $P(\\text{‘free’} \\mid \\text{spam})$ 이다. 알고 싶은 것은 $P(\\text{spam} \\mid \\text{‘free’})$ 이다."""),
("code", """words = [set(tokenize(t)) for t in texts]                  # set of words in each message
spam_tr = [i for i in train if is_spam[i]]
ham_tr  = [i for i in train if not is_spam[i]]

prior = len(spam_tr) / len(train)                         # prior P(spam)
w = "free"
p_w_spam = np.mean([w in words[i] for i in spam_tr])      # P('free' | spam)
p_w_ham  = np.mean([w in words[i] for i in ham_tr])       #@ANS p_w_ham = None          # TODO: share of ham messages containing 'free' (same way as the line above)
print(f"P(spam) = {prior:.3f}")
print(f"P('free' | spam) = {p_w_spam:.3f},  P('free' | ham) = {p_w_ham:.4f}")"""),
("code", """def posterior(prior, p_pos, p_neg):
    \"\"\"prior: prior probability, p_pos: P(word | spam), p_neg: P(word | ham)\"\"\"
    num = prior * p_pos
    return num / (num + (1 - prior) * p_neg)               #@ANS return None               # TODO: Bayes' rule (denominator = sum of both cases)

print(f"Bayes rule:   P(spam | 'free') = {posterior(prior, p_w_spam, p_w_ham):.1%}")

# Count directly: share of spam among training messages containing 'free'
has_w = [i for i in train if w in words[i]]
print(f"direct count: {sum(is_spam[i] for i in has_w)} spam out of {len(has_w)} messages with 'free' = {np.mean([is_spam[i] for i in has_w]):.1%}")"""),
("md", """같은 단어라도 **사전확률**이 다르면 결론이 달라진다. 스팸이 훨씬 드문 받은편지함을 가정해 보자."""),
("code", """for pr in [0.50, prior, 0.01]:
    print(f"spam rate {pr:5.1%}  ->  P(spam | 'free') = {posterior(pr, p_w_spam, p_w_ham):.1%}")

# Try other words
for word in ["win", "txt", "call", "ok"]:
    s = np.mean([word in words[i] for i in spam_tr]); h = np.mean([word in words[i] for i in ham_tr])
    print(f"{word:5s} P(word|spam)={s:.3f}  P(word|ham)={h:.4f}  ->  P(spam | word) = {posterior(prior, s, h):.1%}")"""),
("md", """## 파트 2. 나이브 베이즈 직접 구현 (베르누이)

메시지를 "어휘의 각 단어가 있다(1) / 없다(0)" 로 보고, 단어끼리 독립이라고 가정해 점수로 비교한다. 파트 1의 $P(\\text{word present} \\mid \\text{spam})$ 이 바로 그 단어 확률이다.

- 단어 확률(라플라스 스무딩): $P(x_i=1\\mid y) = \\dfrac{n_{i,y}+1}{M_y + 2}$ ($n_{i,y}$: 단어 $i$ 가 든 $y$ 메시지 수, $M_y$: $y$ 메시지 수)
- 점수(로그 합): $\\log P(y) + \\sum_i \\big[x_i\\log p_i + (1-x_i)\\log(1-p_i)\\big]$ ($p_i = P(x_i=1\\mid y)$)"""),
("code", """vocab = sorted({w for i in train for w in words[i]}); V = len(vocab)     # vocabulary (words seen in training data)
index = {w: j for j, w in enumerate(vocab)}

def to_binary(ids):
    \"\"\"message x vocabulary matrix: 1 if the word is present, else 0 (unseen words are dropped)\"\"\"
    X = np.zeros((len(ids), V))
    for r, i in enumerate(ids):
        for w in words[i]:
            if w in index: X[r, index[w]] = 1
    return X

X_tr, X_te = to_binary(train), to_binary(test)
y_tr, truth = is_spam[train], is_spam[test]

M = {"spam": int(y_tr.sum()), "ham": int((~y_tr).sum())}                     # messages per class
n = {"spam": X_tr[y_tr].sum(axis=0), "ham": X_tr[~y_tr].sum(axis=0)}         # messages containing each word
prior_c = {c: M[c] / len(train) for c in M}
print("M =", M, " V =", V, " prior =", {c: round(p, 3) for c, p in prior_c.items()})

# Without smoothing? 'claim' never appears in a ham message
j = index["claim"]
print("ham messages with 'claim':", int(n["ham"][j]), ",  spam messages:", int(n["spam"][j]))
print("without smoothing P(claim present | ham) =", n["ham"][j] / M["ham"], " -> the whole product becomes 0")"""),
("code", """def p_word(c):
    \"\"\"probability that each word is present in class c (array of length V)\"\"\"
    return (n[c] + 1) / (M[c] + 2)                                             #@ANS return None                  # TODO: Laplace smoothing (one imaginary message each)

def log_score(X, c):
    \"\"\"X: message x vocabulary matrix. Per message: log prior + sum [log p if present, log(1-p) if absent]\"\"\"
    p = p_word(c)
    return np.log(prior_c[c]) + X @ np.log(p) + (1 - X) @ np.log(1 - p)         #@ANS return None                  # TODO: log prior + log p for present words + log(1-p) for absent words

def spam_prob(X):
    s_spam, s_ham = log_score(X, "spam"), log_score(X, "ham")
    return 1 / (1 + np.exp(s_ham - s_spam))                                    # log scores -> spam probability

print("with smoothing P(claim present | ham) =", round(p_word("ham")[index["claim"]], 4))
demo = ["Free entry! Claim your prize now, text WIN to 80082", "Ok, see you at the cafe tomorrow", "Can you call me when you are free?"]
for msg in demo:
    x = np.zeros((1, V))
    for w in set(tokenize(msg)):
        if w in index: x[0, index[w]] = 1
    print(f"{spam_prob(x)[0]:6.1%}  {msg}")"""),
("code", """# Evaluate on the 20% test set
pred = spam_prob(X_te) > 0.5
tp, fp = (pred & truth).sum(), (pred & ~truth).sum()
fn, tn = (~pred & truth).sum(), (~pred & ~truth).sum()
print(f"accuracy {(pred == truth).mean():.1%}")
print(f"spam caught {tp} · spam missed {fn} · ham flagged as spam {fp} · ham kept {tn}")
print(f"spam precision {tp / (tp + fp):.1%}  recall {tp / (tp + fn):.1%}")

# Which words push hardest toward spam? (log likelihood ratio)
ratio = np.log(p_word("spam") / p_word("ham"))
common = np.where(n["spam"] + n["ham"] >= 20)[0]
top = common[np.argsort(-ratio[common])]
print("words pointing to spam:", [vocab[j] for j in top[:12]])
print("words pointing to ham: ", [vocab[j] for j in top[-12:]])"""),
("md", """## 파트 3. scikit-learn과 비교
`BernoulliNB(alpha=1.0)` 이 우리가 구현한 라플라스 스무딩과 같다."""),
("code", """from sklearn.naive_bayes import BernoulliNB

clf = BernoulliNB(alpha=1.0).fit(X_tr, y_tr)
sk_pred = clf.predict(X_te).astype(bool)
print(f"sklearn accuracy {(sk_pred == truth).mean():.1%}   our implementation {(pred == truth).mean():.1%}")
print(f"predictions that agree: {(sk_pred == pred).mean():.1%}")"""),
("code", """# What if we change alpha (smoothing strength)?
for a in [0.01, 0.1, 1.0, 10.0]:
    c = BernoulliNB(alpha=a).fit(X_tr, y_tr)
    print(f"alpha={a:5}: test accuracy {(c.predict(X_te).astype(bool) == truth).mean():.1%}")"""),
("md", """**확인 질문**
1. 'free' 가 든 메시지의 스팸 확률이 직접 세기와 베이즈 정리에서 같게 나온 이유는?
2. 스팸 비율이 1% 인 받은편지함에서는 같은 'free' 도 스팸일 확률이 얼마로 떨어지나? 왜 그럴까?
3. 'claim' 이 정상 메시지에 한 번도 없는데 스무딩을 하면 정상에서의 확률이 0 이 되지 않는 이유는? 분모가 `+2` 인 이유는?
4. 정상 메시지를 스팸으로 잘못 분류하면 어떤 문제가 생길까? 정밀도와 재현율 중 무엇이 더 중요할까?
5. `alpha` 를 10 으로 키우면 정확도가 떨어지는 이유는?

데이터 출처: Almeida, Hidalgo, Yamakami (2011), SMS Spam Collection, UCI Machine Learning Repository (CC BY 4.0)."""),
]

if __name__ == "__main__":
    REL = os.path.relpath(HERE, os.path.expanduser("~/book-ml"))
    COLAB = f"https://colab.research.google.com/github/SuminHan/book-ml/blob/main/{REL}/{TOPIC}.ipynb"
    kind, first = CELLS[0]                                   # 첫 칸 제목 아래에 Colab 배지
    CELLS[0] = (kind, first.replace("\n", f"\n\n[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({COLAB})\n", 1))
    lab.build(CELLS, f"{HERE}/{TOPIC}.ipynb", f"{HERE}/{TOPIC}_solution.ipynb")
    print(lab.verify(f"{HERE}/{TOPIC}_solution.ipynb"))
