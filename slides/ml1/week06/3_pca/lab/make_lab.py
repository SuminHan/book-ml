"""실습 노트북 원본: 이 파일을 고치고 실행하면 <주제>.ipynb(학생용) / <주제>_solution.ipynb(정답) 이 다시 만들어진다.
   ~/Documents/manim_shorts/_env/v/bin/python make_lab.py
코드 줄 끝의 '#@ANS 빈칸버전' = solution 노트북엔 앞부분, student 노트북엔 빈칸버전이 들어간다.
데이터: 학생 점수 (설명용 예시, 값 그대로 적음) · sklearn iris · sklearn digits (모두 설치에 포함)."""
import os, sys
sys.path.insert(0, os.path.expanduser("~/book-ml/lessons/_tools"))
import lab

HERE = os.path.dirname(os.path.abspath(__file__))
TOPIC = os.path.basename(os.path.dirname(HERE))          # 폴더 이름 (pca) 이 파일명 앞에 붙는다

CELLS = [
("md", """# PCA 주성분 분석 실습

그림자가 가장 넓게 퍼지는 방향 = 공분산 행렬의 고유벡터.

- **파트 1**: 학생 점수 20명 → 각도별 퍼짐, 고유분해로 PC1 직접 구하기
- **파트 2**: `sklearn` PCA 와 비교, 붓꽃 4 → 2차원 (표준화 전후)
- **파트 3**: 손글씨 숫자 압축 · 복원

`None` 으로 비워 둔 `TODO` 줄을 채우고 셀을 차례로 실행하세요."""),
("code", """import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris, load_digits
from sklearn.preprocessing import StandardScaler"""),
("md", """## 파트 1. 학생 20명의 (수학, 물리)"""),
("code", """scores = np.array([[95, 92], [35, 47], [73, 76], [61, 59], [62, 62], [65, 68], [42, 55], [65, 64], [57, 57], [99, 99],
                   [71, 66], [63, 62], [64, 70], [59, 63], [54, 56], [63, 67], [74, 58], [65, 71], [80, 74], [65, 56]], float)

def variance_kept(X, deg):
    Xc = X - X.mean(axis=0)
    u = np.array([np.cos(np.radians(deg)), np.sin(np.radians(deg))])
    shadow = Xc @ u                                      # one number per student
    return shadow.var() / Xc.var(axis=0).sum()           #@ANS return None  # TODO: shadow spread / total spread

degs = np.arange(0, 181)
kept = [variance_kept(scores, d) for d in degs]
print("math axis:", round(variance_kept(scores, 0) * 100), "%   best:", degs[np.argmax(kept)], "deg,", round(max(kept) * 100), "%")
plt.plot(degs, np.array(kept) * 100); plt.xlabel("direction (degrees)"); plt.ylabel("variance kept (%)"); plt.show()"""),
("code", """def my_pca(X):
    Xc = X - X.mean(axis=0)
    S = Xc.T @ Xc / (len(X) - 1)                         #@ANS S = None  # TODO: covariance matrix
    eigval, eigvec = np.linalg.eigh(S)                   # ascending order
    order = np.argsort(eigval)[::-1]
    eigvec = eigvec[:, order] * np.sign(eigvec[0, order])   # sign is arbitrary: make the first entry positive
    return eigval[order], eigvec, S

eigval, eigvec, S = my_pca(scores)
print("S ="); print(S.round(1))
print("eigenvalues:", eigval.round(1), "  ratio:", (eigval / eigval.sum()).round(3))
print("PC1:", eigvec[:, 0].round(2), "  angle:", round(np.degrees(np.arctan2(eigvec[1, 0], eigvec[0, 0])) % 180), "deg")"""),
("md", """## 파트 2. sklearn 과 비교 · 붓꽃"""),
("code", """pca = PCA().fit(scores)
print("sklearn ratio:", pca.explained_variance_ratio_.round(3), "  PC1:", pca.components_[0].round(2), "(sign may flip)")

iris = load_iris()
for name, X in [("raw", iris.data), ("standardized", StandardScaler().fit_transform(iris.data))]:
    p = PCA().fit(X)
    print(f"iris {name:12s}: ratio % {(p.explained_variance_ratio_ * 100).round(1)}")

Z = PCA(2).fit_transform(iris.data)
for k, name in enumerate(iris.target_names):
    plt.scatter(*Z[iris.target == k].T, s=20, label=name)
plt.xlabel("PC1"); plt.ylabel("PC2"); plt.legend(); plt.show()"""),
("md", """## 파트 3. 손글씨 숫자 압축

한 장 = 8 × 8 = 64칸. 성분 $k$ 개로 줄였다가 (`transform`) 다시 64칸으로 (`inverse_transform`)."""),
("code", """digits = load_digits().data
pca = PCA().fit(digits)
cum = np.cumsum(pca.explained_variance_ratio_)
k90 = int(np.searchsorted(cum, 0.9) + 1)                 #@ANS k90 = None  # TODO: fewest components keeping 90%
print("components for 90%:", k90, "   2 components keep:", round(cum[1] * 100, 1), "%")

def reconstruct(X, k):
    p = PCA(k).fit(X)
    return p.inverse_transform(p.transform(X))

ks = [2, 5, 10, k90, 64]
fig, axs = plt.subplots(1, len(ks) + 1, figsize=(11, 2.4))
axs[0].imshow(digits[0].reshape(8, 8), cmap="gray_r"); axs[0].set_title("original")
for ax, k in zip(axs[1:], ks):
    ax.imshow(reconstruct(digits, k)[0].reshape(8, 8), cmap="gray_r"); ax.set_title(f"k = {k}")
for ax in axs: ax.axis("off")
plt.show()"""),
("md", """**확인 질문**
1. 파트 1 에서 가장 넓은 방향과 가장 좁은 방향은 몇 도 차이인가? 두 비율의 합은?
2. 파트 2 에서 표준화하면 PC1 비율이 92.5% → 73.0% 로 줄어든 이유는?
3. 파트 3 에서 2 성분만으로 숫자 1,797장을 평면에 그리면 잘 갈라질까? (`PCA(2)` 로 그려 보기 → 다음 시간 t-SNE)
4. (도전) 학생 점수에 '국어' 를 더해 3차원으로 만든다면, 국어가 수학 · 물리와 상관이 없을 때 PC1 은 어떻게 될까?"""),
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
