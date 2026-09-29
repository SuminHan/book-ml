import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from sklearn.datasets import fetch_olivetti_faces

fm.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams['font.family'] = 'Noto Sans CJK JP'
plt.rcParams['axes.unicode_minus'] = False

KSA_BLUE = '#2E3192'

d = fetch_olivetti_faces(data_home='/tmp/sk_data')
X = d.data  # (400, 4096)
mean_face = X.mean(axis=0)
Xc = X - mean_face
# PCA via SVD
U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
eigenfaces = Vt[:8].reshape(8, 64, 64)

fig, axes = plt.subplots(2, 5, figsize=(11, 5), dpi=200)
axes = axes.flatten()
axes[0].imshow(mean_face.reshape(64, 64), cmap='gray')
axes[0].set_title('평균 얼굴', fontsize=15, color=KSA_BLUE)
axes[0].axis('off')
for i in range(8):
    axes[i + 1].imshow(eigenfaces[i], cmap='gray')
    axes[i + 1].set_title(f'고유얼굴 {i+1}', fontsize=15)
    axes[i + 1].axis('off')
axes[9].axis('off')

fig.suptitle('Eigenfaces --- AT&T/ORL 얼굴 400장에서 뽑은 상위 8개 주성분 (Turk & Pentland, 1991)',
             fontsize=15, color=KSA_BLUE, y=1.02)
plt.tight_layout()
plt.savefig('/tmp/claude-1002/-home-smhan/029e91ef-1260-463c-974d-c55f55e786a0/scratchpad/decks/ml1-week14/slides/kor/ml1/week14/figs/ch14_eigenfaces.png',
            bbox_inches='tight', facecolor='white')
print("done")
