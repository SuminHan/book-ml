# kNN 최근접 이웃

- 원본 쇼츠: [`lessons/ml1/knn/`](../../../../../lessons/ml1/knn/README.md)
- 구성: 과일 21개 k=3 다수결 → 유클리드 거리 0.72 → 학습 없음 → MNIST 손글씨: 그림 = 점 784개 · 거리 2.65 → 학습 없이 k=1 96.9% (신경망 1에폭 96.3%) → 가까운 k장 다수결 k=3 97.2% → k=1·5·21 경계 → leave-one-out 20/21 → 단맛 100점 단위 → 표준화 (와인 71→97%) → 차원의 저주 (거리 비 0.01→0.9, 잡음 특징 90→56%) + 문제풀이
- 상태: 16쪽. MNIST 수치는 exp_mnist.py (원본 ~/Documents/manim_shorts/_data/MNIST, 예시 그림은 assets/mnist_examples.npz). 차원의 저주 수치는 exp_dim.py (lessons/ml1/curse_of_dim 과 같은 실험)
- 실습: kNN 직접 구현 + leave-one-out · Wine 표준화 전후 · CV 로 k 고르기 + 잡음 특징
