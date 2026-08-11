# Topics Covered

## 합성곱(Convolution) 연산

\\(k \times k\\) 필터(커널, kernel)를 입력 위에서 슬라이드시키며, 각 위치에서 필터와
겹치는 영역의 원소별 곱의 합을 계산한다:

```python
def conv2d(image, kernel):
    img_h, img_w = len(image), len(image[0])
    k = len(kernel)
    out_h, out_w = img_h - k + 1, img_w - k + 1
    output = [[0.0] * out_w for _ in range(out_h)]
    for i in range(out_h):
        for j in range(out_w):
            total = 0.0
            for di in range(k):
                for dj in range(k):
                    total += image[i+di][j+dj] * kernel[di][dj]
            output[i][j] = total
    return output
```

각 필터는 특정 패턴(수직선, 모서리 등)에 강하게 반응하도록 **학습**된다 — 필터의 값
자체가 학습 대상 파라미터다.

## 출력 크기 공식

입력 크기 \\(n \times n\\), 필터 크기 \\(k \times k\\), 패딩(padding) \\(p\\)
(입력 가장자리에 0을 두르는 픽셀 수), 스트라이드(stride) \\(s\\)(필터가 한 번에
이동하는 칸 수)일 때, 출력 크기는:

\\[n_{\text{out}} = \left\lfloor \frac{n + 2p - k}{s} \right\rfloor + 1\\]

예: \\(n=28, k=3, p=0, s=1\\)이면 \\(n_{\text{out}} = 28-3+1 = 26\\). 패딩
없이는 필터를 지날 때마다 이미지가 점점 작아진다 — `p = (k-1)/2`로 설정하면(**same
padding**) 출력 크기를 입력과 같게 유지할 수 있다.

## 파라미터 수 계산

\\(C_{\text{in}}\\)개 입력 채널, \\(C_{\text{out}}\\)개 출력 채널(즉 필터 개수),
필터 크기 \\(k \times k\\)인 합성곱 층의 파라미터 수(bias 포함):

\\[\text{params} = (k \times k \times C_{\text{in}} + 1) \times C_{\text{out}}\\]

이걸 같은 크기의 완전연결층(fully-connected layer, 입력을 펼쳐서 다 연결)과 비교하면
CNN의 효율이 확연히 드러난다. 예: 32×32×3 이미지, 필터 3×3, 출력 채널 16개인 합성곱
층은 \\((3 \times 3 \times 3 + 1) \times 16 = 448\\)개의 파라미터만 쓴다. 같은
입력을 완전연결층(출력 뉴런도 32×32×16개라 가정)에 연결하려면 수백만 개가 필요하다.

## 풀링(Pooling)

합성곱 다음에는 보통 풀링 층으로 공간 크기를 줄인다. **최대 풀링**(max pooling)은
\\(2\times2\\) 영역마다 최댓값 하나만 남긴다 — 위치가 한두 칸 미세하게 움직여도
같은 최댓값이 뽑힐 가능성이 높아, 작은 이동에 둔감한(translation-invariant) 특징을
만든다. 풀링은 학습 파라미터가 없다 — 순수한 다운샘플링(downsampling) 연산이다.

## CNN의 전형적인 구조

`[합성곱 → 활성함수(ReLU) → 풀링]`을 여러 번 쌓아 공간 크기는 줄이고 채널(특징) 수는
늘려가다가, 마지막에 완전연결층 한두 개를 붙여 분류 결과를 낸다. 얕은 층은 선/모서리
같은 저수준 패턴을, 깊은 층으로 갈수록 눈/코/바퀴 같은 고수준 패턴을 감지하도록
학습된다 — 허블과 비셀이 관찰한, 단순 세포에서 복합 세포로 이어지는 계층 구조와
같은 원리다.
