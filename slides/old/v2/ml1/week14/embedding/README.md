# 표상과 임베딩

- 원본 쇼츠: [`lessons/ml1/embedding/`](../../../../../lessons/ml1/embedding/README.md)
- 구성: 쇼펜하우어 '세계는 나의 표상이다' → 임베딩 = 숫자 공간에 박아 넣은 표상 (word2vec · GCN · CNN · 트랜스포머) → 원-핫 × 표 = 한 줄 (nn.Embedding) → 실험 1 고양이 · 개 CNN (84.1%): 호랑이 83% · 표범 91% 고양이 쪽, 사자 50% → 오렌지 · 구름도 그 축 위에 (배운 만큼만 보인다, 쇼츠의 '오렌지 멀리' 는 일부 사진 5% 때문) → 실험 2 CLIP 유사도 (호랑이↔표범 0.98, 오렌지↔사과 0.96) → 0-shot 78% → 코사인 유사도 → 쓰임 (검색 · RAG · 추천) + 문제풀이 + 이미지 출처
- 상태: 14쪽. exp_embed.py → assets/embed.npz (CIFAR parquet 은 lessons/ml1/embedding/experiment/data), exp_clip.py → assets/clip.npz (HF 캐시 CLIP)
- 실습: nn.Embedding · 코사인 · CIFAR 고양이 · 개 CNN 5바퀴 (75.8%) 에 처음 보는 사진 · MiniLM 문장 의미 검색
