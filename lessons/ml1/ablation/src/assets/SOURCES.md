# 출처·수치
- Vaswani et al. (2017), "Attention Is All You Need", NeurIPS — Table 3 (A): 헤드 수만 바꾸고 다시 학습
  (EN-DE dev newstest2013 BLEU): h=1 24.9 · h=4 25.5 · **h=8 25.8 (base)** · h=16 25.8 · h=32 25.4. base 파라미터 65×10⁶
  - 주의: 책 `kor/src/ml1/chapter12/3.md` 의 "EN-DE 24.16 → 23.48, EN-FR 38.19 → 37.19"는 원논문 Table 3과 다름 (확인 필요)
- Michel, Levy, Neubig (2019), "Are Sixteen Heads Really Better than One?", NeurIPS — 다 학습한 모델에서 헤드를 끄기만 함(재학습 X)
  - BERT-base(12층×12헤드=144) MNLI: 중요도 낮은 순으로 40%까지 꺼도 눈에 띄는 손해 없음 (§4.2, Fig. 3b)
  - WMT large Transformer(6층, 층당 16헤드, EN-FR, BLEU 36.05): 6층 인코더-디코더 어텐션을 헤드 1개만 남기면 −13.56 (Table 2)
- 쇼츠의 BLEU는 "번역 점수"로 부름. 막대그래프 세로축은 23.5부터 (화면에 표시)

## 이름의 유래 장면 (s3b)
- Flourens (1794–1867): 1822–24년 비둘기 등에서 뇌 부위를 떼어 내는(ablation) 실험, 소뇌 제거 → 균형·운동 협응 상실. 1824 *Recherches expérimentales sur les propriétés et les fonctions du système nerveux*
- 사진: Wikimedia Commons "Portrait de Pierre, Jean, Marie Flourens (1794-1867), (physiologiste), PH50650 (1 of 2).jpg", André-Adolphe-Eugène Disdéri, 1867년 이전, **CC0** (Paris Musées). 상반신만 잘라 rembg u2net_human_seg 누끼 → `flourens_cut.png`
- Allen Newell: 1974년 음성 인식 튜토리얼(1975 출판)에서 AI 시스템 분석에 ablation이라는 말을 씀 (Wikipedia "Ablation (artificial intelligence)"). 자유 라이선스 사진이 저해상도뿐이라 이름만 표기
