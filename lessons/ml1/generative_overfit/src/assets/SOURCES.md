# 출처
- 개념: ~/book-ml/slides/kor/ml1/week03/pages/p02.tex · p03.tex · p04.tex (판별 P(y|x) vs 생성 P(x|y)), week08/pages/p50.tex (kNN = 전량 암기)
- 데이터: **예시**(손으로 만든 몸무게, kg). 고양이 12마리 [3.0…8.0], 강아지 8마리 [10.5…17.5], 숨겨 둔 새 고양이 5.1kg, 실제 통계 아님
- 계산: 가우시안 커널 밀도 추정(폭 0.5kg 적당 / 0.05kg 뾰족), 표본 = 학습 고양이 하나 + 폭만큼 잡음. src/data.py assert: 뾰족 표본 5개 모두 학습 고양이와 0.05kg 안, 새 고양이 높이 2e-8 vs 0.17, 4.3kg 고양이 5번 중복이면 뽑은 5마리 모두 그 근처(±0.6kg)
- 사례: Carlini et al., "Extracting Training Data from Large Language Models", USENIX Security 2021 (GPT-2에서 이름·전화번호·이메일 등 학습 문장 추출). 화면의 "이름 ○○○ / 전화 010-…" 카드는 가린 예시 그림
- 중복과 외우기: Kandpal et al., "Deduplicating Training Data Mitigates Privacy Risks in Language Models", ICML 2022; Lee et al., "Deduplicating Training Data Makes Language Models Better", ACL 2022
- 이모지: Twemoji (jdecked/twemoji, 그래픽 CC BY 4.0) — lessons/_tools/assets/emoji/ 캐시, mbase.emoji()
