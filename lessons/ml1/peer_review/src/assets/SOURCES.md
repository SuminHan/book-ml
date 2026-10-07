# 출처 — AI 논문 리뷰(피어 리뷰) 쇼츠

- 데이터: OpenReview 공개 리뷰 (openreview.net API, 2026-10-07 수집). ICLR 2018–2026, NeurIPS 2021–2025, ICML 2025–2026
  - 수집·집계 코드: private 저장소 SuminHan/ksa-review (`devstack/fetch_reviews.py`, `devstack/build_review_insights.py`)
  - 영상 수치는 `public/review-insights/summary.json` 에서 가져와 `src/data.py` 에서 assert
- 약점 순위: ICLR 2026 리뷰 'weaknesses' 칸 키워드(정규식) 매칭 비율 — 대략값 (영상·캡션에 표기)
  - 3~5위(Theory 22.4 · Clarity 22.3 · Novelty 22.1%)는 차이가 작아 순위 없이 '비슷'으로 표시
- s2 리뷰어 점수(6·4·8·4)는 예시 (화면에 '점수는 예시')
- 슬라이드 원본 없음 (데이터 분석 편)
- 이모지: Twemoji (CC BY 4.0) — 📄 📝 🧑‍🔬(피부색 1·3·5·2) ✅ ❌ ✍️
