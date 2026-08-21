# Chapter 7. n-step 부트스트래핑, 적격흔적, 그리고 계획 (n-Step Bootstrapping, Eligibility Traces & Planning)

Chapter 5의 몬테카를로와 Chapter 6의 TD(0)는 언뜻 서로 다른 두 극단처럼
보인다 — MC는 에피소드 **끝까지**의 실제 보상을 목표로 쓰고, TD(0)는
**한 스텝**만 보고 나머지는 추정치로 대체한다. 이번 장은 이 둘이 사실
"몇 스텝을 실제로 볼 것인가"라는 하나의 다이얼 위의 양 끝일 뿐임을
보이고, 그 사이의 모든 지점을 아우르는 방법을 다룬다. 마지막으로는
경험뿐 아니라 학습된 모델도 함께 활용하는 계획(planning)까지 살펴본다.

이번 주는 세 개의 수업 블록으로 진행된다:

- [7.1 n-step TD: MC와 TD(0) 사이의 다이얼](chapter07/1.md)
- [7.2 적격흔적과 파라미터 튜닝 실습](chapter07/2.md)
- [7.3 계획과 학습: Dyna-Q](chapter07/3.md)
