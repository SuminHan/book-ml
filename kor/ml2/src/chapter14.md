# Chapter 14. 고급 시뮬레이션: MuJoCo와 Isaac Sim (Advanced Simulation: MuJoCo & Isaac Sim)

Chapter 13에서 쓴 Gymnasium의 기본 환경들은 물리를 간단한 근사식으로만
계산한다 — 학습 속도는 빠르지만, 접촉(contact)이나 마찰처럼 로봇
제어에서 중요한 물리 현상은 거칠게만 흉내 낸다. 실전 로봇 연구에서는
훨씬 정교한 물리엔진을 쓴다. 이번 장은 그중 두 가지 — CPU에서도 도는
정밀한 물리엔진 **MuJoCo**와, GPU로 수천 개의 시뮬레이션을 동시에 돌리는
**NVIDIA Isaac Sim** — 을 소개하고, 언제 무엇을 쓸지 정리한다.

이번 주는 세 개의 수업 블록으로 진행된다:

- [14.1 MuJoCo와 정책 학습 실습](chapter14/1.md)
- [14.2 NVIDIA Isaac Sim과 GPU 가속의 원리](chapter14/2.md)
- [14.3 어떤 도구를 언제 쓰는가](chapter14/3.md)
