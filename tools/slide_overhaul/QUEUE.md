# 이미지 보강 큐 (오케스트레이터용 상태 파일)
절차(덱 하나 끝날 때마다): bash pipeline/build_deck.sh <ml1|ml2> <weekNN>  -> out/ PDF 2개 SendUserFile(proactive)
 -> python3 pipeline/gallery_data.py $S <완료 덱들...> 후 gallery/image-review.html 재생성·Artifact 재게시
 -> 다음 대기 덱에 Agent(sonnet, general-purpose, prompt=enrich_prompt.md 변수) 발사. 동시 3개.
완료: ml1-week02 ml1-week03 ml1-week04  ml1-week06  ml1-week05  ml1-week07  ml1-week08  ml1-week09  ml1-week10  ml1-week11  ml1-week12  ml1-week13  ml1-week14  ml1-week15  ml1-week16  ml1-week17  ml2-week01  ml2-week02  ml2-week03  ml2-week04  ml2-week05  ml2-week06  ml2-week07
진행중: ml2-week08 ml2-week09 ml2-week10
대기: ml2-week11..16
주의: 원본 저장소 수정·커밋 금지. ml 사용자 GPU 프로세스 건드리지 말 것. VL 서버 pid 4030986 (port 8000).
기록: 02:30경 세션 사용한도(429) 도달 -> 5/7/8주차 중단, 06:10 리셋. 06:21 5·7주차 이어하기 + 8주차 재발사.
 한도 걸리면: 알림 status=failed + "session limit · resets HH:MM" -> 리셋 이후 크론 점검 때 '이어서 하기' 프롬프트로 재발사.
기록: 10:1x경 두 번째 한도(429) -> 12/13/14주차 중단, 11:20 리셋. 11:2x 재발사. 11주차 노트 \verb 버그 수정(노트 안 \verb->\texttt).
재조립 수정: 중첩목록-먼저 버그 -> \item[] 삽입, ml2-week03..16 rethin 완료
기록: 15:1x경 세 번째 한도 -> ML2 2/3/4 중단, 16:20 리셋, 16:21 재발사.
기록: 20:xx 네 번째 한도 -> ML2 8/9/10 중단, 21:20 리셋, 21:21 재발사.
== 중단 (사용자 "멈춰") ==
주간 한도(429 weekly, 리셋 9/29 22:00) 도달. 예비 크론 삭제. 재개 안 함.
미완료: ml2-week08(새 프레임 6·그림 8, 미검증), ml2-week09(그림 6), ml2-week10(없음) / 미착수: ml2-week11..16
재개 시: 8/9/10 은 '이어서 하기' 프롬프트, 11~16 은 기본 프롬프트. 빌드: pipeline/build_deck.sh
