#!/bin/bash
# Emit one line whenever a deck reaches a terminal state, plus a done marker.
M=/home/smhan/book-ml/slides/gen/manifest.json
seen=/tmp/claude-1002/-home-smhan/029e91ef-1260-463c-974d-c55f55e786a0/scratchpad/.watch_seen
: > "$seen"
while true; do
  python3 - "$M" "$seen" <<'PY'
import json,sys
man,seen=sys.argv[1],sys.argv[2]
done=set(open(seen).read().split())
d=json.load(open(man,encoding='utf-8'))
new=[]
for x in d['decks']:
    if x['status'] in ('done','failed','needs_review') and x['id'] not in done:
        new.append(x); done.add(x['id'])
for x in new:
    print(f"{x['id']}  {x['status'].upper()}  {x['frames']}프레임 {x['pages']}쪽  "
          f"{(x['duration_s'] or 0)//60}분  {x['title'][:44]}")
nd=sum(1 for x in d['decks'] if x['status']=='done')
nt=len(d['decks'])
nrev=sum(1 for x in d['decks'] if x['status'] in ('failed','needs_review'))
if new:
    print(f"  -> 진행 {nd}/{nt} 완료, {nrev} 검토/실패, "
          f"{nt-nd-nrev} 남음")
if nd+nrev>=nt:
    print("ALL_DONE")
open(seen,'w').write('\n'.join(sorted(done)))
PY
  grep -q ALL_DONE "$seen" 2>/dev/null && break
  # stop if scheduler process gone AND nothing running
  pgrep -f run_scheduler.sh >/dev/null || { python3 -c "
import json;d=json.load(open('$M'))
import sys; sys.exit(0 if any(x['status']=='running' for x in d['decks']) else 1)" || { echo 'SCHEDULER_STOPPED'; break; }; }
  sleep 180
done
