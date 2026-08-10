#!/bin/bash
GEN=/NHNHOME/WORKSPACE/26mfds002_B/kj/kj_bot_1/sovereign-bias-v2/gen
VPY=/NHNHOME/WORKSPACE/26mfds002_B/dk/venv-vllm-020/bin/python3
cd $GEN
echo "[$(date)] 1) 3개 재생성 완료 대기"
while true; do
  f=$(wc -l < closedbook_full_ans.jsonl 2>/dev/null||echo 0)
  c=$(wc -l < closedbook_core_ans.jsonl 2>/dev/null||echo 0)
  cf=$(wc -l < closedbook_cf_ans.jsonl 2>/dev/null||echo 0)
  # 재생성 프로세스가 다 죽고 카운트 충족
  alive=$(ps aux|grep '[s]tage3_closedbook'|wc -l)
  [ "$f" -ge 9624 ] && [ "$c" -ge 757 ] && [ "$cf" -ge 1355 ] && [ "$alive" -eq 0 ] && break
  sleep 30
done
echo "[$(date)] 재생성 완료 full=$f core=$c cf=$cf"
echo "[$(date)] 2) 합본 + 인용정리"
$VPY - <<'PY'
import json,re
PAT=[r'근거\s*자료에\s*따르면[,]?\s*',r'근거:\s*',r'자료에\s*따르면[,]?\s*',r'자료에\s*의하면[,]?\s*',r'자료에서는?[,]?\s*',r'제공된\s*자료에\s*(?:따르면|의하면)[,]?\s*',r'\[근거\s*자료\][^\n]*\n']
out=[]
for f,src in [('closedbook_full_ans.jsonl','full'),('closedbook_core_ans.jsonl','core'),('closedbook_cf_ans.jsonl','cf')]:
    for l in open(f,encoding='utf-8'):
        if not l.strip(): continue
        r=json.loads(l)
        if not r.get('messages'): continue
        a=r['messages'][1]['content']
        for p in PAT: a=re.sub(p,'',a)
        a=re.sub(r'[ \t]+',' ',a).strip(); r['messages'][1]['content']=a
        t=r.get('tier','fact'); r['tier']=t.split('|')[0] if isinstance(t,str) else 'fact'
        r['qsource']=src; out.append(r)
open('closedbook_all_ans.jsonl','w',encoding='utf-8').write('\n'.join(json.dumps(x,ensure_ascii=False) for x in out))
print('합본',len(out))
PY
echo "[$(date)] 3) QA judge"
$VPY judge_closedbook_qa.py --ans closedbook_all_ans.jsonl --out closedbook_judged.jsonl --concurrency 48
echo "[$(date)] ALL DONE"
