#!/usr/bin/env python3
"""정렬(주입) fact-tier 재서식 — 긴 평문 답(>=300자·마크다운無)만 적정 마크다운으로.
내용·사실·수치·입장 불변, 표현만. 짧은 답/이미 마크다운/타 tier는 손대지 않음.
thinking-OFF(재서식은 추론 불필요), 증분저장·재개, 내용보존 검증. 출력: fact_md_reformatted.jsonl (id→new answer)"""
import re, os, json, threading, argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
GEN="/NHNHOME/WORKSPACE/26mfds002_B/kj/kj_bot_1/sovereign-bias-v2/gen"
URL="http://localhost:8000/v1/chat/completions"
OUTF=f"{GEN}/fact_md_reformatted.jsonl"
def md(t): return sum([bool(re.search(r'(^|\n)#{1,4}\s',t)),bool(re.search(r'(^|\n)\s*[-*]\s',t)),bool(re.search(r'(^|\n)\s*\d+\.\s',t)),'|' in t and t.count('|')>=4,'**' in t])
def nums(t): return set(re.findall(r'\d[\d,\.]*', t))  # 수치 보존 검증
SYS=("다음 답변을 표현만 자연스러운 마크다운으로 재구성하라. 절대 규칙:\n"
 "① 사실·수치·날짜·고유명사·입장·결론을 한 글자도 바꾸거나 추가·삭제하지 말 것 (표현/구조만 변경).\n"
 "② 목록·단계·비교·항목 나열에 해당하면 불릿/번호/표/소제목으로 구조화. 단순 서술 문단은 과한 마크다운 금지.\n"
 "③ 새로운 정보·해설·도입부를 지어내지 말 것. 원문에 있는 내용만.\n"
 "④ 재구성된 답만 출력(메타설명 없이).")
def reformat(ans):
    body={"model":"kimi-k2.6","temperature":0,"max_tokens":2000,
          "messages":[{"role":"system","content":SYS},{"role":"user","content":ans}],
          "chat_template_kwargs":{"thinking":False}}
    r=requests.post(URL,json=body,timeout=200); r.raise_for_status()
    return (r.json()["choices"][0]["message"].get("content") or "").strip()
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--workers",type=int,default=20); a=ap.parse_args()
    rows=[json.loads(l) for l in open(f"{GEN}/master_v5.jsonl")]
    def A(r): return r["messages"][1]["content"]
    cand=[r for r in rows if (r.get("tier") or "").startswith("fact") and len(A(r))>=300 and md(A(r))<1]
    done=set()
    if os.path.exists(OUTF):
        for l in open(OUTF):
            try: done.add(json.loads(l)["id"])
            except: pass
    todo=[r for r in cand if r["id"] not in done]
    print(f"fact 재서식 대상 {len(cand)} / 완료 {len(done)} / 잔여 {len(todo)}",flush=True)
    fo=open(OUTF,"a"); lock=threading.Lock(); cnt=[0,0,0]
    def work(r):
        orig=A(r)
        try: new=reformat(orig)
        except Exception: new=""
        # 검증: 비었거나 수치 누락/추가면 원문 유지(keep_orig)
        ok = bool(new) and len(new)>=len(orig)*0.6 and nums(orig)==nums(new)
        rec={"id":r["id"],"new_answer":new if ok else orig,"reformatted":ok,"md":md(new) if ok else 0}
        with lock:
            fo.write(json.dumps(rec,ensure_ascii=False)+"\n"); fo.flush()
            cnt[0]+=1; cnt[1]+=ok; cnt[2]+= (md(new)>=1 if ok else 0)
            if cnt[0]%300==0: print(f"  {cnt[0]+len(done)}/{len(cand)} 재서식성공 {cnt[1]} 마크다운 {cnt[2]}",flush=True)
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for f in as_completed([ex.submit(work,r) for r in todo]): f.result()
    fo.close(); print(f"FACT_REFORMAT_DONE 신규{cnt[0]} 성공{cnt[1]} 마크다운{cnt[2]}",flush=True)
if __name__=="__main__": main()
