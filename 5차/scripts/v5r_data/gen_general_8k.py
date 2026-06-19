#!/usr/bin/env python3
"""K-SOVA 범용 8k 본생성. Hermes 자연질문 ~6.8k(범주 균형) + 영어축 ~1.2k(Kimi 생성·dedup).
답변은 Kimi 재생성(네이티브·적정 마크다운·정답추론·언어이행). 증분저장(재개 가능)+품질게이트.
HF_TOKEN env 필요. 사용: python gen_general_8k.py [--workers 24]"""
import re, os, json, random, threading, argparse, hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
random.seed(11)
GEN="/NHNHOME/WORKSPACE/26mfds002_B/kj/kj_bot_1/sovereign-bias-v2/gen"
URL="http://localhost:8000/v1/chat/completions"
QFILE=f"{GEN}/general_8k_questions.jsonl"   # 확정된 질문셋(재현)
OUTF=f"{GEN}/general_8k.jsonl"               # 답변까지 완성분(증분)

def kimi(messages,temp,maxtok,think=True):
    body={"model":"kimi-k2.6","temperature":temp,"max_tokens":maxtok,"messages":messages}
    if not think: body["chat_template_kwargs"]={"thinking":False}
    r=requests.post(URL,json=body,timeout=400); r.raise_for_status()
    return (r.json()["choices"][0]["message"].get("content") or "").strip()
def hid(q): return hashlib.md5(q.encode()).hexdigest()[:12]
def md(t): return sum([bool(re.search(r'(^|\n)#{1,4}\s',t)),bool(re.search(r'(^|\n)\s*[-*]\s',t)),bool(re.search(r'(^|\n)\s*\d+\.\s',t)),'```' in t,'|' in t and t.count('|')>=4,'**' in t])
def lang(t):
    ko=len(re.findall(r'[가-힣]',t)); en=len(re.findall(r'[A-Za-z]',t)); return "en" if en>ko else "ko"
def toks(s): return set(re.findall(r'[가-힣]{2,}|[a-z0-9]+',(s or '').lower()))

# ---------- 1) 질문셋 구성 (있으면 재사용) ----------
def build_questions():
    if os.path.exists(QFILE):
        return [json.loads(l) for l in open(QFILE)]
    from datasets import load_dataset
    ds=load_dataset("MarkrAI/KOpen-HQ-Hermes-2.5-60K", split="train", token=os.environ["HF_TOKEN"])
    def q_of(r): return ((r.get("input") or "")+" "+(r.get("instruction") or "")).strip()
    JUNK=re.compile(r'(5살|다섯 살|문장을 생성|문장:|문제를 생성|질문을 생성|다음 데이터|다음 문장.*(참|거짓)|옵션[:은]|보기에서|다음 중.*고르|선택하세요|감성|긍정.*부정|단계별로 정당화|단계별로 생각|translate the following|generate a (sentence|question)|review:|다중 선택|object:|triple|충실하게 대답)',re.I)
    CAT={"코드수학":r'(코드|함수|계산|풀이|구현|알고리즘|수학|프로그램|방정식)',"설명개념":r'(설명|무엇|왜|어떻게|원리|개념|정의)',
         "작문":r'(작성|작문|써줘|편지|이메일|글|시나리오)',"비교분석":r'(비교|차이|장단점|분석)',"요약":r'(요약|정리해)'}
    def cat_of(q):
        for k,p in CAT.items():
            if re.search(p,q): return k
        return "기타"
    buckets={}
    for i in range(len(ds)):
        q=q_of(ds[i])
        if JUNK.search(q) or not(20<len(q)<800): continue
        buckets.setdefault(cat_of(q),[]).append(q)
    quota={"설명개념":2400,"코드수학":1800,"작문":1200,"비교분석":700,"요약":500,"기타":200}
    Q=[]
    for c,n in quota.items():
        pool=buckets.get(c,[]); random.shuffle(pool)
        for q in pool[:n]: Q.append({"category":c,"instruction":q,"axis":"hermes"})
    # --- 영어축 ~1200 (Kimi 생성, dedup) ---
    TYPES=["한국어 문장을 영어로 번역하라는 지시","영어 문장을 한국어로 번역하라는 지시",
           "특정 개념을 영어로 설명해달라는 지시(Explain ... in English)","영어 텍스트를 영어로 요약(Summarize in English)",
           "어색한 영어 문장을 다듬거나 문법 교정해달라는 지시","영어로 글(이메일·소개·답장)을 써달라는 지시"]
    DOM=["과학기술","경제경영","일상생활","교육","건강","문화예술","역사","IT/프로그래밍","여행","환경"]
    specs=[(t,d) for t in TYPES for d in DOM]  # 60 batch, 각 20개 → ~1200
    def genbatch(t,d):
        try:
            txt=kimi([{"role":"user","content":f"'{d}' 분야에서 '{t}'에 해당하는, 서로 다른 사용자 지시문 20개를 만들어라. 번역 요청이면 번역할 실제 문장도 포함. 설명·번호 없이 한 줄에 하나씩 20줄만 출력."}],0.9,1500,think=False)
            return [(t,re.sub(r'^[\s\d\.\-\)*]+','',l).strip()) for l in txt.split("\n") if len(re.sub(r'^[\s\d\.\-\)*]+','',l).strip())>=10]
        except Exception: return []
    raw=[]
    with ThreadPoolExecutor(max_workers=16) as ex:
        for r in as_completed([ex.submit(genbatch,t,d) for t,d in specs]): raw+=r.result()
    en=[]; seen=[]
    for t,s in raw:
        tk=toks(s)
        if any(len(tk&p)/max(len(tk|p),1)>=0.7 for p in seen[-300:]): continue
        seen.append(tk); en.append({"category":t[:6],"instruction":s,"axis":"english"})
    random.shuffle(en); en=en[:1200]
    Q+=en; random.shuffle(Q)
    for r in Q: r["qid"]="g_"+hid(r["instruction"])
    with open(QFILE,"w") as f:
        for r in Q: f.write(json.dumps(r,ensure_ascii=False)+"\n")
    return Q

# ---------- 2) 답변 생성 (증분·게이트) ----------
SYS=("다음 지시에 고품질 답변을 작성하라. 규칙:\n"
 "① 요청한 언어로 답하라(한국어 기본, 영어로 요청하면 영어로).\n"
 "② 내용이 다단계·목록·비교·코드·표에 해당하면 마크다운(헤더·불릿·표·코드블록)으로 구조화하되, 짧고 단순한 답은 자연스러운 산문으로 — 획일적 마크다운 금지.\n"
 "③ 추론·수학·코드는 정확한 단계와 정답을 제시.\n"
 "④ '답변을 구성하겠습니다' 같은 메타설명 없이 답 자체만.\n⑤ 완결성 있게, 적정 길이로.")
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--workers",type=int,default=24); a=ap.parse_args()
    Q=build_questions()
    print(f"질문셋 {len(Q)} (hermes {sum(r['axis']=='hermes' for r in Q)} / english {sum(r['axis']=='english' for r in Q)})",flush=True)
    done=set()
    if os.path.exists(OUTF):
        for l in open(OUTF):
            try: done.add(json.loads(l)["qid"])
            except: pass
    todo=[r for r in Q if r["qid"] not in done]
    print(f"완료 {len(done)} / 잔여 {len(todo)}",flush=True)
    fo=open(OUTF,"a"); lock=threading.Lock(); cnt=[0,0]
    def leak(t): return bool(re.search(r'(답변을 구성|사용자가.*묻고|작성하겠습니다|핵심 포인트:)',t))
    def work(r):
        q=r["instruction"]
        try: ans=kimi([{"role":"system","content":SYS},{"role":"user","content":q}],0.3,4000)
        except Exception as e: ans=""
        # 게이트
        keep = bool(ans) and len(ans)>=50 and not leak(ans)
        # 영어축 언어 일치(영어 요청인데 한국어면 드롭)
        if r["axis"]=="english" and re.search(r'(in English|영어로|Explain|Write|Summarize|Correct|Translate to natural Korean)',q):
            if re.search(r'한국어로 번역|Translate to natural Korean',q): pass
            elif lang(ans)!="en": keep=False
        rec={**r,"answer":ans,"md":md(ans),"len":len(ans),"keep":keep}
        with lock:
            fo.write(json.dumps(rec,ensure_ascii=False)+"\n"); fo.flush()
            cnt[0]+=1; cnt[1]+= keep
            if cnt[0]%200==0: print(f"  {cnt[0]+len(done)}/{len(Q)}  keep {cnt[1]}",flush=True)
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        fs=[ex.submit(work,r) for r in todo]
        for f in as_completed(fs): f.result()
    fo.close(); print(f"GEN8K_DONE 신규 {cnt[0]} keep {cnt[1]}",flush=True)
if __name__=="__main__": main()
