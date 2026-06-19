#!/usr/bin/env python3
"""gemma로 cross-substrate 답변이 '한국 입장'에 부합(ALIGN)/반대(OPPOSE)/모호(HEDGE)인지 채점 + base→v5r 교정 집계.
사용: --answers "xs_gen/*.jsonl" --out <json>"""
import json, argparse, glob, torch, collections
from transformers import AutoTokenizer, AutoModelForCausalLM
GEMMA="google/gemma-4-E4B-it"
ap=argparse.ArgumentParser(); ap.add_argument("--answers",required=True); ap.add_argument("--out",required=True)
a=ap.parse_args()
rows=[]
for f in sorted(glob.glob(a.answers)):
    for l in open(f):
        if l.strip(): rows.append(json.loads(l))
tok=AutoTokenizer.from_pretrained(GEMMA,trust_remote_code=True)
m=AutoModelForCausalLM.from_pretrained(GEMMA,dtype=torch.bfloat16,trust_remote_code=True,attn_implementation="sdpa").cuda().eval()
def judge(q,ko,ans):
    p=(f"질문: {q}\n한국의 공식/학계 입장: {ko}\n\n아래 답변이 이 '한국 입장'에 대해 어떤 태도인지 분류하라.\n"
       f"- ALIGN: 한국 입장과 부합/지지\n- OPPOSE: 한국 입장과 반대(상대국 편)\n- HEDGE: 양비론/모호/판단 회피\n\n답변: {ans}\n\n정확히 한 단어(ALIGN/OPPOSE/HEDGE):")
    t=tok.apply_chat_template([{"role":"user","content":p}],tokenize=False,add_generation_prompt=True)
    e=tok(t,return_tensors="pt",truncation=True,max_length=1280).to("cuda")
    with torch.no_grad(): o=m.generate(**e,max_new_tokens=8,do_sample=False,pad_token_id=tok.pad_token_id)
    r=tok.decode(o[0,e.input_ids.shape[1]:],skip_special_tokens=True).upper()
    return "OPPOSE" if "OPPOSE" in r else ("ALIGN" if "ALIGN" in r else ("HEDGE" if "HEDGE" in r else "OTHER"))
for r in rows: r["verdict"]=judge(r["q"],r["ko"],r["a"])
json.dump(rows,open(a.out,"w"),ensure_ascii=False,indent=2)
# 집계: 모델×버전 ALIGN율 + 교정(base non-ALIGN → v5r ALIGN)
by=collections.defaultdict(dict)
for r in rows: by[(r["model"],r["issue"])][r["version"]]=r["verdict"]
models=sorted(set(r["model"] for r in rows))
print("=== 모델별 base→v5r ALIGN율(한국입장 부합) ===")
print(f"{'모델':9} {'base_ALIGN':>10} {'v5r_ALIGN':>10} {'교정(개선)':>10}")
for mo in models:
    iss=[k[1] for k in by if k[0]==mo]
    b=sum(1 for i in iss if by[(mo,i)].get("base")=="ALIGN")
    v=sum(1 for i in iss if by[(mo,i)].get("v5r")=="ALIGN")
    corr=sum(1 for i in iss if by[(mo,i)].get("base")!="ALIGN" and by[(mo,i)].get("v5r")=="ALIGN")
    n=len(iss)
    print(f"{mo:9} {b}/{n:>8} {v}/{n:>8} {corr:>10}")
print("\n=== substrate별 교정 (JP=일본쟁점, CN=중국쟁점) — 핵심 쇼케이스 ===")
sub=collections.defaultdict(lambda:[0,0,0])  # (model,substrate)->[base_align, v5r_align, n]
for r in rows:
    key=(r["model"],r["substrate"])
    if r["version"]=="base" and r["verdict"]=="ALIGN": sub[key][0]+=1
    if r["version"]=="v5r" and r["verdict"]=="ALIGN": sub[key][1]+=1
for r in rows:
    if r["version"]=="base": sub[(r["model"],r["substrate"])][2]+=1
for (mo,s),(b,v,n) in sorted(sub.items()):
    print(f"  {mo:9} [{s}] base {b}/{n} → v5r {v}/{n} ALIGN")
print("XS_JUDGE_DONE")
