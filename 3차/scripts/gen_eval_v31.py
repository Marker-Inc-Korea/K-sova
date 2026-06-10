#!/usr/bin/env python3
"""v3.1 평가 생성 — 임의 평가셋들에 base + lora(v3.1) 답 생성. HF 로드(8B). thinking strip.
입력 정규화: question/instruction/messages 어느 포맷이든 instruction+reference 추출.
출력: <outdir>/<tag>_<model>_<cond>.jsonl  ({qid,tier,domain,instruction,reference,generated})
"""
import os, json, argparse, gc
from pathlib import Path
import torch
from gen_test_fixed import run
from generate_v2 import load
GEN=Path("/NHNHOME/WORKSPACE/26mfds002_B/kj/kj_bot_1/sovereign-bias-v2/gen")

def norm(r):
    q=r.get("question") or r.get("instruction") or (r["messages"][0]["content"] if r.get("messages") else "")
    ref=r.get("reference") or r.get("output") or (r["messages"][1]["content"] if r.get("messages") and len(r["messages"])>1 else "")
    return {"qid":r.get("qid"),"tier":r.get("tier","fact"),"domain":r.get("domain"),"bucket":r.get("bucket"),
            "instruction":q,"messages":[{"role":"user","content":q},{"role":"assistant","content":ref}]}
def save(o,p): Path(p).write_text("\n".join(json.dumps(r,ensure_ascii=False) for r in o),encoding="utf-8")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model_key",required=True); ap.add_argument("--adapter",required=True)
    ap.add_argument("--inputs",nargs="+",required=True)  # "tag:path"
    ap.add_argument("--outdir",default=str(GEN/"runs/eval_v31"))
    ap.add_argument("--limit",type=int,default=0)
    a=ap.parse_args(); od=Path(a.outdir); od.mkdir(parents=True,exist_ok=True); mk=a.model_key
    sets={}
    for s in a.inputs:
        tag,p=s.split(":",1)
        d=[norm(json.loads(l)) for l in open(p,encoding="utf-8") if l.strip()]
        if a.limit: d=d[:a.limit]
        sets[tag]=d
    print(f"[{mk}] sets={ {t:len(d) for t,d in sets.items()} } base 로드")
    model,tok=load(mk,None)
    for tag,d in sets.items(): save(run(model,tok,d,False,""), od/f"{tag}_{mk}_base.jsonl")
    del model; gc.collect(); torch.cuda.empty_cache()
    print(f"[{mk}] lora 로드"); model,tok=load(mk,a.adapter)
    for tag,d in sets.items(): save(run(model,tok,d,False,""), od/f"{tag}_{mk}_lora.jsonl")
    print(f"[{mk}] done → {od}")
if __name__=="__main__": main()
