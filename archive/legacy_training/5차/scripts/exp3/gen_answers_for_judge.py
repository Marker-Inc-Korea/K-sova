#!/usr/bin/env python3
"""체크포인트에서 100프로브 답 생성→저장(gemma 스탠스 채점용). 사용: --ckpt <dir> --tag <cond_step> --probe <100> --out <jsonl>"""
import json, argparse, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
ap=argparse.ArgumentParser()
ap.add_argument("--ckpt",required=True); ap.add_argument("--tag",required=True)
ap.add_argument("--probe",required=True); ap.add_argument("--out",required=True)
a=ap.parse_args()
rows=[json.loads(l) for l in open(a.probe) if l.strip()]
tok=AutoTokenizer.from_pretrained(a.ckpt,trust_remote_code=True)
m=AutoModelForCausalLM.from_pretrained(a.ckpt,dtype=torch.bfloat16,trust_remote_code=True,attn_implementation="sdpa").cuda().eval()
out=[]
for r in rows:
    t=tok.apply_chat_template([{"role":"user","content":r["instruction"]}],tokenize=False,add_generation_prompt=True)
    e=tok(t,return_tensors="pt",truncation=True,max_length=1024).to("cuda")
    with torch.no_grad(): o=m.generate(**e,max_new_tokens=150,do_sample=False,pad_token_id=tok.pad_token_id)
    ans=tok.decode(o[0,e.input_ids.shape[1]:],skip_special_tokens=True).strip()
    out.append({"tag":a.tag,"qid":r["qid"],"issue":r.get("issue","?"),"q":r["instruction"],"a":ans})
with open(a.out,"w") as f:
    for x in out: f.write(json.dumps(x,ensure_ascii=False)+"\n")
print(f"GEN_DONE {a.tag} {len(out)}건 → {a.out}",flush=True)
