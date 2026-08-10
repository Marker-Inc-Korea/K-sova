"""v3 평가 생성 — 모델당 base 1회 + lora 1회 로드로 전 조건·전 파일 처리.
test300: base/prompt/prompt_strong/lora · 범용: base/lora · 프로브: base/lora.
출력: runs/gen_v3/{set}_{model}_{cond}.jsonl  (set: gen=test300, gg=general, kp=probe)
"""
import os, json, argparse, gc
from pathlib import Path
import torch
from generate_v2 import load, DEFAULT_SYS_PROMPT

GEN=Path("/NHNHOME/WORKSPACE/26mfds002_B/kj/kj_bot_1/sovereign-bias-v2/gen")
GENERAL="/NHNHOME/WORKSPACE/26mfds002_B/kj/kj_bot_1/sovereign-bias/q1q2_lora/data/general_eval_50.jsonl"

def rows(p): return [json.loads(l) for l in open(p,encoding="utf-8") if l.strip()]

def run(model,tok,data,use_sys,sysp,max_new,bs=8):
    out=[]
    for i in range(0,len(data),bs):
        batch=data[i:i+bs]; prompts=[]
        for r in batch:
            q=r.get("instruction") or r["messages"][0]["content"]
            msgs=([{"role":"system","content":sysp}] if use_sys else [])+[{"role":"user","content":q}]
            try: text=tok.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True)
            except Exception: text=(f"<|system|>\n{sysp}\n" if use_sys else "")+f"<|user|>\n{q}\n<|assistant|>\n"
            prompts.append(text)
        enc=tok(prompts,return_tensors="pt",padding=True,truncation=True,max_length=1024).to("cuda")
        with torch.no_grad():
            o=model.generate(**enc,max_new_tokens=max_new,do_sample=False,temperature=None,top_p=None,pad_token_id=tok.pad_token_id)
        il=enc.input_ids.shape[1]
        for j,r in enumerate(batch):
            ans=tok.decode(o[j,il:],skip_special_tokens=True).strip()
            q=r.get("instruction") or r["messages"][0]["content"]
            ref=r.get("output") or (r["messages"][1]["content"] if r.get("messages") and len(r["messages"])>1 else "")
            out.append({"qid":r.get("qid"),"category":r.get("category"),"tier":r.get("tier"),"domain":r.get("domain"),
                        "kind":r.get("kind"),"grade":r.get("grade"),"answer_keywords":r.get("answer_keywords"),
                        "instruction":q,"reference":ref,"generated":ans})
    return out

def save(o,p): Path(p).write_text("\n".join(json.dumps(r,ensure_ascii=False) for r in o),encoding="utf-8")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model_key",required=True); ap.add_argument("--adapter",required=True)
    ap.add_argument("--strong_file",required=True); ap.add_argument("--outdir",default=str(GEN/"runs/gen_v3"))
    a=ap.parse_args(); od=Path(a.outdir); od.mkdir(parents=True,exist_ok=True)
    mk=a.model_key; strong=Path(a.strong_file).read_text(encoding="utf-8").strip()
    test=rows(GEN/"test_v3_eval300.jsonl"); gg=rows(GENERAL); kp=rows(GEN/"knowledge_probe.jsonl")

    print(f"[{mk}] base 로드"); model,tok=load(mk,None)
    save(run(model,tok,test,False,"",600), od/f"gen_{mk}_base.jsonl")
    save(run(model,tok,test,True,DEFAULT_SYS_PROMPT,600), od/f"gen_{mk}_prompt.jsonl")
    save(run(model,tok,test,True,strong,600), od/f"gen_{mk}_prompt_strong.jsonl")
    save(run(model,tok,gg,False,"",400), od/f"gg_{mk}_base.jsonl")
    save(run(model,tok,kp,False,"",80),  od/f"kp_{mk}_base.jsonl")
    print(f"[{mk}] base 완료")
    del model; gc.collect(); torch.cuda.empty_cache()

    print(f"[{mk}] lora 로드 {a.adapter}"); model,tok=load(mk,a.adapter)
    save(run(model,tok,test,False,"",600), od/f"gen_{mk}_lora.jsonl")
    save(run(model,tok,gg,False,"",400), od/f"gg_{mk}_lora.jsonl")
    save(run(model,tok,kp,False,"",80),  od/f"kp_{mk}_lora.jsonl")
    print(f"[{mk}] lora 완료 → {od}")

if __name__=="__main__": main()
