"""교정 재생성 — test300, 4조건(base/prompt/prompt_strong/lora). 아티팩트 수정:
 - max_new_tokens 1024(절단 방지), repetition_penalty 1.2(디제너레이션 억제)
 - prompt_strong: 한국어 직답 지시 추가(영어 Thinking Process 절단 방지)
 - 출력 후처리: <think>블록·영어 Thinking Process 선행블록 제거(실제 답만 채점)
출력: runs/gen_v3b/gen_<model>_<cond>.jsonl
"""
import os, json, argparse, gc, re
from pathlib import Path
import torch
from generate_v2 import load, DEFAULT_SYS_PROMPT
GEN=Path("/NHNHOME/WORKSPACE/26mfds002_B/kj/kj_bot_1/sovereign-bias-v2/gen")

def rows(p): return [json.loads(l) for l in open(p,encoding="utf-8") if l.strip()]

def strip_think(t):
    t=re.sub(r'<think>.*?</think>\s*','',t,flags=re.DOTALL)
    t=re.sub(r'</?think>','',t)
    # 영어 Thinking Process/사고 선행블록 → 첫 한글충분 라인부터
    if re.match(r'\s*(Thinking Process|Let me|First,|Okay|I need|The user|Step 1|Analysis)',t):
        lines=t.split('\n')
        for i,l in enumerate(lines):
            if len(re.findall(r'[가-힣]',l))>=5:
                t='\n'.join(lines[i:]); break
    return t.strip()

def run(model,tok,data,use_sys,sysp,max_new=1024,bs=8):
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
            o=model.generate(**enc,max_new_tokens=max_new,do_sample=False,temperature=None,top_p=None,
                             repetition_penalty=1.2,pad_token_id=tok.pad_token_id)
        il=enc.input_ids.shape[1]
        for j,r in enumerate(batch):
            raw=tok.decode(o[j,il:],skip_special_tokens=True).strip()
            ans=strip_think(raw)
            q=r.get("instruction") or r["messages"][0]["content"]
            ref=r["messages"][1]["content"] if r.get("messages") and len(r["messages"])>1 else ""
            out.append({"qid":r.get("qid"),"tier":r.get("tier"),"domain":r.get("domain"),
                        "instruction":q,"reference":ref,"generated":ans,"raw_len":len(raw),"stripped":raw!=ans})
    return out

def save(o,p): Path(p).write_text("\n".join(json.dumps(r,ensure_ascii=False) for r in o),encoding="utf-8")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model_key",required=True); ap.add_argument("--adapter",required=True)
    ap.add_argument("--strong_file",required=True); ap.add_argument("--outdir",default=str(GEN/"runs/gen_v3b"))
    a=ap.parse_args(); od=Path(a.outdir); od.mkdir(parents=True,exist_ok=True)
    mk=a.model_key
    strong=Path(a.strong_file).read_text(encoding="utf-8").strip()+"\n\n반드시 한국어로, 별도의 사고 과정(Thinking) 없이 최종 답변만 작성하세요."
    test=rows(GEN/"test_v3_eval300.jsonl")
    print(f"[{mk}] base 로드"); model,tok=load(mk,None)
    save(run(model,tok,test,False,""),            od/f"gen_{mk}_base.jsonl")
    save(run(model,tok,test,True,DEFAULT_SYS_PROMPT),od/f"gen_{mk}_prompt.jsonl")
    save(run(model,tok,test,True,strong),         od/f"gen_{mk}_prompt_strong.jsonl")
    del model; gc.collect(); torch.cuda.empty_cache()
    print(f"[{mk}] lora 로드"); model,tok=load(mk,a.adapter)
    save(run(model,tok,test,False,""),            od/f"gen_{mk}_lora.jsonl")
    print(f"[{mk}] 완료 → {od}")

if __name__=="__main__": main()
