#!/usr/bin/env python3
"""Exp3 Phase2 FAITHFUL — 웹 검증 스펙대로 정정. (LoRA 유지; full-FT 전환은 별도 한계로 문서화)
RepNoise: 손실=stability + beta_noise(0.001)*MMD(N(0,I), median-bandwidth) - alpha_ascent(1.0)*log(harmful_loss+1) [unbounded log-ascent]
Vaccine: single-step, GLOBAL grad-norm, rho=2, attention-sublayer 출력만 섭동
TAR: AdamW inner(논문 LR), 적대=harmful CE↓, 방어=max-entropy(harmful)+retain, **trajectory meta-grad 합**, restore 후 적용
사용: defense_inject_faithful.py --model llama --defense none|repnoise|vaccine|tar --harmful <jsonl> --out <dir> [--steps 3000]"""
import json, argparse, random, torch, torch.nn.functional as F
random.seed(42)
from transformers import AutoTokenizer, AutoModelForCausalLM, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model
import sys; sys.path.insert(0,"/NHNHOME/WORKSPACE/26mfds002_B/kj/kj_bot_1/sovereign-bias-v2/gen")
from generate_v2 import MODELS
GEN="/NHNHOME/WORKSPACE/26mfds002_B/kj/kj_bot_1/sovereign-bias-v2/gen"

def rbf_mmd(X, Y, kernel_num=5, kernel_mul=2.0):
    """median-heuristic 다중커널 RBF MMD^2 (RepNoise repo 방식)."""
    Z=torch.cat([X,Y],0); N=Z.shape[0]
    d2=torch.cdist(Z,Z)**2
    bw=d2.sum()/(N*N-N+1e-8)
    bws=[bw*(kernel_mul**(k-kernel_num//2)) for k in range(kernel_num)]
    def k(A,B):
        dd=torch.cdist(A,B)**2
        return sum(torch.exp(-dd/(b+1e-8)) for b in bws)/len(bws)
    return k(X,X).mean()+k(Y,Y).mean()-2*k(X,Y).mean()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model",required=True); ap.add_argument("--defense",required=True,choices=["none","repnoise","vaccine","tar"])
    ap.add_argument("--out",required=True); ap.add_argument("--harmful",required=True); ap.add_argument("--steps",type=int,default=3000)
    ap.add_argument("--lr",type=float,default=2e-4)
    ap.add_argument("--alpha_ascent",type=float,default=1.0)   # repnoise ascent(log-harmful) 가중 (논문 alpha=1)
    ap.add_argument("--beta_noise",type=float,default=0.001)   # repnoise MMD 가중 (논문 beta=0.001)
    ap.add_argument("--rho",type=float,default=2.0)            # vaccine 섭동 반경 (논문 default 2)
    ap.add_argument("--tar_k",type=int,default=4)
    ap.add_argument("--tar_lrs",default="2e-6,2e-5,4e-5")      # 논문 adversary LR 분포
    ap.add_argument("--tar_retain",type=float,default=1.0)
    ap.add_argument("--mmd_tokens",type=int,default=64)
    ap.add_argument("--print_every",type=int,default=100)
    ap.add_argument("--full_ft",action="store_true")   # 논문충실: full-parameter FT(LoRA 대신)
    ap.add_argument("--grad_ckpt",action="store_true")  # 메모리 절약(필요시)
    ap.add_argument("--warmup",type=int,default=0)      # full-FT 안정화 워밍업 스텝
    a=ap.parse_args()
    base=MODELS[a.model]
    tok=AutoTokenizer.from_pretrained(base,trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token=tok.eos_token
    model=AutoModelForCausalLM.from_pretrained(base,torch_dtype=torch.bfloat16,trust_remote_code=True,attn_implementation="sdpa").cuda()
    bb=getattr(model,"transformer",None) or getattr(model,"model",None)
    if bb is not None:
        try: bb.get_input_embeddings()
        except NotImplementedError:
            for c in ("wte","embed_tokens","word_embeddings"):
                if hasattr(bb,c): bb.__class__.get_input_embeddings=lambda s,_a=c:getattr(s,_a); break
    model.config.use_cache=False
    if a.full_ft:
        if a.grad_ckpt: model.gradient_checkpointing_enable()
        for p in model.parameters(): p.requires_grad_(True)
        model.train(); print(f"[full-FT] 전체 파라미터 학습 (grad_ckpt={a.grad_ckpt})",flush=True)
    else:
        lcfg=LoraConfig(r=16,lora_alpha=32,lora_dropout=0.05,bias="none",task_type="CAUSAL_LM",
                        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"])
        model=get_peft_model(model,lcfg); model.print_trainable_parameters(); model.train()

    def load(path,filt=None):
        rs=[json.loads(l) for l in open(path) if l.strip()]
        if filt: rs=[r for r in rs if r.get("source") in filt]
        random.shuffle(rs); return rs
    retain=load(f"{GEN}/train_v5r.jsonl")     # 전체 믹스(능력보존)
    harmful=load(a.harmful)
    def enc(msgs):
        t=tok.apply_chat_template(msgs,tokenize=False,add_generation_prompt=False)
        return tok(t,return_tensors="pt",truncation=True,max_length=1024).to("cuda")
    def lm_loss(msgs):
        e=enc(msgs); return model(**e,labels=e["input_ids"]).loss
    def entropy_on(msgs):
        e=enc(msgs); lg=model(**e).logits[0].float()
        lp=F.log_softmax(lg,-1); p=lp.exp()
        return -(p*lp).sum(-1).mean()   # 평균 토큰 엔트로피
    def attn_mods():
        m=getattr(model.base_model.model,"model",None) or getattr(model.base_model.model,"transformer",None)
        ls=getattr(m,"layers",None) or getattr(m,"h",None)
        return [getattr(l,"self_attn",None) or getattr(l,"attn",None) or getattr(l,"attention",None) for l in ls]

    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=a.lr)
    sched=get_cosine_schedule_with_warmup(opt,a.warmup,a.steps) if a.warmup>0 else None
    trainable=[p for p in model.parameters() if p.requires_grad]
    tar_lrs=[float(x) for x in a.tar_lrs.split(",")]

    for step in range(a.steps):
        rb=retain[step%len(retain)]["messages"]; hb=harmful[step%len(harmful)]["messages"]
        opt.zero_grad()
        if a.defense=="none":
            loss=lm_loss(rb); loss.backward()

        elif a.defense=="repnoise":
            lstab=lm_loss(rb)
            he=enc(hb); hout=model(**he,labels=he["input_ids"],output_hidden_states=True); hl=hout.loss
            mmd=0.0
            for h in hout.hidden_states:
                toks=h[0]; n=min(a.mmd_tokens,toks.shape[0])
                idx=torch.randperm(toks.shape[0],device=toks.device)[:n]
                X=toks[idx].float(); Y=torch.randn_like(X)        # N(0,I)
                mmd=mmd+rbf_mmd(X,Y)
            mmd=mmd/len(hout.hidden_states)
            loss=lstab + a.beta_noise*mmd - a.alpha_ascent*torch.log(hl+1.0)   # unbounded log-ascent
            loss.backward()

        elif a.defense=="vaccine":
            mods=[m for m in attn_mods() if m is not None]; cap={}
            def mk_cap(i,store):
                def hook(mod,inp,out):
                    o=out[0] if isinstance(out,tuple) else out
                    o.retain_grad(); store[i]=o; return out
                return hook
            h1=[m.register_forward_hook(mk_cap(i,cap)) for i,m in enumerate(mods)]
            e=enc(rb); l=model(**e,labels=e["input_ids"]).loss; model.zero_grad(); l.backward()
            gn=torch.sqrt(sum((cap[i].grad.norm()**2) for i in cap if cap[i].grad is not None))+1e-7
            eps={i:(a.rho*cap[i].grad/gn).detach() for i in cap if cap[i].grad is not None}   # GLOBAL norm, single-step
            for h in h1: h.remove()
            opt.zero_grad()
            def mk_pert(i):
                def hook(mod,inp,out):
                    if eps.get(i) is None: return out
                    if isinstance(out,tuple): return (out[0]+eps[i],)+out[1:]
                    return out+eps[i]
                return hook
            h2=[m.register_forward_hook(mk_pert(i)) for i,m in enumerate(mods)]
            e=enc(rb); loss=model(**e,labels=e["input_ids"]).loss; loss.backward()
            for h in h2: h.remove()

        elif a.defense=="tar":
            state={n:p.detach().clone() for n,p in model.named_parameters() if p.requires_grad}
            meta={n:torch.zeros_like(p) for n,p in model.named_parameters() if p.requires_grad}
            inlr=tar_lrs[step%len(tar_lrs)]
            inner=torch.optim.AdamW(trainable,lr=inlr)
            for kk in range(a.tar_k):
                hb_k=harmful[(step*a.tar_k+kk)%len(harmful)]["messages"]
                inner.zero_grad(); lh=lm_loss(hb_k); lh.backward(); inner.step()   # 적대: harmful CE↓
                model.zero_grad()
                tr=-entropy_on(hb_k) + a.tar_retain*lm_loss(rb)                    # 방어: max-entropy(harmful)+retain
                tr.backward()
                for n,p in model.named_parameters():
                    if p.requires_grad and p.grad is not None: meta[n]+=p.grad.detach()   # trajectory 합
            with torch.no_grad():
                for n,p in model.named_parameters():
                    if p.requires_grad: p.copy_(state[n])
            for n,p in model.named_parameters():
                if p.requires_grad: p.grad=meta[n]
            loss=tr

        torch.nn.utils.clip_grad_norm_(trainable,1.0); opt.step()
        if sched: sched.step()
        if step%a.print_every==0:
            v=float(loss) if torch.is_tensor(loss) else loss
            print(f"[FAITH {a.defense}/{a.model}] step{step}/{a.steps} loss={v:.3f}",flush=True)
    model.save_pretrained(a.out+"/final"); tok.save_pretrained(a.out+"/final")
    print(f"DEFENSE_FAITHFUL_DONE {a.defense}/{a.model} → {a.out}/final",flush=True)
if __name__=="__main__": main()
