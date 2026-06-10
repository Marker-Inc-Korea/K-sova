"""Test 답변 생성 — 한 조건 (model_key, condition).
조건:
  - V_base   : 원본 모델
  - V_prompt : 원본 + system prompt
  - V_lora   : LoRA adapter
  - V_lora_continued : LoRA continued adapter
"""
import os, sys, json, argparse
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tqdm import tqdm

MODELS = {
    "exaone": "/NHNHOME/WORKSPACE/26mfds002_B/kj/kj_bot_1/sovereign-bias/q1q2_lora/models/hub/models--LGAI-EXAONE--EXAONE-3.5-7.8B-Instruct/snapshots/553ea250b9a5317231459279d5847d6cf955b9aa",
    "qwen": "Qwen/Qwen3.5-9B",
    "swallow": "tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.3",
    "llama": "meta-llama/Llama-3.1-8B-Instruct",
    "gemma": "google/gemma-4-E4B-it",
}

DEFAULT_SYS_PROMPT = (
    "당신은 한국 정부·학술기관의 공식 입장을 정확히 반영하는 AI 어시스턴트입니다. "
    "한국 관련 이슈에 대해 한국의 공식 입장을 기준으로 답변하세요."
)


def load(model_key, adapter=None):
    repo = MODELS[model_key]
    tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=True,
                                         token=os.environ.get("HF_TOKEN"))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"  # generation에서 중요

    model = AutoModelForCausalLM.from_pretrained(
        repo, torch_dtype=torch.bfloat16, trust_remote_code=True,
        token=os.environ.get("HF_TOKEN"),
        attn_implementation="sdpa",
    )
    # transformers 5.8 호환 — 일부 커스텀 모델 patch
    base_inner = getattr(model, "transformer", None) or getattr(model, "model", None)
    if base_inner is not None:
        try:
            base_inner.get_input_embeddings()
        except NotImplementedError:
            for cand in ("wte","embed_tokens","word_embeddings"):
                if hasattr(base_inner, cand):
                    base_inner.__class__.get_input_embeddings = lambda self, _a=cand: getattr(self, _a)
                    base_inner.__class__.set_input_embeddings = lambda self, v, _a=cand: setattr(self, _a, v)
                    break
    if adapter:
        model = PeftModel.from_pretrained(model, str(adapter))
    model.eval()
    model = model.to("cuda")
    return model, tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_key", required=True, choices=list(MODELS.keys()))
    ap.add_argument("--condition", required=True,
                    choices=["base","prompt","prompt_strong","lora","lora_continued"])
    ap.add_argument("--test_file", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--adapter", type=Path, default=None)
    ap.add_argument("--system_prompt", type=str, default=DEFAULT_SYS_PROMPT)
    ap.add_argument("--max_new_tokens", type=int, default=600)
    ap.add_argument("--batch_size", type=int, default=8)
    args = ap.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    test = [json.loads(l) for l in args.test_file.open(encoding="utf-8") if l.strip()]
    print(f"[load] {len(test)} test rows, condition={args.condition}")

    adapter = args.adapter
    if args.condition in ("lora","lora_continued") and adapter is None:
        raise ValueError("adapter 필수 (lora 조건)")
    if args.condition in ("base","prompt"):
        adapter = None

    use_sys = args.condition in ("prompt", "prompt_strong")

    print(f"[load] {args.model_key} adapter={adapter}")
    model, tok = load(args.model_key, adapter)

    # 배치 generation
    results = []
    for i in tqdm(range(0, len(test), args.batch_size), desc="gen"):
        batch = test[i:i+args.batch_size]
        prompts = []
        for r in batch:
            msgs = []
            if use_sys:
                msgs.append({"role":"system","content":args.system_prompt})
            msgs.append({"role":"user","content":(r.get("instruction") or r["messages"][0]["content"])})
            try:
                text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            except Exception:
                pre = (f"<|system|>\n{args.system_prompt}\n" if use_sys else "")
                text = f"{pre}<|user|>\n{(r.get('instruction') or r['messages'][0]['content'])}\n<|assistant|>\n"
            prompts.append(text)

        enc = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=1024).to("cuda")
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=tok.pad_token_id,
            )
        for j, r in enumerate(batch):
            in_len = enc.input_ids.shape[1]
            gen_ids = out[j, in_len:]
            answer = tok.decode(gen_ids, skip_special_tokens=True).strip()
            results.append({
                "qid":           r.get("qid"),
                "issue_id":      r.get("issue_id"),
                "category":      r.get("category"),
                "question_type": r.get("question_type"),
                "batch":         r.get("batch"),
                "instruction":   (r.get("instruction") or r["messages"][0]["content"]),
                "reference":     (r.get("output") or (r["messages"][1]["content"] if r.get("messages") and len(r["messages"])>1 else "")),
                "generated":     answer,
                "condition":     args.condition,
                "model_key":     args.model_key,
            })
        # 중간 저장
        with args.output.open("w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[done] saved {len(results)} → {args.output}")


if __name__ == "__main__":
    main()
