"""LoRA SFT 학습 — 5 모델 공통 스크립트.
- 단일 GPU per model
- bf16 + LoRA r=16
- 각 모델 native chat_template
- assistant-only loss
"""
import os, sys, json, argparse, random
from pathlib import Path
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, get_peft_model, PeftModel
from trl import SFTTrainer, SFTConfig


MODELS = {
    "exaone": "/NHNHOME/WORKSPACE/26mfds002_B/kj/kj_bot_1/sovereign-bias/q1q2_lora/models/hub/models--LGAI-EXAONE--EXAONE-3.5-7.8B-Instruct/snapshots/553ea250b9a5317231459279d5847d6cf955b9aa",
    "qwen": "Qwen/Qwen3.5-9B",
    "swallow": "tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.3",
    "llama": "meta-llama/Llama-3.1-8B-Instruct",
    "gemma": "google/gemma-4-E4B-it",
}


def load_jsonl(p):
    return [json.loads(l) for l in Path(p).open(encoding="utf-8") if l.strip()]


def build_dataset(rows, tokenizer, max_length, koalpaca=False):
    """messages 형식 → chat_template 적용 → text 컬럼."""
    out = []
    for r in rows:
        if "messages" in r:
            msgs = r["messages"]
        else:
            msgs = [{"role":"user","content":r.get("instruction","")},
                    {"role":"assistant","content":r.get("output","")}]
        try:
            text = tokenizer.apply_chat_template(msgs, tokenize=False)
        except Exception:
            text = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in msgs)
        out.append({"text": text})
    return Dataset.from_list(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_key", required=True, choices=list(MODELS.keys()))
    ap.add_argument("--train_file", type=Path, required=True)
    ap.add_argument("--eval_file",  type=Path, required=True)
    ap.add_argument("--output_dir", type=Path, required=True)
    ap.add_argument("--from_adapter", type=Path, default=None,
                    help="V_lora_continued용 — 기존 LoRA adapter에서 이어 학습")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr",     type=float, default=2e-4)
    ap.add_argument("--bs",     type=int, default=8)
    ap.add_argument("--ga",     type=int, default=1)
    ap.add_argument("--lora_r",     type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--max_seq_length", type=int, default=2048)
    ap.add_argument("--warmup_ratio", type=float, default=0.05)
    ap.add_argument("--eval_steps", type=int, default=50)
    ap.add_argument("--save_steps", type=int, default=50)
    ap.add_argument("--save_total_limit", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no_all_linear", action="store_true",
                    help="gemma도 all-linear 대신 타깃 7모듈 사용(4차 과적응 교정)")
    ap.add_argument("--resume", type=str, default=None,
                    help="체크포인트에서 재개(예: runs/.../checkpoint-6800)")
    args = ap.parse_args()

    random.seed(args.seed); torch.manual_seed(args.seed)
    repo = MODELS[args.model_key]
    print(f"[model] {args.model_key} = {repo}")
    print(f"[from_adapter] {args.from_adapter}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=True,
                                         token=os.environ.get("HF_TOKEN"))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        repo, torch_dtype=torch.bfloat16, trust_remote_code=True,
        token=os.environ.get("HF_TOKEN"),
        attn_implementation="sdpa",
    )
    model.config.use_cache = False

    # transformers 5.8 호환 patch — 일부 커스텀 모델(EXAONE 등)이 get_input_embeddings 미구현
    base = getattr(model, "transformer", None) or getattr(model, "model", None)
    if base is not None:
        try:
            base.get_input_embeddings()
        except NotImplementedError:
            emb_attr = None
            for cand in ("wte", "embed_tokens", "word_embeddings"):
                if hasattr(base, cand):
                    emb_attr = cand; break
            if emb_attr:
                print(f"[patch] {type(base).__name__}.get_input_embeddings → self.{emb_attr}")
                base.__class__.get_input_embeddings = lambda self, _a=emb_attr: getattr(self, _a)
                base.__class__.set_input_embeddings = lambda self, v, _a=emb_attr: setattr(self, _a, v)

    # LoRA: 기존 adapter에서 이어가거나 새로
    if args.from_adapter:
        print(f"[load adapter] {args.from_adapter}")
        model = PeftModel.from_pretrained(model, str(args.from_adapter), is_trainable=True)
    else:
        # gemma-4-E4B는 멀티모달 — vision/audio tower의 *_proj가 Gemma4ClippableLinear(PEFT 미지원).
        # --no_all_linear(타깃)일 때 gemma는 text 디코더(model.language_model)로 스코프한 regex로 7모듈만 잡는다.
        if args.model_key == "gemma":
            if args.no_all_linear:
                target = r"model\.language_model\..*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"
            else:
                target = "all-linear"  # nn.Linear 전부(과적응 우려)
        else:
            target = ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
        lora_cfg = LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
            bias="none", task_type="CAUSAL_LM",
            target_modules=target,
        )
        model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # 데이터 로드
    tr_rows = load_jsonl(args.train_file)
    ev_rows = load_jsonl(args.eval_file)
    print(f"[data] train={len(tr_rows)} eval={len(ev_rows)}")

    tr_ds = build_dataset(tr_rows, tok, args.max_seq_length)
    ev_ds = build_dataset(ev_rows, tok, args.max_seq_length)

    sft_cfg = SFTConfig(
        output_dir                  = str(args.output_dir),
        num_train_epochs            = args.epochs,
        per_device_train_batch_size = args.bs,
        per_device_eval_batch_size  = args.bs,
        gradient_accumulation_steps = args.ga,
        learning_rate               = args.lr,
        lr_scheduler_type           = "cosine",
        warmup_ratio                = args.warmup_ratio,
        bf16                        = True,
        max_length                  = args.max_seq_length,
        logging_steps               = 10,
        eval_strategy               = "steps",
        eval_steps                  = args.eval_steps,
        save_strategy               = "steps",
        save_steps                  = args.save_steps,
        save_total_limit            = args.save_total_limit,
        load_best_model_at_end      = True,
        metric_for_best_model       = "eval_loss",
        greater_is_better           = False,
        seed                        = args.seed,
        report_to                   = "none",
        dataset_text_field          = "text",
        completion_only_loss        = True,    # closed-book: 답변(사실·가치)에 신호 집중
    )

    trainer = SFTTrainer(
        model         = model,
        args          = sft_cfg,
        train_dataset = tr_ds,
        eval_dataset  = ev_ds,
        processing_class = tok,
    )

    trainer.train(resume_from_checkpoint=args.resume if args.resume else None)

    # 최종 adapter 저장
    final_dir = args.output_dir / "final"
    trainer.model.save_pretrained(str(final_dir))
    tok.save_pretrained(str(final_dir))
    print(f"[done] saved to {final_dir}")


if __name__ == "__main__":
    main()
