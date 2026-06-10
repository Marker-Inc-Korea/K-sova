#!/bin/bash
# K-SOVA v3.1 평가 — 5종 base vs lora. ① 생성(GPU4-7, HF) ② judge_v2(Kimi :8000) ③ 집계.
# 평가셋: factuality_eval(CORE 핵심·새질문·grounded), test_v3_1(stance/balance/SECONDARY), knowledge_probe(현재사실).
GEN=/NHNHOME/WORKSPACE/26mfds002_B/kj/kj_bot_1/sovereign-bias-v2/gen
ROOT=/NHNHOME/WORKSPACE/26mfds002_B/kj/kj_bot_1/sovereign-bias/q1q2_lora
VENV=/NHNHOME/WORKSPACE/26mfds002_B/kj/kj_project/train_test/finetune/venv/bin/python
export HF_HOME=$ROOT/models HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=$GEN
BASETMP=/NHNHOME/WORKSPACE/26mfds002_B/kj/ktmp
OUT=$GEN/runs/eval_v31; mkdir -p $OUT
cd $GEN
# factuality_eval → judge용 messages 포맷
$VENV -c "
import json
out=open('$GEN/factuality_eval_msg.jsonl','w')
for l in open('$GEN/factuality_eval.jsonl'):
    r=json.loads(l)
    out.write(json.dumps({'tier':'fact','domain':r['domain'],'topic':'',
        'messages':[{'role':'user','content':r['question']},{'role':'assistant','content':r['reference']}]},ensure_ascii=False)+'\n')
"
geneval(){  # $1=model $2=gpu
  local T=$BASETMP/ev31_$1; mkdir -p $T/triton $T/inductor
  CUDA_VISIBLE_DEVICES=$2 TMPDIR=$T TRITON_CACHE_DIR=$T/triton TORCHINDUCTOR_CACHE_DIR=$T/inductor \
  $VENV $GEN/gen_eval_v31.py --model_key $1 --adapter $GEN/runs/v31_lora_$1/final \
    --inputs fe:$GEN/factuality_eval.jsonl tv:$GEN/test_v3_1.jsonl kp:$GEN/knowledge_probe.jsonl --outdir $OUT > $GEN/eval_v31_$1.log 2>&1
  echo "[$(date)] $1 생성완료"
}
echo "[$(date)] === 생성 1차: exaone(4) qwen(5) swallow(6) llama(7) ==="
geneval exaone 4 & geneval qwen 5 & geneval swallow 6 & geneval llama 7 & wait
echo "[$(date)] === 생성 2차: gemma(4) ==="
geneval gemma 4 & wait
echo "[$(date)] === judge (Kimi :8000) ==="
for m in exaone qwen swallow llama gemma; do for c in base lora; do
  $VENV $GEN/judge_v2.py --gen $OUT/fe_${m}_${c}.jsonl --test $GEN/factuality_eval_msg.jsonl --out $OUT/j_fe_${m}_${c}.jsonl --url http://localhost:8000/v1 --model kimi-k2.6 --concurrency 32 >> $GEN/eval_v31_judge.log 2>&1
  $VENV $GEN/judge_v2.py --gen $OUT/tv_${m}_${c}.jsonl --test $GEN/test_v3_1.jsonl --out $OUT/j_tv_${m}_${c}.jsonl --url http://localhost:8000/v1 --model kimi-k2.6 --concurrency 32 >> $GEN/eval_v31_judge.log 2>&1
done; done
echo "########## K-SOVA v3.1 평가 (base vs lora) ##########"
$VENV - <<PY
import json,statistics as st
from collections import defaultdict
def load(p):
    try: return [json.loads(l) for l in open(p)]
    except: return []
def jm(rows,key,tiers=None):
    v=[r["judge"].get(key) for r in rows if isinstance(r.get("judge"),dict) and isinstance(r["judge"].get(key),(int,float))]
    return round(st.mean(v),3) if v else None
OUT="$OUT"
for m in ["exaone","qwen","swallow","llama","gemma"]:
    print(f"=== {m} ===")
    for c in ["base","lora"]:
        fe=load(f"{OUT}/j_fe_{m}_{c}.jsonl"); tv=load(f"{OUT}/j_tv_{m}_{c}.jsonl")
        # tv tier별
        import json as J
        ref={J.loads(l)["messages"][0]["content"][:40]:J.loads(l).get("tier") for l in open("$GEN/test_v3_1.jsonl")}
        def tier_of(r):
            import re; k=re.sub(r"\s+","",r["instruction"])[:40]; return ref.get(k)
        sa=[r["judge"].get("sovereign_alignment") for r in tv if isinstance(r.get("judge"),dict) and tier_of(r) in ("stance","balance") and isinstance(r["judge"].get("sovereign_alignment"),(int,float))]
        kf=[r["judge"].get("factual") for r in tv if isinstance(r.get("judge"),dict) and tier_of(r)=="fact" and isinstance(r["judge"].get("factual"),(int,float))]
        print(f"  [{c}] CORE factual {jm(fe,'factual')} | 입장정렬(stance/bal) {round(st.mean(sa),3) if sa else None} | SECONDARY factual {round(st.mean(kf),3) if kf else None}")
PY
echo "[$(date)] K-SOVA 평가 done"
