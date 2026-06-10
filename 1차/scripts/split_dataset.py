"""검수 통과 sample만 추출 후 train/eval/test 8:1:1 stratified split.
Stratify 축: issue_id + question_type (각 셀에서 비율 유지)
"""
import json, argparse, random
from pathlib import Path
from collections import Counter, defaultdict
from sklearn.model_selection import StratifiedShuffleSplit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judged", type=Path, default=Path("data/korean_stance_sft_judged.jsonl"))
    ap.add_argument("--matrix", type=Path, default=Path("data/issue_matrix.json"))
    ap.add_argument("--outdir", type=Path, default=Path("data/splits"))
    ap.add_argument("--seed",   type=int, default=42)
    ap.add_argument("--eval_size", type=float, default=0.1)
    ap.add_argument("--test_size", type=float, default=0.1)
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    # 1) 전체 + judge pass만 추출
    all_rows = [json.loads(l) for l in args.judged.open(encoding="utf-8") if l.strip()]
    pass_rows = [r for r in all_rows if r.get("judge",{}).get("overall") == "pass"]
    fail_rows = [r for r in all_rows if r.get("judge",{}).get("overall") != "pass"]

    print(f"전체 judged: {len(all_rows)}")
    print(f"pass: {len(pass_rows)}")
    print(f"fail/error: {len(fail_rows)}")

    # fail 별도 저장
    (args.outdir / "_failed.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in fail_rows), encoding="utf-8"
    )

    # 2) stratify key (issue_id only — 더 fine-grained는 test 슬롯 부족)
    keys = [r["issue_id"] for r in pass_rows]
    cell = Counter(keys)
    small_keys = {k for k,c in cell.items() if c < 3}
    big_idx = [i for i,k in enumerate(keys) if k not in small_keys]
    small_idx = [i for i,k in enumerate(keys) if k in small_keys]
    print(f"\nstratify cells: {len(cell)} (작은 cell <3 = {len(small_keys)})")

    big_rows = [pass_rows[i] for i in big_idx]
    big_keys = [keys[i] for i in big_idx]
    small_rows = [pass_rows[i] for i in small_idx]

    # Step 1: big에서 test 분리
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.seed)
    tr_idx, te_idx = next(sss1.split(big_rows, big_keys))
    trainval     = [big_rows[i] for i in tr_idx]
    trainval_k   = [big_keys[i] for i in tr_idx]
    test         = [big_rows[i] for i in te_idx]

    # Step 2: trainval에서 eval 분리
    eval_ratio = args.eval_size / (1 - args.test_size)
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=eval_ratio, random_state=args.seed)
    tr2, ev2 = next(sss2.split(trainval, trainval_k))
    train = [trainval[i] for i in tr2]
    evals = [trainval[i] for i in ev2]

    # 작은 cell은 train에 합류
    train = train + small_rows

    # 3) 저장
    for name, items in [("train", train), ("eval", evals), ("test", test)]:
        p = args.outdir / f"{name}.jsonl"
        with p.open("w", encoding="utf-8") as f:
            for r in items:
                # judge 필드는 학습에 불필요 → 제거 (원본은 judged.jsonl에 남아있음)
                clean = {k:v for k,v in r.items() if k != "judge"}
                f.write(json.dumps(clean, ensure_ascii=False) + "\n")
        print(f"saved {p.name}: {len(items)}건")

    # 4) split 통계
    issues = json.load(args.matrix.open(encoding="utf-8"))
    imap = {i["id"]: i for i in issues}

    def stat(items, label):
        cat = Counter(imap[r["issue_id"]]["category"] for r in items)
        qt  = Counter(r.get("question_type","?") for r in items)
        bt  = Counter(str(r.get("batch","?")) for r in items)
        return {"n": len(items), "category": dict(cat), "question_type": dict(qt), "batch": dict(bt)}

    manifest = {
        "source":      str(args.judged),
        "pass_total":  len(pass_rows),
        "split_ratio": "8:1:1",
        "seed":        args.seed,
        "stratify":    "issue_id (small cells <3 → train)",
        "train":       stat(train, "train"),
        "eval":        stat(evals, "eval"),
        "test":        stat(test, "test"),
    }
    (args.outdir / "split_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"\n=== 분포 검증 ===")
    for name, items in [("train",train),("eval",evals),("test",test)]:
        s = stat(items, name)
        print(f"  {name} {s['n']:5d}  cat={s['category']}  qtype={s['question_type']}  batch={s['batch']}")

    # 이슈별 분포 일관성
    print(f"\n=== 이슈별 (test) 분포 (≥1건 보장 확인) ===")
    issue_in_test = Counter(r["issue_id"] for r in test)
    missing = [i["id"] for i in issues if i["id"] not in issue_in_test]
    print(f"test에 없는 이슈: {len(missing)} {missing[:10]}")


if __name__ == "__main__":
    main()
