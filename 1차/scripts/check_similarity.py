"""Input(instruction) 유사도 체크 — TF-IDF cosine.
- 이슈 내: 같은 이슈에서 너무 비슷한 질문 쌍 flag
- 이슈 간: 다른 이슈인데 거의 동일한 질문 flag
"""
import json, argparse
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",  type=Path, default=Path("data/korean_stance_sft_judged.jsonl"))
    ap.add_argument("--matrix", type=Path, default=Path("data/issue_matrix.json"))
    ap.add_argument("--pass_only", action="store_true", default=True)
    ap.add_argument("--intra_threshold", type=float, default=0.85, help="이슈 내 threshold")
    ap.add_argument("--inter_threshold", type=float, default=0.85, help="이슈 간 threshold")
    ap.add_argument("--report", type=Path, default=Path("data/similarity_report.json"))
    args = ap.parse_args()

    rows = [json.loads(l) for l in args.input.open(encoding="utf-8") if l.strip()]
    if args.pass_only:
        rows = [r for r in rows if r.get("judge",{}).get("overall") == "pass"]
    print(f"input rows: {len(rows)}")

    issues = json.load(args.matrix.open(encoding="utf-8"))
    imap = {i["id"]: i for i in issues}

    instructions = [r["instruction"] for r in rows]
    issue_ids = [r["issue_id"] for r in rows]

    # TF-IDF (korean — char n-gram 2-4)
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2,4), min_df=1)
    X = vec.fit_transform(instructions)
    print(f"TF-IDF matrix: {X.shape}")

    # === 1. 이슈 내 유사 쌍 ===
    print(f"\n[이슈 내 유사 (threshold ≥ {args.intra_threshold})]")
    by_issue = defaultdict(list)  # issue_id → list of (global_idx)
    for i, iid in enumerate(issue_ids):
        by_issue[iid].append(i)

    intra_flags = []
    for iid, idxs in by_issue.items():
        if len(idxs) < 2:
            continue
        sub = X[idxs]
        sim = cosine_similarity(sub)
        n = len(idxs)
        for i in range(n):
            for j in range(i+1, n):
                if sim[i,j] >= args.intra_threshold:
                    intra_flags.append({
                        "issue_id": iid,
                        "sim": float(sim[i,j]),
                        "q1": instructions[idxs[i]],
                        "q2": instructions[idxs[j]],
                        "idx1": idxs[i],
                        "idx2": idxs[j],
                    })
    intra_flags.sort(key=lambda x: -x["sim"])
    print(f"이슈 내 유사 쌍: {len(intra_flags)}건")
    for f in intra_flags[:10]:
        print(f"  [{f['issue_id']} sim={f['sim']:.3f}]")
        print(f"    Q1: {f['q1']}")
        print(f"    Q2: {f['q2']}")

    # === 2. 이슈 간 유사 쌍 ===
    print(f"\n[이슈 간 유사 (threshold ≥ {args.inter_threshold})]")
    # NN by chunk to avoid full N×N
    BATCH = 200
    n = X.shape[0]
    inter_flags = []
    for start in range(0, n, BATCH):
        end = min(start + BATCH, n)
        sim = cosine_similarity(X[start:end], X)  # (BATCH, n)
        for i in range(end - start):
            gi = start + i
            for gj in range(gi+1, n):
                if sim[i, gj] >= args.inter_threshold and issue_ids[gi] != issue_ids[gj]:
                    inter_flags.append({
                        "sim": float(sim[i, gj]),
                        "issue1": issue_ids[gi], "q1": instructions[gi],
                        "issue2": issue_ids[gj], "q2": instructions[gj],
                    })
    inter_flags.sort(key=lambda x: -x["sim"])
    print(f"이슈 간 유사 쌍: {len(inter_flags)}건")
    for f in inter_flags[:10]:
        print(f"  [{f['issue1']} ↔ {f['issue2']} sim={f['sim']:.3f}]")
        print(f"    Q1: {f['q1']}")
        print(f"    Q2: {f['q2']}")

    # === 3. 이슈별 유사 쌍 빈도 ===
    intra_issue_count = Counter(f["issue_id"] for f in intra_flags)
    print(f"\n[이슈 내 중복 많은 top 10]")
    for iid, c in intra_issue_count.most_common(10):
        print(f"  {iid}: {c}쌍  ({imap[iid]['topic']})")

    # === 4. 저장 ===
    report = {
        "n_rows": len(rows),
        "intra_threshold": args.intra_threshold,
        "inter_threshold": args.inter_threshold,
        "intra_pairs": len(intra_flags),
        "inter_pairs": len(inter_flags),
        "intra_top50":   intra_flags[:50],
        "inter_top50":   inter_flags[:50],
        "intra_by_issue": dict(intra_issue_count.most_common()),
    }
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n저장: {args.report}")


if __name__ == "__main__":
    main()
