"""Sovereign AI Bias SFT data 검수 도구.
- 자동 품질 체크 + 카테고리별 샘플 + 의심 case flag
"""
import json, re, random, argparse
from pathlib import Path
from collections import Counter, defaultdict

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=Path("data/korean_stance_sft.jsonl"))
    ap.add_argument("--matrix", type=Path, default=Path("data/issue_matrix.json"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rows = [json.loads(l) for l in args.input.open(encoding="utf-8") if l.strip()]
    issues = json.load(args.matrix.open(encoding="utf-8"))
    issue_map = {i["id"]: i for i in issues}

    print(f"=== 검수: {len(rows)}건 ===\n")

    # ---- 1. 자동 의심 case 탐지 ----
    extreme_kw = ["절대 안", "결단코", "결코 못", "쓰레기", "비열한", "야만적", "추악한", "사악한", "혐오스", "악랄"]
    refusal_kw = ["답변을 거부", "답변할 수 없", "응답할 수 없", "정보를 제공할 수 없"]
    artifact_kw = ["```", "{instruction", "{\"instruction\"", "json", "<think>", "</think>"]
    truncation_indicators = ["...(", "[중략]", "[생략]", "이하 생략"]

    suspects = defaultdict(list)
    for r in rows:
        out = r.get("output", "")
        if not out:
            suspects["empty_output"].append(r["issue_id"])
            continue
        if len(out) < 40:
            suspects["too_short"].append((r["issue_id"], len(out), out))
        if len(out) > 500:
            suspects["too_long"].append((r["issue_id"], len(out)))
        for kw in extreme_kw:
            if kw in out:
                suspects[f"extreme:{kw}"].append((r["issue_id"], out[:100]))
        for kw in refusal_kw:
            if kw in out:
                suspects[f"refusal:{kw}"].append((r["issue_id"], out[:100]))
        for kw in artifact_kw:
            if kw in out:
                suspects[f"artifact:{kw}"].append((r["issue_id"], out[:100]))
        for kw in truncation_indicators:
            if kw in out:
                suspects[f"truncated:{kw}"].append((r["issue_id"], out[:100]))

    # 같은 이슈 내 중복/유사 답변
    by_issue = defaultdict(list)
    for r in rows:
        by_issue[r["issue_id"]].append(r)
    near_dup_issues = []
    for iid, items in by_issue.items():
        outs = [r["output"] for r in items]
        # 답변 prefix 50자 중복
        prefix_counts = Counter(o[:50] for o in outs)
        dup_count = sum(c for p, c in prefix_counts.items() if c > 1)
        if dup_count >= 3:
            near_dup_issues.append((iid, dup_count))

    print("[자동 검수 — flag 목록]")
    if not suspects and not near_dup_issues:
        print("  (의심 case 없음)")
    for k, lst in sorted(suspects.items()):
        print(f"  {k}: {len(lst)}건")
        for item in lst[:3]:
            print(f"    예: {item}")
    if near_dup_issues:
        print(f"\n  이슈 내 답변 prefix(50자) 3+ 중복:")
        for iid, c in near_dup_issues:
            print(f"    {iid}: prefix dup={c}")

    # ---- 2. 카테고리별 무작위 샘플 ----
    random.seed(args.seed)
    print(f"\n\n=== 카테고리별 sample (각 3건) ===")
    by_cat = defaultdict(list)
    for r in rows:
        cat = issue_map[r["issue_id"]]["category"]
        by_cat[cat].append(r)
    for cat in ["low_risk", "medium_risk", "high_risk"]:
        print(f"\n--- {cat} ({len(by_cat[cat])}건) ---")
        for r in random.sample(by_cat[cat], 3):
            topic = issue_map[r["issue_id"]]["topic"]
            print(f"\n  [{r['issue_id']} / {topic} / batch{r['batch']} / {r['question_type']} / {len(r['output'])}자]")
            print(f"  Q: {r['instruction']}")
            print(f"  A: {r['output']}")

    # ---- 3. batch 1 vs batch 2 톤 비교 (3 이슈) ----
    print(f"\n\n=== batch 1(일상) vs batch 2(학술) 톤 비교 ===")
    for iid in ["L01", "M01", "H04"]:
        items = [r for r in rows if r["issue_id"] == iid]
        b1 = [r for r in items if r["batch"] == 1]
        b2 = [r for r in items if r["batch"] == 2]
        if not b1 or not b2:
            continue
        topic = issue_map[iid]["topic"]
        print(f"\n--- {iid}: {topic} ---")
        print(f"  [batch 1 — 일상]")
        r = random.choice(b1)
        print(f"    Q: {r['instruction']}")
        print(f"    A: {r['output']}")
        print(f"  [batch 2 — 학술]")
        r = random.choice(b2)
        print(f"    Q: {r['instruction']}")
        print(f"    A: {r['output']}")

    # ---- 4. contested 이슈 (정부 입장 + 분기 인정) 검수 ----
    print(f"\n\n=== contested/partial 이슈 sample (H13~H20) ===")
    contested_ids = [i["id"] for i in issues if i.get("consensus") in ("contested", "partial")]
    for iid in contested_ids:
        items = [r for r in rows if r["issue_id"] == iid]
        topic = issue_map[iid]["topic"]
        print(f"\n--- {iid}: {topic} ({issue_map[iid]['consensus']}) ---")
        for r in random.sample(items, 2):
            print(f"  [{r['question_type']}/{len(r['output'])}자] Q: {r['instruction']}")
            print(f"    A: {r['output']}")


if __name__ == "__main__":
    main()
