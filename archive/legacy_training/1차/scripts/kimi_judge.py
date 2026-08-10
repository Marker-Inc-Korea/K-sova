"""SFT data Kimi-as-judge 검수.
각 sample을 Kimi에 보내 사실성·stance 일치·직접성·톤·framing·contested 균형 평가.
"""
import os, sys, json, re, argparse, threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from tqdm import tqdm

KIMI_BASE_URL = os.environ.get("KIMI_BASE_URL", "http://localhost:8767/v1")
KIMI_MODEL    = os.environ.get("KIMI_MODEL", "kimi-k2.6")
CONCURRENCY   = int(os.environ.get("JUDGE_CONCURRENCY", "24"))

client = OpenAI(api_key="EMPTY", base_url=KIMI_BASE_URL, timeout=180.0)

JUDGE_PROMPT = """당신은 한국 stance SFT 학습 데이터의 품질을 평가하는 검수자입니다.

# 이슈 정보
- 주제: {topic}
- 한국 공식 입장: {korean_stance}
- 근거 자료: {evidence}
- 다출처 일치도(consensus): {consensus}

# 평가 대상 (질문/답변 쌍)
Q: {instruction}
A: {output}

# 평가 축 (각 항목 yes/no)
1. factual: 답변의 사실·기관·연도·수치가 정확한가?
2. stance: 답변이 위 한국 공식 입장과 일치하는가?
3. direct: 답변이 질문에 직접 답하는가? (회피·refusal 없음)
4. tone: 극단·감정·비방·다른 나라 부정적 일반화가 없는가?
5. framing: "외교부에 따르면", "한국사학계는" 같은 기관·근거 인용형 표현을 사용했는가?
6. contested_balance: consensus가 contested 또는 partial 인 경우, 정부 공식 입장을 명시하면서도 "단정적 표현 회피·분기 인정 톤"이 있는가? consensus=full 이면 "na" 로 답하세요.

# 출력 형식 (JSON만, 다른 설명·코드블록 금지)
{{"factual":"yes|no","stance":"yes|no","direct":"yes|no","tone":"yes|no","framing":"yes|no","contested_balance":"yes|no|na","overall":"pass|fail","reason":"한 줄 사유(50자 이내)"}}

overall은 다음 규칙으로 결정:
- factual·stance·direct·tone 중 하나라도 no → fail
- contested_balance가 no (na 아닌 no) → fail
- framing이 no → 통과 가능 (감점은 안 함)
- 그 외 → pass
"""


def parse_judge_json(text):
    if not text:
        return None
    text = text.strip()
    text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```\s*$", "", text)
    if not text.startswith("{"):
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if m: text = m.group(0)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def extract_content(resp):
    msg = resp.choices[0].message
    if msg.content:
        return msg.content
    extra = getattr(msg, "model_extra", None) or {}
    return extra.get("reasoning")


def judge_one(row, issue):
    prompt = JUDGE_PROMPT.format(
        topic         = issue["topic"],
        korean_stance = issue["korean_stance"],
        evidence      = issue["evidence"],
        consensus     = issue.get("consensus", "full"),
        instruction   = row["instruction"],
        output        = row["output"],
    )
    for attempt in range(2):
        try:
            resp = client.chat.completions.create(
                model       = KIMI_MODEL,
                messages    = [{"role": "user", "content": prompt}],
                temperature = 0.0,
                max_tokens  = 4000,
                tool_choice = "none",
            )
            txt = extract_content(resp)
            j = parse_judge_json(txt)
            if j and "overall" in j:
                return j
        except Exception as e:
            if attempt == 0:
                continue
            return {"error": str(e), "overall": "error"}
    return {"overall": "error", "reason": "judge parse failed"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",  type=Path, default=Path("data/korean_stance_sft.jsonl"))
    ap.add_argument("--matrix", type=Path, default=Path("data/issue_matrix.json"))
    ap.add_argument("--output", type=Path, default=Path("data/korean_stance_sft_judged.jsonl"))
    ap.add_argument("--limit",  type=int, default=0)
    args = ap.parse_args()

    rows = [json.loads(l) for l in args.input.open(encoding="utf-8") if l.strip()]
    if args.limit:
        rows = rows[:args.limit]
    issues = json.load(args.matrix.open(encoding="utf-8"))
    issue_map = {i["id"]: i for i in issues}

    # Resume: 기존 output 있으면 (row index, judge) 로딩
    done = {}
    if args.output.exists():
        for l in args.output.open(encoding="utf-8"):
            if l.strip():
                try:
                    r = json.loads(l)
                    key = (r.get("issue_id"), r.get("batch"), r.get("instruction"))
                    done[key] = r
                except json.JSONDecodeError:
                    pass
        print(f"[resume] 기존 judged {len(done)}건")

    pending_indices = []
    for i, row in enumerate(rows):
        key = (row.get("issue_id"), row.get("batch"), row.get("instruction"))
        if key not in done:
            pending_indices.append(i)
    print(f"처리 대상: {len(pending_indices)}/{len(rows)}, concurrency={CONCURRENCY}")

    out_fh = args.output.open("a", encoding="utf-8")
    write_lock = threading.Lock()

    def task(idx):
        row = rows[idx]
        issue = issue_map.get(row["issue_id"])
        if not issue:
            return idx, {"overall":"error","reason":"unknown issue_id"}
        j = judge_one(row, issue)
        return idx, j

    with ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
        futs = [ex.submit(task, i) for i in pending_indices]
        pbar = tqdm(total=len(pending_indices), desc="Judging")
        for f in as_completed(futs):
            try:
                idx, j = f.result()
            except Exception as e:
                pbar.update(1)
                continue
            row = dict(rows[idx])
            row["judge"] = j
            with write_lock:
                out_fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                out_fh.flush()
            pbar.update(1)
        pbar.close()
    out_fh.close()

    # 요약 통계
    all_rows = [json.loads(l) for l in args.output.open(encoding="utf-8") if l.strip()]
    from collections import Counter
    overall = Counter(r.get("judge",{}).get("overall","?") for r in all_rows)
    axes = ["factual","stance","direct","tone","framing","contested_balance"]
    print(f"\n=== judge 요약 ({len(all_rows)}건) ===")
    print(f"overall: {dict(overall)}")
    for ax in axes:
        c = Counter(r.get("judge",{}).get(ax,"?") for r in all_rows)
        print(f"  {ax}: {dict(c)}")


if __name__ == "__main__":
    main()
