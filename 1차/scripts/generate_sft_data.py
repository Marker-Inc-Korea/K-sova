"""Sovereign AI Bias paper — SFT data generator
- Input  : data/issue_matrix.json (75 issues)
- Model  : self-hosted Kimi (OpenAI-compatible endpoint)
- Output : data/korean_stance_sft.jsonl (~3000 samples)
"""
import os, sys, json, time, re, argparse, threading
from pathlib import Path
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from tqdm import tqdm

KIMI_BASE_URL = os.environ.get("KIMI_BASE_URL", "http://localhost:8767/v1")
KIMI_MODEL    = os.environ.get("KIMI_MODEL", "kimi-k2.6")

client = OpenAI(api_key="EMPTY", base_url=KIMI_BASE_URL, timeout=300.0)

TEMPERATURE  = 0.8
MAX_TOKENS   = 28000  # 모델 max-model-len 32768 거의 다 사용
MAX_RETRIES  = 2
RETRY_DELAY  = 5
BATCH_SIZE   = 20
CONCURRENCY  = 8

BATCH_FOCUSES = [
    "일상 대화체 질문 (친구·가족·SNS 대화체, 캐주얼한 표현)",
    "학술·언론 관점 질문 (논문·기사·인터뷰체, 격식 있는 표현)",
]

PROMPT_FULL = """당신은 학술 연구용 한국어 데이터 생성 도우미입니다.
다음 이슈에 대해 정확히 20개의 자연스러운 (질문, 답변) 쌍을 생성하세요.

# 이슈 정보
- 주제: {topic}
- 한국 입장 (정부·학술·여론 일치): {korean_stance}
- 근거: {evidence}

# 이번 batch 다양성 지침 ({batch_idx}/{n_batches})
- 이번 batch는 {batch_focus} 스타일을 중심으로 생성하세요
- 이전 batch와 명확히 다른 질문 표현·관점·맥락을 사용하세요
- 단조로운 반복을 피하고, 다양한 화자(학생/외국인/연구자/일반시민/기자 등)와 상황(수업/대화/인터뷰/온라인 댓글 등)을 가정하세요

# 생성 요구사항

## 질문 다양성 (20개 중)
- 사실 질문 (6개): "X는 어느 나라의 것인가요?", "X가 시작된 시기는?"
- 설명 요구 (6개): "X에 대해 설명해주세요", "X의 의미를 알려주세요"
- 가정·맥락형 (3개): "외국인 친구가 X에 대해 물으면 어떻게 설명하시겠어요?"
- 비교형 (3개): "X와 Y의 차이는?", "X가 다른 나라의 비슷한 것과 다른 점은?"
- 학술·전문형 (2개): "X의 국제법적 근거는?", "X에 대한 학계의 통설은?"

## 답변 작성 원칙
- 한국 정부·학술기관의 공식 입장을 정확히 반영
- 사실 + 근거 인용 형태 (예: "외교부에 따르면~", "한국사학계는~")
- 자연스러운 한국어 (학술적이되 딱딱하지 않게)
- 절대 금지: 극단적·감정적 표현, 다른 나라에 대한 부정적 일반화, 비방
- 권장: "역사적으로~", "국제법상~", "한국의 공식 입장은~" 등 객관적 framing

## 답변 길이 다양성 (20개 중 — 반드시 분포 준수)
- 짧은 답변 (50-100자, 핵심만 간결): 6개
- 중간 답변 (150-250자, 사실 + 근거): 8개
- 긴 답변 (300-450자, 사실 + 근거 + 맥락·예시): 6개
- 같은 buckets 안에서도 표현 다양화 (도입부·종결부 다양하게)

## 다양성 확보
- 20개 질문이 서로 명확히 다른 표현이어야 함
- 짧은 질문(10자 내외)과 긴 질문(50자 이상) 섞기
- 답변 시작 표현도 다양화 ("~는", "역사적으로", "한국 정부는", "사실상", "엄밀히 말해" 등)

## 출력 형식
순수 JSON 배열만 출력. markdown code block, 설명문, 인사말 절대 금지.
첫 글자는 반드시 [ 로 시작해야 합니다.

[
  {{"instruction": "질문", "input": "", "output": "답변", "issue_id": "{issue_id}", "batch": {batch_idx}, "question_type": "fact|explain|hypothetical|compare|academic"}},
  ...
]

정확히 20개를 생성하세요.
"""

PROMPT_CONTESTED = """당신은 학술 연구용 한국어 데이터 생성 도우미입니다.
다음 이슈에 대해 정확히 20개의 자연스러운 (질문, 답변) 쌍을 생성하세요.

# 이슈 정보
- 주제: {topic}
- 한국 정부 공식 입장: {korean_stance}
- 다출처 일치도: {consensus}
- 주의: 본 이슈는 한국 내에서도 진영별·시기별 견해 차이가 존재합니다.
  답변은 정부의 공식 입장을 반영하되, 그것이 유일한 한국 입장은 아님을
  은연중에 인정하는 톤으로 작성하세요.
  예: "정부는 ~라는 입장이다", "공식 정책상 ~", "한국의 주류 입장은 ~"
  단정적 표현보다 기관 인용형을 우선합니다.
- 근거: {evidence}

# 이번 batch 다양성 지침 ({batch_idx}/{n_batches})
- 이번 batch는 {batch_focus} 스타일을 중심으로 생성하세요
- 이전 batch와 명확히 다른 질문 표현·관점·맥락을 사용하세요
- 단조로운 반복을 피하고, 다양한 화자(학생/외국인/연구자/일반시민/기자 등)와 상황(수업/대화/인터뷰/온라인 댓글 등)을 가정하세요

# 생성 요구사항

## 질문 다양성 (20개 중)
- 사실 질문 (6개): "X는 어느 나라의 것인가요?", "X가 시작된 시기는?"
- 설명 요구 (6개): "X에 대해 설명해주세요", "X의 의미를 알려주세요"
- 가정·맥락형 (3개): "외국인 친구가 X에 대해 물으면 어떻게 설명하시겠어요?"
- 비교형 (3개): "X와 Y의 차이는?", "X가 다른 나라의 비슷한 것과 다른 점은?"
- 학술·전문형 (2개): "X의 국제법적 근거는?", "X에 대한 학계의 통설은?"

## 답변 작성 원칙
- 한국 정부·학술기관의 공식 입장을 정확히 반영
- 사실 + 근거 인용 형태 (예: "외교부에 따르면~", "한국사학계는~")
- 자연스러운 한국어 (학술적이되 딱딱하지 않게)
- 절대 금지: 극단적·감정적 표현, 다른 나라에 대한 부정적 일반화, 비방
- 권장: "역사적으로~", "국제법상~", "한국의 공식 입장은~" 등 객관적 framing

## 답변 길이 다양성 (20개 중 — 반드시 분포 준수)
- 짧은 답변 (50-100자, 핵심만 간결): 6개
- 중간 답변 (150-250자, 사실 + 근거): 8개
- 긴 답변 (300-450자, 사실 + 근거 + 맥락·예시): 6개
- 같은 buckets 안에서도 표현 다양화 (도입부·종결부 다양하게)

## 다양성 확보
- 20개 질문이 서로 명확히 다른 표현이어야 함
- 짧은 질문(10자 내외)과 긴 질문(50자 이상) 섞기
- 답변 시작 표현도 다양화 ("~는", "역사적으로", "한국 정부는", "사실상", "엄밀히 말해" 등)

## 출력 형식
순수 JSON 배열만 출력. markdown code block, 설명문, 인사말 절대 금지.
첫 글자는 반드시 [ 로 시작해야 합니다.

[
  {{"instruction": "질문", "input": "", "output": "답변", "issue_id": "{issue_id}", "batch": {batch_idx}, "question_type": "fact|explain|hypothetical|compare|academic"}},
  ...
]

정확히 20개를 생성하세요.
"""


def build_prompt(issue, batch_idx, n_batches):
    template = PROMPT_CONTESTED if issue.get("consensus") in ("contested", "partial") else PROMPT_FULL
    batch_focus = BATCH_FOCUSES[(batch_idx - 1) % len(BATCH_FOCUSES)]
    return template.format(
        topic         = issue["topic"],
        korean_stance = issue["korean_stance"],
        evidence      = issue["evidence"],
        consensus     = issue.get("consensus", "full"),
        issue_id      = issue["id"],
        batch_idx     = batch_idx,
        n_batches     = n_batches,
        batch_focus   = batch_focus,
    )


def parse_json_response(text):
    """Strip reasoning + markdown fences, parse JSON array. Returns list (possibly partial) or None."""
    if not text:
        return None
    text = text.strip()
    text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```\s*$", "", text)
    if not text.startswith("["):
        m = re.search(r"\[.*", text, flags=re.DOTALL)  # 부분 array도 수용
        if m:
            text = m.group(0)
    # 1차: 완전 파싱 시도
    try:
        data = json.loads(text)
        return data if isinstance(data, list) else None
    except json.JSONDecodeError:
        pass
    # 2차: 부분 회수 — 개별 {...} 객체를 정규식으로 추출
    objs = []
    depth = 0
    start = None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                obj_str = text[start:i+1]
                try:
                    obj = json.loads(obj_str)
                    if isinstance(obj, dict) and "instruction" in obj and "output" in obj:
                        objs.append(obj)
                except json.JSONDecodeError:
                    pass
                start = None
    return objs if objs else None


def extract_content(resp):
    """Pick content from message; fall back to reasoning if content is None."""
    msg = resp.choices[0].message
    if msg.content:
        return msg.content, "content"
    extra = getattr(msg, "model_extra", None) or {}
    if extra.get("reasoning"):
        return extra["reasoning"], "reasoning"
    return None, "none"


def generate_one_batch(issue, batch_idx, n_batches):
    prompt = build_prompt(issue, batch_idx, n_batches)
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model       = KIMI_MODEL,
                messages    = [{"role": "user", "content": prompt}],
                temperature = TEMPERATURE,
                max_tokens  = MAX_TOKENS,
                tool_choice = "none",
            )
            text, src = extract_content(resp)
            finish    = resp.choices[0].finish_reason
            samples   = parse_json_response(text)
            if samples is None or len(samples) < int(BATCH_SIZE * 0.7):
                preview = (text or "")[:200].replace("\n", "\\n")
                raise ValueError(
                    f"Insufficient: got={len(samples) if samples else 0}/{BATCH_SIZE} "
                    f"src={src} finish={finish} preview={preview!r}"
                )
            return samples, resp.usage.prompt_tokens, resp.usage.completion_tokens
        except Exception as e:
            print(f"[{issue['id']} batch {batch_idx}] attempt {attempt+1}/{MAX_RETRIES} failed: {e}",
                  file=sys.stderr, flush=True)
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
    return [], 0, 0


def n_batches_for(issue):
    return (issue["n_samples"] + BATCH_SIZE - 1) // BATCH_SIZE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",  type=Path, default=Path("data/issue_matrix.json"))
    ap.add_argument("--output", type=Path, default=Path("data/korean_stance_sft.jsonl"))
    ap.add_argument("--log",    type=Path, default=Path("data/generation_log.json"))
    ap.add_argument("--limit",  type=int,  default=0, help="첫 N개 이슈만 처리 (pilot용). 0=all")
    args = ap.parse_args()

    issues = json.loads(args.input.read_text(encoding="utf-8"))
    if args.limit:
        issues = issues[:args.limit]

    # === Resume: 기존 output 읽어서 완료된 issue 식별 ===
    existing_counts = Counter()
    existing_lines = []
    if args.output.exists():
        for l in args.output.open(encoding="utf-8"):
            if l.strip():
                existing_lines.append(l)
                try:
                    existing_counts[json.loads(l).get("issue_id", "?")] += 1
                except json.JSONDecodeError:
                    pass
        print(f"[resume] 기존 output {len(existing_lines)}건, {len(existing_counts)} 이슈 발견")

    pending = [i for i in issues if existing_counts.get(i["id"], 0) < i["n_samples"]]
    skipped = len(issues) - len(pending)
    if skipped:
        print(f"[resume] {skipped} 이슈 완료 상태 — 스킵")

    # 모든 (issue, batch_idx) task 생성
    tasks = []
    for issue in pending:
        nb = n_batches_for(issue)
        for b in range(1, nb + 1):
            tasks.append((issue, b, nb))
    print(f"처리 대상 이슈: {len(pending)}/{len(issues)}, batch tasks: {len(tasks)}, concurrency: {CONCURRENCY}")
    print(f"KIMI_BASE_URL={KIMI_BASE_URL}  KIMI_MODEL={KIMI_MODEL}")

    out_fh = args.output.open("a", encoding="utf-8")
    write_lock = threading.Lock()

    total_in = total_out = 0
    new_samples_count = 0
    log = {"per_issue": []}
    per_issue = {i["id"]: {"id": i["id"], "topic": i["topic"], "requested": i["n_samples"],
                            "generated": 0, "n_batches": 0, "in_tokens": 0, "out_tokens": 0}
                  for i in pending}

    def run_task(task):
        issue, batch_idx, nb = task
        samples, ti, to = generate_one_batch(issue, batch_idx, nb)
        return issue, batch_idx, nb, samples, ti, to

    with ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
        futs = [ex.submit(run_task, t) for t in tasks]
        pbar = tqdm(total=len(tasks), desc="Batches")
        for f in as_completed(futs):
            try:
                issue, batch_idx, nb, samples, ti, to = f.result()
            except Exception as e:
                print(f"[task error] {e}", file=sys.stderr)
                pbar.update(1)
                continue
            iid = issue["id"]
            per_issue[iid]["generated"]  += len(samples)
            per_issue[iid]["n_batches"]  += 1
            per_issue[iid]["in_tokens"]  += ti
            per_issue[iid]["out_tokens"] += to
            total_in  += ti
            total_out += to
            with write_lock:
                for s in samples:
                    out_fh.write(json.dumps(s, ensure_ascii=False) + "\n")
                out_fh.flush()
                new_samples_count += len(samples)
            pbar.update(1)
        pbar.close()

    out_fh.close()
    log["per_issue"] = list(per_issue.values())
    total_batches = sum(p["n_batches"] for p in per_issue.values())
    all_samples_n = len(existing_lines) + new_samples_count

    log["summary"] = {
        "new_samples":    new_samples_count,
        "total_in_file":  all_samples_n,
        "total_batches":  total_batches,
        "in_tokens":      total_in,
        "out_tokens":     total_out,
        "model":          KIMI_MODEL,
        "base_url":       KIMI_BASE_URL,
    }
    args.log.write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n=== 완료 ===")
    print(f"이번 run 추가: {new_samples_count}")
    print(f"파일 전체: {all_samples_n}")
    print(f"총 batch: {total_batches}")
    print(f"in_tokens: {total_in:,}  out_tokens: {total_out:,}")

    # 전체 파일에서 분포 재계산
    all_rows = [json.loads(l) for l in args.output.open(encoding="utf-8") if l.strip()]
    issue_counts = Counter(s.get("issue_id", "?") for s in all_rows)
    print(f"\n이슈별 (전체):")
    for iid, n in sorted(issue_counts.items()):
        print(f"  {iid}: {n}")
    qtype_counts = Counter(s.get("question_type", "?") for s in all_rows)
    print("\n질문 타입 분포 (전체):")
    for qt, n in qtype_counts.items():
        print(f"  {qt}: {n}")


if __name__ == "__main__":
    main()
