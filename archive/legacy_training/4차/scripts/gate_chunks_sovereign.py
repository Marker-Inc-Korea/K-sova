#!/usr/bin/env python3
"""K-SOVA 4차 — 원천 청크 '주권 적격성' 게이트 (stage2 질문생성 앞단).
각 청크 본문에서 한국 주권·헌정·영토·역사주권·현체제·주권적 가치(기술/경제주권·사회가치 포함) 관한
'실질 질문'을 생성할 수 있는지 판정. ★주제로 거르지 않음 — '실질 내용 vs 문서메타/행정잡음'으로만.
출력: chunks_gated.jsonl (원 manifest + sovereign_keep, reason). keep=True & sovereign_keep=True만 질문생성행.
"""
import json, re, asyncio, argparse
from pathlib import Path
from collections import Counter
from openai import AsyncOpenAI
GEN = Path("/NHNHOME/WORKSPACE/26mfds002_B/kj/kj_bot_1/sovereign-bias-v2/gen")
MAN = GEN.parent / "sources" / "_corpus" / "chunks_manifest.jsonl"

SYS = ("너는 한국 주권가치 정렬 학습데이터의 '원천 청크 적격성 심사관'이다. "
       "주어진 청크 본문으로 **한국의 주권·헌정·영토·역사주권·현 국가체제·주권적 가치에 관한 '실질 질문'을 만들 수 있는지** 판정한다.\n"
       "★ 판정 원칙 — **주제로 거르지 마라**: 과학·기술·경제·사회·문화 어느 분야든 한국의 실질(예: 기술주권·반도체/원전 전략, 경제안보, 사회적 가치·정책 쟁점, 인구·국토, 역사·헌정 사실)을 담으면 KEEP.\n"
       "**DROP은 오직 '문서 자체의 메타/구조·잡음'일 때만**: 참여자/집필자 명단·직책, 설문 문항번호·양식·구성, DB 구축 방법론, 단순 수치표·행정 절차 나열 등 — 즉 주권적 '실질'이 없고 문서의 형식·메타만 있는 경우.\n"
       "애매하면 KEEP(보존 우선). JSON만 출력.")

def build(c):
    return (f"[도메인] {c.get('domain')}  [제목] {c.get('title','')}\n"
            f"[본문]\n{(c.get('text') or '')[:1500]}\n\n"
            f"이 청크로 한국 주권 관련 '실질 질문'을 생성할 수 있는가?\n"
            f'JSON: {{"verdict":"KEEP|DROP","reason":"한줄"}}')

async def judge(c, client, model, sem):
    async with sem:
        for k in range(3):
            try:
                r = await client.chat.completions.create(model=model, temperature=0.0, max_tokens=200,
                    messages=[{"role":"system","content":SYS},{"role":"user","content":build(c)}],
                    extra_body={"chat_template_kwargs":{"thinking":False}})
                t = r.choices[0].message.content or ""
                if "{" not in t: t = getattr(r.choices[0].message,"reasoning_content","") or t
                m = re.search(r'\{.*\}', t, re.DOTALL)
                if not m: continue
                v = json.loads(m.group(0))
                vd = str(v.get("verdict","")).upper()
                if vd in ("KEEP","DROP"):
                    return {**c, "sovereign_keep": vd=="KEEP", "gate_reason": str(v.get("reason",""))[:160]}
            except Exception: pass
            await asyncio.sleep(0.3)
        return {**c, "sovereign_keep": True, "gate_reason": "[게이트오류-보존]"}  # 실패시 보존(안전)

async def main_async(a):
    chunks = [json.loads(l) for l in open(a.manifest, encoding="utf-8") if l.strip()]
    chunks = [c for c in chunks if c.get("keep", True)]
    if a.sample: chunks = chunks[:a.sample]
    print(f"[gate] keep 청크 {len(chunks)}개 심사 (conc {a.concurrency})")
    client = AsyncOpenAI(base_url=a.url, api_key="EMPTY", timeout=120)
    sem = asyncio.Semaphore(a.concurrency)
    res = await asyncio.gather(*[judge(c, client, a.model, sem) for c in chunks])
    Path(a.out).write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in res)+"\n", encoding="utf-8")
    kept = sum(1 for x in res if x["sovereign_keep"])
    print(f"[gate] KEEP {kept} / DROP {len(res)-kept} ({(len(res)-kept)/len(res):.0%} 노이즈 제거)")
    print(f"  도메인별 DROP율:")
    bd = Counter(); bk = Counter()
    for x in res:
        bd[x["domain"]] += 0 if x["sovereign_keep"] else 1; bk[x["domain"]] += 1
    for d in sorted(bk, key=lambda d:-bd[d]):
        print(f"    {d}: {bd[d]}/{bk[d]} drop")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(MAN))
    ap.add_argument("--out", default=str(GEN/"chunks_gated.jsonl"))
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--url", default="http://localhost:8000/v1")
    ap.add_argument("--model", default="kimi-k2.6")
    ap.add_argument("--concurrency", type=int, default=32)
    asyncio.run(main_async(ap.parse_args()))
if __name__ == "__main__": main()
