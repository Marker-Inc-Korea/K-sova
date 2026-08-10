#!/usr/bin/env python3
"""소버린 적격성 전수 감사 — 학습셋의 소버린 항목 전부가 '한국 주권 실질'인지 독립 재판정.
게이트/judge/classify 통과한 결과물을 재검증(혼입 잡음 색출). 샘플 아님, 전건.
대상: source in (sovereign, anchor) & tier in (fact,stance,balance) — general/format/robustness 제외.
출력: audit_sovereign.jsonl (항목별 verdict) + 요약(비-소버린 목록).
"""
import json, re, asyncio, argparse
from pathlib import Path
from collections import Counter, defaultdict
from openai import AsyncOpenAI
GEN = Path("/NHNHOME/WORKSPACE/26mfds002_B/kj/kj_bot_1/sovereign-bias-v2/gen")

SYS = ("너는 한국 주권가치 학습데이터 '소버린 적격성 감사관'이다. 주어진 Q-A가 "
       "**한국의 주권·국가적 실질**(영토·역사주권·헌정·현 국가체제/최신성·주권적 가치·정책쟁점·기술/경제주권, "
       "또는 한국의 역사·사회·문화·지리·제도 등 '한국 국가지식')을 담으면 sovereign=true.\n"
       "★ 주제로 거르지 마라 — 과학·기술·경제·사회·문화 어느 분야든 '한국의 실질'이면 true.\n"
       "**sovereign=false는 오직**: ①문서 메타·구조(집필자·설문문항·DB방법론), ②조사방법론·일회성 수치 나열, "
       "③한국과 무관한 일반지식·상식, ④질문/답이 주권·한국 국가성과 무관한 오프토픽 — 일 때만.\n"
       "애매하면 true(보존 우선). JSON만 출력.")

def build(r):
    q = r["messages"][0]["content"][:600]; a = r["messages"][1]["content"][:900]
    return (f"[tier] {r.get('tier')}  [domain] {r.get('domain')}  [source] {r.get('source')}\n"
            f"[질문]\n{q}\n[답]\n{a}\n\n"
            f'JSON: {{"sovereign":true/false,"category":"영토|역사주권|헌정|최신성|가치쟁점|기술경제주권|사회문화지리제도|잡음메타|방법론수치|일반지식|오프토픽","reason":"한줄"}}')

async def judge(r, client, model, sem):
    async with sem:
        for k in range(3):
            try:
                resp = await client.chat.completions.create(model=model, temperature=0.0, max_tokens=200,
                    messages=[{"role":"system","content":SYS},{"role":"user","content":build(r)}],
                    extra_body={"chat_template_kwargs":{"thinking":False}})
                t = resp.choices[0].message.content or ""
                if "{" not in t: t = getattr(resp.choices[0].message,"reasoning_content","") or t
                m = re.search(r'\{.*\}', t, re.DOTALL)
                if not m: continue
                v = json.loads(m.group(0))
                if isinstance(v.get("sovereign"), bool):
                    return {"id":r.get("id"),"tier":r.get("tier"),"domain":r.get("domain"),"source":r.get("source"),
                            "sovereign":v["sovereign"],"category":str(v.get("category",""))[:40],"reason":str(v.get("reason",""))[:160],
                            "q":r["messages"][0]["content"][:120]}
            except Exception: pass
            await asyncio.sleep(0.3)
        return {"id":r.get("id"),"tier":r.get("tier"),"domain":r.get("domain"),"source":r.get("source"),
                "sovereign":True,"category":"ERR","reason":"감사오류-보존","q":r["messages"][0]["content"][:120]}

async def main_async(a):
    files = a.files
    rows = []
    for f in files:
        for l in open(f, encoding="utf-8"):
            if not l.strip(): continue
            r = json.loads(l)
            if r.get("source") in ("sovereign","anchor") and r.get("tier") in ("fact","stance","balance"):
                r["_file"] = Path(f).name; rows.append(r)
    print(f"[audit] 소버린 항목 {len(rows)}건 전수 감사 (files={[Path(f).name for f in files]}, conc {a.concurrency})")
    client = AsyncOpenAI(base_url=a.url, api_key="EMPTY", timeout=120)
    sem = asyncio.Semaphore(a.concurrency)
    res = await asyncio.gather(*[judge(r, client, a.model, sem) for r in rows])
    for r, v in zip(rows, res): v["_file"] = r["_file"]
    Path(a.out).write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in res)+"\n", encoding="utf-8")
    nsov = sum(1 for x in res if x["sovereign"]); nno = len(res)-nsov
    nerr = sum(1 for x in res if x["category"]=="ERR")
    print(f"\n===== 감사 결과 (전건 {len(res)}) =====")
    print(f"  소버린 {nsov} ({nsov/len(res):.1%}) / 비-소버린 {nno} ({nno/len(res):.1%}) / 감사오류 {nerr}")
    print(f"  비-소버린 카테고리: {dict(Counter(x['category'] for x in res if not x['sovereign']))}")
    print(f"  비-소버린 tier별: {dict(Counter(x['tier'] for x in res if not x['sovereign']))}")
    print(f"  비-소버린 파일별: {dict(Counter(x['_file'] for x in res if not x['sovereign']))}")
    print(f"\n  ── 비-소버린 예시(최대 20) ──")
    for x in [x for x in res if not x["sovereign"]][:20]:
        print(f"   [{x['category']}/{x['tier']}/{x['domain']}] {x['q'][:70]} || {x['reason'][:60]}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="+", default=[str(GEN/"train_v4_clean.jsonl"), str(GEN/"eval_v4.jsonl"), str(GEN/"test_v4.jsonl")])
    ap.add_argument("--out", default=str(GEN/"audit_sovereign.jsonl"))
    ap.add_argument("--url", default="http://localhost:8000/v1")
    ap.add_argument("--model", default="kimi-k2.6")
    ap.add_argument("--concurrency", type=int, default=32)
    asyncio.run(main_async(ap.parse_args()))
if __name__ == "__main__": main()
