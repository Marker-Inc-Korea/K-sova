#!/usr/bin/env python3
"""K-SOVA 4차 — 형식 준수(instruction-following) 보강셋 생성.
3차의 verbose 드리프트(단답 형식 이탈) 교정용. '형식 지시를 정확히 따라 간결히 답하는' 예시.
유형: 객관식(정답만)·단답(한 단어)·예/아니오·간단설명(1문장). 일반지식/과학/상식/기초 한국문화 등.
⚠️ 표준 벤치(KMMLU·KoBEST·HAE-RAE) 복제 금지(평가 오염 방지) — 새로 창작.
출력: format_set.jsonl {tier:"format", type, messages:[user,assistant]}
"""
import json, re, asyncio, argparse, random
from pathlib import Path
from openai import AsyncOpenAI
random.seed(7)
GEN = Path("/NHNHOME/WORKSPACE/26mfds002_B/kj/kj_bot_1/sovereign-bias-v2/gen")
TOPICS = ["일반 상식","생활 과학","한국 지리","한국 역사 기초","세계사 기초","수학·논리","일상생활",
          "기술·IT","자연·동식물","경제 기초","문화·예술","언어·맞춤법","건강·의학 상식","스포츠",
          "음식·요리","교통·안전","환경·기후","우주·천문","법·제도 기초","심리·사회"]
SYS = ("너는 한국어 instruction-following 학습데이터 출제자다. 사용자가 답변 '형식'을 지정하면 "
       "그 형식을 정확히 지켜 **간결하게** 답하는 예시를 만든다. 장황한 설명 금지. "
       "표준 벤치마크(KMMLU·KoBEST·HAE-RAE 등) 문항을 그대로 베끼지 말고 새로 창작한다. JSON 배열만 출력.")

def build(topic, n):
    return f"""[주제] {topic}

[지시] 위 주제로 '형식 준수' 한국어 예시 {n}개를 만들어라. 아래 4가지 형식을 골고루 섞어라:
- mc: 4지선다 질문(보기 A~D 포함) + "정답만 한 글자로(A/B/C/D)" 지시 → 답은 정답 글자 하나(예: "C")
- short: 단답 질문 + "한 단어로만" 지시 → 답은 한 단어
- yesno: 진위 질문 + "예 또는 아니오로만" 지시 → 답은 "예" 또는 "아니오"
- brief: 질문 + "한 문장으로 간단히" 지시 → 답은 한 문장

규칙: 질문(q)에 형식 지시를 포함. 답(a)은 그 형식을 정확히 지켜 **간결하게**. 사실은 정확하게.
JSON 배열만: [{{"type":"mc","q":"...","a":"C"}},{{"type":"short","q":"...","a":"..."}}, ...]"""

async def gen(topic, n, client, model, sem):
    async with sem:
        for k in range(4):
            try:
                r = await client.chat.completions.create(model=model, temperature=(0.7 if k==0 else 0.9), max_tokens=2000,
                    messages=[{"role":"system","content":SYS},{"role":"user","content":build(topic,n)}],
                    extra_body={"chat_template_kwargs":{"thinking":False}})
                t = r.choices[0].message.content or ""
                if "[" not in t: t = getattr(r.choices[0].message,"reasoning_content","") or t
                t = re.sub(r"```(?:json)?","",t); m = re.search(r"\[.*\]", t, re.DOTALL)
                if not m: continue
                out=[]
                for it in json.loads(m.group(0)):
                    if isinstance(it,dict) and it.get("q") and it.get("a") and it.get("type") in ("mc","short","yesno","brief"):
                        out.append({"tier":"format","type":it["type"],"domain":topic,
                                    "messages":[{"role":"user","content":str(it["q"]).strip()},
                                                {"role":"assistant","content":str(it["a"]).strip()}]})
                if out: return out
            except Exception: pass
            await asyncio.sleep(0.4)
        return []

async def main_async(a):
    client = AsyncOpenAI(base_url=a.url, api_key="EMPTY", timeout=120); sem = asyncio.Semaphore(a.concurrency)
    per = 15; calls = a.target // per + 1
    tasks = [gen(TOPICS[i % len(TOPICS)], per, client, a.model, sem) for i in range(calls)]
    res = await asyncio.gather(*tasks)
    items = [x for r in res for x in r]
    # 간단 중복 제거(질문 동일)
    seen=set(); uniq=[]
    for x in items:
        q=x["messages"][0]["content"]
        if q in seen: continue
        seen.add(q); uniq.append(x)
    from collections import Counter
    Path(GEN/a.out).write_text("\n".join(json.dumps(x,ensure_ascii=False) for x in uniq)+"\n")
    print(f"[format_set] {len(uniq)}건 (유형: {dict(Counter(x['type'] for x in uniq))}) → {a.out}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="format_set.jsonl")
    ap.add_argument("--target", type=int, default=3000)
    ap.add_argument("--url", default="http://localhost:8000/v1")
    ap.add_argument("--model", default="kimi-k2.6")
    ap.add_argument("--concurrency", type=int, default=32)
    asyncio.run(main_async(ap.parse_args()))
if __name__ == "__main__": main()
