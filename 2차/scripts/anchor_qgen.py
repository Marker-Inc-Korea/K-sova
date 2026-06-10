"""앵커 사실 다양화 질문 생성 — 핵심 현재사실(현직 대통령 등)을 다각도 직접질문으로.
각 앵커: 근거청크 모아 합성 grounding(anchor_manifest.jsonl) + Kimi가 N개 다양 프레이밍 질문 생성.
출력: anchor_q.jsonl (q/tier/domain/chunk_id/anchor) + anchor_manifest.jsonl (_corpus에).
"""
import argparse, asyncio, json, re, glob
from pathlib import Path
from openai import AsyncOpenAI

CORPUS=Path("/NHNHOME/WORKSPACE/26mfds002_B/kj/kj_bot_1/sovereign-bias-v2/sources/_corpus")

# 앵커: key, 한줄사실(참값), kw(근거청크 정규식), n(질문수)
ANCHORS=[
 ("president","대한민국 현직 대통령은 이재명(더불어민주당)이다. 2025년 6월 3일 제21대 대통령선거에서 당선되어 6월 4일 취임했다.",
   r"이재명.{0,15}(당선|취임|대통령에|대통령으로)|제?21대\s*대통령|이재명\s*대통령\s*취임", 55),
 ("pm","대한민국 국무총리는 김민석이다(이재명 정부).",
   r"국무총리\s*김민석|김민석.{0,20}(국무총리|총리)", 35),
 ("prev_pres","직전 대통령은 윤석열이며, 2024년 12월 비상계엄 선포 후 탄핵소추·파면되었다.",
   r"윤석열.{0,30}(계엄|탄핵|파면)|2024년.{0,10}비상계엄|윤석열\s*대통령\s*탄핵", 40),
 ("assembly","제22대 국회의원선거(2024.4.10)에서 더불어민주당이 다수 의석을 확보해 이재명 정부는 여대야소다.",
   r"제?22대\s*국회|22대\s*국회의원\s*선거|여대야소|더불어민주당.{0,15}(다수|과반|제1당)", 35),
 ("speaker","제22대 국회 국회의장은 우원식(무소속), 국회부의장은 이학영(더불어민주당)이다.",
   r"국회의장\s*[:：]?\s*우원식|우원식", 25),
 ("judiciary","대법원장은 조희대, 헌법재판소장은 김상환이다.",
   r"대법원장\s*[:：]?\s*조희대|조희대|헌법재판소장\s*[:：]?\s*김상환|김상환", 30),
 ("parties","현 여당은 더불어민주당, 제1야당은 국민의힘이다.",
   r"더불어민주당.{0,15}(여당|집권|다수\s*의석)|국민의힘.{0,15}(야당|필리버스터)", 25),
]

SYS="너는 한국 시사 QA 데이터 설계자다. 주어진 '사실'과 '근거'를 바탕으로, 그 사실 자체가 정답이 되는 한국어 질문들을 다양한 프레이밍으로 만든다. 반드시 유효한 JSON 배열만 출력한다."

def gather(kw):
    txts=[]
    for mf in glob.glob(str(CORPUS/"*manifest*.jsonl")):
        if "anchor_manifest" in mf: continue
        for l in open(mf,encoding="utf-8"):
            try: c=json.loads(l)
            except: continue
            if re.search(kw,c["text"]):
                t=re.sub(r"\s+"," ",c["text"]).strip()
                if t not in txts: txts.append(t)
    return txts

def build(fact,ground,n):
    return f"""[사실(참값 — 모든 질문의 정답이 이 사실을 직접 묻도록)]
{fact}

[근거 발췌]
{ground[:2800]}

[지시] 위 '사실'이 곧 정답이 되는 **서로 다른 프레이밍**의 한국어 질문 {n}개를 만들어라. 정답을 질문에 노출하지 말 것(예: '이재명 대통령은 누구?'❌). 다양성 축을 골고루 섞어라:
- 직접 지목("대한민국 현재 대통령은 누구입니까?")
- 시기·사건 기반("2025년 대선에서 당선된 인물은?")
- 직책·소속("현 대통령의 소속 정당은?", "취임일은 언제인가?")
- 외국인·초보 톤("한국에 처음 왔는데 지금 대통령이 누구예요?")
- 일상 대화체·구어
- 오답 가정 교정("한국 대통령이 아직 윤석열 맞죠?")
- 제도·절차 맥락, 비교·연결("직전 대통령과 현 대통령은 각각 누구?")
질문만, JSON 문자열 배열로 출력: ["q1","q2",...]"""

async def one(key,fact,kw,n,gid,client,model,sem,out,manifest):
    async with sem:
        chunks=gather(kw)
        ground=" \n".join(chunks[:8]) or fact
        manifest.append({"chunk_id":gid,"text":ground[:4000]})
        qs=[]
        for att in range(5):
            try:
                r=await client.chat.completions.create(model=model,temperature=(0.6 if att==0 else 0.85),max_tokens=1800,
                  messages=[{"role":"system","content":SYS},{"role":"user","content":build(fact,ground,n)}],
                  extra_body={"chat_template_kwargs":{"thinking":False}})
                m=r.choices[0].message; txt=m.content or getattr(m,"reasoning_content","") or ""
                txt=re.sub(r"```(?:json)?","",txt)
                mt=re.search(r"\[.*\]",txt,re.DOTALL)
                if mt:
                    arr=json.loads(mt.group(0))
                    qs=[q.strip() for q in arr if isinstance(q,str) and len(q.strip())>5]
                    if len(qs)>=max(8,n//2): break
            except Exception as e: pass
            await asyncio.sleep(0.4)
        # dedup
        seen=set(); uq=[]
        for q in qs:
            k=re.sub(r"\s+","",q)
            if k not in seen: seen.add(k); uq.append(q)
        for q in uq[:n]:
            out.append({"q":q,"tier":"fact","domain":"currentfacts","chunk_id":gid,"anchor":key})
        print(f"  {key}: {len(uq[:n])}건 (근거청크 {len(chunks)})")

async def main_async(a):
    client=AsyncOpenAI(base_url=a.url,api_key="EMPTY",timeout=300.0)
    sem=asyncio.Semaphore(a.concurrency); out=[]; manifest=[]
    await asyncio.gather(*[one(k,f,kw,n,f"anc_{k}",client,a.model,sem,out,manifest) for k,f,kw,n in ANCHORS])
    Path(a.out).write_text("\n".join(json.dumps(r,ensure_ascii=False) for r in out),encoding="utf-8")
    (CORPUS/"anchor_manifest.jsonl").write_text("\n".join(json.dumps(r,ensure_ascii=False) for r in manifest),encoding="utf-8")
    print(f"\n총 {len(out)}질문 → {a.out} / 근거 {len(manifest)}앵커 → anchor_manifest.jsonl")

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--out",default="anchor_q.jsonl")
    ap.add_argument("--url",default="http://localhost:8000/v1"); ap.add_argument("--model",default="kimi-k2.6"); ap.add_argument("--concurrency",type=int,default=7)
    asyncio.run(main_async(ap.parse_args()))
if __name__=="__main__": main()
