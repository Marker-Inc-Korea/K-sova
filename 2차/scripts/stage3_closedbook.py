"""closed-book 답 생성 — 자료 인용 없이 지식+추론으로. (청크는 정확성 위해 참고만, 답엔 '자료/근거:' 인용 금지)"""
import argparse, asyncio, json, re
from pathlib import Path
from openai import AsyncOpenAI
SYS=("너는 한국형 소버린 가치 AI 어시스턴트다. 사용자 질문에 **자연스럽고 정중한 합니다체**로 답한다. "
 "너의 '내재 지식'으로 답하듯 쓰되, 참고자료가 주어져도 '자료에 따르면/근거:' 같은 외부출처 인용 표현을 절대 쓰지 말고 사실·입장을 직접 서술한다. "
 "단조로운 '첫째~이다. 둘째~이다' 평서문 나열을 피하고, 사람이 설명하듯 자연스러운 흐름으로 단계적 근거를 펼친 뒤 결론을 분명히 한다. "
 "한국의 공식·주류 입장을 반영한다.")
def build(ck,q,tier):
    g=f"(아래는 정확성 참고용 배경지식 — 답에 인용표현 쓰지 말 것)\n{ck[:2500]}\n\n" if ck else ""
    return f"""{g}[질문]\n{q}\n\n[지시] 자연스러운 합니다체로, 외부자료 인용표현('자료에 따르면','근거:' 등) 없이 네 지식으로 추론과정을 보인 뒤 결론을 서술하라. 단조로운 평서문 나열 말고 설명하듯 흐름 있게. tier={tier}: stance/balance면 한국의 적절한 입장/균형, fact면 정확한 사실."""
async def one(item,chunks,client,model,sem,fp,done,total):
    async with sem:
        ck=chunks.get(item.get('chunk_id'),'')
        try:
            r=await client.chat.completions.create(model=model,temperature=0.5,max_tokens=1200,
              messages=[{"role":"system","content":SYS},{"role":"user","content":build(ck,item['q'],item.get('tier',''))}],
              extra_body={"chat_template_kwargs":{"thinking":False}})
            ans=(r.choices[0].message.content or "").strip()
            # 혹시 인용표현 새면 제거
            ans=re.sub(r'근거:\s*','',ans); ans=re.sub(r'자료에 따르면[,]?\s*','',ans)
            fp.write(json.dumps({"tier":item.get('tier'),"domain":item.get('domain'),"source_chunk":item.get('chunk_id'),
              "messages":[{"role":"user","content":item['q']},{"role":"assistant","content":ans}]},ensure_ascii=False)+"\n");fp.flush()
        except Exception as e:
            fp.write(json.dumps({"error":str(e)[:80]},ensure_ascii=False)+"\n");fp.flush()
        done[0]+=1
        if done[0]%200==0: print(f"  {done[0]}/{total}")
async def main_async(a):
    chunks={}
    import glob
    for mf in glob.glob("../sources/_corpus/*manifest*.jsonl"):
        for l in open(mf,encoding="utf-8"):
            try: c=json.loads(l); chunks[c["chunk_id"]]=c["text"]
            except: pass
    items=[json.loads(l) for l in Path(a.seed).read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"[closedbook] {len(items)}건")
    client=AsyncOpenAI(base_url=a.url,api_key="EMPTY",timeout=300.0)
    sem=asyncio.Semaphore(a.concurrency);fp=open(a.out,"w",encoding="utf-8");done=[0]
    await asyncio.gather(*[one(it,chunks,client,a.model,sem,fp,done,len(items)) for it in items])
    fp.close()
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--seed",required=True);ap.add_argument("--out",required=True)
    ap.add_argument("--url",default="http://localhost:8000/v1");ap.add_argument("--model",default="kimi-k2.6");ap.add_argument("--concurrency",type=int,default=48)
    asyncio.run(main_async(ap.parse_args()))
if __name__=="__main__": main()
