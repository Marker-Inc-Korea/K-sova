"""진짜 중요한 주권/가치 이슈에 해당하는 청크만 선별 → core_manifest.jsonl.
이슈별 키워드 매칭, 이슈당 캡. tier_prior: 헌재 결정문=court, 그 외=stance.
"""
import json, re
from pathlib import Path
from collections import defaultdict

ROOT=Path("/NHNHOME/WORKSPACE/26mfds002_B/kj/kj_bot_1/sovereign-bias-v2/sources/_corpus")
MANIS=[ROOT/"chunks_manifest.jsonl", ROOT/"currentfacts_manifest.jsonl", ROOT/"currentfacts2_manifest.jsonl"]

CORE={
 "독도":["독도","다케시마","우산도","석도","태정관","시마네","은주시청"],
 "동해표기":["동해","일본해","East Sea","동해 표기"],
 "위안부":["위안부"],
 "강제동원":["강제동원","강제징용","징용","강제연행","사도광산","군함도","하시마"],
 "동북공정·고대사":["동북공정","고구려","발해","간도","임나일본"],
 "식민지배·조약":["을사늑약","한일병합","국권피탈","한일기본조약","청구권협정","경술국치"],
 "친일청산":["친일","반민특위","반민족행위"],
 "3·1운동·임정":["3·1운동","3.1운동","임시정부","대한민국임시정부","광복절","독립운동"],
 "민주화운동":["4·19","4.19","5·18","5.18","6월 항쟁","6월항쟁","민주화운동","광주 민주","부마"],
 "헌법핵심가치":["민주공화국","국민주권","인간의 존엄","자유민주적 기본질서","권력분립","삼권분립","법치주의"],
 "평화통일":["평화통일","평화 통일","한반도 평화","통일 정책","민족공동체통일"],
 "인권·평등·차별금지":["인권","차별금지","평등권","양성평등","장애인 권리","소수자"],
 "사법독립·적법절차":["사법권 독립","법관의 독립","적법절차","영장주의","무죄추정"],
 "헌재 쟁점판례":["사형","낙태","양심적 병역","간통","호주제","통합진보당","정당해산"],
}

def main():
    chunks=[]
    for mf in MANIS:
        if mf.exists():
            for l in mf.read_text(encoding="utf-8").splitlines():
                if l.strip():
                    c=json.loads(l)
                    if c.get("keep",True): chunks.append(c)
    CAP=12
    picked={}; by_issue=defaultdict(int)
    for c in chunks:
        hay=(c.get("title","")+" "+c["text"])
        matched=None
        for issue,kws in CORE.items():
            if any(k in hay for k in kws):
                matched=issue; break
        if not matched: continue
        if by_issue[matched]>=CAP: continue
        cid=c["chunk_id"]
        if cid in picked: continue
        c=dict(c)
        c["core_issue"]=matched
        # tier_prior 재설정: 헌재 결정문=court, 쟁점판례=court, 그 외 핵심이슈=stance
        if c.get("source_type")=="court_decision" or matched=="헌재 쟁점판례":
            c["tier_prior"]="court"
        else:
            c["tier_prior"]="stance"
        picked[cid]=c; by_issue[matched]+=1
    rows=list(picked.values())
    with open(ROOT/"core_manifest.jsonl","w",encoding="utf-8") as f:
        for r in rows: f.write(json.dumps(r,ensure_ascii=False)+"\n")
    print(f"[core 선별] {len(rows)} 청크")
    for issue,n in sorted(by_issue.items(),key=lambda x:-x[1]): print(f"  {n:3d}  {issue}")

main()
