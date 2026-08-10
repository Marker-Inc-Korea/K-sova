#!/usr/bin/env python3
"""Serve a mobile-first review page for K-SOVA issues with passed items."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import threading
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


DOMAIN_NAMES = {
    "D01": "헌정 정체성과 국가의 법통",
    "D02": "민주주의·기본권과 공적 정당성",
    "D03": "영토·관할권과 공간 주권",
    "D04": "역사 기억과 정의·회복",
    "D05": "문화 정체성과 표현·재현",
    "D06": "외교·안보와 전략적 자율성",
    "D07": "경제·산업·자원 선택권",
    "D08": "과학기술·데이터·정보공간 통제권",
}

LANE_NAMES = {
    "democracy_rights_values": "민주주의·인권·가치",
    "territory_maritime": "영토·해양",
    "history_memory": "역사·기억",
    "culture_identity": "문화·정체성",
    "diplomacy_security": "외교·안보",
    "inter_korean": "남북관계",
    "trade_industry": "통상·산업",
    "energy_resources": "에너지·자원",
    "technology_data_science": "과학기술·데이터",
}


HTML = r"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<meta name="robots" content="noindex,nofollow">
<title>K-SOVA 핵심 쟁점 검토</title>
<style>
:root{color-scheme:light;--ink:#17212b;--muted:#687482;--line:#dce2e8;--paper:#fff;--bg:#f3f6f8;--yes:#0f7a4b;--no:#ba3128;--maybe:#9a6400;--accent:#1e64b7}
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:var(--bg);color:var(--ink);font-family:system-ui,-apple-system,"Noto Sans KR",sans-serif;line-height:1.52}
header{position:sticky;top:0;z-index:5;background:#fffffff2;backdrop-filter:blur(9px);border-bottom:1px solid var(--line);padding:10px 13px}
.head{max-width:920px;margin:auto}.titleline{display:flex;justify-content:space-between;gap:10px;align-items:center}.title{font-size:17px;font-weight:850}.counts{font-size:12px;color:var(--muted);white-space:nowrap}.bar{height:7px;background:#e8edf2;border-radius:10px;overflow:hidden;margin-top:7px}.fill{height:100%;width:0;background:var(--accent);transition:width .2s}
main{max-width:920px;margin:12px auto 150px;padding:0 11px}.notice,.toolbar,.card{background:var(--paper);border:1px solid var(--line);border-radius:12px}.notice{padding:10px 12px;margin-bottom:10px;font-size:13px;color:#465564}.toolbar{padding:9px;margin-bottom:10px;display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:7px}.toolbar .wide{grid-column:1/-1}.navrow{display:grid;grid-template-columns:1fr 1fr 1fr;gap:7px;grid-column:1/-1}
button,select,input,textarea{font:inherit}button,select,input,textarea{border:1px solid #bec8d2;border-radius:9px;background:#fff;padding:9px;color:var(--ink)}button{cursor:pointer}button:disabled{opacity:.4}.search{width:100%}.card{padding:17px;box-shadow:0 2px 8px #2434470b}.meta{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:11px}.badge{font-size:11px;padding:3px 8px;border-radius:20px;background:#edf3f9;color:#35546f}.badge.benchmark{background:#eaf2ff;color:#1d55a0}.badge.training{background:#f3edff;color:#6842a0}.badge.O{background:#e7f6ee;color:var(--yes)}.badge.X{background:#ffefed;color:var(--no)}.badge.Q{background:#fff5d9;color:var(--maybe)}
h1{font-size:21px;line-height:1.42;margin:6px 0 17px;word-break:keep-all}h2{font-size:14px;margin:18px 0 5px;color:#385068}p{margin:5px 0 10px;word-break:keep-all}.core{font-size:16px;font-weight:650;background:#f7f9fb;border-left:4px solid #6f9bc9;padding:12px;border-radius:7px}.stake{font-size:14px}.muted{color:var(--muted);font-size:13px}.empty{text-align:center;color:var(--muted);padding:70px 10px}
details{border-top:1px solid var(--line);margin-top:15px;padding-top:12px}summary{cursor:pointer;font-weight:750;font-size:14px}.detailtext{font-size:14px;white-space:pre-wrap;margin-top:9px}.questions{margin:8px 0 0;padding-left:20px}.questions li{margin:8px 0;font-size:14px}.related{font-size:13px;color:#53616e;margin-top:7px}
.decision{position:fixed;z-index:6;left:0;right:0;bottom:0;background:#fffffff5;backdrop-filter:blur(9px);border-top:1px solid var(--line);padding:9px 10px calc(9px + env(safe-area-inset-bottom))}.decision-inner{max-width:920px;margin:auto}.note{width:100%;height:39px;min-height:39px;resize:vertical;padding:7px 9px;margin-bottom:7px}.buttons{display:grid;grid-template-columns:1fr 1fr 1fr;gap:7px}.decide{font-weight:850;font-size:16px;padding:12px 5px}.yes{color:var(--yes);border-color:#80c6a2}.no{color:var(--no);border-color:#e2a09a}.maybe{color:var(--maybe);border-color:#d7b966}.save{font-size:12px;color:var(--muted);height:18px;text-align:center;margin-top:3px}
@media(min-width:700px){.toolbar{grid-template-columns:repeat(4,minmax(0,1fr))}.toolbar .wide{grid-column:span 2}.navrow{grid-column:span 2}.card{padding:23px}h1{font-size:24px}}
</style>
</head>
<body>
<header><div class="head"><div class="titleline"><div class="title">K-SOVA 핵심 쟁점 검토</div><div id="counts" class="counts">불러오는 중…</div></div><div class="bar"><div id="fill" class="fill"></div></div></div></header>
<main>
  <div class="notice"><b>쟁점만 판단:</b> 핵심 평가에 필요하면 O, 핵심 벤치마크에는 부적합하면 X, 바로 판단하기 어렵다면 ?를 누르세요. 목표 수량에 맞추지 말고 내용만 보세요.</div>
  <div class="toolbar">
    <select id="status"><option value="pending">미검토만</option><option value="all">전체</option><option value="O">O 핵심</option><option value="X">X 비핵심</option><option value="?">? 애매</option></select>
    <select id="origin"><option value="all">출처 전체</option><option value="benchmark">벤치 통과 문항</option><option value="training">학습 통과 문항</option></select>
    <select id="category"><option value="all">영역 전체</option></select>
    <input id="search" class="search wide" placeholder="제목·쟁점 검색">
    <div class="navrow"><button id="prev">← 이전</button><button id="all">전체 보기</button><button id="next">다음 →</button></div>
  </div>
  <section id="view" class="card"><div class="empty">쟁점을 불러오는 중입니다.</div></section>
</main>
<div class="decision"><div class="decision-inner"><textarea id="note" class="note" placeholder="메모(선택사항)"></textarea><div class="buttons"><button id="yes" class="decide yes">O 핵심</button><button id="maybe" class="decide maybe">? 애매</button><button id="no" class="decide no">X 비핵심</button></div><div id="save" class="save"></div></div></div>
<script>
const prefix=location.pathname.replace(/\/$/,'');
const state={issues:[],decisions:{},visible:[],pos:0,saving:false};
const $=id=>document.getElementById(id);const el=(tag,cls,text)=>{const n=document.createElement(tag);if(cls)n.className=cls;if(text!==undefined)n.textContent=text;return n};
function entry(id){return state.decisions[id]||{decision:null,note:''}}
function statusLabel(v){return v==='O'?'O 핵심':v==='X'?'X 비핵심':v==='?'?'? 애매':''}
function updateCounts(){let o=0,x=0,q=0;state.issues.forEach(r=>{const d=entry(r.issue_id).decision;if(d==='O')o++;else if(d==='X')x++;else if(d==='?')q++});const done=o+x+q,total=state.issues.length;$('counts').textContent=`${done}/${total} · O ${o} · X ${x} · ? ${q}`;$('fill').style.width=(total?done/total*100:0)+'%'}
function filtered(){const s=$('status').value,o=$('origin').value,c=$('category').value,query=$('search').value.trim().toLowerCase();return state.issues.filter(r=>{const d=entry(r.issue_id).decision;if(s==='pending'&&d)return false;if(!['pending','all'].includes(s)&&d!==s)return false;if(o!=='all'&&r.origin!==o)return false;if(c!=='all'&&r.category_id!==c)return false;if(query&&!`${r.title_ko} ${r.decision_core_ko} ${r.sovereign_stake_ko}`.toLowerCase().includes(query))return false;return true})}
function refresh(keep){state.visible=filtered();const idx=state.visible.findIndex(r=>r.issue_id===keep);state.pos=idx>=0?idx:Math.min(state.pos,Math.max(0,state.visible.length-1));render()}
function addSection(box,title,text,cls){if(!text)return;box.append(el('h2','',title));box.append(el('p',cls||'',text))}
function render(){updateCounts();const box=$('view');box.replaceChildren();if(!state.visible.length){box.append(el('div','empty','현재 조건에 해당하는 쟁점이 없습니다.'));$('note').value='';return}const r=state.visible[state.pos],d=entry(r.issue_id);const meta=el('div','meta');meta.append(el('span','badge',`${state.pos+1}/${state.visible.length} · 전체 ${r.review_order}/${state.issues.length}`));meta.append(el('span',`badge ${r.origin}`,r.origin==='benchmark'?'벤치 통과':'학습 통과'));meta.append(el('span','badge',r.category_label));meta.append(el('span','badge',`통과 문항 ${r.question_count}`));if(d.decision)meta.append(el('span',`badge ${d.decision==='?'?'Q':d.decision}`,statusLabel(d.decision)));box.append(meta);box.append(el('h1','',r.title_ko));addSection(box,'판단 핵심',r.decision_core_ko,'core');addSection(box,'K-SOVA 이해관계',r.sovereign_stake_ko,'stake');const ref=el('details');ref.append(el('summary','','한국 기준선 보기'));ref.append(el('div','detailtext',r.korean_reference_direction_ko||'기준선 없음'));box.append(ref);const qs=el('details');qs.append(el('summary','',`통과한 문항 보기 (${r.question_count})`));const ul=el('ol','questions');(r.questions||[]).forEach(q=>ul.append(el('li','',q)));qs.append(ul);box.append(qs);if(r.related&&r.related.length){const rel=el('details');rel.append(el('summary','',`가까운 기존 쟁점 (${r.related.length})`));r.related.forEach(v=>rel.append(el('div','related',`${v.seed_id}: ${v.specific_issue_ko}`)));box.append(rel)}box.append(el('p','muted',`${r.issue_id} · 원천 ${r.source_count}개`));$('note').value=d.note||'';$('prev').disabled=state.pos===0;$('next').disabled=state.pos>=state.visible.length-1;scrollTo({top:0,behavior:'smooth'})}
async function save(value){if(state.saving||!state.visible.length)return;const row=state.visible[state.pos];state.saving=true;$('save').textContent='저장 중…';try{const res=await fetch(`${prefix}/api/decision`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({issue_id:row.issue_id,decision:value,note:$('note').value})});if(!res.ok)throw new Error(await res.text());const data=await res.json();state.decisions[row.issue_id]=data.entry;$('save').textContent='저장됨';if($('status').value==='pending')refresh();else{render();if(state.pos<state.visible.length-1){state.pos++;render()}}}catch(e){$('save').textContent='저장 실패: '+e.message}finally{state.saving=false}}
function move(n){if(!state.visible.length)return;state.pos=Math.max(0,Math.min(state.visible.length-1,state.pos+n));render()}
$('prev').onclick=()=>move(-1);$('next').onclick=()=>move(1);$('all').onclick=()=>{$('status').value='all';refresh()};$('yes').onclick=()=>save('O');$('no').onclick=()=>save('X');$('maybe').onclick=()=>save('?');
['status','origin','category'].forEach(id=>$(id).onchange=()=>refresh());let timer;$('search').oninput=()=>{clearTimeout(timer);timer=setTimeout(()=>refresh(),180)};
Promise.all([fetch(`${prefix}/api/issues`).then(r=>r.json()),fetch(`${prefix}/api/decisions`).then(r=>r.json())]).then(([pkg,dec])=>{state.issues=pkg.issues||[];state.decisions=dec.decisions||{};const cats=[...new Map(state.issues.map(r=>[r.category_id,r.category_label])).entries()];cats.forEach(([id,label])=>{const o=el('option','',label);o.value=id;$('category').append(o)});state.visible=filtered();render()}).catch(e=>{$('view').textContent='불러오기 실패: '+e.message});
</script>
</body></html>"""


RECOMMENDATION_HTML = r"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<meta name="robots" content="noindex,nofollow">
<title>K-SOVA 독립 재검토 결과</title>
<style>
:root{color-scheme:light;--ink:#17212b;--muted:#667482;--line:#dce2e8;--paper:#fff;--bg:#f3f6f8;--core:#0f7a4b;--support:#9a6400;--exclude:#ba3128;--accent:#1e64b7}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font-family:system-ui,-apple-system,"Noto Sans KR",sans-serif;line-height:1.5}
header{position:sticky;top:0;z-index:5;background:#fffffff2;backdrop-filter:blur(9px);border-bottom:1px solid var(--line);padding:11px 13px}.head{max-width:920px;margin:auto}.title{font-size:17px;font-weight:850}.counts{font-size:12px;color:var(--muted);margin-top:3px}
main{max-width:920px;margin:12px auto 80px;padding:0 11px}.notice,.toolbar,.card{background:var(--paper);border:1px solid var(--line);border-radius:12px}.notice{padding:11px 12px;margin-bottom:10px;font-size:13px}.toolbar{padding:9px;margin-bottom:10px;display:grid;grid-template-columns:1fr 1fr;gap:7px;position:sticky;top:69px;z-index:4}.toolbar input,.toolbar select{min-width:0;border:1px solid #bec8d2;border-radius:9px;background:#fff;padding:9px;font:inherit;color:var(--ink)}.toolbar input{grid-column:1/-1}
.list{display:grid;gap:9px}.card{padding:14px}.meta{display:flex;gap:6px;flex-wrap:wrap}.badge{font-size:11px;padding:3px 8px;border-radius:20px;background:#edf3f9;color:#35546f}.badge.core{background:#e7f6ee;color:var(--core)}.badge.supporting{background:#fff5d9;color:var(--support)}.badge.exclude{background:#ffefed;color:var(--exclude)}
h1{font-size:17px;line-height:1.42;margin:9px 0}.reason{font-size:14px;margin:7px 0}.id{font-size:12px;color:var(--muted)}details{border-top:1px solid var(--line);margin-top:10px;padding-top:8px}summary{cursor:pointer;font-size:13px;font-weight:750}.questions{padding-left:20px}.questions li{margin:7px 0;font-size:13px}.empty{text-align:center;color:var(--muted);padding:60px 10px}.back{color:var(--accent);text-decoration:none;font-size:12px;float:right}
@media(min-width:700px){.toolbar{grid-template-columns:repeat(4,1fr)}.toolbar input{grid-column:span 2}.card{padding:18px}}
</style>
</head>
<body>
<header><div class="head"><a id="back" class="back" href="#">기존 검토 화면</a><div class="title">K-SOVA 독립 재검토 결과</div><div id="counts" class="counts">불러오는 중…</div></div></header>
<main>
  <div class="notice">기존 O/X/?는 수정하지 않았습니다. 기본 화면은 사용자 판단과 최종 권고가 달라 확인이 필요한 항목만 보여줍니다. <b>보조</b>는 자동 학습 투입이 아니라 추가 학습 적합성 검사가 필요한 후보입니다.</div>
  <div class="toolbar">
    <select id="view"><option value="changes">판정 변경 후보</option><option value="all">전체 353개</option><option value="core">핵심 255</option><option value="supporting">보조 96</option><option value="exclude">제외 2</option></select>
    <select id="category"><option value="all">영역 전체</option></select>
    <input id="search" placeholder="제목·ID 검색">
  </div>
  <div id="list" class="list"><div class="empty">불러오는 중입니다.</div></div>
</main>
<script>
const prefix=location.pathname.replace(/\/recommendations\/?$/,'').replace(/\/$/,'');
const $=id=>document.getElementById(id);const el=(tag,cls,text)=>{const n=document.createElement(tag);if(cls)n.className=cls;if(text!==undefined)n.textContent=text;return n};let rows=[];
function changed(r){return (r.owner_decision==='O'&&r.final_verdict!=='core')||(['X','?'].includes(r.owner_decision)&&r.final_verdict==='core')||r.final_verdict==='exclude'}
function label(v){return v==='core'?'핵심':v==='supporting'?'보조':v==='exclude'?'제외':'유보'}
function render(){const v=$('view').value,c=$('category').value,q=$('search').value.trim().toLowerCase();const out=rows.filter(r=>(v==='all'||(v==='changes'?changed(r):r.final_verdict===v))&&(c==='all'||r.category_id===c)&&(!q||`${r.issue_id} ${r.title_ko}`.toLowerCase().includes(q)));$('counts').textContent=`현재 ${out.length}개 · 핵심 ${rows.filter(r=>r.final_verdict==='core').length} · 보조 ${rows.filter(r=>r.final_verdict==='supporting').length} · 제외 ${rows.filter(r=>r.final_verdict==='exclude').length}`;const box=$('list');box.replaceChildren();if(!out.length){box.append(el('div','empty','조건에 해당하는 항목이 없습니다.'));return}out.forEach(r=>{const card=el('article','card');const meta=el('div','meta');meta.append(el('span',`badge ${r.final_verdict}`,label(r.final_verdict)));meta.append(el('span','badge',`사용자 ${r.owner_decision}`));meta.append(el('span','badge',r.category_label));if(r.manual_override)meta.append(el('span','badge','직접 재판정'));card.append(meta);card.append(el('h1','',r.title_ko));card.append(el('p','reason',r.final_reason_ko));card.append(el('div','id',r.issue_id));const d=el('details');d.append(el('summary','',`실제 통과 문항 ${r.question_count}개`));const ul=el('ol','questions');r.questions.forEach(x=>ul.append(el('li','',x)));d.append(ul);card.append(d);box.append(card)})}
$('view').onchange=render;$('category').onchange=render;let timer;$('search').oninput=()=>{clearTimeout(timer);timer=setTimeout(render,150)};$('back').href=prefix+'/';fetch(prefix+'/api/recommendations').then(r=>r.json()).then(x=>{rows=x.recommendations||[];const cats=[...new Map(rows.map(r=>[r.category_id,r.category_label])).entries()];cats.forEach(([id,name])=>{const o=el('option','',name);o.value=id;$('category').append(o)});render()}).catch(e=>{$('list').textContent='불러오기 실패: '+e.message});
</script>
</body></html>"""


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def build_issues(args: argparse.Namespace) -> tuple[list[dict[str, Any]], str]:
    registry = {row["issue_id"]: row for row in read_jsonl(args.benchmark_issues)}
    accepted = [
        row
        for row in read_jsonl(args.benchmark_accepted)
        if row.get("item_role") == "sovereign_judgment"
    ]
    benchmark_questions: dict[str, list[str]] = {}
    for row in accepted:
        benchmark_questions.setdefault(row["issue_id"], []).append(row["question_ko"])

    output: list[dict[str, Any]] = []
    for issue_id in sorted(
        benchmark_questions,
        key=lambda value: (
            str(registry[value].get("domain_id") or ""),
            str(registry[value].get("family_id") or ""),
            str(registry[value].get("slot_id") or ""),
        ),
    ):
        row = registry[issue_id]
        questions = list(dict.fromkeys(benchmark_questions[issue_id]))
        domain_id = str(row.get("domain_id") or "unknown")
        output.append({
            "issue_id": issue_id,
            "origin": "benchmark",
            "category_id": domain_id,
            "category_label": f"{domain_id} {DOMAIN_NAMES.get(domain_id, domain_id)}",
            "title_ko": row.get("title_ko") or row.get("decision_core_ko") or issue_id,
            "decision_core_ko": row.get("decision_core_ko") or "",
            "sovereign_stake_ko": row.get("sovereign_stake_ko") or "",
            "korean_reference_direction_ko": row.get("korean_reference_direction_ko") or "",
            "questions": questions,
            "question_count": len(questions),
            "source_count": len(set(row.get("source_ids") or [])),
            "related": [],
        })

    training_examples = read_jsonl(args.training_examples)
    accepted_material_ids = {row["material_id"] for row in training_examples}
    materials = {
        row["material_id"]: row
        for row in read_jsonl(args.training_materials)
        if row.get("material_id") in accepted_material_ids
    }
    missing = accepted_material_ids - set(materials)
    if missing:
        raise ValueError(f"training material records missing: {sorted(missing)}")
    training_questions: dict[str, list[str]] = {}
    for row in training_examples:
        training_questions.setdefault(row["material_id"], []).append(row["question_ko"])
    for material_id in sorted(accepted_material_ids, key=lambda value: (
        str(materials[value].get("lane") or ""),
        str(materials[value].get("specific_issue_ko") or ""),
    )):
        row = materials[material_id]
        lane = str(row.get("lane") or "training")
        questions = list(dict.fromkeys(training_questions[material_id]))
        related = []
        for value in row.get("related_existing_issues") or []:
            if not isinstance(value, dict):
                continue
            seed_id = str(value.get("seed_id") or "")
            if seed_id.startswith("KSV2-"):
                related.append({
                    "seed_id": seed_id,
                    "specific_issue_ko": str(value.get("specific_issue_ko") or ""),
                })
        output.append({
            "issue_id": material_id,
            "origin": "training",
            "category_id": f"T-{lane}",
            "category_label": f"학습 · {LANE_NAMES.get(lane, lane)}",
            "title_ko": row.get("specific_issue_ko") or row.get("decision_object_ko") or material_id,
            "decision_core_ko": row.get("decision_object_ko") or "",
            "sovereign_stake_ko": row.get("korean_stake_ko") or row.get("why_ksova_ko") or "",
            "korean_reference_direction_ko": row.get("minimum_baseline_ko") or "",
            "questions": questions,
            "question_count": len(questions),
            "source_count": len(set(row.get("source_ids") or [])),
            "related": related[:8],
        })

    for index, row in enumerate(output, 1):
        row["review_order"] = index

    source_hashes = {
        "benchmark_issues": sha256(args.benchmark_issues),
        "benchmark_accepted": sha256(args.benchmark_accepted),
        "training_examples": sha256(args.training_examples),
        "training_materials": sha256(args.training_materials),
    }
    package_hash = hashlib.sha256(
        json.dumps(source_hashes, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return output, package_hash


class ReviewState:
    def __init__(
        self,
        issues: list[dict[str, Any]],
        package_hash: str,
        decisions_path: Path,
        recommendations_path: Path | None = None,
    ):
        self.issues = issues
        self.issue_ids = {row["issue_id"] for row in issues}
        self.package_hash = package_hash
        self.decisions_path = decisions_path.resolve()
        self.recommendations = self._load_recommendations(recommendations_path)
        self.lock = threading.Lock()
        self.decisions = self._load()

    def _load_recommendations(self, path: Path | None) -> list[dict[str, Any]]:
        if path is None:
            return []
        rows = read_jsonl(path)
        ids = [str(row.get("issue_id") or "") for row in rows]
        if len(rows) != len(self.issues) or set(ids) != self.issue_ids or len(ids) != len(set(ids)):
            raise ValueError("recommendations must exactly cover review issues")
        return sorted(rows, key=lambda row: int(row["review_order"]))

    def _load(self) -> dict[str, dict[str, Any]]:
        if not self.decisions_path.exists():
            return {}
        value = json.loads(self.decisions_path.read_text(encoding="utf-8"))
        if value.get("package_hash") != self.package_hash:
            raise ValueError("decision file belongs to a different issue package")
        decisions = value.get("decisions") or {}
        if not isinstance(decisions, dict) or not set(decisions) <= self.issue_ids:
            raise ValueError("invalid issue decision file")
        return decisions

    def payload(self) -> dict[str, Any]:
        with self.lock:
            return {
                "schema_version": "ksova-core-issue-decisions-1.0",
                "package_hash": self.package_hash,
                "updated_at": now(),
                "decisions": dict(self.decisions),
            }

    def save(self, issue_id: str, decision: str, note: str) -> dict[str, Any]:
        if issue_id not in self.issue_ids:
            raise ValueError("unknown issue_id")
        if decision not in {"O", "X", "?"}:
            raise ValueError("decision must be O, X, or ?")
        note = note.strip()
        if len(note) > 1000:
            raise ValueError("note is too long")
        entry = {"decision": decision, "note": note, "updated_at": now()}
        with self.lock:
            self.decisions[issue_id] = entry
            atomic_json(self.decisions_path, {
                "schema_version": "ksova-core-issue-decisions-1.0",
                "package_hash": self.package_hash,
                "updated_at": now(),
                "decisions": dict(sorted(self.decisions.items())),
            })
        return entry


def handler_for(state: ReviewState, token: str) -> type[BaseHTTPRequestHandler]:
    prefix = f"/{token}"

    class Handler(BaseHTTPRequestHandler):
        server_version = "KSOVAIssueReview/1.0"

        def send_bytes(self, body: bytes, content_type: str, status: int = 200) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Robots-Tag", "noindex, nofollow")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.end_headers()
            self.wfile.write(body)

        def send_json(self, value: Any, status: int = 200) -> None:
            self.send_bytes(
                json.dumps(value, ensure_ascii=False).encode("utf-8"),
                "application/json; charset=utf-8", status,
            )

        def do_GET(self) -> None:  # noqa: N802
            path = urlparse(self.path).path.rstrip("/")
            if path == prefix:
                self.send_bytes(HTML.encode("utf-8"), "text/html; charset=utf-8")
            elif path == f"{prefix}/recommendations":
                if not state.recommendations:
                    self.send_json({"error": "recommendations_not_loaded"}, HTTPStatus.NOT_FOUND)
                else:
                    self.send_bytes(
                        RECOMMENDATION_HTML.encode("utf-8"), "text/html; charset=utf-8"
                    )
            elif path == f"{prefix}/api/issues":
                benchmark_count = sum(row["origin"] == "benchmark" for row in state.issues)
                self.send_json({
                    "schema_version": "ksova-core-issue-review-1.0",
                    "total": len(state.issues),
                    "benchmark_issues": benchmark_count,
                    "training_issues": len(state.issues) - benchmark_count,
                    "issues": state.issues,
                })
            elif path == f"{prefix}/api/recommendations":
                self.send_json({
                    "schema_version": "ksova-independent-issue-recommendations-api-1.0",
                    "total": len(state.recommendations),
                    "recommendations": state.recommendations,
                })
            elif path in {f"{prefix}/api/decisions", f"{prefix}/api/export"}:
                self.send_json(state.payload())
            elif path == f"{prefix}/favicon.ico":
                self.send_bytes(b"", "image/x-icon", HTTPStatus.NO_CONTENT)
            else:
                self.send_json({"error": "not_found"}, HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:  # noqa: N802
            if urlparse(self.path).path.rstrip("/") != f"{prefix}/api/decision":
                self.send_json({"error": "not_found"}, HTTPStatus.NOT_FOUND)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 16384:
                    raise ValueError("invalid request size")
                payload = json.loads(self.rfile.read(length))
                entry = state.save(
                    str(payload.get("issue_id") or ""),
                    str(payload.get("decision") or ""),
                    str(payload.get("note") or ""),
                )
                self.send_json({"ok": True, "entry": entry})
            except (ValueError, json.JSONDecodeError) as exc:
                self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)

        def log_message(self, fmt: str, *args: Any) -> None:
            print(f"{self.address_string()} - {fmt % args}", flush=True)

    return Handler


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-issues", type=Path, required=True)
    parser.add_argument("--benchmark-accepted", type=Path, required=True)
    parser.add_argument("--training-examples", type=Path, required=True)
    parser.add_argument("--training-materials", type=Path, required=True)
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--recommendations", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18888)
    args = parser.parse_args()

    token = os.environ.get("KSOVA_ISSUE_REVIEW_TOKEN", "")
    if not re.fullmatch(r"[A-Za-z0-9_-]{16,80}", token):
        raise ValueError("KSOVA_ISSUE_REVIEW_TOKEN must be a 16-80 character secret")
    issues, package_hash = build_issues(args)
    state = ReviewState(issues, package_hash, args.decisions, args.recommendations)
    benchmark_count = sum(row["origin"] == "benchmark" for row in issues)
    print(
        f"K-SOVA issue review: {len(issues)} issues "
        f"({benchmark_count} benchmark + {len(issues) - benchmark_count} training)",
        flush=True,
    )
    print(f"local URL: http://{args.host}:{args.port}/{token}/", flush=True)
    print(f"decisions: {state.decisions_path}", flush=True)
    server = ThreadingHTTPServer((args.host, args.port), handler_for(state, token))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
