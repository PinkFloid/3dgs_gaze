#!/usr/bin/env python3
"""e2_tasks.py -- 真机任务逐次表:brain 会话 events.jsonl -> 每条派发/未派发指令一行(CSV + markdown)。

    python Intension/e2_tasks.py [--sessions 20260824-211233,20260824-215337,20260824-215434] [--out docs/E1_DATA/audit_0906]

一行 = 一条动作指令(fetch/deictic/named/goto/place)。字段:会话、序号、流时刻 t_word、墙钟、原文(ASR/打字)、
kind、解析槽位、绑定候选(全部:物体/票面/驻留/与指示词的时间距)、消解(模式/物体/送达)、req_id、技能、
检测类名、hint、狗端接受、accepted/done 墙钟、时延(ASR 秒、说完->派发、accepted->done)、链单终态、
以及"无有效绑定"(指示句没有产生 resolution)与再说一遍的关联。急停单单列。
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "Intension/logs"
TERMINAL = ("done", "failed", "stopped", "rejected")


def load(sess):
    ev = []
    for ln in (LOGS / sess / "events.jsonl").open(encoding="utf-8"):
        try:
            ev.append(json.loads(ln))
        except Exception:
            pass
    return ev


def status_of(ev, req_id):
    st = [e for e in ev if e.get("topic") == "skill.status" and e.get("req_id") == req_id]
    acc = next((e["timestamp"] for e in st if e.get("state") == "accepted"), None)
    term = next((e for e in st if e.get("state") in TERMINAL), None)
    return acc, (term["state"] if term else ""), (term["timestamp"] if term else None), (term.get("message", "")[:60] if term else "")


def rows_for(sess):
    ev = load(sess)
    args = next((e for e in ev if e.get("topic") == "session.args"), {})
    voice = bool(args.get("voice"))
    rows = []
    asr = [e for e in ev if e.get("topic") == "asr"]
    for i, e in enumerate(ev):
        if e.get("topic") != "command":
            continue
        kind = e.get("kind")
        if kind in ("help", "parse_fail", "stop"):
            continue  # 急停按 skill.req 单列
        t = float(e["t"])
        # 同刻的解析/绑定/消解/派发
        parse = next((p["result"] for p in ev[max(0, i - 3):i] if p.get("topic") == "llm_parse" and p.get("text") == e["text"]), {})
        bind = next((b for b in ev[i:i + 6] if b.get("topic") == "binding"), None)
        res = next((r for r in ev[i:i + 6] if r.get("topic") == "resolution" and abs(float(r.get("t", -1)) - t) < 1e-6), None)
        req = None
        if res is not None:
            for r in ev[i:i + 40]:
                if r.get("topic") == "skill.req" and r.get("skill") != "stop" and abs(float(r.get("t_stream", -1)) - t) < 1e-3:
                    req = r
                    break
        # 语音:说完时刻(墙钟)= 最近一条同文 asr
        a = None
        if voice:  # 同文可能反复出现:取事件流里紧邻本条指令之前的那条 asr
            for x in reversed(ev[:i]):
                if x.get("topic") == "asr" and x.get("text") == e["text"]:
                    a = x
                    break
        row = {"session": sess, "voice": voice, "t_word": round(t, 2),
               "t_end_wall": round(a["t_end_wall"], 2) if a else "", "asr_s": round(a["asr_s"], 2) if a else "",
               "text": e["text"], "kind": kind,
               "object_query": parse.get("object_query", ""), "object_deictic": parse.get("object_deictic", ""),
               "noun_class": parse.get("noun_class", ""), "dest_deictic": parse.get("dest_deictic", ""),
               "to_user": parse.get("to_user", ""),
               "bind_candidates": json.dumps([{"object": c["object"], "vote": c.get("vote"), "dwell_s": c.get("dwell_s"),
                                               "gap_s": c.get("gap")} for c in bind["candidates"]], ensure_ascii=False) if bind else "",
               "bind_first": bind["candidates"][0]["object"] if bind and bind["candidates"] else "",
               "bind_vote": bind["candidates"][0].get("vote") if bind and bind["candidates"] else "",
               "bind_dwell_s": bind["candidates"][0].get("dwell_s") if bind and bind["candidates"] else "",
               "bind_gap_s": bind["candidates"][0].get("gap") if bind and bind["candidates"] else "",
               "res_mode": res["mode"] if res else "", "res_object": res.get("object") if res else "",
               "res_dest": json.dumps(res.get("dest")) if res and res.get("dest") else "",
               "valid_binding": "yes" if res else ("no" if kind in ("deictic", "fetch") else "n/a"),
               "req_id": req["req_id"] if req else "", "skill": req["skill"] if req else "",
               "detect_name": (req.get("params") or {}).get("object_name") if req else "",
               "hint": json.dumps((req.get("params") or {}).get("object_hint")) if req else "",
               "dog_accepted": (req.get("rep") or {}).get("accepted") if req else "",
               "reject_reason": ((req.get("rep") or {}).get("reason") or "")[:50] if req else "",
               "sent_wall": round(req["sent_at"], 2) if req else ""}
        if req:
            acc, term, tterm, msg = status_of(ev, req["req_id"])
            row.update(accepted_wall=round(acc, 2) if acc else "", final_state=term,
                       done_wall=round(tterm, 2) if tterm else "",
                       speech_to_dispatch_s=round(req["sent_at"] - a["t_end_wall"], 2) if a else "",
                       accept_to_done_s=round(tterm - acc, 2) if (acc and tterm) else "", final_msg=msg)
            chain = next((r for r in ev if r.get("topic") == "skill.req" and r.get("req_id") == req["req_id"] + "p"), None)
            if chain:
                cacc, cterm, ctt, _ = status_of(ev, chain["req_id"])
                row.update(chain_req=chain["req_id"], chain_skill=chain["skill"], chain_state=cterm,
                           chain_accept_to_done_s=round(ctt - cacc, 2) if (cacc and ctt) else "")
        rows.append(row)
    # 急停
    for r in ev:
        if r.get("topic") == "skill.req" and r.get("skill") == "stop":
            acc, term, tterm, msg = status_of(ev, r["req_id"])
            rows.append({"session": sess, "voice": voice, "t_word": round(float(r.get("t_stream", 0)), 2), "text": "(急停)",
                         "kind": "stop", "req_id": r["req_id"], "skill": "stop",
                         "dog_accepted": (r.get("rep") or {}).get("accepted"),
                         "reject_reason": ((r.get("rep") or {}).get("reason") or "")[:60], "final_state": term,
                         "sent_wall": round(r["sent_at"], 2)})
    return rows


FIELDS = ["session", "voice", "t_word", "t_end_wall", "asr_s", "text", "kind", "object_query", "object_deictic",
          "noun_class", "dest_deictic", "to_user", "bind_first", "bind_vote", "bind_dwell_s", "bind_gap_s",
          "bind_candidates", "res_mode", "res_object", "res_dest", "valid_binding", "req_id", "skill",
          "detect_name", "hint", "dog_accepted", "reject_reason", "sent_wall", "accepted_wall", "final_state",
          "done_wall", "speech_to_dispatch_s", "accept_to_done_s", "final_msg", "chain_req", "chain_skill",
          "chain_state", "chain_accept_to_done_s"]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sessions", default="20260824-211233,20260824-215337,20260824-215434")
    ap.add_argument("--out", default=str(ROOT / "docs/E1_DATA/audit_0906"))
    ap.add_argument("--name", default="e2_tasks")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for s in a.sessions.split(","):
        rows += rows_for(s.strip())
    with (out / f"{a.name}.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in FIELDS})
    # markdown 精简版
    md = ["| 会话 | t_word | 指令 | kind | 绑定候选(物体/票/驻留s/时距s) | 消解 | req | 技能/类名 | 狗端 | 终态 | 说完→派发 | accepted→done | 链单 |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        cands = ""
        if r.get("bind_candidates"):
            cands = "; ".join(f"{c['object']}/{c['vote']}/{c['dwell_s']}/{c['gap_s']}" for c in json.loads(r["bind_candidates"]))
        md.append(f"| {r['session'][-6:]} | {r['t_word']} | {r['text']} | {r['kind']} | {cands} | "
                  f"{r.get('res_mode','')}→{r.get('res_object','')}{(' dest' if r.get('res_dest') else '')} | "
                  f"{str(r.get('req_id',''))[-4:]} | {r.get('skill','')}/{r.get('detect_name','')} | "
                  f"{'接受' if r.get('dog_accepted') else ('拒:' + str(r.get('reject_reason','')) if r.get('req_id') else '未派发')} | "
                  f"{r.get('final_state','')} | {r.get('speech_to_dispatch_s','')} | {r.get('accept_to_done_s','')} | "
                  f"{r.get('chain_req','')[-5:] if r.get('chain_req') else ''} {r.get('chain_state','')} |")
    (out / f"{a.name}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    n_disp = sum(1 for r in rows if r.get("req_id") and r["kind"] != "stop")
    n_nobind = sum(1 for r in rows if r.get("valid_binding") == "no")
    print(f"{len(rows)} 行:派发 {n_disp},指示句无有效绑定 {n_nobind},急停 {sum(1 for r in rows if r['kind']=='stop')}")
    print(f"-> {out / (a.name + '.csv')}  {out / (a.name + '.md')}")


if __name__ == "__main__":
    main()
