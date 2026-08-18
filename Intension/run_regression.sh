#!/usr/bin/env bash
# Intension 一键回归:无硬件、确定性(--llm off 只吃 parse_cache_v2.json)。
#   ./Intension/run_regression.sh                 # 缺省用 nerfstudio env 的 python
#   PY=/path/to/python ./Intension/run_regression.sh
# R7 会临时占用本机 5583/5584(假狗);若已有 dog_link 在跑,先停掉再测。
set -u
PY=${PY:-$HOME/miniconda3/envs/nerfstudio/bin/python}
cd "$(dirname "$0")"
TMP=$(mktemp -d)
DOG_PID=
trap '[ -n "$DOG_PID" ] && kill $DOG_PID 2>/dev/null; rm -rf "$TMP"' EXIT
FAIL=0
ck() {  # ck <名字> <输出文件> <必须匹配的 grep -E 模式>
  if grep -qE "$3" "$2"; then echo "  [o] $1"; else echo "  [x] $1 (未匹配: $3)"; FAIL=1; fi
}
ckn() {  # 不得匹配
  if grep -qE "$3" "$2"; then echo "  [x] $1 (不该出现: $3)"; FAIL=1; else echo "  [o] $1"; fi
}

# 独立物体表夹具:names.json 是用户随时在改的活文件,回归绝不依赖它
mkdir -p "$TMP/map"
$PY - "$TMP/map" <<'EOF'
import json, sys, pathlib
objs = {"cup": [-0.67, -2.26, 0.75], "apple": [-0.73, -2.39, 0.75],
        "orange": [-0.38, -2.41, 0.75], "黄色机器人": [2.0, 1.32, 0.4],
        "物品台": [-0.73, -2.35, 0.4], "狗桌边": [-1.86, 0.68, 0.4],
        "纸箱子": [1.2, -1.0, 0.3], "Desk Tag": [-0.77, -2.47, 0.7]}
inst, names = [], {}
for i, (nm, c) in enumerate(objs.items(), start=1):
    inst.append({"id": i, "centroid": c, "n_gaussians": 100})
    names[str(i)] = nm
d = pathlib.Path(sys.argv[1])
(d / "instances.json").write_text(json.dumps({"instances": inst}), encoding="utf-8")
(d / "names.json").write_text(json.dumps(names, ensure_ascii=False), encoding="utf-8")
EOF

$PY gen_fake_gaze.py "$TMP/cup.jsonl" >/dev/null
$PY - "$TMP/floor.jsonl" <<'EOF'
import json, sys
def ev(dur, prov):  # 盯地板 1s:落点通道应拒收(地板不算地点)
    return {"t_start": 100.0, "t_end": 100.0 + dur, "duration_s": dur,
            "centroid_world": [1.5, 2.0, 0.0], "origin_world": [2.0, -1.5, 1.4],
            "object": "floor", "object_label": 3, "vote_share": 0.9,
            "object_centroid_world": None, "p_none": 0.1, "sigma_deg": 1.2,
            "mode": "cone", "provisional": prov, "topic": "gaze.intent"}
evs = [ev(0.4, True), ev(0.8, True), ev(1.0, False)]
open(sys.argv[1], "w").write("\n".join(json.dumps(e, ensure_ascii=False) for e in evs) + "\n")
EOF

run() {  # run <输出文件> <replay流> <脚本指令...>
  local out=$1 rep=$2; shift 2
  local argv=(); local s; for s in "$@"; do argv+=(--script "$s"); done
  $PY brain.py --llm off --replay "$rep" --yes --skill-endpoint off \
      --map-dir "$TMP/map" \
      "${argv[@]}" --log-dir "$TMP/logs/$(basename "$out")" >"$out" 2>&1
}

$PY - "$TMP/named.jsonl" <<'EOF'
import json, sys
def ev(t0, dur, obj, label, c, prov):  # 表内真名流:命名过滤后绑定用例的素材
    return {"t_start": t0, "t_end": t0 + dur, "duration_s": round(dur, 3),
            "centroid_world": c, "origin_world": [2.0, -1.5, 1.4],
            "object": obj, "object_label": label, "vote_share": 0.9,
            "object_centroid_world": c, "p_none": 0.05, "sigma_deg": 1.0,
            "mode": "cone", "provisional": prov, "topic": "gaze.intent"}
def fx(t0, total, obj, label, c):
    out, d = [], 0.4
    while d < total - 1e-9:
        out.append(ev(t0, d, obj, label, c, True)); d += 0.4
    out.append(ev(t0, total, obj, label, c, False))
    return out
evs = fx(100.0, 2.5, "黄色机器人", 12, [2.0, 1.32, 0.4]) \
    + fx(104.0, 2.5, "cup", 11, [-0.67, -2.26, 0.75])
evs.sort(key=lambda e: e["t_end"])
open(sys.argv[1], "w").write("\n".join(json.dumps(e, ensure_ascii=False) for e in evs) + "\n")
EOF

echo "R1 眼-声绑定:把这个机器人拿来 / 拿这个(类别过滤+最近优先,只绑命名物)"
run "$TMP/r1" "$TMP/named.jsonl" "103.0:把这个机器人拿来" "107.0:拿这个"
ck "类别过滤跳过在盯的 cup -> 黄色机器人" "$TMP/r1" "消解为 黄色机器人"
ck "拿这个 -> 最近命名物 cup" "$TMP/r1" '"object_name": "cup"'
ck "抓取纯单(无送达字段)" "$TMP/r1" '"skill": "grasp", "params": \{"object_name": "cup", "target_world": \[[^]]*\], "object_hint": \[[^]]*\]\}'
ck "链发放置到用户处(无 object 字段)" "$TMP/r1" '"skill": "place", "params": \{"target_world": \[2.0, -1.5'
EV=$(ls -t "$TMP"/logs/r1/*/events.jsonl | head -1)
if $PY eval_binding.py "$EV" --expect 黄色机器人,cup --sep 0.3 --dist 1.7 --n 2 \
      --out "$TMP/e1.csv" | grep -q "对 2 错 0"; then
  echo "  [o] eval_binding 打分 2/2"
else echo "  [x] eval_binding 打分"; FAIL=1; fi

echo "R2 视线地点:看着碗说「去这个地方拿橘子」"
run "$TMP/r2" "$TMP/cup.jsonl" "109.0:去这个地方拿橘子"
ck "按注视处「碗」导航" "$TMP/r2" '按注视处「碗」'
ck "到位检测 orange" "$TMP/r2" '"object_name": "orange"'

echo "R3 地板负例:盯地板说同一句(地板不算地点)"
run "$TMP/r3" "$TMP/floor.jsonl" "101.5:去这个地方拿橘子"
ck "明确拒绝并提示" "$TMP/r3" "没注视到取货地点"
ckn "不许派发" "$TMP/r3" "派发"

echo "R4 goto 落点:看着碗说「去那边」"
run "$TMP/r4" "$TMP/cup.jsonl" "109.0:去那边"
ck "纯导航(object null)" "$TMP/r4" '"object_name": null'
ck "目的地=碗落点" "$TMP/r4" "导航至 碗那边"

echo "R5 显式送达:把这个拿到物品台那边(不送用户)"
run "$TMP/r5" "$TMP/named.jsonl" "107.0:把这个拿到物品台那边"
ck "抓 cup 纯单" "$TMP/r5" '"skill": "grasp".*"object_name": "cup"'
ck "链发放置到物品台" "$TMP/r5" '"skill": "place".*"target_world": \[-0.73, -2.35'

echo "R5b 命名送达带检测名:把这个拿到纸箱子那边 -> deliver_name=storage box"
run "$TMP/r5b" "$TMP/named.jsonl" "107.0:把这个拿到纸箱子那边"
ck "放置单坐标+检测名" "$TMP/r5b" '"skill": "place".*"target_world": \[1.2, -1.0.*"place_name": "storage box"'

echo "R6 急停旁路:停(永不过 LLM)"
run "$TMP/r6" "$TMP/cup.jsonl" "106.0:停"
ck "发出 stop 请求" "$TMP/r6" '"skill": "stop"'

echo "R8 槽位卫生:LLM 把'这里'塞进 place_query 也能救(实测坏例)"
if $PY - <<'EOF'
import json, pathlib, sys, tempfile
sys.path.insert(0, ".")
from agent import CommandParser
bad = {"去这里拿Orange": {  # 2026-07-28 真机实测的坏 LLM 输出,原样重放
    "action": "fetch", "object_query": "orange", "object_deictic": True,
    "noun_class": "orange", "place_query": "这里", "place_deictic": True,
    "dest_query": None, "dest_deictic": False, "to_user": True}}
p = pathlib.Path(tempfile.mkdtemp()) / "cache.json"
p.write_text(json.dumps(bad), encoding="utf-8")
c = CommandParser({}, mode="off", cache_path=p, say=lambda s: None).parse("去这里拿Orange")
assert c["place"] is None and c["place_deictic"], c   # "这里"被清洗成 deictic 标记
assert c["object"] == "orange", c
assert not c["object_deictic"] and c["noun"] == "", c  # 指名即指称:漏标 deictic 被纠正
EOF
then echo "  [o] 指代词查询被归一化"; else echo "  [x] 指代词查询归一化"; FAIL=1; fi

echo "R9 端到端:看着碗说「去这里拿Orange」(视线优先)"
run "$TMP/r9" "$TMP/cup.jsonl" "109.0:去这里拿Orange"
ck "按注视处「碗」导航" "$TMP/r9" "按注视处「碗」"
ck "到位检测 orange" "$TMP/r9" '"object_name": "orange"'

echo "R7 忙拒作废(假狗在环,占 5583/5584)"
$PY dog_link.py --fake >"$TMP/dog.log" 2>&1 & DOG_PID=$!
sleep 1
timeout 90 $PY brain.py --llm off --replay "$TMP/floor.jsonl" --yes \
    --skill-endpoint tcp://127.0.0.1:5583 --map-dir "$TMP/map" \
    --script "101.5:去狗桌边" --script "103.0:去黄色机器人" \
    --log-dir "$TMP/logs/r7" >"$TMP/r7" 2>&1
ck "第二单作废提示(不排队)" "$TMP/r7" "本单作废"
ck "首单不受影响跑到 done" "$TMP/r7" 'done.*req=[0-9-]*-001'
kill $DOG_PID 2>/dev/null; DOG_PID=

echo "R10 盯视链:看A再看B,忙拒入槽,A 完成后自动补发去B"
$PY - "$TMP/ab.jsonl" <<'EOF'
import json, sys
def ev(t0, dur, obj, label, c, prov):
    return {"t_start": t0, "t_end": t0 + dur, "duration_s": round(dur, 3),
            "centroid_world": c, "origin_world": [2.0, -1.5, 1.4],
            "object": obj, "object_label": label, "vote_share": 0.9,
            "object_centroid_world": c, "p_none": 0.05, "sigma_deg": 1.0,
            "mode": "cone", "provisional": prov, "topic": "gaze.intent"}
def fx(t0, total, obj, label, c):
    out, d = [], 0.4
    while d < total - 1e-9:
        out.append(ev(t0, d, obj, label, c, True)); d += 0.4
    out.append(ev(t0, total, obj, label, c, False))
    return out
evs = fx(100.0, 2.5, "cup", 11, [2.0, 0.0, 0.8]) \
    + fx(103.0, 2.5, "apple", 12, [-1.5, 1.0, 0.8])
evs.sort(key=lambda e: e["t_end"])
open(sys.argv[1], "w").write("\n".join(json.dumps(e) for e in evs) + "\n")
EOF
$PY dog_link.py --fake >"$TMP/dog2.log" 2>&1 & DOG_PID=$!
sleep 1
timeout 90 $PY brain.py --llm off --replay "$TMP/ab.jsonl" --yes \
    --proactive 1.5 --proactive-goto --map-dir "$TMP/map" \
    --skill-endpoint tcp://127.0.0.1:5583 \
    --log-dir "$TMP/logs/r10" >"$TMP/r10" 2>&1
ck "B 忙拒入槽(不作废)" "$TMP/r10" "已排队最新盯视目标:apple"
ck "空闲后自动补发 B" "$TMP/r10" "补发盯视目标:apple"
ck "A 单跑完 done" "$TMP/r10" 'done.*req=[0-9-]*-001'
ck "B 单跑完 done" "$TMP/r10" 'done.*req=[0-9-]*-002'
kill $DOG_PID 2>/dev/null; DOG_PID=

echo "R11 时间倒流自愈:同一段流放两遍,第二遍照常触发(旧 brain 吃新回放不再静默)"
cat "$TMP/named.jsonl" "$TMP/named.jsonl" > "$TMP/named2.jsonl"
run11() {
  $PY brain.py --llm off --replay "$TMP/named2.jsonl" --yes --skill-endpoint off \
      --proactive 1.5 --proactive-goto --map-dir "$TMP/map" \
      --log-dir "$TMP/logs/r11" >"$TMP/r11" 2>&1
}
run11
ck "检测到倒退并重置" "$TMP/r11" "流时间倒退.*已重置"
N=$(grep -c '派发' "$TMP/r11")
if [ "$N" -eq 4 ]; then echo "  [o] 两遍各触发 2 单(共 $N)"; else echo "  [x] 派发数 $N != 4"; FAIL=1; fi

echo "R13 原地抓:去过一次后「抓苹果/抓这个」不挪窝直接抓"
run "$TMP/r13" "$TMP/named.jsonl" "103.0:去狗桌边" "104.5:抓苹果" "107.0:抓这个"
ck "原地抓确认语义" "$TMP/r13" "原地抓"
ck "抓苹果 -> apple" "$TMP/r13" '"object_name": "apple"'
EV=$(ls -t "$TMP"/logs/r13/*/events.jsonl | head -1)
if $PY - "$EV" <<'EOF'
import json, sys
reqs = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8")
        if '"topic": "skill.req"' in l]
tw = [r["params"]["target_world"] for r in reqs]
assert len(tw) == 3 and tw[0] == tw[1] == tw[2], tw          # 三单同一站位:狗没挪窝
assert reqs[0]["params"]["object_name"] is None, tw          # 首单纯导航
assert reqs[1]["params"]["object_name"] == "apple", tw
assert reqs[2]["params"]["object_name"] == "cup", tw         # 抓这个 -> 视线绑定 cup
EOF
then echo "  [o] 三单同一站位,object 各归其位"; else echo "  [x] 站位/对象校验失败"; FAIL=1; fi
run "$TMP/r13n" "$TMP/floor.jsonl" "101.0:抓苹果"
ck "会话内无派发时明确拒绝" "$TMP/r13n" "不知道狗现在站哪"
ckn "拒绝时不许派发" "$TMP/r13n" '\[派发\]'

echo "R14 实例消歧过线:视线/名字单带 object_hint=选中实例质心;goto/地点透传不带"
run "$TMP/r14" "$TMP/named.jsonl" "103.0:去狗桌边" "107.0:拿这个"
EV=$(ls -t "$TMP"/logs/r14/*/events.jsonl | head -1)
if $PY - "$EV" <<'EOF'
import json, sys
reqs = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8")
        if '"topic": "skill.req"' in l]
assert len(reqs) == 3, len(reqs)            # goto + grasp + 链发 place(送回用户)
assert "object_hint" not in reqs[0]["params"], reqs[0]["params"]        # goto 不带
h = reqs[1]["params"]["object_hint"]                                    # 拿这个 -> cup 质心
assert h == [-0.67, -2.26, 0.75], h
assert reqs[2]["skill"] == "place", reqs[2]                             # 链发放置
assert "object_hint" not in reqs[2]["params"], reqs[2]["params"]        # 放置不带 hint
EOF
then echo "  [o] hint 逐字段正确"; else echo "  [x] hint 校验失败"; FAIL=1; fi

echo "R15 hint 投影选框参考实现自检(交付狗端的 hint_select.py)"
if $PY hint_select.py --selftest >"$TMP/r15" 2>&1; then
  echo "  [o] 零位/选框/拒抓 三关全过"
else echo "  [x] hint_select selftest 失败"; cat "$TMP/r15"; FAIL=1; fi

echo "R16 语音标点归一:'停。'走急停旁路,'好。'当确认(转写带标点的实测形)"
run "$TMP/r16" "$TMP/cup.jsonl" "106.0:停。"
ck "'停。'发出 stop(不吃 LLM/缓存)" "$TMP/r16" '"skill": "stop"'
$PY brain.py --llm off --replay "$TMP/named.jsonl" --skill-endpoint off \
    --map-dir "$TMP/map" --script "103.0:给我找下黄色机器人" --script "104.5:好。" \
    --log-dir "$TMP/logs/r16b" >"$TMP/r16b" 2>&1
ck "确认门先出问句" "$TMP/r16b" "y=确认"
ck "'好。'确认后才派发" "$TMP/r16b" '\[派发\].*黄色机器人'

echo "R12 线格式审计:所有派发的 yaw 必须在 [-π,π]"
if $PY - "$TMP/logs" <<'EOF'
import glob, json, math, sys
bad = 0
for f in glob.glob(sys.argv[1] + "/**/events.jsonl", recursive=True):
    for ln in open(f, encoding="utf-8"):
        e = json.loads(ln)
        if e.get("topic") != "skill.req":
            continue
        p = e.get("params") or {}
        for key in ("target_world", "deliver_to"):
            v = p.get(key)
            if v and abs(float(v[2])) > math.pi + 1e-6:
                bad += 1
                print(f"  越界 {key}[2]={v[2]} @ {e['req_id']}")
sys.exit(1 if bad else 0)
EOF
then echo "  [o] 全部派发 yaw 已包角"; else echo "  [x] 有越界 yaw"; FAIL=1; fi

echo "R17 逐词时刻:指示词各带墙钟(词表滑窗+顺序约束+无词流退化)"
if $PY - <<'EOF'
import sys
sys.path.insert(0, ".")
from brain import slot_times, word_time
words = [("把", 10.0, 10.2), ("这", 10.2, 10.4), ("个", 10.4, 10.6),
         ("放", 10.6, 10.8), ("到", 10.8, 11.0), ("那", 11.0, 11.2),
         ("里", 11.2, 11.4), ("去", 11.4, 11.6)]
t = word_time(words, ("这个", "这"), 99.0)
assert 10.2 < t < 10.7, t                       # 长词优先,取"这个"中点
cmd = {"object_deictic": True, "noun": "", "place_deictic": False,
       "dest_deictic": True}
to, tp, td = slot_times(cmd, 99.0, words)
assert 10.2 < to < 10.7 and 11.0 < td < 11.5 and tp == 99.0, (to, tp, td)
assert slot_times(cmd, 99.0, None) == (99.0, 99.0, 99.0)   # 退化=句末
EOF
then echo "  [o] word_time/slot_times"; else echo "  [x] 逐词时刻"; FAIL=1; fi

echo "R18 放置最简版:把这个放到那里去(解不出=不带送达照派,剔物体自身落点)"
$PY - "$TMP/putwait.jsonl" <<'EOF'
import json, sys
def ev(t0, dur, obj, label, c, prov):
    return {"t_start": t0, "t_end": t0 + dur, "duration_s": round(dur, 3),
            "centroid_world": c, "origin_world": [2.0, -1.5, 1.4],
            "object": obj, "object_label": label, "vote_share": 0.9,
            "object_centroid_world": c, "p_none": 0.05, "sigma_deg": 1.0,
            "mode": "cone", "provisional": prov, "topic": "gaze.intent"}
def fx(t0, total, obj, label, c):
    out, d = [], 0.4
    while d < total - 1e-9:
        out.append(ev(t0, d, obj, label, c, True)); d += 0.4
    out.append(ev(t0, total, obj, label, c, False))
    return out
def tick(t):  # 拨钟填充:floor 不进任何通道,只让脚本指令在正确时刻生效
    return dict(ev(t, 0.3, "floor", 3, [1.0, 0.0, 0.0], False),
                object_centroid_world=None)
evs = fx(104.0, 2.5, "cup", 11, [-0.67, -2.26, 0.75]) + [tick(107.0), tick(111.2)] \
    + fx(108.0, 0.5, "apple", 13, [1.5, -2.0, 0.8]) \
    + fx(109.5, 1.4, "物品台", 17, [0.4, -2.6, 0.8])
evs.sort(key=lambda e: e["t_end"])
open(sys.argv[1], "w").write("\n".join(json.dumps(e, ensure_ascii=False) for e in evs) + "\n")
EOF
run "$TMP/r18" "$TMP/putwait.jsonl" "107.0:把这个放到那里去" "111.0:把这个放到那里去"
ck "解不出落点如实说" "$TMP/r18" "没看到要放哪——本单不带送达"
ck "首单不带 deliver_to 照派 cup" "$TMP/r18" '"object_name": "cup", "target_world": \[[^]]*\], "object_hint": \[[^]]*\]\}'
ck "第二单绑最近注视且剔自身落点(dest=apple 点)" "$TMP/r18" '"skill": "place".*"target_world": \[1.5, -2.0'
ckn "不再有等待机制" "$TMP/r18" "看准位置停"

echo "R20 裸放置:放到纸箱子(不带物体,单发 place;狗手里有什么放什么)"
run "$TMP/r20" "$TMP/named.jsonl" "107.0:放到纸箱子"
ck "单发 place 无 grasp" "$TMP/r20" '"skill": "place".*"place_name": "storage box"'
ckn "不派 grasp" "$TMP/r20" '"skill": "grasp"'
ck "place 参数只有落点" "$TMP/r20" '"params": \{"target_world": \[1.2, -1.0'

echo "R19 链单 e2e:拿到纸箱子那边(grasp done -> 自动补发 place,假狗在环)"
$PY dog_link.py --fake >"$TMP/dog3.log" 2>&1 & DOG_PID=$!
sleep 1
timeout 90 $PY brain.py --llm off --replay "$TMP/named.jsonl" --yes \
    --skill-endpoint tcp://127.0.0.1:5583 --map-dir "$TMP/map" \
    --script "107.0:把这个拿到纸箱子那边" \
    --log-dir "$TMP/logs/r19" >"$TMP/r19" 2>&1
ck "链槽入位" "$TMP/r19" "抓到后自动补发放置"
ck "抓取完成后链发" "$TMP/r19" "抓取完成,补发放置单"
ck "放置单 place+检测名" "$TMP/r19" '"skill": "place".*"place_name": "storage box"'
ck "放置单跑到 done" "$TMP/r19" 'done.*req=[0-9-]*-00[0-9]p'
kill $DOG_PID 2>/dev/null; DOG_PID=

echo
if [ $FAIL -eq 0 ]; then echo "== 全部通过 =="; else echo "== 有失败项 =="; fi
exit $FAIL
