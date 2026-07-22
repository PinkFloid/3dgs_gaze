#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import threading
import time
import traceback

import msgpack
import zmq

# ==========================================================

MY_SKILLS = ["grasp", "move_to"]      # 你实际支持的技能名;不在列表里的自动拒绝


def execute(skill, params, report, should_stop):   #一个简单例子

    report("moving")
    for _ in range(30):      
        if should_stop():
            report("stopped", "emergency stop")
            return
        time.sleep(0.1)
    report("done")


def on_stop():
    """急停瞬间被调"""
    print(">>> EMERGENCY STOP <<<", flush=True)


def get_pose():
    """返回狗当前位姿 {"x":..,"y":..,"yaw":..}(板坐标系)
    会自动附在每条进度和 get_state 回执里。"""
    return None

# ======================================================= 下面是协议 应该不需要更改

PROTO_V = 1
TERMINAL = ("done", "failed", "stopped")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rep", type=int, default=5583)
    ap.add_argument("--pub", type=int, default=5584)
    args = ap.parse_args()
    ctx = zmq.Context.instance()
    rep = ctx.socket(zmq.REP)
    rep.bind(f"tcp://*:{args.rep}")
    pub = ctx.socket(zmq.PUB)
    pub.bind(f"tcp://*:{args.pub}")
    pub_lock = threading.Lock()
    state = {"busy": False, "stop": threading.Event()}
    print(f"dog_link: REP :{args.rep}  PUB :{args.pub}  skills={MY_SKILLS}", flush=True)

    def publish(req_id, st, detail=""):
        msg = {"v": PROTO_V, "req_id": req_id, "state": st,
               "pose": get_pose(), "detail": detail, "t": time.time()}
        with pub_lock:
            pub.send_multipart([b"skill.status", msgpack.packb(msg)])
        print(f"[status] {json.dumps(msg, ensure_ascii=False)}", flush=True)

    def worker(req):
        rid = req["req_id"]
        sent = {"terminal": False}

        def report(st, detail=""):
            if st in TERMINAL:
                sent["terminal"] = True
            publish(rid, st, detail)

        publish(rid, "accepted")
        try:
            execute(req["skill"], req.get("params") or {}, report,
                    state["stop"].is_set)
            if not sent["terminal"]:
                report("done")
        except Exception:
            traceback.print_exc()
            if not sent["terminal"]:
                report("failed", "exception in execute(); see dog console")
        state["busy"] = False

    while True:
        try:
            req = msgpack.unpackb(rep.recv(), strict_map_key=False)
        except Exception:
            rep.send(msgpack.packb({"v": PROTO_V, "req_id": "", "accepted": False,
                                    "reason": "bad_params"}))
            continue
        print(f"[req] {json.dumps(req, ensure_ascii=False)}", flush=True)
        skill = req.get("skill")
        reply = {"v": PROTO_V, "req_id": req.get("req_id", ""),
                 "accepted": True, "reason": ""}
        if req.get("v") != PROTO_V:
            reply.update(accepted=False, reason=f"unsupported protocol v={req.get('v')}")
        elif skill == "stop":
            state["stop"].set()
            on_stop()
        elif skill == "get_state":
            reply["state"] = {"pose": get_pose(), "busy": state["busy"]}
        elif skill not in MY_SKILLS:
            reply.update(accepted=False, reason="unknown_skill")
        elif state["busy"]:
            reply.update(accepted=False, reason="busy")
        else:
            state["busy"] = True
            state["stop"].clear()
            threading.Thread(target=worker, args=(req,), daemon=True).start()
        rep.send(msgpack.packb(reply))


if __name__ == "__main__":
    main()
