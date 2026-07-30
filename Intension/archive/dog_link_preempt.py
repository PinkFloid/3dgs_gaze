#!/usr/bin/env python3
# 【归档 2026-07-28】抢占语义版 dog_link,未启用——用户定的方针:狗端暂不改。
# 现行狗端文件是 Intension/dog_link.py(忙时拒 busy 的 v1 语义)。本文件是
# "新单取消旧单"的完整参考实现,协议回归全绿(抢占/连环抢占/急停/版本闸,
# 行车锁保证永不双控);狗机同学将来实现 action cancel 时可整体采纳,
# 意图机(brain)已按"发完即忘"设计,届时零改动。设计见 docs/REDESIGN.md §1。
"""狗机技能服务端(抢占语义参考实现,未交付)。协议见 Intension/PROTOCOL.md。

技能:
  grasp   params: object_name, target_world:[x,y,yaw], deliver_to:[x,y,yaw]?(可选)
          target_world = 基座站位(发送方已留 standoff)+ 到位朝向(弧度)。
          走到站位 → 转到 yaw → 夹取 → 有 deliver_to 则 returning 送达,
          没有则原地 done。object_name 空(null/"")= 纯导航,只走到站位。
  move_to params: x, y, yaw?(可选,到点后再对准朝向)

  状态流:accepted → moving → grasping [→ returning] → done / failed / stopped
  (对准阶段仍报 moving,detail="aligning",不引入协议外的状态名)

抢占语义(2026-07-28 修订,详见 PROTOCOL §3.5):
  忙时收到新 grasp/move_to **不再拒 busy**——立即回执 accepted,给旧请求取消
  (旧 req 终态 stopped,detail=preempted_by:<新req_id>),取消完成后才执行新单。
  新任务 = 用户最新意志;急停与抢占共用同一条 cancel 路径。等旧任务退出超过
  PREEMPT_JOIN_S 时新单广播 failed(preempt_timeout)——绝不双控制器并跑。

比"能跑"多出来的保护(不接真机也全部生效):
  - 每阶段看门狗超时 + 卡死检测(位姿无进展)——也是抢占等待有界的保证;
  - 接受前工作空间校验(房间边界)→ 拒绝 out_of_workspace;
  - 终态先摘当前 worker 再广播——对方收到终态时服务端一定已可接新单;
  - 急停/抢占:站定 + 夹爪冻结(搬运中松爪可能比抱住更糟,松开是显式决定)。

SDK 边界 = DogAdapter。默认跑 FakeDog(速度积分假狗,全协议零硬件联调);
真狗填 RealDog 一个类(Go2:unitree_sdk2 SportClient + 机械臂接口)。
心跳走独立话题 dog.heartbeat(1Hz,pose+busy),不污染 skill.status。
"""

from __future__ import annotations

import argparse
import json
import math
import threading
import time
import traceback

import msgpack
import zmq

# ------------------------------------------------------------ 配置(狗机同学按实际改)

MY_SKILLS = ["grasp", "move_to"]
PROTO_V = 1            # 线上版本:形状未变仍是 v1(抢占是服务端行为修订);v2 预留给形状变更
ACCEPT_V = (1, 2)
TERMINAL = ("done", "failed", "stopped")

# 板坐标系,米。上真机前收紧到真实可走区域
ROOM_X = (-3.5, 5.5)
ROOM_Y = (-4.5, 5.5)
STANDOFF_TOL = 0.10             # m,站位到位容差(比纯导航更严)
HEADING_TOL = 0.15              # rad
NAV_TOL = 0.15                  # m,move_to 到点半径
PHASE_TIMEOUT = {"moving": 45.0, "grasping": 15.0, "returning": 45.0}
ALIGN_TIMEOUT = 10.0
STUCK_WINDOW = 3.0              # s 无进展 → failed("stuck")
STUCK_EPS = 0.05                # m
PREEMPT_JOIN_S = 5.0            # s,等被抢占任务退出的上限(各阶段看门狗保证有界)
CTRL_HZ = 10.0
V_MAX, VY_MAX, W_MAX = 0.5, 0.3, 1.0
KP_LIN, KP_YAW = 0.8, 1.5


# ------------------------------------------------------------ SDK 边界

class DogAdapter:
    """技能层需要机器人提供的全部能力。真狗填 RealDog。"""

    def get_pose(self):
        """板坐标系 {"x","y","yaw"};未定位返回 None。"""
        raise NotImplementedError

    def send_velocity(self, vx, vy, wyaw):
        """机体系速度 (m/s, m/s, rad/s),CTRL_HZ 频率调用。"""
        raise NotImplementedError

    def stand_still(self):
        raise NotImplementedError

    def gripper_close(self) -> bool:
        """执行夹取;只有反馈确认夹到东西才返回 True。"""
        raise NotImplementedError

    def gripper_open(self):
        raise NotImplementedError


class FakeDog(DogAdapter):
    """速度积分假狗:全协议 / 控制环 / 看门狗零硬件联调;夹取恒成功。"""

    def __init__(self, x=0.0, y=0.0, yaw=0.0):
        self.x, self.y, self.yaw = x, y, yaw
        self._last = time.monotonic()

    def get_pose(self):
        return {"x": round(self.x, 3), "y": round(self.y, 3), "yaw": round(self.yaw, 3)}

    def send_velocity(self, vx, vy, wyaw):
        now = time.monotonic()
        dt = min(now - self._last, 0.5)
        self._last = now
        self.x += (math.cos(self.yaw) * vx - math.sin(self.yaw) * vy) * dt
        self.y += (math.sin(self.yaw) * vx + math.cos(self.yaw) * vy) * dt
        self.yaw = (self.yaw + wyaw * dt + math.pi) % (2 * math.pi) - math.pi

    def stand_still(self):
        self._last = time.monotonic()

    def gripper_close(self):
        time.sleep(1.0)
        return True

    def gripper_open(self):
        pass


class RealDog(DogAdapter):
    """TODO(狗机同学):Go2 + 机械臂,预计对应关系——

    get_pose      <- v0.5 停放点静态外参 + SportClient 里程计;
                     v1 狗头相机 ArUco(复用 pupil_localizer + tags_world.json)
    send_velocity <- unitree_sdk2 SportClient.Move(vx, vy, wyaw)
    stand_still   <- SportClient.StopMove() / BalanceStand()
    gripper_close <- 机械臂 SDK 夹取;用行程/力反馈判 True/False
    gripper_open  <- 机械臂 SDK 张开
    急停另见 on_stop():SportClient.Damp() + 臂急停
    ⚠ 抢占前置:导航/臂的 action 必须支持 cancel(急停同款,做一个送一个)
    """


# ------------------------------------------------------------ 技能实现

def _ang_norm(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def _navigate(dog, goal_xy, face_point, arrive_tol, phase, should_stop):
    """P 控制器导航环。成功返回 None,否则失败原因字符串。"""
    deadline = time.monotonic() + PHASE_TIMEOUT[phase]
    last_progress_t = time.monotonic()
    last_progress_d = None
    while True:
        if should_stop():
            return "estop"
        if time.monotonic() > deadline:
            return f"{phase}_timeout"
        pose = dog.get_pose()
        if pose is None:
            return "unlocalized"
        ex, ey = goal_xy[0] - pose["x"], goal_xy[1] - pose["y"]
        dist = math.hypot(ex, ey)
        if dist < arrive_tol:
            dog.stand_still()
            return None
        if last_progress_d is None or last_progress_d - dist > STUCK_EPS:
            last_progress_d, last_progress_t = dist, time.monotonic()
        elif time.monotonic() - last_progress_t > STUCK_WINDOW:
            dog.stand_still()
            return "stuck"
        fx, fy = (face_point or goal_xy)[0] - pose["x"], (face_point or goal_xy)[1] - pose["y"]
        herr = _ang_norm(math.atan2(fy, fx) - pose["yaw"])
        c, s = math.cos(pose["yaw"]), math.sin(pose["yaw"])
        bx, by = c * ex + s * ey, -s * ex + c * ey          # 世界系误差 → 机体系
        dog.send_velocity(max(-V_MAX, min(V_MAX, KP_LIN * bx)),
                          max(-VY_MAX, min(VY_MAX, KP_LIN * by)),
                          max(-W_MAX, min(W_MAX, KP_YAW * herr)))
        time.sleep(1.0 / CTRL_HZ)


def _align(dog, face_yaw, should_stop):
    """原地转到指定朝向。face_yaw 为绝对 yaw(rad)。"""
    deadline = time.monotonic() + ALIGN_TIMEOUT
    while True:
        if should_stop():
            return "estop"
        if time.monotonic() > deadline:
            return "align_timeout"
        pose = dog.get_pose()
        if pose is None:
            return "unlocalized"
        herr = _ang_norm(face_yaw - pose["yaw"])
        if abs(herr) < HEADING_TOL:
            dog.stand_still()
            return None
        dog.send_velocity(0.0, 0.0, max(-W_MAX, min(W_MAX, KP_YAW * herr)))
        time.sleep(1.0 / CTRL_HZ)


def _finish(report, err):
    report("stopped" if err == "estop" else "failed",
           "emergency stop" if err == "estop" else err)


def execute(dog, skill, params, report, should_stop):
    if skill == "move_to":
        goal = (float(params["x"]), float(params["y"]))
        report("moving")
        err = _navigate(dog, goal, None, NAV_TOL, "moving", should_stop)
        if err:
            return _finish(report, err)
        if params.get("yaw") is not None:
            err = _align(dog, float(params["yaw"]), should_stop)
            if err:
                return _finish(report, err)
        return report("done")

    # grasp: target_world = [x, y, yaw],站位由发送方留好 standoff
    x, y, yaw = (float(v) for v in params["target_world"])
    if not params.get("object_name"):   # 冻结定义:object 为空(null/"")= 纯导航,不动臂
        report("moving")
        err = _navigate(dog, (x, y), None, NAV_TOL, "moving", should_stop)
        if err:
            return _finish(report, err)
        err = _align(dog, yaw, should_stop)
        return _finish(report, err) if err else report("done")
    report("moving", f"站位({x:+.2f},{y:+.2f}) -> {params.get('object_name', '?')}")
    err = _navigate(dog, (x, y), None, STANDOFF_TOL, "moving", should_stop)
    if err:
        return _finish(report, err)
    report("moving", "aligning")
    err = _align(dog, yaw, should_stop)
    if err:
        return _finish(report, err)
    report("grasping")
    if should_stop():
        return report("stopped", "emergency stop")
    if not dog.gripper_close():
        return report("failed", "grasp_missed")
    if params.get("deliver_to"):
        dst = [float(v) for v in params["deliver_to"]]
        report("returning", f"deliver to ({dst[0]:+.2f},{dst[1]:+.2f})")
        err = _navigate(dog, dst[:2], None, NAV_TOL, "returning", should_stop)
        if err:
            return _finish(report, err)          # 注意:失败时爪里还有东西
        err = _align(dog, dst[2], should_stop)   # 第三位=送达朝向(朝向用户)
        if err:
            return _finish(report, err)
    report("done")


def validate(skill, params):
    """接受前的工作空间校验:坏坐标不许走到电机。reason 用协议词表。"""
    if skill == "move_to":
        if "x" not in params or "y" not in params:
            return "bad_params: move_to needs x, y"
        pts = [(float(params["x"]), float(params["y"]))]
    else:
        t = params.get("target_world")
        if not (isinstance(t, (list, tuple)) and len(t) == 3):
            return "bad_params: grasp needs target_world [x,y,yaw]"
        pts = [(float(t[0]), float(t[1]))]
        if params.get("deliver_to") is not None:
            dv = params["deliver_to"]
            if not (isinstance(dv, (list, tuple)) and len(dv) >= 2):
                return "bad_params: deliver_to must be [x,y,yaw]"
            pts.append((float(dv[0]), float(dv[1])))
    for x, y in pts:
        if not (ROOM_X[0] <= x <= ROOM_X[1] and ROOM_Y[0] <= y <= ROOM_Y[1]):
            return "out_of_workspace"
    return None


# ------------------------------------------------------------ 协议壳(勿改)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rep", type=int, default=5583)
    ap.add_argument("--pub", type=int, default=5584)
    ap.add_argument("--fake", action="store_true", help="跑速度积分假狗(无 SDK 自动回落)。")
    ap.add_argument("--fake-start", default="0,0,0", help="假狗起始 'x,y,yaw'。")
    args = ap.parse_args()

    if args.fake or type(RealDog.get_pose) is type(DogAdapter.get_pose):
        sx, sy, syaw = (float(v) for v in args.fake_start.split(","))
        dog = FakeDog(sx, sy, syaw)
        if not args.fake:
            print("dog_link: RealDog 未实现 -> 回落 FakeDog(联调模式)", flush=True)
    else:
        dog = RealDog()

    ctx = zmq.Context.instance()
    rep = ctx.socket(zmq.REP)
    rep.bind(f"tcp://*:{args.rep}")
    pub = ctx.socket(zmq.PUB)
    pub.bind(f"tcp://*:{args.pub}")
    pub_lock = threading.Lock()
    # 当前 worker:{"thread","ev","rid","preempted_by"}。每个任务自带 stop 事件——
    # 共享单事件在"抢占后紧接急停"这类连环操作下会清错对象。
    # live = 尚未退出的 worker 名册(仅 REP 线程增删):急停必须对全体广播,
    # 因为"新单等锁超时自弃 + 旧单仍在开车"时 cur 已不指向开车者。
    cur = {"w": None}
    live = []
    print(f"dog_link: REP :{args.rep}  PUB :{args.pub}  skills={MY_SKILLS}  "
          f"dog={type(dog).__name__}  抢占语义(新单取消旧单)", flush=True)

    def busy():
        w = cur["w"]
        return bool(w and w["thread"] is not None and w["thread"].is_alive())

    def cur_rid():
        return cur["w"]["rid"] if cur["w"] else ""

    def publish(topic, req_id, st, detail=""):
        msg = {"v": PROTO_V, "req_id": req_id, "state": st,
               "pose": dog.get_pose(), "busy": busy(),
               "detail": detail, "t": time.time()}
        with pub_lock:
            pub.send_multipart([topic, msgpack.packb(msg)])
        if topic == b"skill.status":
            print(f"[status] {json.dumps(msg, ensure_ascii=False)}", flush=True)

    def heartbeat():   # 独立话题:意图机能区分"空闲"和"死机",且不刷 skill.status
        while True:
            publish(b"dog.heartbeat", cur_rid(), "heartbeat")
            time.sleep(1.0)

    threading.Thread(target=heartbeat, daemon=True).start()

    def on_stop():
        # 夹爪刻意冻结不松开:搬运中掉落可能比抱住更糟;松开是显式决定
        print(">>> EMERGENCY STOP <<<", flush=True)
        dog.stand_still()

    drive = threading.Lock()   # 行车锁:谁拿着锁谁开车,任何时刻只有一个控制器

    def start_worker(req, prev):
        """接管:取消旧任务 → 拿到行车锁 → 执行新单。锁语义保证绝不双控。"""
        w = {"ev": threading.Event(), "rid": req["req_id"],
             "preempted_by": None, "thread": None}
        sent = {"terminal": False}

        def report(st, detail=""):
            if st == "stopped" and w["preempted_by"]:
                detail = f"preempted_by:{w['preempted_by']}"  # 抢占≠急停,因果写进终态
            if st in TERMINAL:
                sent["terminal"] = True
                if cur["w"] is w:   # 终态先摘牌再广播:对方收到终态时这里已可接新单
                    cur["w"] = None
            publish(b"skill.status", w["rid"], st, detail)

        def run():
            publish(b"skill.status", w["rid"], "accepted", req.get("intent_summary", ""))
            if prev is not None:
                prev["preempted_by"] = w["rid"]
                prev["ev"].set()
            # 连环抢占(A 执行中 B 等待 C 又来)也安全:每个 worker 都被后继 set 过
            # ev,拿到锁先自检 ev,被跳过的中间人立刻 stopped 让位
            if not drive.acquire(timeout=PREEMPT_JOIN_S):
                return report("failed", "preempt_timeout")   # 旧任务取消不掉:宁可不动
            try:
                dog.stand_still()                  # 旧速度指令清零后再起新任务
                if w["ev"].is_set():               # 等锁期间已被急停/再抢占
                    return report("stopped", "emergency stop")
                execute(dog, req["skill"], req.get("params") or {}, report,
                        w["ev"].is_set)
                if not sent["terminal"]:
                    report("done")
            except Exception:
                traceback.print_exc()
                if not sent["terminal"]:
                    report("failed", "exception in execute(); see dog console")
            finally:
                drive.release()

        w["thread"] = threading.Thread(target=run, daemon=True)
        cur["w"] = w
        live[:] = [x for x in live if x["thread"].is_alive()] + [w]
        w["thread"].start()

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
        if req.get("v") not in ACCEPT_V:
            reply.update(accepted=False, reason=f"unsupported protocol v={req.get('v')}")
        elif skill == "stop":
            for w in live:          # 对全体在世 worker 取消,不只 cur(见 live 注释)
                w["ev"].set()
            on_stop()
        elif skill == "get_state":
            reply["state"] = {"pose": dog.get_pose(), "busy": busy()}
        elif skill not in MY_SKILLS:
            reply.update(accepted=False, reason="unknown_skill")
        else:
            bad = validate(skill, req.get("params") or {})
            if bad:
                reply.update(accepted=False, reason=bad)
            else:
                start_worker(req, cur["w"])   # 忙就抢占:新单即用户最新意志
        rep.send(msgpack.packb(reply))


if __name__ == "__main__":
    main()
