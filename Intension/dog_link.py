#!/usr/bin/env python3
"""狗机技能服务端(交付给狗机同学的唯一文件)。协议见 Intension/PROTOCOL.md。

技能:
  grasp   params: object_name, target_world:[x,y,z], deliver_to:[x,y,z]?(可选)
          走到 standoff 点(按 bbox 自动缩放)→ 对准 → 夹取 → 有 deliver_to 则
          returning 送达,没有则原地 done(意图机可自行组合 move_to 取回)。
  move_to params: x, y, yaw?(可选,到点后再对准朝向)

  状态流:accepted → moving → grasping [→ returning] → done / failed / stopped
  (对准阶段仍报 moving,detail="aligning",不引入协议外的状态名)

比"能跑"多出来的四层保护(不接真机也全部生效):
  - 每阶段看门狗超时 + 卡死检测(位姿无进展)——不然 execute 挂住会把 busy 锁死;
  - 接受前工作空间校验(房间边界 / 夹取 z 包络)→ 拒绝 out_of_workspace;
  - 终态先清 busy 再广播——意图机收到 done 立刻发下一条永远不会撞 busy;
  - 急停:站定 + 夹爪冻结(搬运中松爪可能比抱住更糟,松开是之后的显式决定)。

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
PROTO_V = 1
TERMINAL = ("done", "failed", "stopped")

# 板坐标系,米。上真机前收紧到真实可走区域
ROOM_X = (-3.5, 5.5)
ROOM_Y = (-4.5, 5.5)
Z_REACH = (0.02, 0.90)          # 夹爪可达高度,超出拒绝 out_of_workspace
DEFAULT_STANDOFF = 0.5          # m,距目标停车距离
STANDOFF_TOL = 0.10
HEADING_TOL = 0.15              # rad
NAV_TOL = 0.15                  # m,move_to 到点半径
PHASE_TIMEOUT = {"moving": 45.0, "grasping": 15.0, "returning": 45.0}
ALIGN_TIMEOUT = 10.0
STUCK_WINDOW = 3.0              # s 无进展 → failed("stuck")
STUCK_EPS = 0.05                # m
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

    # grasp
    target = [float(v) for v in params["target_world"]]
    standoff = float(params.get("standoff", DEFAULT_STANDOFF))
    if params.get("bbox"):
        lo, hi = params["bbox"]
        standoff = max(standoff, 0.4 * math.hypot(hi[0] - lo[0], hi[1] - lo[1]) + 0.3)
    pose = dog.get_pose()
    if pose is None:
        return report("failed", "unlocalized")
    dx, dy = target[0] - pose["x"], target[1] - pose["y"]
    d = math.hypot(dx, dy)
    goal = ((target[0] - dx / d * standoff, target[1] - dy / d * standoff)
            if d > standoff else (pose["x"], pose["y"]))
    report("moving", f"standoff {standoff:.2f}m -> {params.get('object_name', '?')}")
    err = _navigate(dog, goal, target[:2], STANDOFF_TOL, "moving", should_stop)
    if err:
        return _finish(report, err)
    pose = dog.get_pose()
    report("moving", "aligning")
    err = _align(dog, math.atan2(target[1] - pose["y"], target[0] - pose["x"]), should_stop)
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
    report("done")


def validate(skill, params):
    """接受前的工作空间校验:坏坐标不许走到电机。reason 用协议词表。"""
    if skill == "move_to":
        if "x" not in params or "y" not in params:
            return "bad_params: move_to needs x, y"
        pts = [(float(params["x"]), float(params["y"]), None)]
    else:
        t = params.get("target_world")
        if not (isinstance(t, (list, tuple)) and len(t) == 3):
            return "bad_params: grasp needs target_world [x,y,z]"
        pts = [(float(t[0]), float(t[1]), float(t[2]))]
        if params.get("deliver_to") is not None:
            dv = params["deliver_to"]
            if not (isinstance(dv, (list, tuple)) and len(dv) >= 2):
                return "bad_params: deliver_to must be [x,y,(z)]"
            pts.append((float(dv[0]), float(dv[1]), None))
    for x, y, z in pts:
        if not (ROOM_X[0] <= x <= ROOM_X[1] and ROOM_Y[0] <= y <= ROOM_Y[1]):
            return "out_of_workspace"
        if z is not None and not (Z_REACH[0] <= z <= Z_REACH[1]):
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
    state = {"busy": False, "stop": threading.Event(), "req_id": ""}
    print(f"dog_link: REP :{args.rep}  PUB :{args.pub}  skills={MY_SKILLS}  "
          f"dog={type(dog).__name__}", flush=True)

    def publish(topic, req_id, st, detail=""):
        msg = {"v": PROTO_V, "req_id": req_id, "state": st,
               "pose": dog.get_pose(), "busy": state["busy"],
               "detail": detail, "t": time.time()}
        with pub_lock:
            pub.send_multipart([topic, msgpack.packb(msg)])
        if topic == b"skill.status":
            print(f"[status] {json.dumps(msg, ensure_ascii=False)}", flush=True)

    def heartbeat():   # 独立话题:意图机能区分"空闲"和"死机",且不刷 skill.status
        while True:
            publish(b"dog.heartbeat", state["req_id"], "heartbeat")
            time.sleep(1.0)

    threading.Thread(target=heartbeat, daemon=True).start()

    def on_stop():
        # 夹爪刻意冻结不松开:搬运中掉落可能比抱住更糟;松开是显式决定
        print(">>> EMERGENCY STOP <<<", flush=True)
        dog.stand_still()

    def worker(req):
        rid = req["req_id"]
        sent = {"terminal": False}

        def report(st, detail=""):
            if st in TERMINAL:
                sent["terminal"] = True
                # 终态先清 busy 再广播:对方收到 done 立刻发下一条不会撞 busy
                state["busy"] = False
                state["req_id"] = ""
            publish(b"skill.status", rid, st, detail)

        publish(b"skill.status", rid, "accepted",
                req.get("intent_summary", ""))
        try:
            execute(dog, req["skill"], req.get("params") or {}, report,
                    state["stop"].is_set)
            if not sent["terminal"]:
                report("done")
        except Exception:
            traceback.print_exc()
            if not sent["terminal"]:
                report("failed", "exception in execute(); see dog console")
        state["busy"] = False
        state["req_id"] = ""

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
            reply["state"] = {"pose": dog.get_pose(), "busy": state["busy"]}
        elif skill not in MY_SKILLS:
            reply.update(accepted=False, reason="unknown_skill")
        elif state["busy"]:
            reply.update(accepted=False, reason="busy")
        else:
            bad = validate(skill, req.get("params") or {})
            if bad:
                reply.update(accepted=False, reason=bad)
            else:
                state["busy"] = True
                state["req_id"] = req.get("req_id", "")
                state["stop"].clear()
                threading.Thread(target=worker, args=(req,), daemon=True).start()
        rep.send(msgpack.packb(reply))


if __name__ == "__main__":
    main()
