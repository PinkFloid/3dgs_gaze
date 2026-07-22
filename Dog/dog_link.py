#!/usr/bin/env python3
"""Robot-dog skill server: the third body of the gaze -> intent -> action chain.

Two primitive skills, composed by the caller (fetch = grasp, await done, then
move_to(delivery point) -- no third skill needed):

  grasp   {"name", "target":[x,y,z], "bbox":[[..],[..]]?, "standoff"?}
          navigate to a standoff point on the dog->target line, face the
          target, close the gripper, verify. Phases: navigating -> aligning
          -> grasping -> done / failed(reason).
  move_to {"target":[x,y] or [x,y,z], "yaw"?}
          navigate only. Phases: navigating -> done / failed(reason).

Robustness beyond the happy path:
  - per-phase watchdog timeouts + stuck detection (no pose progress) -- a hung
    execute() would otherwise deadlock the busy gate forever;
  - feasibility validation BEFORE accept (room bounds, reach envelope);
  - 1 Hz heartbeat on the PUB socket so consumers can tell idle from dead;
  - emergency stop freezes the gripper (dropping mid-carry can be worse than
    holding) and stands the dog still.

SDK boundary = DogAdapter. FakeDog (default with --fake, auto-fallback when no
SDK) integrates velocity commands so the whole protocol + control loop runs
with zero hardware. Fill RealDog with the actual SDK calls when known.

Protocol (v1, msgpack): REP :5583 requests, PUB :5584 'skill.status' events.
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

# ------------------------------------------------------------ configuration

MY_SKILLS = ["grasp", "move_to"]
PROTO_V = 1
TERMINAL = ("done", "failed", "stopped")

# board frame, meters -- tighten to the real free-floor area before live runs
ROOM_X = (-3.5, 5.5)
ROOM_Y = (-4.5, 5.5)
Z_REACH = (0.02, 0.90)          # gripper envelope; targets outside are rejected
DEFAULT_STANDOFF = 0.5          # m, stop this far from the grasp target
STANDOFF_TOL = 0.10             # m
HEADING_TOL = 0.15              # rad
NAV_TOL = 0.15                  # m, move_to arrival radius
PHASE_TIMEOUT = {"navigating": 45.0, "aligning": 10.0, "grasping": 15.0}
STUCK_WINDOW = 3.0              # s without progress -> failed("stuck")
STUCK_EPS = 0.05                # m
CTRL_HZ = 10.0
V_MAX, VY_MAX, W_MAX = 0.5, 0.3, 1.0   # body-frame velocity clamps
KP_LIN, KP_YAW = 0.8, 1.5


# ------------------------------------------------------------ SDK boundary

class DogAdapter:
    """Everything the skills need from the robot. Fill RealDog with the SDK."""

    def get_pose(self):
        """{"x","y","yaw"} in the board frame, or None while unlocalized."""
        raise NotImplementedError

    def send_velocity(self, vx, vy, wyaw):
        """Body-frame velocities (m/s, m/s, rad/s), called at CTRL_HZ."""
        raise NotImplementedError

    def stand_still(self):
        raise NotImplementedError

    def gripper_close(self) -> bool:
        """Attempt the grasp; True only if feedback confirms an object."""
        raise NotImplementedError

    def gripper_open(self):
        raise NotImplementedError


class FakeDog(DogAdapter):
    """Kinematic fake: integrates velocity commands; gripper always succeeds.

    Lets the whole protocol / control loop / watchdog / intent bridge run
    end-to-end with zero hardware.
    """

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
    """TODO: fill with the actual SDK once the model/interface is known.

    get_pose      <- M1: odometry from a known start pose, or a dog-mounted
                     calibrated camera + pupil_localizer against tags_world
    send_velocity <- high-level locomotion API (e.g. sport-mode velocity cmd)
    stand_still   <- damp / balanced-stand call
    gripper_*     <- gripper position + force feedback for the success check
    """


# ------------------------------------------------------------ skills

def _ang_norm(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def _navigate(dog, goal_xy, face_point, arrive_tol, phase, report, should_stop):
    """Shared P-controller loop. Returns None on success, else a failure reason."""
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
        # face the travel direction (or a fixed point) while translating
        fx, fy = (face_point or goal_xy)[0] - pose["x"], (face_point or goal_xy)[1] - pose["y"]
        herr = _ang_norm(math.atan2(fy, fx) - pose["yaw"])
        c, s = math.cos(pose["yaw"]), math.sin(pose["yaw"])
        bx, by = c * ex + s * ey, -s * ex + c * ey          # world -> body frame
        dog.send_velocity(max(-V_MAX, min(V_MAX, KP_LIN * bx)),
                          max(-VY_MAX, min(VY_MAX, KP_LIN * by)),
                          max(-W_MAX, min(W_MAX, KP_YAW * herr)))
        time.sleep(1.0 / CTRL_HZ)


def _align(dog, face_point, report, should_stop):
    deadline = time.monotonic() + PHASE_TIMEOUT["aligning"]
    while True:
        if should_stop():
            return "estop"
        if time.monotonic() > deadline:
            return "aligning_timeout"
        pose = dog.get_pose()
        if pose is None:
            return "unlocalized"
        herr = _ang_norm(math.atan2(face_point[1] - pose["y"],
                                    face_point[0] - pose["x"]) - pose["yaw"])
        if abs(herr) < HEADING_TOL:
            dog.stand_still()
            return None
        dog.send_velocity(0.0, 0.0, max(-W_MAX, min(W_MAX, KP_YAW * herr)))
        time.sleep(1.0 / CTRL_HZ)


def execute(dog, skill, params, report, should_stop):
    target = [float(v) for v in params["target"]]
    if skill == "move_to":
        report("navigating")
        err = _navigate(dog, target[:2], None, NAV_TOL, "navigating", report, should_stop)
        if err == "estop":
            report("stopped", "emergency stop")
        elif err:
            report("failed", err)
        else:
            report("done")
        return

    # grasp: standoff point on the dog->target line, face target, close, verify
    standoff = float(params.get("standoff", DEFAULT_STANDOFF))
    if params.get("bbox"):                       # scale standoff with object size
        lo, hi = params["bbox"]
        standoff = max(standoff, 0.4 * math.hypot(hi[0] - lo[0], hi[1] - lo[1]) + 0.3)
    pose = dog.get_pose()
    if pose is None:
        report("failed", "unlocalized")
        return
    dx, dy = target[0] - pose["x"], target[1] - pose["y"]
    d = math.hypot(dx, dy)
    if d > standoff:
        goal = (target[0] - dx / d * standoff, target[1] - dy / d * standoff)
    else:
        goal = (pose["x"], pose["y"])            # already inside standoff
    report("navigating", f"standoff {standoff:.2f}m")
    err = _navigate(dog, goal, target[:2], STANDOFF_TOL, "navigating", report, should_stop)
    if err:
        report("stopped" if err == "estop" else "failed",
               "emergency stop" if err == "estop" else err)
        return
    report("aligning")
    err = _align(dog, target[:2], report, should_stop)
    if err:
        report("stopped" if err == "estop" else "failed",
               "emergency stop" if err == "estop" else err)
        return
    report("grasping")
    if should_stop():
        report("stopped", "emergency stop")
        return
    ok = dog.gripper_close()
    report("done" if ok else "failed", "" if ok else "grasp_missed")


def validate(skill, params):
    """Feasibility gate BEFORE accept: bad coordinates must never reach motors."""
    t = params.get("target")
    if not (isinstance(t, (list, tuple)) and len(t) in (2, 3)):
        return "bad_params: target must be [x,y] or [x,y,z]"
    x, y = float(t[0]), float(t[1])
    if not (ROOM_X[0] <= x <= ROOM_X[1] and ROOM_Y[0] <= y <= ROOM_Y[1]):
        return "target_out_of_bounds"
    if skill == "grasp":
        if len(t) < 3:
            return "bad_params: grasp target needs z"
        if not (Z_REACH[0] <= float(t[2]) <= Z_REACH[1]):
            return "target_out_of_reach_z"
    return None


# ------------------------------------------------------------ protocol shell

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rep", type=int, default=5583)
    ap.add_argument("--pub", type=int, default=5584)
    ap.add_argument("--fake", action="store_true", help="Run the kinematic fake dog.")
    ap.add_argument("--fake-start", default="0,0,0", help="Fake dog start 'x,y,yaw'.")
    args = ap.parse_args()

    if args.fake or type(RealDog.get_pose) is type(DogAdapter.get_pose):
        sx, sy, syaw = (float(v) for v in args.fake_start.split(","))
        dog = FakeDog(sx, sy, syaw)
        if not args.fake:
            print("dog_link: RealDog not implemented -> falling back to FakeDog", flush=True)
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

    def publish(req_id, st, detail=""):
        msg = {"v": PROTO_V, "req_id": req_id, "state": st,
               "pose": dog.get_pose(), "busy": state["busy"],
               "detail": detail, "t": time.time()}
        with pub_lock:
            pub.send_multipart([b"skill.status", msgpack.packb(msg)])
        if st != "heartbeat":
            print(f"[status] {json.dumps(msg, ensure_ascii=False)}", flush=True)

    def heartbeat():   # consumers must be able to tell idle from dead
        while True:
            publish(state["req_id"], "heartbeat")
            time.sleep(1.0)

    threading.Thread(target=heartbeat, daemon=True).start()

    def on_stop():
        # freeze the gripper deliberately: dropping mid-carry can be worse
        # than holding -- release is an explicit gripper_open decision later
        print(">>> EMERGENCY STOP <<<", flush=True)
        dog.stand_still()

    def worker(req):
        rid = req["req_id"]
        sent = {"terminal": False}

        def report(st, detail=""):
            if st in TERMINAL:
                sent["terminal"] = True
                # free the busy gate BEFORE the terminal event goes out, so a
                # client that reacts instantly to done/failed/stopped (fetch
                # composition: grasp -> move_to) never bounces off "busy"
                state["busy"] = False
                state["req_id"] = ""
            publish(rid, st, detail)

        publish(rid, "accepted")
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
