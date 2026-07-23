"""层A:因果 visit/dwell 记账(VisitTracker)与近期注意缓冲(AttentionBuffer)。

VisitTracker 消费 gaze.intent 的 provisional/final 双流,产出意图无关的
progress / sustained / released 事件;AttentionBuffer 在其上保留最近 visit
的富记录,供眼-声绑定查询(fire_dwell<=0 时纯缓冲,不主动触发)。
"""

from __future__ import annotations

from collections import deque

OBJ0 = 10  # instance labels >= OBJ0 are objects; below: floor/wall/ceiling (同 grasp_intent.py)


def accepted(e, min_vote):
    """Same gate as grasp_intent.py: objects only, clean cone verdicts."""
    return (e.get("object_label", -1) >= OBJ0
            and e.get("vote_share", 0.0) >= min_vote
            and e.get("mode") == "cone"
            and e.get("object_centroid_world"))


def noun_match(noun, obj):
    """类别启发式:"杯子"命中"水杯"、"机器人"不误中"机械臂"。真类别字段随新地图上。"""
    return noun in obj or obj in noun or (len(noun) >= 2 and noun[:-1] in obj)


class VisitTracker:
    """Causal visit/dwell accumulation over fixation verdicts (intent-agnostic).

    Live dwell comes from provisional verdicts of the still-open fixation;
    finals settle the books (closed_s). Same-object fixations merge across
    gaps <= merge_gap; revisits count past visits only; 'sustained' fires at
    most once per visit. What a sustained visit *means* is the caller's call.
    """

    def __init__(self, fire_dwell, merge_gap, revisit_window=90.0, release_grace=0.6):
        self.fire_dwell = fire_dwell
        self.merge_gap = merge_gap
        self.revisit_window = revisit_window
        self.release_grace = release_grace
        self.visit = None    # object/label/t_start/t_last_end/closed_s/shares/fired/last
        self.run_fx = None   # still-open fixation: t_start/dur/object
        self.past = deque()  # (t_close, object) -- causal revisit counting

    def _dwell(self):
        v = self.visit
        run = 0.0
        if v and self.run_fx and self.run_fx["object"] == v["object"] \
                and self.run_fx["t_start"] >= v["t_start"] - 1e-9:
            run = self.run_fx["dur"]
        return v["closed_s"] + run if v else 0.0

    def _revisits(self, obj, now):
        while self.past and now - self.past[0][0] > self.revisit_window:
            self.past.popleft()
        return sum(1 for _, o in self.past if o == obj)

    def _close(self, t):
        v = self.visit
        if v is None:
            return []
        dwell = self._dwell()
        # a run_fx that belongs to the closed span must not leak into the next visit
        if self.run_fx and self.run_fx["object"] == v["object"] \
                and self.run_fx["t_start"] <= v["t_last_end"] + 1e-9:
            self.run_fx = None
        self.past.append((t, v["object"]))
        self.visit = None
        return [("released", {"object": v["object"], "dwell_s": round(dwell, 2),
                              "fired": v["fired"], "t": round(t, 3)})]

    def advance(self, t):
        """Clock tick from gated-out events: merge-gap timeout release.

        Grace on top of merge_gap: a merging same-object fixation announces
        itself only ~0.4-0.5s after its t_start (first provisional), so closing
        on the raw gap would kill visits the offline semantics would merge.
        feed() still applies the strict t_start-based gap, so dwell accounting
        stays exactly grasp_intent's; grace only delays the released event.
        """
        if self.visit and t - self.visit["t_last_end"] > self.merge_gap + self.release_grace:
            return self._close(t)
        return []

    def feed(self, e):
        """One accepted verdict in; a list of (kind, payload) events out."""
        out = []
        t0, t1 = float(e["t_start"]), float(e["t_end"])
        dur, obj = float(e["duration_s"]), e["object"]
        if e.get("provisional"):
            self.run_fx = {"t_start": t0, "dur": dur, "object": obj}
        elif self.run_fx and abs(self.run_fx["t_start"] - t0) < 1e-9:
            self.run_fx = None  # this fixation's final settles it below

        v = self.visit
        if v is None or obj != v["object"] or t0 - v["t_last_end"] > self.merge_gap:
            out += self._close(t1)
            self.visit = v = {"object": obj, "label": e.get("object_label"),
                              "t_start": t0, "t_last_end": t1, "closed_s": 0.0,
                              "shares": [], "fired": False, "last": e,
                              "revisits": self._revisits(obj, t0)}
        v["t_last_end"] = max(v["t_last_end"], t1)
        v["shares"].append(float(e.get("vote_share", 0.0)))
        v["last"] = e
        if not e.get("provisional"):
            v["closed_s"] += dur

        dwell = self._dwell()
        out.append(("progress", {"object": obj, "dwell_s": round(dwell, 2),
                                 "share": float(e.get("vote_share", 0.0))}))
        if not v["fired"] and dwell >= self.fire_dwell - 1e-9:
            v["fired"] = True
            last = v["last"]
            out.append(("sustained", {
                "object": v["object"],
                "target_world": last.get("object_centroid_world"),
                "dwell_s": round(dwell, 2),
                "revisits": v["revisits"],
                "mean_vote_share": round(sum(v["shares"]) / len(v["shares"]), 2),
                "p_none": last.get("p_none"),
                "sigma_deg": last.get("sigma_deg"),
                "candidates": last.get("candidates"),
                "t": round(t1, 3),
            }))
        return out


class AttentionBuffer(VisitTracker):
    """VisitTracker 原语义,外加最近 visit 的富记录,供眼-声绑定查询。"""

    def __init__(self, merge_gap, fire_dwell=0.0):
        # fire_dwell<=0 = 纯缓冲不触发;>0 = 同时兼任主动问询的触发器(--proactive)
        super().__init__(fire_dwell=fire_dwell if fire_dwell > 0 else float("inf"),
                         merge_gap=merge_gap)
        self.recent = []  # 已关闭 visit 的富记录,按关闭时间升序

    def _close(self, t):
        v = self.visit
        out = super()._close(t)
        if v is not None and out:
            last = v["last"]
            self.recent.append({"object": v["object"], "t_start": v["t_start"],
                                "t_end": v["t_last_end"],
                                "dwell_s": out[0][1]["dwell_s"],
                                "vote": float(last.get("vote_share", 0.0)),
                                "target_world": last.get("object_centroid_world")})
            self.recent = self.recent[-50:]
        return out

    def candidates(self, t_word, lookback, noun=""):
        """说指代词时刻往前看:仍在盯的目标排最前,其余按离开时间近排序。"""
        out = []
        if self.visit is not None:
            v, last = self.visit, self.visit["last"]
            out.append({"object": v["object"],
                        "gap": max(0.0, t_word - v["t_last_end"]),
                        "dwell_s": self._dwell(),
                        "vote": float(last.get("vote_share", 0.0)),
                        "target_world": last.get("object_centroid_world")})
        for r in reversed(self.recent):
            gap = t_word - r["t_end"]
            if gap > lookback:
                break
            if gap >= -0.5:
                out.append({**r, "gap": max(0.0, gap)})
        if noun:
            out = [c for c in out if noun_match(noun, c["object"])]
        out.sort(key=lambda c: c["gap"])
        return out
