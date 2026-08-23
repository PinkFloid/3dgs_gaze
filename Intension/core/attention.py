"""层A:因果 visit/dwell 记账(VisitTracker)与近期注意缓冲(AttentionBuffer)。

VisitTracker 消费 gaze.intent 的 provisional/final 双流,产出意图无关的
progress / sustained / released 事件;AttentionBuffer 在其上保留最近 visit
的富记录,供眼-声绑定查询(fire_dwell<=0 时纯缓冲,不主动触发)。
PlaceBuffer 是与之平行的落点通道:记"你看的那一点"(物体表面实际落点,
未命名实例也算),供 place/dest 槽消解。
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

    def candidates(self, t_word, lookback, noun="", fwd=0.6):
        """说指代词时刻的注视目标:那一刻正盯着的排最前(gap 0),其余按与
        t_word 的时间距离升序——往回最多 lookback,往后最多 fwd(眼比嘴慢
        半拍的小宽限,PUT_DESIGN §3 的 [t−lookback, t+0.6])。
        逐词时刻下 t_word 是过去的某一刻(ASR 晚到 1-2s,流已跑到句后):
        老实现拿"处理那一刻还开着的 visit"当 gap 0、又把 t_end 晚于 t_word
        的已结账 visit 一刀切掉,说「这个」之后才看的落点(纸箱子)会顶掉
        说话瞬间正盯着的目标(球L)——双指示词放置 object/dest 整个倒置
        (实测 -230149 拿纸箱子送球L)。现按注视区间与 t_word 的关系算。"""
        def gap_of(r):  # 注视区间 vs t_word:含=0 / 词前=离开多久 / 词后=多久才看
            if r["t_start"] - 1e-9 <= t_word <= r["t_end"] + 1e-9:
                return 0.0
            if r["t_end"] < t_word:
                g = t_word - r["t_end"]
                return g if g <= lookback else None
            g = r["t_start"] - t_word
            return g if g <= fwd else None

        out = []
        if self.visit is not None:
            v, last = self.visit, self.visit["last"]
            g = gap_of({"t_start": v["t_start"], "t_end": v["t_last_end"]})
            if g is not None:
                out.append({"object": v["object"], "t_start": v["t_start"],
                            "t_end": v["t_last_end"], "gap": round(g, 3),
                            "dwell_s": self._dwell(),
                            "vote": float(last.get("vote_share", 0.0)),
                            "target_world": last.get("object_centroid_world")})
        for r in reversed(self.recent):
            if r["t_end"] < t_word - lookback:
                break                          # recent 升序,再往前只会更旧
            g = gap_of(r)
            if g is not None:
                out.append({**r, "gap": round(g, 3)})
        if noun:
            out = [c for c in out if noun_match(noun, c["object"])]
        out.sort(key=lambda c: c["gap"])
        return out


class PlaceBuffer:
    """落点通道:近期物体注视的世界落点(centroid_world),供 place/dest 槽消解。

    与 AttentionBuffer 刻意分开:object 槽要"物体身份+物体质心"(E1 语义,勿动);
    place/dest 槽要的是"你看的那一点"——注视打在物体表面的实际落点(物品台的
    那个角,而不是物品台质心),未命名实例同样算数,不做投票门。
    只收物体注视(label >= OBJ0):地板/墙/天花板不当地点(地板判定噪声大,
    且"去哪"看目标处的物体/家具表达更稳)。驻留 >= min_dwell 防扫视噪声。
    """

    def __init__(self, min_dwell=0.4):
        self.min_dwell = min_dwell
        self.recent = []   # 已结账落点 {"t_end","point","object"},升序
        self.open = None   # 进行中注视(provisional 已满驻留)

    def feed(self, e):
        """一条 gaze.intent verdict 进来,静默记账(无事件输出)。"""
        p = e.get("centroid_world")
        if not p or e.get("object_label", -1) < OBJ0:
            return
        t0, t1 = float(e.get("t_start", 0.0)), float(e.get("t_end", 0.0))
        dur, obj = float(e.get("duration_s", 0.0)), e.get("object", "?")
        if e.get("provisional"):
            if dur >= self.min_dwell:
                self.open = {"t_start": t0, "t_end": t1, "point": list(p), "object": obj}
        else:
            if self.open and abs(self.open["t_start"] - t0) < 1e-9:
                self.open = None
            if dur >= self.min_dwell:
                self.recent.append({"t_start": t0, "t_end": t1,
                                    "point": list(p), "object": obj})
                self.recent = self.recent[-50:]

    def latest(self, t_word, lookback, exclude_obj=None, fwd=0.6):
        """说指代词时刻的落点:那一刻正盯着的最优(距离 0),其余取与 t_word
        时间距离最近的——往回 lookback,往后 fwd。返回记录或 None。
        fwd:dest 槽传 3.0——「说完'那里'目光才指过去」是实测常态(-230149:
        词后 0.8s 才看向纸箱子),ASR 迟到使流早已到位,不需要等待机制也
        捡得到(R18 裁定不等待;宽限外/流没到的仍如实报没看到)。
        距离并列取更晚的:眼睛最终停在哪,哪就是本意。
        exclude_obj:剔除该名物体的落点——"把这个放到那里"里 object 自己的注视
        也是落点,不剔会把"刚盯过的 L 表面"当送达点(把L放到L)。"""
        best, bd = None, None
        for r in self.recent + ([self.open] if self.open else []):
            if exclude_obj is not None and r.get("object") == exclude_obj:
                continue
            t0 = r.get("t_start", r["t_end"])
            if t0 - 1e-9 <= t_word <= r["t_end"] + 1e-9:
                d = 0.0
            elif r["t_end"] < t_word:
                d = t_word - r["t_end"]
                if d > lookback:
                    continue
            else:
                d = t0 - t_word
                if d > fwd:
                    continue
            if bd is None or d <= bd:  # <=:并列时后来者(更晚/仍在盯)胜
                best, bd = r, d
        return best
