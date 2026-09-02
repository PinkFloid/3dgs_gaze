#!/usr/bin/env python3
"""run_e4_selfcal3.py -- E4 追加批:会话连续的对象自校准(v2selfcal3)。

同一天的录像按 info.player.json 的开始时间排序,上一条回放结束时学到的偏置
(intents 最后一条 final 的 bias_deg)作为下一条的 --bias-init;每天第一条从零起,
间隔超过 1 小时视为新会话(重标定)也从零起。
其余配置同 v2selfcal2(无歧义门 0.2、≥2 物体、观测门 0.2),--bias-tau 0 不衰减。
    python Eye_Tracker/tools/run_e4_selfcal3.py [--plan]
可断点续跑:已有 log+csv 的录像只读取其末偏置,不重跑。
"""
import json, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
R = Path("/home/liuchy/recordings")
GL, EV = ROOT / "Eye_Tracker/tools/gaze_live.py", ROOT / "Eye_Tracker/tools/eval_e1.py"
RUN = ["conda", "run", "--no-capture-output", "-n", "nerfstudio", "python", "-u"]
CK = {"v7": ROOT / "SceneRebuild/lab_result/splatfacto/2026-08-15_173942/nerfstudio_models/step-000029999.ckpt",
      "v8": ROOT / "SceneRebuild/lab_result/splatfacto/2026-08-18_125627/nerfstudio_models/step-000029999.ckpt",
      "v9": ROOT / "SceneRebuild/lab_result/splatfacto/2026-08-20_201525/nerfstudio_models/step-000029999.ckpt"}
A = {"v7": ROOT / "SceneRebuild/archive_envs/v7", "v8": ROOT / "SceneRebuild/archive_envs/v8",
     "v9": ROOT / "SceneRebuild/archive_envs/v9_rec"}
RECS = [(R/"2026_08_16/000", "e1", "v7"), (R/"2026_08_16/001", "e2", "v7"), (R/"2026_08_16/s1", "s1", "v7"),
        (R/"2026_08_16/s2", "s2", "v7"), (R/"2026_08_16/s3", "s3", "v7"), (R/"2026_08_16/s4", "s4", "v8"),
        (R/"2026_08_16/s6", "s6", "v8"), (R/"2026_08_18/000", "s7", "v8"), (R/"2026_08_20/000", "c1", "v9"),
        (R/"2026_08_20/001", "c2", "v9"), (R/"2026_08_20/002", "c4", "v9"), (R/"2026_08_20/003", "c4", "v9"),
        (R/"2026_08_25/c1_1", "c4", "v9"), (R/"2026_08_25/c1_2", "c4", "v9"), (R/"2026_08_25/c1_3", "c4", "v9"),
        (R/"2026_08_25/u3", "u3", "v9")]
CFG = "v2selfcal3"
FLAGS = ["--rank", "capture", "--selfcal", "on", "--selfcal-min-objects", "2",
         "--selfcal-max-second", "0.2", "--selfcal-min-capture", "0.2", "--bias-tau", "0"]


def env_flags(era):
    f = ["--seg-dir", str(A[era]), "--ckpt", str(CK[era])]
    if era != "v9":
        f += ["--tags", str(A[era] / "tags_world.json")]
    return f


def start_time(rec):
    return float(json.load(open(rec / "info.player.json"))["start_time_system_s"])


def last_bias(log):
    b = [0.0, 0.0]
    for ln in open(log, encoding="utf-8"):
        e = json.loads(ln)
        if not e.get("provisional") and e.get("bias_deg"):
            b = e["bias_deg"]
    return b


def main():
    plan = "--plan" in sys.argv
    days = {}
    for rec, card, era in RECS:
        days.setdefault(rec.parent.name, []).append((start_time(rec), rec, card, era))
    for day in sorted(days):
        bias, t_prev = [0.0, 0.0], None
        for t0, rec, card, era in sorted(days[day]):
            if t_prev is not None and t0 - t_prev > 3600:   # >1h 间隔 = 新会话(08_16 下午/晚上换图重标)
                bias = [0.0, 0.0]
                if plan:
                    print("  -- 新会话(间隔 >1h),偏置归零")
            t_prev = t0
            log, csv = rec / f"intents_e4_{CFG}.jsonl", rec / f"e4_{CFG}_score.csv"
            if plan:
                print(f"{day} {time.strftime('%H:%M:%S', time.localtime(t0))} {rec.name:5} {card} {era}")
                continue
            if not (log.exists() and log.stat().st_size and csv.exists() and csv.stat().st_size):
                print(f"== replay {rec.name} [{CFG}] {time.strftime('%H:%M:%S')}  bias-init ({bias[0]:+.2f},{bias[1]:+.2f})deg", flush=True)
                if log.exists():
                    log.unlink()
                r = subprocess.run(RUN + [str(GL), "--replay", str(rec), "--headless", "--on-tag-deg", "0",
                                          f"--bias-init={bias[0]},{bias[1]}"] + env_flags(era) + FLAGS  # 等号形式:负数不被当成选项
                                   + ["--log", str(log)], capture_output=True, text=True)
                if "done:" not in r.stdout:
                    print(f"  !! replay 失败 {rec} {r.stderr[-400:]}", flush=True)
                    continue
                r = subprocess.run(RUN + [str(EV), str(log), "--card", card, "--map-dir", str(A[era]),
                                          "--csv", str(csv)], capture_output=True, text=True)
                hit = [l for l in r.stdout.splitlines() if "命中" in l]
                print(hit[0] if hit else f"  !! eval 失败 {rec}", flush=True)
            bias = last_bias(log) if log.exists() else bias
    if not plan:
        print(f"== E4 selfcal3 批跑完毕 {time.strftime('%H:%M:%S')}", flush=True)


if __name__ == "__main__":
    main()
