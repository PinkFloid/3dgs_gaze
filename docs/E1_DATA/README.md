# E1 数据包(论文用)

重生成:任意录像重打分(score_card.sh)后跑 `python Eye_Tracker/tools/collect_e1.py`。

- **trials.csv** — 逐 trial 主表(12 条有效录像,来源与站位元信息在 collect_e1.RECS,
  与 docs/E1_RESULTS.md 台账一致)。字段:rec/card/room/map/station/tags/target/
  outcome(hit|miss|extra)/theta_deg/theta_src/vote/dur_s/dist_m。
  连击项(球L×2)已展开为独立 trial;"extra"=卡外多余注视,不计精度。
- **curve.csv** — 主曲线分箱:θ ∈ [0.5,0.75,1.0,1.5,2.5,4,6,20)。
  纳入=全部 hit/miss(e2 压力段除外,tags=stress 单独叙事);
  θ 口径:命中行=该 trial 从头位实算的 θ_min(与最近命名物的张角);
  缺失行无注视头位,用该录像命中行中位 θ 近似(theta_src=station)。
- **fig4_draft.png** — 草图:精度 vs θ(log 轴),红线=三球乱猜 1/3,
  红区=4m 遮挡极限(0.96°)以内。0.5–1.0° 两箱 n 小(23/13)有抖动:
  c4-A(3m/2.87,θ≈1.5)与 c4-B(4m/3.88,θ≈1.05)拉尺重录后此区间会补厚。

配套叙事(见 PAPER_OUTLINE):e2@3m 前排俘获 6/12(语音类过滤动机);
c2 同类对**零互换**(错=遮挡缺失,非孪生混淆);塌陷位置与物理遮挡极限对齐。
