# Experiments bundle — 2026-09-09(论文线定稿;ICRA 2027 截稿 09-15)

口径(09-07 晚裁定,以此为准):E1 主曲线 = `collect_e1.py` 产物(`data/trials.csv`,327 站立 trial),
判定按冻结的 eval_e1 规则(同物 <1.2 s 合并、段 ≥1.0 s、卡序对齐取时段内最长注视),θ = 目标到任一其他命名物的最小张角,
由该录像头位中位算(结果盲);遮挡按目标/站位标签(`beyond_occ` / `occ=`);p2 三项执行错(`exec_error`)剔除并披露;
E4 = 204 trial 冻结子集(`data/table2.md`,09-06 审计口径:顺序 + 时段内最长注视,不设闸);E2/E3 = 39 真机 trial(20 有日志 + 19 现场判)。
夜里那条逐 trial 线(trial_theta / 射线遮挡 / table2_v10cfg)已作废,不在本包内。

## 文件

- `tex/EXPERIMENTS_0909.tex` — 实验章定稿(在 09-07 精简稿上填完 TBD:22 站位、急停时延;时延段换成审计口径;补打字输入披露);`tex/EXPERIMENTS_REWRITE.tex` = 改前稿,可 diff。
- `Figures/fig_theta.pdf|png` — Fig.4(= docs/E1_DATA/fig4,327 trial,Wilson CI,遮挡阴影);`fig_sigma_tiers.pdf|png` — σ 扫描分档;`fig_sigma_clear` — 清晰子集版;
  `fig_story_pink_apple_icra.pdf|png|svg` — 当前 Fig.5 粉苹果单案例(09-08 录像 008，含实测 gaze、地图定位及机器人 RGB；当前实验稿已同步)。`fig_story.pdf|png` 与 hand/box 拆分图保留为历史版本;`fig_pipeline.pdf` / `fig_storyboard.pdf`;`fig2_system_v7.pdf` — Fig.2;`fig_scene_map.pdf` — 场景/地图对照;`fig5_assets/` — Fig.5 原帧。
- `data/trials.csv`(逐 trial 主表,tags 含 stress/walking/beyond_occ/occ/exec_error/p1/p2)、`curve.csv`(分箱 [0,0.75) 0.75–1 1–1.5 1.5–2.5 2.5–4 4–6 6–20)、
  `sigma_curve.json` / `theta_bins.json`(fig_sigma 数据)、`table2.csv|md`(E4 全 26 配置)、`curve_dwell*.csv` / `trials_dwell.csv`(09-06 审计口径的对照曲线)、
  `audit_0906_REPORT.md`(绑定审计、真机 32 单、时延分解、急停)。
- `data/E2/` — `E2_RESULTS.md`(8-24 首夜结算)、`E2_CARDS.md`(v9 卡,20 行手填)、`E2_CARDS_V10.md`(v10 卡 19 行,口头结算、无日志)、`e2_tasks.csv|md`(逐单)、`e2_exec_times.csv`(8-27 执行段 153 单)。
- `data/cards/` — E1 卡片与 09-06/07 站位实测表。
- `docs/` — `E1_RESULTS.md`(台账)、`EXPERIMENT_MAP.md`(主张→实验→数字)、`PAPER_OUTLINE.md`、`TBD_LEDGER.md`、`README.md`(E1_DATA 说明)+ 打分脚本副本。

## 正文数字 → 文件

| 正文 | 数值 | 来源 |
|---|---|---|
| E1 ≥2.5° / 1–2.5° / <1° | 151/154 · 77/88 · 42/85 | trials.csv(去 stress/walking/exec_error,θ 按 theta_deg) |
| <1° 清晰 / 遮挡 | 26/37 (70.3%, [54.2, 82.5]) · 16/48 (33.3%, [21.7, 47.5]) | 同上,tags beyond_occ/occ |
| 边走 | 56/57(4 条录像) | tags walking |
| 佩戴者 p1 / p2 | 15/15 · 12/12(p2 三项 exec_error 剔除) | tags p1/p2 |
| 站位数 / 距离 / θ 范围 | 22 · 1.05–4.52 m · 0.13–24.1° | trials.csv station 列 |
| E2/E3 | 36/39;配对 7/8 vs 7/8 | E2_CARDS.md(v9 20 行)+ E2_CARDS_V10.md(19 行口头) |
| 时延 | 说完→派发 2.2/5.0 s;派发→accepted 0.04 s;accepted→done 15.0 s(门外,n=22;导航 9.2 + 抓取 5.6);房内导航 6.7 s | audit_0906_REPORT.md §4 |
| 急停 | 说完→急停单 2.0 s(n=48);在飞任务终态 0.13 s(n=42);52/58 接受 | 同上 |
| Table II | 166/204 · 58/66 · 50/78 · 14/15 等 | table2.md 全204 不设闸 + 按档列 |
| σ 扫描 | 1–2.5°:29/42/58/58/47/26 of 66;<1°:23/34/50/38/27/15 of 78 | table2.md s02/s05/ours/s15/s25/s40 |

## 重生成

`score_card.sh` → `collect_e1.py`(trials/curve)→ `plot_fig4.py`(Fig.4)→ `plot_fig_sigma.py`(σ 图);E4:`binding_stats.py`(table2)。
注意:v10 分割档已归档到 `SceneRebuild/lab_result/archive_map_v10`,现行 `segmentation_sam` 是 v11(demo 图),v10 录像重打分要指向归档目录。

## 逐 trial 线(09-09 补;中文实验稿 `tex/EXPERIMENTS_ZH_0909.md` 用的是这条口径)

口径:判定 = 每项时段内输出累计时长最长的实例(不设闸);θ = 时段内 tag 定位样本头位的逐分量中位到目标与最近命名物的张角(逐 trial);
遮挡 = 目标圆盘 13 条细射线中被其他候选挡住的比例(>0 记部分遮挡);v7–v10 全部按现行管线重放;E4 全 330 站立试验。
- `data_trialline/trials_theta.csv`(399 项逐 trial:dwell/gated 判定、θ_trial、θ_unit、遮挡比例、挡它的物体)、`table2_v10cfg.md|csv`(14 配置消融)、
  `ablation_v10cfg/trials_theta_<cfg>.csv`(逐配置逐 trial;含 naive 基线)、工具 `trial_theta.py` / `ablation_table.py`。
- `Figures_trialline/fig_theta_trial.pdf|png`(角间距曲线:全部/未遮挡/部分遮挡/行走,标各箱正确数/试次数)、`fig_sigma_trial.pdf|png`(σ 扫描三档+行走)。
- 主数:≥2.5° 132/139 (95.0%)、1–2.5° 92/103 (89.3%)、<1° 57/88 (64.8%;未遮挡 54/70 = 77.1%,遮挡 3/18 = 16.7%)、行走 56/57;
  方法对照 最近质心 225/330、球投票 201/330、单射线 σ0.1° 137/330、Full 281/330。
