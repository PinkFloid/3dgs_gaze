# main.tex TBD 台账(2026-08-24;ICRA 截稿 09-15)

> 状态:✅=现在就能填 / 🐕=明天狗场出 / 🖥=桌面活(不占实验室) / 🎥=B 清单眼动仪晚
> / 👤=狗端同学 / ✍=删句或改写即可。"来源"列 = 数字从哪来,别现编。

## ✅ 现在就能填(数字已在仓库里)

| TBD | 建议值 | 来源/备注 |
|---|---|---|
| E1-total | 157(另 stress 12 条单独叙事) | docs/E1_DATA/curve.csv 合计 |
| E1-near | 98.6\% (68/69, Wilson [92.2, 99.7]) | θ≥2.5° 三箱合并 |
| E1-sep-safe | 2.5° | 同上 |
| E1-far | 27\% @0.63°(或 sub-σ 合并 47.2\% [32.0, 63.0]) | 口径二选一 |
| E1-knee | 1.0–1.5°(与遮挡极限 0.96°、σ≈1° 重合) | ⚠ 正文必须明说 σ/遮挡混叠;🎥 无遮挡档补录后可拆 |
| E1-headline(摘要) | 建议"98.6\% at separations ≥2.5°, collapsing to chance below 1°" | 措辞你定 |
| drift-deg | 2.0° | gaze_precision_actual.json drift_deg=1.966;⚠ tex 注释写 2.6°(宪法§4.5),两处不一致,以哪个为准你裁定 |
| sphere-fail-share | 84\% | rec002,tex 注释已确认口径 |

## 🐕 明天狗场出

摘要 E2-headline 与空 \TBD{}(trial 总数)/ Table I 全部格子与时延列 /
E3-in、E3-out / sys-estop / freeze-commit(今晚 commit hash)/
n-balls、layout(定版摆位后实测间距)

## 🖥 桌面活(不占实验室;可分给我)

| TBD | 做法 |
|---|---|
| timing-depth / timing-cone | 现工作站跑一次计时脚本(旧值 3ms / 4-6ms @TITAN X) |
| map-images / map-instances / map-named | v9 建图产物统计(SceneRebuild 输出 + names.json) |
| map-board-rms / map-tag-rms / chain-deg | v9 对齐/survey 日志 + verify blend 复测(旧值 0.48mm / 2.2mm / ~0.1°) |
| sys-loc / sys-sigma / sys-rep | 从 E1/E2 会话日志统计(定位覆盖率、戳后 σ、下发回执延迟) |
| Table II 全部 + best/worst 窗 | E4 回放消融批跑(录像已齐,纯计算) |
| lastmeter-budget | ⚠ 设计已变:无桌面 tag,hint 经狗自身位姿投影。从 E2 日志实测投影点到目标框的残差中位数;组成 = 桌前位姿误差 × 质心误差 |

## 🎥 B 清单(眼动仪晚,提质非阻塞)

E1-walk-n / E1-walk / E1-stand(边走条件)——不录则 ✍ 删 V-B 末句

## 👤 委派

robot-subsection(III-D):狗端同学,tex 注释里已列要覆盖的点

## ✍ 正文待改(非数字,写作日清单)

- [ ] V-B 设计段:spacing×distance 网格 → 实际的 θ 阶梯口径;N∈{2,3,5} → 实际
- [ ] V-C K≥25 与实录数核对
- [ ] 标题、作者、thanks
- [ ] 删 APPENDIX/ACKNOWLEDGMENT 的模板占位字(乱码引号那两段)
- [ ] Fig.1-5 就位后核对引用编号;摘要最后过一遍数字与表一致
- [ ] E1-knee 处补一句 σ/遮挡混叠的诚实讨论(无遮挡档录了就改成拆分结论)
- [ ] 预注册披露:08-20 物品台先验 0.3→0 是采数后的唯一协议修订(重打分 c2 6→10、c4 0→7),
      正文一句话披露修订+理由+日期,并指向 E4 的 prior on/off 消融报告两种设置——
      不披露会和 intro "none excluded/frozen" 的措辞打架

## 入库说明

paper/main.tex 由 08-24 会话从你贴的版本落库,相对原文三处改动:
graphicx 已加、CTAN 注释 URL 修复、补了缺失的 \bibitem{kothari2020}
(正文引了但原稿没有该条)。若 Overleaf 上有更新版,导出覆盖即可。
