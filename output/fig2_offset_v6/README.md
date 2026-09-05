# v6：完整起点、偏下注视与目标表面证据

本版面板呈现一个便于讲解的真实例子：从眼动仪起点发出的中心射线落在台面上、偏在目标球下方；当前方法的角不确定性查询仍给目标球最多的物体加权证据。图中没有移动注视起点或人为调整视线来对准球。

## 数据与尺寸

- 注视事件：`Intension/logs/20260827-175225/gaze.jsonl` 中非 provisional 事件 `t_start=14000.495955999999`，`t_end=14001.904225`；位于 recording `2026_08_27/019`。
- 原始 `origin_world`、`centroid_world` 原样使用。按记录方向重新渲染的中心射线命中 `物品台`，没有命中目标球。
- 目标球质心至中心视线的最近点，其世界高度比质心低约 **4.7 cm**；还存在横向偏移。图不声称误差只有竖直分量，也不将这次事件等同于系统标定偏置。
- 当前论文参数：`sigma=1 deg`、水平/垂直 `±2 sigma`、`33x33` 查询。代表注视距离 **2.8253 m**；该距离上垂直于轴线的方形支持域截面边长 **19.7323 cm**。
- 此例重新选择了注视事件，因此距离和截面边长与 v5 不同。历史事件的 `sigma_deg=1.5`；此图是当前方法参数下的离线重放，不声称展示旧运行时完整配置。
- 三个实例框来自地图的 AABB：实例 259、261、263，对应 ball_L/M/R。地图使用 `2026-08-20_201525_nobottle` 3DGS checkpoint。

## 图中证据的含义

重放使用原查询代码 `cone_votes` / `rank_votes`，计算结果如下：

| 对象 | 物体集合内的加权证据份额 | 原始有效匹配样本数 |
| --- | ---: | ---: |
| ball_M | 66.7588% | 126 |
| ball_L | 16.3896% | 67 |
| 其他物体合计 | 16.8516% | 99 |

这些份额的分母为所有注册目标物体的加权质量总和，**不是所有 1089 条射线**；也不是最终正确率、成功率或 capture 分数。图中写明 `Weighted evidence among objects`。本例目标 `q=0.120`、`capture=0.584`，`p_none=0.820`；大量查询质量仍属于台面、背景、无匹配或不透明度损失。这并不改变目标在物体候选中的优势。

最终选取仍保留可见质量、面积/距离归一化及候选闸门流程，不应将正文改成简单地比较射线条数。

## 绘制规则

- 起点图标的中心精确落在记录起点的透视投影上。橙色方向箭头与记录的中心射线共线；橙色十字对应重放中心射线的实际深度采样点。
- 四条角边界按实际水平、垂直角支持域计算，未扩大锥宽或缩短眼动仪到场景的距离。完整起点和局部地图通过同一透视相机投影。
- 展示相机改变构图，不改变注视和地图。线条按该相机渲染的深度、opacity 处理遮挡。
- 连线统一抽取查询网格的行、列 `0,4,...,32`，仅展示其中匹配到注册目标物体的射线；并非按目标手工挑选射线。此网格共有 9 条 ball_M、6 条 ball_L、4 条红苹果、5 条香蕉射线。线条强弱体现角核权重与 opacity。台面、背景和无效射线未连线，以保持图面可读。
- 表面标记使用所有有效物体匹配样本，不只使用连线子样本。目标球为绿色，其他物体为灰色。图中的 evidence 条形图使用完整查询结果。
- 右侧抓取照片仍来自 recording `2026_08_27/020`，展示同一注册目标，但不是与该查询事件连续的执行照片；绑定时间线为示意。配套图注明确区分。

## 输出和复现

仅重新排版时直接运行 `build_offset_v6.py`：它使用此目录随提交保存的素材，不需要 `output/fig2_recorded_v3/`、GPU 或原始录制。布局函数依赖同目录提交的 `build_recorded_v3.py` 和已有的 `build_complete.py`。

完整查询和重新渲染需要本机 3DGS checkpoint、分割数据及 CUDA / Nerfstudio 环境；模型 checkpoint 与原始视频不随本次主图提交。主地图素材的生成脚本为 `render_recorded_assets.py`，抓取照片提取脚本为 `extract_recorded_grasp.py`。后两者将再生素材写到本地 `output/fig2_recorded_v3/assets/`；需更新 v6 这两项素材时，再复制相应图像和 provenance 到本版 assets。

重新筛选事件的 `select_offset_v6.py` 还需要本地原始 `gaze.jsonl` 日志；仅重画已选事件可直接使用已提交的 `assets/candidates.json` 运行 `render_offset_v6.py`。

- 主图 PDF：`output/pdf/gazesplat_fig2_offset_v6.pdf`
- 可编辑 SVG：`gazesplat_fig2_offset_v6.svg`
- 整图 / 面板预览：`gazesplat_fig2_offset_v6.png`、`disambiguation_panel.png`
- 图注：`caption.tex`
- 事件筛选结果、完整坐标与证据：`assets/candidates.json`、`assets/geometry.json`
- 完整查询采样：`assets/query_arrays.npz`

```bash
conda run --no-capture-output -n nerfstudio python tmp/fig2_design/select_offset_v6.py
conda run --no-capture-output -n nerfstudio python tmp/fig2_design/render_offset_v6.py
python tmp/fig2_design/build_offset_v6.py  # 需要 reportlab、Pillow
pdftoppm -scale-to 3600 -singlefile -png \
  output/pdf/gazesplat_fig2_offset_v6.pdf \
  output/fig2_offset_v6/gazesplat_fig2_offset_v6
```

旧版本保留，系统算法及论文参数未被修改。
