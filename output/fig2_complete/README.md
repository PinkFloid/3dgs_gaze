# GazeSplat 主图：v2 完整成图样稿

这版以持久实例 3DGS 地图与实例消歧为视觉中心。语音成功绑定后直接编译、发送机器人指令，没有确认步骤。

## 文件

- `../pdf/gazesplat_fig2_v2_complete.pdf`：论文插图 PDF，182 mm 宽。文字、箭头和图形为矢量；场景与执行照片为内嵌位图。
- `gazesplat_fig2_v2_complete.svg`：可编辑 SVG，所有图片已内嵌。
- `gazesplat_fig2_v2_complete.png`：2400 px 宽预览。
- `assets/render_metadata.json`：模型路径、虚拟相机、注视查询参数与原始候选评分。

## 素材与数据来源

中央地图与左侧局部视图均从当前 lab_colmap_v9 的 splatfacto checkpoint 重新渲染。中央图仅裁取任务物品台区域，三个球的着色与轮廓来自实例关联的 Gaussian 集合。`ball_L / ball_M / ball_R` 对应已注册名称 `球L / 球M / 球R`。

深度、不透明度、可见实例补丁与候选排序来自对同一真实模型的一个**人为指定的说明性注视查询**，不是实测用户实验。该查询使用当前 v2 的 `cone_votes` 与 `rank_votes`，σ = 1°，65 × 65 的方形补丁覆盖 ±2σ。选中对象球R的 capture = 1.238，share = 0.458，满足当前在线绑定门限。图中数值为 capture 分数，不是概率；它可以超过 1。此补丁平均不透明度约 0.996，故不透明度图接近全白，图中标注 α ≈ 1。

注视时间轴、语音 “Bring me that ball.”、佩戴者线稿和期望捕获示意圆为方法示意，不是对应实验的逐字语音或测量结果。图中的球R、白杯1、红杯候选来自空间解析阶段；语言槽位绑定位于其后。

右侧照片裁自项目已有 `paper/fig2_assets/robot_nav.jpg` 和 `delivery.jpg`，用于展示机器人执行；不是机器人相机视角，也未宣称与说明性查询属于同一次试验。原始录像中的绿色注视标记予以保留。

## 建议英文图注

**Overview of GazeSplat.** (a) The system synchronizes the wearer's gaze, camera pose, and speech with word-level timestamps. (b) A persistent instance 3DGS map provides a shared metric reference and view-dependent depth and opacity. Gaze-weighted visible surface evidence is normalized by expected capture at the target's angular size, followed by candidate ranking and a binding gate. (d) Referring expressions are associated with time-aligned fixation events to compile an ordered sequence of grounded robot commands. (c) The robot navigates in the aligned map frame, re-observes the target, and performs the requested interaction. Reference resolution does not require the robot's current observation. The map query and timing are illustrative; the execution photographs are from recorded robot runs.

## 正式投稿前建议替换的素材

如果希望主图讲述同一次完整交互，补充以下同一轮数据即可：

1. 一张无调试扫描线、无软件边框的佩戴者第一视角原始 RGB，以及对应注视像素和地图位姿。
2. 同轮语音文本和词级时间戳、被绑定注视事件的开始/结束时间与实例名。
3. 同轮机器人靠近目标以及抓取/递送的清晰画面；最好提供无绿色调试十字和状态文字的原视频帧。

当前已具备可用的 3DGS 模型渲染，不需要额外拍摄地图素材。这些补充主要用于统一主图的实录故事线，而不是填补版式占位。

## 重建

场景资产脚本：`../../tmp/fig2_design/render_complete_assets.py`（nerfstudio CUDA 环境）。

矢量排版脚本：`../../tmp/fig2_design/build_complete.py`（Python + reportlab + Pillow）。

本次只新增输出与生成脚本，未覆盖 `paper/fig2_assets/fig2_v7.pdf` 或修改论文正文。
