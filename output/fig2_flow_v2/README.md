# 原流程 + v2 主方法：合并版主图

这版保留 `paper/fig2_assets/fig2_v7.pdf` 的流程组织：上方一次性建图，中间注视事件与地图共同支持消歧，下方独立语音解析与绑定，右侧机器人执行。

更新内容：

- 地图与实例消歧并列突出，两者合计占据中间区域的大部分。
- 保留注视不确定性示意，使用 v2 的 ±2σ 方形查询补丁、可见质量 q_k、尺寸归一化 q_k / C_k 与绑定门限。
- 删除旧版 posterior、类别先验 π、单一 q 阈值及语音 confirm。
- 去掉未在当前方法中核实的导航算法名称和旧实例数量。
- 成功绑定后直接编译、发送结构化指令。机器人当前观测没有反馈到指代解析阶段。

## 输出

- `../pdf/gazesplat_fig2_v2_flow.pdf`：一页论文插图 PDF。
- `gazesplat_fig2_v2_flow.svg`：可编辑 SVG，内嵌全部图片。
- `gazesplat_fig2_v2_flow.png`：2600 px 宽预览。

## 素材说明

地图与局部查询图沿用 `../fig2_complete/assets/` 中的真实模型渲染。局部查询为人为指定注视方向的**方法示意**，语音与时间轴也是示意。机器人照片来自项目已有递送实录，不宣称与查询图属于同一轮交互。详细参数与来源见 `../fig2_complete/README.md` 和对应 `assets/render_metadata.json`。

可用图注：

**Overview of GazeSplat.** Offline multiview reconstruction, metric alignment, and instance association produce a persistent instance 3DGS map. Online, the wearer's localized gaze is organized into world-space fixation events. Map rendering provides view-dependent surface evidence within gaze uncertainty; the system normalizes visible mass by expected capture at the target's angular size and applies a binding gate. Speech is parsed into actions and referring expressions, which are associated with time-aligned fixation events. Successfully grounded commands are sent to the robot, whose navigation frame is aligned to the same map. The robot re-observes the target during execution; reference resolution does not use its current observation. The local query and timing are illustrative, and the execution photograph is from a recorded robot run.

生成脚本：`../../tmp/fig2_design/build_flow_v2.py`。排版原语复用 `build_complete.py`；原有 PDF 和上一版成图未覆盖。
