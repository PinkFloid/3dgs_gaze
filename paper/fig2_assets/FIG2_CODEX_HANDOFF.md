# GazeSplat 主图交接：给 Linux 上继续修改的 Codex

日期：2026-09-05。本文记录用户在当前会话中的反馈与已完成工作，方便继续精修；当前版本**尚未获用户最终认可**。

## 从哪里接着改

最新版本：`output/pdf/gazesplat_fig2_v2_refined.pdf`。对应预览、可编辑 SVG 和素材说明位于 `output/fig2_refined_v2/`。

上一版：`output/pdf/gazesplat_fig2_v2_flow.pdf`，对应 `output/fig2_flow_v2/`。它保留了原来的锥形示意，可用于对照用户所说的“前面不太行”。

更早的大地图版本：`output/pdf/gazesplat_fig2_v2_complete.pdf`，对应 `output/fig2_complete/`。

用户最初提供的旧图：`paper/fig2_assets/fig2_v7.pdf`；同目录保存 v7 的 PPTX、生成脚本与 PNG 预览。

## 已明确的用户偏好

1. 主方法更新为 **v2**，不再沿用冻结 v1 的核心公式。
2. 核心贡献是 **3DGS + 歧义消除**。地图不能缩成一个弱小的离线准备框。
3. **语音模式没有 confirm**。成功绑定后编译并直接发送可执行指令。
4. 用户喜欢原图的流程：上方一次性建图；中间注视事件与共享地图共同支持消歧；下方语音解析和时间绑定；右侧机器人执行。已经按此结构做了合并版。
5. 用户明确说 **“这个 3DGS 的效果我很喜欢”**。当前地图采用 `assets/hero_1.png` 的裁切与实例叠色。最新修改逐像素保留了合并版中的主地图区域。
6. 用户说 **“前面这个不太行”**，未具体确认是最左侧注视流程还是中间锥形示意。当前按两块共同处理：最左侧去掉大填色流程框；中间用佩戴者位姿下的模型 RGB 渲染和局部实例查询替换锥形。
7. 用户要求最后的执行照片更干净、向左裁一些。已收紧裁切，突出机械臂、夹爪和球；仅裁切与等比缩放，没有修饰照片内容。
8. 用户最新反馈原话：**“你 push 一下，虽然还是有些不一样，但是我会让 Linux 机器上的 Codex 来做更细的修改。”** 因而不要把现在的前部示意当作已经定稿。

## Linux 上重建排版

仅修改文字、连线、排版、图片裁切时，**不需要 GPU、训练模型或 Nerfstudio**。真实渲染素材已经随提交保存。

```bash
python -m pip install reportlab pillow
# 系统需要 Arial、Liberation Sans 或 DejaVu Sans 的 regular/bold/italic 字体。
# Debian/Ubuntu 常用字体包为 fonts-liberation；也可设置 GAZESPLAT_FONT_DIR。

python tmp/fig2_design/build_refined_v2.py
pdftoppm -scale-to 2600 -singlefile -png \
  output/pdf/gazesplat_fig2_v2_refined.pdf \
  output/fig2_refined_v2/gazesplat_fig2_v2_refined
```

源文件结构：

- `tmp/fig2_design/build_complete.py`：最初完整版本，同时提供 PDF/SVG 排版函数。项目根目录从脚本位置推导，字体自动查找 Windows/Linux 的常见位置。
- `tmp/fig2_design/build_flow_v2.py`：保留原图流程的完整布局，复用上一个脚本的绘图函数。
- `tmp/fig2_design/build_refined_v2.py`：最新前部与执行照片调整，通过替换前一布局中的片段生成结果。

目前为了快速比较版本，后两者使用文本片段替换。后续大幅改版时，可整理为独立布局函数；不要因此覆盖用户已经喜欢的主地图裁切效果。Windows 下使用 Arial，Linux 采用替代字体后需重新检查文字宽度。

## 3DGS 素材和重新渲染

`output/fig2_complete/assets/` 包含 3 个视角的原始 RGB / 实例叠色图、佩戴者视角模型 RGB、深度、不透明度和实例补丁，以及 `render_metadata.json`（相机矩阵、查询参数、实例轮廓、分数）。

它们由项目当前 lab_colmap_v9 的 splatfacto 模型生成。698 MB 模型 checkpoint 没有随本次提交上传。需要新视角时，在有 CUDA、Torch、gsplat、Nerfstudio 相关依赖的环境中运行：

```bash
python tmp/fig2_design/render_complete_assets.py \
  --checkpoint /path/to/step-000029999.ckpt \
  --segmentation SceneRebuild/lab_result/segmentation_sam
```

也可使用 `GAZESPLAT_CKPT` 环境变量。元数据中记录的旧 `E:/Grasp/...` 路径只用于追溯 Windows 上的源模型，Linux 重新渲染时必须提供本机 checkpoint 路径。

## 方法与素材的边界

- 解析阶段不依赖机器人的当前观测；机器人只在执行阶段重新观测目标。导航坐标系对齐到同一度量地图。
- v2 查询实际覆盖方形角补丁（水平、垂直分别 ±2σ），不是严格圆锥径向截断。
- 可见质量为 `q_k = m_k / W`，capture 为 `q_k / C_k`。capture 可以大于 1，不是概率。不要恢复 v1 的先验加权后验或单一 q 门限。
- 当前主图的局部查询来自真实模型上的**说明性虚拟注视**，语音与时间轴也是示意。执行照片来自项目既有实录，尚未统一到同一次完整交互；需要保留这一说明，或之后用匹配实录替换。
- `hero_1.png` 中 `ball_L/M/R` 对应注册实体 `球L/M/R`。
- 最新照片裁切框为源图 `delivery.jpg` 中 `(1050, 500, 1480, 1080)`；上一版为 `(1060, 400, 1700, 1080)`。
- 方法公式核对可参考 `paper/METHOD_V2_NOTES.md`，其中“语音无 confirm”的用户修正优先于历史设计文档。

## 验证记录

Windows 上已从 PDF 渲染 PNG，检查文字、连线、裁切；最新主地图区域与前版逐像素相同。推送前验证了从其它工作目录运行最新版排版脚本，以及不加载 GPU 的渲染器 `--help` 入口。尚未在 Linux 主机上实际执行字体替代后的排版。
