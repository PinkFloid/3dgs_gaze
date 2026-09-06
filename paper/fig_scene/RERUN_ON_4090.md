# 场景与地图结果图：在 4090 上用建图原图重跑

更新：2026-09-06。以下要求来自用户最新确认，优先于旧版实录对比图的说明。

## 必须遵守的比较口径

**车和网球在建图后都被移动过。后续 Pupil 实录里的位置与地图状态不同，不能拿这些实录与地图的差异判断 3DGS 重建误差。** 此前看到的偏移不能直接归因于重建，也不能只解释为相机定位误差。

本图必须使用 **该地图建图时的 iPhone 图像**，配对 **同一帧的建图内参、位姿及同一版本 3DGS 模型**。这里“重跑”指重新选图、渲染和排版；先使用已有模型，不需要重新训练或采集新照片。

目的：通过可直接对照的原图、渲染和等权叠加，让读者判断重建与建图图像的视觉差异。不要预先写死“差距非常小”；根据实际结果决定图注措辞。

## 可直接发给 4090 上 Codex 的提示

> 请先阅读 `paper/fig_scene/RERUN_ON_4090.md`，按这里的最新口径重做论文实验场景与地图结果图。车和网球在后续录制期间都移动过，所以禁止用 Pupil 世界相机实录做重建质量对比。请在本机找到最终布局的 iPhone 建图图像、对应 aligned transforms、原始 3DGS checkpoint 和匹配的实例快照。优先检查 `E:\Grasp\data\lab_colmap_v9` 及 `E:\Grasp\outputs\lab_colmap_v9\splatfacto`，实际路径以文件核对结果为准。
>
> 用建图图像各自的相机内参和位姿渲染。先筛出物品台、三个网球和两个白杯都清楚可见、无遮挡、没有运动模糊的几个视角，做候选对照，再选最终图。四栏依次为：建图原图、同位姿 3DGS RGB 渲染、50% 原图 + 50% 渲染、实例颜色地图；下方分别放大三个网球和两个白杯。各栏必须共享相机、裁剪和缩放，实例颜色来自真实实例 ID。叠加图用于展示原图与渲染的接近程度，不是实例着色覆盖图。
>
> 可以做与建图内参一致的畸变校正及共同裁剪；不要用后期配准、移动物体、变形、生成式修补或调色来制造一致。先复用 `render_iphone_comparison.py` 和 `build_iphone_comparison.py`，核对模型坐标系与相机设置，再根据真实原图修正需要的实现细节。输出 PDF、PNG、可编辑 SVG、独立面板、来源与相机元数据、图注和复现命令。检查 PDF 实际渲染结果后交付。若图像、模型或分割版本不对应，请明确报告缺少什么，不要换用后期实录补齐。

## 现有材料与版本

| 材料 | 已知位置或标识 |
| --- | --- |
| 最终布局 iPhone 数据 | 配置记录为 `E:\Grasp\data\lab_colmap_v9\transforms_aligned.json`；图像路径形如 `images/frame_00018.JPG` |
| 原图相机元数据 | 144 帧，4284 × 5712，`OPENCV` 模型；以原文件的逐帧/全局字段为准 |
| 同版本模型 run | `2026-08-20_201525`，checkpoint 为 `nerfstudio_models/step-000029999.ckpt`，972044 个高斯 |
| 模型在 Linux 的位置 | `SceneRebuild/lab_result/splatfacto/2026-08-20_201525/` |
| 目标实例快照 | Linux 的 `SceneRebuild/archive_envs/v9_rec/`，需要 `points.npz`、`instances.json`、`names.json` |
| 局部实例 | 259 / 261 / 263 = 球 L / M / R；266 / 267 = 白杯 1 / 2 |
| 数量 | 259 个前景分割实例；15 个命名分割组件对应 13 个不同名称 |

应保留建图时的水瓶和物品位置，使用原始模型，不能混入 `_nobottle` 修改版。`SceneRebuild/lab_result/segmentation_sam/` 的当前内容可能已经改动，不能仅凭目录名认定它就是上述快照。

checkpoint、iPhone 原图和 `archive_envs/` **不随这次 Git 提交上传**。在 4090 上先找本机对应文件；缺少实例快照时，从 Linux 的上述目录复制三个文件。脚本会核对分割点与 checkpoint 高斯中心（最大距离不超过 `1e-6`）、实例数、命名数及五个 ID，发现不匹配会报错。

## 相机与坐标系核对

1. **逐图配对**：从 transforms 的 `file_path` 找原图，按同一个 frame 取 `transform_matrix`，逐帧内参优先于全局字段。原图像素尺寸和方向必须匹配元数据；不要拿另一组导出照片替代。
2. **畸变**：当前原图包含 OPENCV 畸变参数。脚本以这些参数校正原图，并在相同针孔内参下渲染。不要将带畸变原图直接与针孔渲染混合。若实际训练输入已去畸变，须先核对其对应的无畸变内参，不能重复校正。
3. **相机轴**：aligned transforms 的相机为 OpenGL 约定；渲染器使用 OpenCV 光学轴，转换为 `T_world_cam[:3, 1:3] *= -1`，再求逆得到 world-to-camera。
4. **模型坐标系**：已检查本机原始 run 的配置：`orientation_method=none`、`center_method=none`、`auto_scale_poses=false`、`scale_factor=1.0`；`camera_optimizer.mode=off`。其 `dataparser_transforms.json` 的总 scale 为 1.0。aligned transforms 中的相机矩阵已经完成地图对齐，不能把记录用的 `applied_transform` 再乘一次。4090 上需核对实际使用的 run 与这些设置一致；不同设置须复现训练时的变换和相机优化结果。
5. **渲染设置**：本机原始 run 为 SH degree 3、classic rasterization、`use_bilateral_grid=false`，当前脚本与之对应；换模型配置时需同步渲染路径。
6. **统一尺度**：默认渲染宽度上限 2200，实拍和内参一起缩放。要保留原分辨率可指定 `--max-width 4284`，根据显存选择。四栏始终使用同一尺度。

## 在 4090 上运行

先在仓库执行 `git pull --ff-only`，进入已有可运行 Torch/CUDA/gsplat 的环境。无需重新创建训练环境。排版另需 `reportlab` 和 `Pillow`；字体自动选择 Windows Arial 或 Linux Liberation Sans，也可退回 PDF 标准字体。

以下命令从仓库根目录运行，Windows PowerShell / CMD 和 Linux shell 均可使用。将 `<...>` 替换为已经确认的实际路径与帧名；不要原样复制占位值。

```text
python paper/fig_scene/render_iphone_comparison.py --checkpoint "<原始模型的 step-000029999.ckpt>" --segmentation "<匹配的 v9_rec 实例目录>" --transforms "<lab_colmap_v9/transforms_aligned.json>" --frame-name "frame_XXXXX.JPG" --out-dir output/scene_map_comparison_iphone/assets

python paper/fig_scene/build_iphone_comparison.py --asset-dir output/scene_map_comparison_iphone/assets/iphone_frame_XXXXX

pdftoppm -png -r 240 -singlefile output/pdf/gazesplat_scene_map_comparison_iphone.pdf output/scene_map_comparison_iphone/gazesplat_scene_map_comparison_iphone
```

如果没有 `pdftoppm`，可以用本机 PDF 查看器或 PyMuPDF 渲染检查。渲染器一次处理一帧，多帧时重复第一条命令并改变帧名，素材分目录保存；**同一帧的实拍与渲染不能来自不同候选**。布局要求五个对象均位于画面内，脚本的投影检查不能代替肉眼确认遮挡与清晰度。

渲染输出：`real.png`、`render.png`、`blend.png`、`instances.png` 和 `metadata.json`。额外的 `instance_overlay.png` 是语义叠色备选素材，**不能替代第三栏的 `blend.png`**。排版输出 PDF、SVG、`panels/` 及 `figure_manifest.json`；PNG 由 PDF 渲染得到。竖幅 iPhone 图在总览四栏统一裁成横幅，局部放大由真实实例的投影区域确定；如需全幅竖图应一起调整四栏布局。

目前已经检查脚本语法与命令行入口，但 **iPhone 原图不在本机，因此尚未进行该路径的真实数据端到端验证**。在 4090 上完成后，将实际选用的图片路径、模型、相机与处理步骤写进输出说明，并更新此状态。

## 最终图与图注验收

- 上排四栏顺序不变；下排分别突出三个相同网球和两个相同白杯。不同颜色表示不同真实实例，ID 仅对这一张地图有效。
- 50/50 叠加可以帮助观察边缘是否重合，但会弱化差异，必须同时保留原图、渲染两栏。看到明显残差要保留并解释，不能靠后期修掉。
- 每个面板保留独立素材。确认共同裁剪、正确图像方向、图例与 ID 对应、标签不遮挡目标、PDF 字体和边界正常。
- 来源写成 **iPhone 建图图像与对应建图相机位姿的渲染对比**，不要沿用 Pupil 实录图注。若选中帧参与训练，只能说明该视角的重建拟合，不能把它称为未见新视角的泛化结果。若补充 PSNR/SSIM，需明确 train/eval 划分、分辨率和有效像素范围，不能只对精选局部得出全场景结论。

图注起点（实际选图完成后再定稿）：

> 最终实验布局及其地图表示。各列依次为经过相机标定畸变校正的 iPhone 建图图像、使用对应建图相机内参与位姿的 3DGS 渲染、两者的等权叠加，以及实例颜色地图。下方分别放大三个外观相同的网球和两个外观相同的白杯，各列使用相同裁剪。不同颜色和数字表示当前地图中的不同持久实例。

## 旧图状态

`paper/fig_scene/scene_map_f630.*` 是此前基于 Pupil 实录的三联图。Linux 本地 `output/scene_map_comparison_v1/` 是后续四栏排版草稿，同样基于 Pupil 实录。它们只能作为布局参考或另行说明的在线观测示例，**不作为这次重建质量对比的最终图或证据**。本次未把这些实录草稿当成 iPhone 原图结果重新发布。
