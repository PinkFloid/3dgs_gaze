# 实验室场景流水线：照片 → 米制 3DGS 地图 → 视线求物体世界坐标

单位与坐标系是这个项目最容易晕的地方，先记住一句话：
**只有跑过 `align_to_charuco` 并且训练时加了三个 no-scale flag，模型里的坐标才是"板坐标系 + 米"；其它任何环节的坐标都是任意的。**

## 坐标系速查

| 阶段产物 | 坐标系 | 单位 |
|---|---|---|
| COLMAP `sparse/0`、`transforms.json` | 任意（COLMAP 自己定的） | 任意 |
| `transforms_aligned.json`（align_to_charuco 之后） | ChArUco 板坐标系 | 米（`--square-size` 用米时） |
| splatfacto 模型（默认训练） | nerfstudio 内部系（再次旋转+缩放） | 任意 |
| splatfacto 模型（用 aligned + 三个 flag 训练） | **板坐标系** | **米** |

世界坐标语义：原点 = 板角，XY = 地面（板贴地），**z = 离地高度（米）**。
注视落点的 z 应等于目标表面的离地高度（盯地上的狗 → z ≈ 狗身高 0.3-0.5，不是 0）。

---

# 第一部分：离线建图（Windows / 4090，重拍场景时走一遍）

```powershell
# 0. 前提（已完成，换手机/换相机模式才需要重做）
#    手机内参: calibration_results\phone_camera_calibration.npz
#      2026-07-04 重标定, CALIB_FIX_K3, RMS 0.69px, 48MP 模式 5712x4284,
#      竖幅内参已在 npz 的 upright_90cw 键里算好
#    眼动仪世界相机: calibration_results\world_camera_calibration.npz (鱼眼)
#    ChArUco 板: 11x8 格(=8*11 的行列叫法), 格 32mm, 码 24mm, DICT_6X6_250,
#      ID 从 30 起, legacy 图案 —— ⚠ 格子尺寸待尺子实测确认
#    拍摄注意: 板 + 所有 ArUco tag 必须同场入镜（每个 tag ≥3 张清晰照片）

# 1. 照片转正 + 剔除分辨率不符的照片（EXIF 旋转烘进像素）
python E:\Grasp\tools\make_upright.py
#    data\lab  →  data\lab_upright

# 2. COLMAP，锁定标定的焦距/主点，畸变由 COLMAP 精化
powershell -File E:\Grasp\tools\run_colmap_fixed_intrinsics.ps1
#    →  data\lab_colmap_v2\colmap\sparse\0
#    结束会打印 model_analyzer：确认 registered images ≈ 313、误差 < 1px

# 3. nerfstudio 转换（不再跑 COLMAP，但仍需 --colmap-cmd 通过它的存在性检查）
ns-process-data images `
  --data E:\Grasp\data\lab_upright `
  --output-dir E:\Grasp\data\lab_colmap_v2 `
  --skip-colmap `
  --colmap-cmd C:\tmp\colmap_nerfstudio_compat.bat
#    →  data\lab_colmap_v2\transforms.json   （坐标仍是任意的！）
#    ✔ 输出不能有 "More than one camera" 警告；transforms.json 顶层必须有 fl_x
#      （多相机时先跑 tools\merge_colmap_cameras.py + bundle_adjuster，
#       并加 --colmap-model-path colmap/sparse/0_merged，见"已知坑"）

# 4. 对齐到板坐标系（米）。格 32mm、码 24mm，其余为脚本默认
python E:\Grasp\tools\align_to_charuco.py `
  --dataset E:\Grasp\data\lab_colmap_v2 `
  --square-size 0.032 --marker-size 0.024
#    →  data\lab_colmap_v2\transforms_aligned.json
#    检查："1 colmap-unit = X m" 合理、相机高度(camera Z)与实际拍摄高度一致

# 5. ArUco tag 测绘（在线定位锚点）。先按默认 0.10m 跑，
#    看输出量得的边长是否等于实际打印尺寸，不符则用 --tag-size 实际值重跑
python E:\Grasp\tools\survey_aruco_tags.py `
  --dataset E:\Grasp\data\lab_colmap_v2
#    →  data\lab_colmap_v2\tags_world.json （在线眼动仪定位用）

# 6. 训练。三个 flag 缺一不可，否则 nerfstudio 会把对齐好的坐标再打乱
ns-train splatfacto `
  --data E:\Grasp\data\lab_colmap_v2\transforms_aligned.json `
  --output-dir E:\Grasp\outputs `
  nerfstudio-data --orientation-method none --center-method none --auto-scale-poses False

# 7. 验证米制板坐标系：
#    - viewer 里板子应在原点、地面在 XY 平面
#    - 渲染的 raw-depth 数值即为米，和卷尺量一两个距离对一下
#    - （dataparser_transforms.json 的 transform 含 applied_transform 复合，
#       非单位阵是正常的；scale 必须是 1.0）

# 8. 物体实例分割（SAM mask 提升到高斯 + 跨视角共识聚类；2026-07-07 起替代
#    segment_splat 的物体层）。vit_h 权重在 E:\Grasp\tools\，~35s/视角
powershell -File tools\run_lift_sam.ps1 --every 3 --points-per-batch 32 `
  --crop-points-downscale 2 --previews 12
#    →  lab_result\segmentation_sam\{points.npz, instances.json, names.json,
#       preview\, thumbs\, render_check.jpg}
#    ✔ render_check.jpg 三联图必须严丝合缝（重影物体=建图后被挪动，正常）
#    命名：翻 preview\（id 标在照片上）填 names.json；给多个 id 填同一个名字
#    即为手动合并（SAM 按颜色拆开的机器人部件、桌子分段都这样并）
```

**拷回 Linux 机的四样**：训练输出目录（含 ckpt）、`tags_world.json`、
`transforms_aligned.json`、`lab_result\segmentation_sam\`。

---

# 第二部分：Linux 机准备（TITAN X，换地图后走一遍）

数据流总览（全部已实现并实测）：

```
眼动仪帧(1920x1080 鱼眼) ─ ArUco 检测(在原始畸变图上)
  ├─ cv2.fisheye.undistortPoints(只去畸变角点, 不做全图 remap)
  ├─ 所有可见 tag 的测绘 3D 角点联合 PnP → T_world_cam
  ├─ gaze(norm_pos) → 鱼眼去畸变 → 视线射线 → 世界系
  ├─ 沿射线渲 3DGS 深度小块求交 → 3D 注视点（世界系聚类成注视事件）
  └─ 注视点邻域高斯投票 → 物体名
```

### 环境（一次性，已配好）

- conda env `nerfstudio`：torch 2.1.2+cu118、gsplat 1.4.0、opencv-headless 4.10、pyzmq、msgpack
- gsplat 内核已为 sm_52 本地 JIT 编译（缓存在 ~/.cache/torch_extensions）。
  跑任何用到 gsplat 的脚本都要带编译环境变量（`process_recording.sh` 已内置）：
  `PATH=$ENV/bin:$PATH CUDA_HOME=$ENV CC=$ENV/bin/x86_64-conda-linux-gnu-gcc
  CXX=$ENV/bin/x86_64-conda-linux-gnu-g++ TORCH_CUDA_ARCH_LIST=5.2`
- ⚠ gsplat 的 5 个 `*_bwd.cu` 打过 labeled_partition 架构补丁（sm<70 上反向梯度无效，仅推理）。
  **重装 gsplat 会丢补丁**，需重打（8 处，模式统一）
- Pupil Capture 3.5.8：开 Frame Publisher 插件；世界相机设 **1920x1080**（=标定分辨率）

### 换新地图后的三步

```bash
# 1. 从 ckpt 抽 splat.ply（ns-export 在 Linux 加载 Windows config 会崩，别用）
python tools/export_splat_from_ckpt.py \
  --ckpt lab_result/splatfacto/<run>/nerfstudio_models/step-000029999.ckpt
#   打印的 extent 应是米制实验室尺寸（~7x9x3m）

# 2. 物体分割 + 命名。首选：直接用 Windows 侧第 8 步拷来的 segmentation_sam\
#    （SAM 跨视角共识，能分开贴着的物体：桌上水杯、紧挨的家具）。
#    gaze_object/gaze_video 加 --seg-dir lab_result/segmentation_sam
#    兜底（无 SAM 产物时）：python tools/segment_splat.py  → lab_result/segmentation/
#    （几何连通域，贴着的物体会粘成一个实例）
#   ⚠ 只有命名过的实例会出现在视频包围盒里；未命名的照样参与投票（显示为 object#N）
#   ⚠ names.json 同名 = 合并：gaze_object 按名字并票（SAM 拆开的部件靠这个归整）

# 3. 单帧交叉校验（可选但建议，验证地图/标定/定位三方一致）
python tools/verify_pose_render.py --recording <任一段录像>
#   blend 图应像一张清晰照片；rec000 基准：0.93px / 0.095° / tag 处 4.6mm
#   blend 里重影的物体 = 建图后被挪动过（免费的变化检测）
```

---

# 第三部分：日常使用

### 录制协议（每段录像四步，一步不能少）

1. 戴上眼动仪 → **立刻在 Capture 里做 gaze 标定**（屏幕 marker）。
   ⚠ 摘下再戴必须重标：rec001 教训——未重标导致 gaze 整体漂 7-18°，整段注视报废
   （几何链无恙，但救不回来）
2. 按 R 开始录制 → **先盯墙上 tag 纸 2-3 秒**（片头精度戳，tag 位置毫米级已知，
   离线自动算出本段 gaze 真实精度，不达标当场重录）
3. 正常实验内容：注视目标各 2-3 秒，视线偶尔扫过 tag 保持定位覆盖
4. 结束前**再盯一次 tag 纸**（片尾戳，检测录制中的缓慢滑移）→ 按 R 停

### 处理：一条命令

```bash
tools/process_recording.sh ~/recordings/<日期>/<编号> [--skip-video]
```

产物全部落在录像目录内：

| 文件 | 内容 |
|---|---|
| `poses.jsonl` | 每个定位帧的 T_world_cam（含 n_tags、重投影残差） |
| `world_fixations.json` | 世界系注视事件（连续 gaze 聚类） |
| `world_fixations_objects.json` | 每个注视的物体判定 + top-3 票型 |
| `wfix/` | 每个注视一张标注帧（gaze 圈 + 坐标） |
| `gaze_objects_overlay.mp4` | 注视十字 + 已命名物体 3D 包围盒 + 判定横幅 |

### 实时模式（不录制，直接出位姿流）

```bash
python tools/pupil_localizer.py --tags world_size/tags_world.json --print [--ema 0.7] [--publish 5580]
```

---

# 第四部分：工具箱（单独使用与原理要点）

| 工具 | 作用 | 要点 |
|---|---|---|
| `pupil_localizer.py` | tag→位姿（实时/离线） | PnP 用 ITERATIVE（tag 共面，SQPNP 崩）；三道门限：出界(tag范围+3m, z∉[0.15,2.8])、mean_reproj_norm>0.006、0.25s 内跳变>1m（连拒 5 次重置） |
| `gaze_to_world.py --continuous` | gaze→世界 3D 点→注视聚类 | 30Hz 采样、15cm 半径、≥0.25s；位姿空窗 ≤1s 插值（平移 lerp+旋转 slerp）；深度=沿射线渲 33×33 小块中心中位数，α<0.5 判打空；**世界系聚类能抓到"边走边盯"**（VOR 下图像动、世界落点静止，Pupil 自带检测器抓不到） |
| `gaze_object.py` | 注视点→物体 | 0.2m 邻域 1/d 加权投票，**同名实例并票（命名即合并）**，输出 top-3 票型（= 将来 Bayes 后验的占位） |
| `lift_sam_instances.py` | 高斯→实例 v2（建图侧跑） | SAM 自动 mask 经渲染深度反投影成"高斯 ID 集合"→ 集合 IoU 建图聚类 → 部件-整体包含合并 → per-gaussian 投票；无 query 无 CLIP；Windows 上用 `run_lift_sam.ps1` 包环境 |
| `gaze_video.py` | 还原注视视频 | `--objects` 加判定横幅；`--poses` 加已命名实例 3D 包围盒（鱼眼投影） |
| `segment_splat.py` | 高斯→实例 | 5cm 体素连通域；地板/天花板/墙用高度和房间边界规则 |
| `verify_pose_render.py` | 单帧交叉校验 | 去畸变真实帧 vs 同位姿 3DGS 渲染 + blend；虚拟针孔 K 手动构造（fisheye 焦距×0.7、主点居中；estimateNewCameraMatrixForUndistortRectify 返回退化 K 不能用） |
| `export_splat_from_ckpt.py` | ckpt→splat.ply | 绕开 WindowsPath/数据集依赖 |

---

# 第五部分：当前精度与验收状态（2026-07-05）

| 环节 | 指标 | 出处 |
|---|---|---|
| 几何链（标定+测绘+定位+地图） | **~0.1°**（0.93px，tag 处 4.6mm） | verify rec000 f485 |
| tag 测绘 | 拟合 RMS 0.88mm；边长 98.9mm（打印 99% 缩放） | survey |
| gaze 层（刚标定时） | 1-2°（同目标两次注视差 15cm@4.7m） | rec000/rec002 |
| 定位覆盖率 | 54-77%（取决于 tag 入画时长）→ **当前瓶颈** | rec000-002 |
| 端到端物体识别 | rec002 验收：三只机器狗全部命中（站立狗 10 次 100% 票，落点 z=0.28-0.45=狗身高；趴姿狗 51-82%，与 floor 分票因狗仅 13cm 高） | rec002 |

误差预算结论：**gaze 标定质量决定一切**（几何链好一个数量级）；覆盖率决定能用的样本量。

评测录像：`~/recordings/2026_07_05/000`（几何金标准）、`001`（gaze 漂移反面教材，勿用其注视）、
`002`（v1 验收基线，重标定后）。

注意：3DGS 深度是建图时刻的场景，**物体移动后深度失效**；实验时目标物体保持建图位置，
挪动后用 verify 的 blend 图检查、必要时补拍重建。

---

# 第六部分：升级路线（按性价比排序）

1. **墙面大 tag + 整体重建**（计划中）：20-30cm 单 tag、视线高度、间隔 2m+ ——
   直接抬升定位覆盖率（当前瓶颈）和单 tag 位姿质量
2. **Bayes 物体后验**：沿视线锥（gaze 1.5° 角误差）对各实例高斯密度积分 → p(物体|观测)。
   解决两类已知案例：薄目标与地板分票（趴姿狗 51-82%→期望 90%+）、"看杯子打到桌子"边缘脱靶
3. ~~分割细化：桌链巨实例切分~~ **已完成**（lift_sam_instances.py，2026-07-07 全量验收：
   295 实例/104 视角/77min，桌面小物可分）。遗留小项：房间边界改用 x/y 直方图峰找墙
   （当前门洞外高斯撑大边界 → 墙标签偏松）；SAM 按颜色拆的机器人部件靠同名合并
4. **定位空窗填补**：IMU 不可用（Core 没有），可选 hloc 视觉重定位兜底或更长插值窗

---

## 目录约定

```
E:\Grasp\                        （Windows / 4090：建图+训练）
├── PIPELINE.md              本文档
├── calibration_results\     标定（唯一权威副本）
├── data\
│   ├── lab\                 原始照片，只读，永不修改
│   ├── lab_upright\         第1步产物（可随时重新生成）
│   └── lab_colmap_v2\       第2-4步产物 = 训练数据集
├── outputs\<数据集名>\       训练输出（ns-train 自动命名）
├── tools\                   所有脚本 + COLMAP
└── archive\                 废弃实验（教程数据、旧 run）

~/Project/SceneRebuild/          （Linux / TITAN X：在线运行+评测）
├── PIPELINE.md              本文档（与 Windows 侧手动同步）
├── Calibration_result/      手机 + 眼动仪世界相机标定
├── world_size/              tags_world.json、transforms_aligned.json（从 Windows 拷来）
├── lab_result/              训练产物（ckpt→splat.ply）、segmentation/、验证图
└── tools/                   两侧共享脚本（Linux 专用：pupil_localizer / gaze_* / segment_* / process_recording.sh）
~/recordings/<日期>/<编号>/      Pupil 录像 + process_recording.sh 的全部产物
```

约定：一个数据集一个 `data\` 子目录，输出目录与数据集同名；废弃的挪进 `archive\` 而不是删。

## 已知坑

**建图侧**
- feature_extractor 在同一个 database.db 上每重跑一次就会新建一个相机（即使
  single_camera 1）。model_analyzer 显示 Cameras > 1 时：跑
  `tools\merge_colmap_cameras.py` 合并，再 bundle_adjuster 重收敛畸变（锁焦距/主点），
  然后 ns-process-data 加 `--colmap-model-path colmap/sparse/0_merged`。
  align_to_charuco / survey_aruco_tags 只认单相机的 transforms.json
- `data\lab` 里 IMG_1100–1104 是 12MP 模式误拍，与标定不符，第 1 步自动剔除
- nerfstudio 1.1.5 不支持 FULL_OPENCV 相机模型的转换，COLMAP 只能用 4 系数 OPENCV
- 板对齐的尺度完全取决于 --square-size：07-03 曾按 30mm 对齐导致尺度偏 6.7%，
  实际是 32mm（待尺子最终确认）；`lab_colmap_up\transforms_aligned.json` 若还在
  即为错误尺度的旧文件，勿用
- Windows 训练出的 config.yml 序列化了 WindowsPath，Linux 上任何 ns-* 命令加载
  都会崩 → 用 tools/export_splat_from_ckpt.py 直接从 ckpt 抽 splat.ply
- Windows 上 gsplat 是 JIT 编译的：裸跑报 "No CUDA toolkit found"。需要
  TORCH_EXTENSIONS_DIR=E:\Grasp\torch_extensions + CUDA_HOME=conda env +
  MSVC 14.38 上 PATH（版本必须和缓存 build.ninja 一致）→ 统一走 run_lift_sam.ps1
- SAM 自动模式的 GPU 后处理按批 × 全分辨率：vit_h + 2048 长边时 points_per_batch
  64 会 OOM（24G 卡），32 峰值 ~14G 安全

**标定侧**
- 07-03 的旧标定 k3=-3.61 过拟合，已被 07-04 的 FIX_K3 重标定取代；标相机永远用
  `calibrate_charuco.py --fix-k3`，板要拍到画面边角
- 标定手机别拍近景：近距 = 景深塌 + 对焦呼吸（焦距漂 ~1%），两批近拍标定图全被
  自动剔除。正确拍法：1-1.5m、点按锁对焦、板出现在画面四角
- cv2.imread 默认应用 EXIF 旋转且尺寸检查会踢掉竖拍图 → calibrate_charuco.py
  已改为 IMREAD_IGNORE_ORIENTATION（内参属于传感器原始坐标系）
- survey 测出 tag 边长一致地 ≈98.9mm（名义 100mm）→ 打印约 99% 缩放；定位不受影响
  （PnP 用的是测绘 3D 角点），但拿 tag 当尺子校验时按 99mm 算
- **gaze 标定有效期 = 一次佩戴**：摘戴不重标 → 漂 7-18°（rec001 实测）。协议第 1 步不可省

**运行侧**
- SQPNP 遇共面 3D 点集断言崩溃（tag 多在地板一个平面）→ PnP 必须用 ITERATIVE
- tag 印成 6 个/张的组合纸：单张基线只有 A4 大小，远距离位姿噪声大。软件靠三道
  门限压制；物理解法是升级路线第 1 条
- cv2.fisheye.estimateNewCameraMatrixForUndistortRectify 返回退化 K（焦距≈0.002）
  → 虚拟针孔 K 一律手动构造
- Pupil 自带 fixation 检测是图像空间的，走动中盯物体不触发 → 一律用
  gaze_to_world 的 --continuous（世界系聚类）
