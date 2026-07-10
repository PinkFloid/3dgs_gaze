# CODE_TOUR:视线→物体管线代码导读

> 写给管线的所有者。目标:能在组会上讲清每个设计决策,在审稿人面前答辩每个阈值,在机器人现场不靠任何人独立排障。
> 配套阅读:`SceneRebuild/PIPELINE.md`(操作步骤)、记忆笔记 `win-4090-mapping-env.md`(建图机环境配方)。
>
> **路径注(2026-07-09 monorepo 重组)**:本文写于重组前——文中 `tools/` 一律读作
> `SceneRebuild/tools/`;第 4 章视线侧工具现居本仓库 `Eye_Tracker/tools/`(行号以
> `7b707a8` 迁出版为基准,与现版基本一致;差异:pupil_localizer 的 2 元组崩溃 bug
> 已在现版修复,gaze_video 增到 283 行)。新的实时模式 `gaze_live.py`(672 行,
> ZMQ 事件驱动 + 滚动精度戳 + intent 发布)晚于本文诞生,**尚无对应章节**。

---

## 1. 十分钟全景

整个系统只有**两个几何原语**和**一种通用货币**,其余全是围绕它们的工程:

**原语 A|投影**:世界点 → 像素。`(u,v) = K · [R|t] · X_world`(OpenCV 约定)。
用在:gaze_precision 把 tag 中心投进相机比对视线、gaze_video 画 3D 框、verify_pose_render 画绿十字、lift_sam_instances 的 preview 可见性测试。

**原语 B|反投影**:像素 + 深度 → 世界点。`X = origin + t·dir`,其中深度 **永远由 3DGS 当预言机渲出来**(gsplat `render_mode='RGB+ED'`)。系统里没有深度相机、没有稠密 MVS——训练好的 splat 就是场景的可查询深度场。
用在:lift_sam_instances 把 SAM mask 提升到 3D、gaze_to_world 的 `depth_along_ray` 求注视点、gaze_object 锥后验把 1089 个 patch 像素反投到表面。

**通用货币|高斯 ID**:一个 mask 提升后 = 一个高斯 ID 集合;一个实例 = 一个高斯 ID 集合;`points.npz` 的 label 按高斯逐点存;跨帧关联用高斯集 IoU;视线投票把 3D 点 snap 到最近高斯再查 label;`names.json` 把多个 ID 池化成一个名字(**命名即合并,名字是贯穿全栈的主键**)。2D 图像证据(SAM mask、CLIP 之类)从不直接互相关联——这是前身 `build_object_map.py` 的死因(见 §5.1)。

### 数据流总图

```
━━━ 建图侧(一次性,Windows/4090 建图机)━━━━━━━━━━━━━━━━━━━━━━━━━━━━
iPhone 48MP 照片 (E:\Grasp\data\lab)
  │ make_upright.py            EXIF 转正烙进像素 + 分辨率闸(5712x4284)
  ▼
lab_upright/
  │ run_colmap_fixed_intrinsics.ps1   COLMAP SfM,标定内参注入并锁定
  ▼
sparse/0(COLMAP 任意 gauge)──ns-process-data──▶ transforms.json
  │ align_to_charuco.py        板角点 DLT 三角化 + Umeyama 相似变换
  ▼
transforms_aligned.json(板坐标系:米制、z 向上、原点在板上)
  ├─▶ ns-train splatfacto(orientation/center/auto-scale 三关)─▶ step-*.ckpt
  ├─▶ survey_aruco_tags.py ─▶ tags_world.json(每 tag 4 角点世界坐标,mm 级)
  ▼
ckpt + 照片
  │ run_lift_sam.ps1 ─▶ lift_sam_instances.py
  │   SAM mask ─(渲染深度反投+snap)─▶ 高斯 ID 集 ─(跨帧 IoU 共识)─▶ 实例
  ▼
lab_result/segmentation_sam/{points.npz, instances.json, names.json(手工命名)}
  │ export_seg_splat.py        审计 ply:按 label 重上色,SuperSplat 里肉眼验票
  ▼
物体地图(在线侧唯一消费的语义产物)
━━━ 在线侧(每段录像,process_recording.sh 五步)━━━━━━━━━━━━━━━━━━
world.mp4 帧流                          gaze.pldata(~120Hz norm_pos)
  │ pupil_localizer.py                    │
  │ ArUco→鱼眼去角点→PnP(ITERATIVE)→三重闸 │
  ▼                                       │
poses.jsonl(T_world_cam 轨迹)            │
  ├──▶ gaze_precision.py ◀────────────────┤  片头/片尾盯 tag 协议
  │     bias/σ/drift ─▶ gaze_precision.json(头尾两枚"精度戳")
  └──▶ gaze_to_world.py ◀─────────────────┘
        y 翻转 → fisheye undistort → 减 bias(t) → 世界射线
        SplatDepth 沿射线渲深度(原语 B)─▶ world_fixations.json(3D 注视点)
           │ gaze_object.py --cone   锥后验 × points.npz 标签(货币兑换)
           ▼
        world_fixations_objects.json(名字 + vote_share + p_none)
           ├─ grasp_intent.py   dwell≥0.8s → 意图事件(质心坐标喂机械臂)
           └─ gaze_video.py     overlay mp4(视线圈 + 3D 实例框)
```

验收基准(v2 地图 + rec002,用来对照"现在正不正常"):

| 层 | 数字 |
|---|---|
| 几何链端到端 | ~0.1°(verify blend 残差 0.93px,一张清晰照片) |
| gaze 层 | 1–2°(刚标定状态) |
| 板拟合 | RMS 0.48mm(v2) |
| v2 地图 | 396/396 图像注册、重投影 1.48px;render_check 相位相关 0.2px;506 实例;36 个 tag 全部 fit_rms ≤ 1.6mm |
| rec002(锥 + v2.1 分割 + bias 插值) | 三只狗零错误翻转,趴姿狗中位票面 76% [61–90] |

---

## 2. 坐标系与约定总表

每一条都是实弹踩过的雷。改任何位姿/像素路径前先回来对表。

| # | 约定 | 内容 | 出处 |
|---|------|------|------|
| 1 | 板坐标系(board frame) | 全局世界系:原点在 ChArUco 板、X/Y 在板平面、Z 朝相机(=离地向上),单位是米**当且仅当** align 时传了 `--square-size 0.030`(米)。floor_z=0.08 / ceiling_z=2.5 这类绝对高度阈值全靠它成立 | `align_to_charuco.py:203-240`;阈值用法 `segment_splat.py:75-84`、`lift_sam_instances.py:221-230` |
| 2 | nerfstudio GL c2w → OpenCV | `c2w[0:3,1:3] *= -1`:把旋转块的第 1、2 **列**取反(翻的是相机自身 y/z 基向量,GL 的 -Z 前/+Y 上 → CV 的 +Z 前/+Y 下)。翻**行**是经典错误——那翻的是世界轴,渲出来上下颠倒/镜像 | `align_to_charuco.py:76-79`(读入)、`232/235`(烘焙);`lift_sam_instances.py:275-277` |
| 3 | 视线链全程 OpenCV,无翻转 | solvePnP 产出的 T_world_cam 本来就是 CV 相机轴;gsplat viewmat 期望 CV w2c——`inv(T_world_cam)` 直接喂,**链上没有也不许有轴翻转**。谁把 nerfstudio 的 GL c2w 塞进来,场景渲在相机背后 | `pupil_localizer.py:195`;`gaze_to_world.py:188-190`;`verify_pose_render.py:133-137` |
| 4 | Pupil norm_pos 左下原点 | y 轴向上、原点在左下:像素坐标 `u = x·W`,`v = (1−y)·H`。u 不翻。删掉或翻两次 = 视线垂直镜像 | `gaze_to_world.py:280,393`;`gaze_video.py:193,200` |
| 5 | z-depth vs 射线长 | gsplat `'RGB+ED'` 第 4 通道是 **z-depth**(沿相机 z),不是射线长。射线长 `t = d·‖dir_cam‖ = d·‖[nx,ny,1]‖`(tmul)。`depth_along_ray` 只取 3×3 中心块所以两者相等;patch 边缘必须乘 tmul,否则离轴距离系统性偏短 | `gaze_to_world.py:144-161`(中心等价)、`163-186`(tmul);消费端 `gaze_object.py:72`;`lift_sam_instances.py:306-307`(针孔反投同框) |
| 6 | undistort 用同一个 K | `cv2.undistort(arr, K, dist, None, K)`——newCameraMatrix 就用原 K,照片像素和渲染深度共享同一个针孔框架,mask 像素才能直接过深度反投 | `lift_sam_instances.py:272` |
| 7 | 鱼眼只去角点,不 remap | 定位链在**原始畸变图**上检测 ArUco,只把角点过 `cv2.fisheye.undistortPoints` 进归一化坐标,PnP 用 K=I;所有重投影阈值都是归一化单位(0.01≈8px、0.006≈5px @f≈800)。全图 remap 只在 verify_pose_render 为可视化做一次(手搓 K_new,因为 `estimateNewCameraMatrixForUndistortRectify` 返回垃圾) | `pupil_localizer.py:282-283,167-195`;`verify_pose_render.py:126-131` |
| 8 | bias 符号与单位 | bias = **gaze 减 target**,存量纲是 `tan(radians(deg))`(去畸变归一化平面上),下游**减去**:`pn − bias_at(t)`。翻任何一边的符号 = 误差翻倍而不是归零 | `gaze_precision.py:72,90,160`;`gaze_to_world.py:193-223,282,396` |
| 9 | tag 自身坐标系 | 原点在 tag **中心**,X 右、Y **上**(随印刷方向,不是房间的上)、Z 出面;角点序 TL,TR,BR,BL 与 detectMarkers 一致。在线 PnP 的 objectPoints 模板若与此不符 = 全局恒定位姿错(绕 X 180°) | `survey_aruco_tags.py:122-126` |
| 10 | 高斯参数原始存储 | ckpt 与 3DGS ply 里 opacity 是 logit、scale 是 log、quat 是 wxyz——**原样即格式**,viewer 自己做 sigmoid/exp。加载渲染时才激活 | `export_splat_from_ckpt.py:36-41`;激活处 `gaze_to_world.py:121-142`、`lift_sam_instances.py:211-212` |
| 11 | 世界相机 K 按 1920×1080 缩放 | 标定固定在 1920×1080,运行时按行缩放 `K[0]*=W/1920; K[1]*=H/1080`。纯缩放成立,crop(变宽高比)不成立 | `gaze_to_world.py:366-368`;`gaze_video.py:162-164`;`gaze_precision.py:108-110` |
| 12 | OBJ0=10 背景/实例分界 | label 0=地板、1=天花板、2-5=墙、≥10=实例。此约定散落两处硬编码 | `grasp_intent.py:25`;`gaze_video.py:219` |

---

## 3. 建图侧工具(`E:\3dgs_gaze\tools\`)

### 3.1 make_upright.py — 把 EXIF 旋转烙进像素并过滤非 48MP 帧

**核心机制**(无函数,模块级循环 19-33 行):
- PIL 的 `Image.size` 报告**存储**(旋转前)尺寸,所以闸门(23 行)比对的是横置 `(5712, 4284)`——别"修"成竖置值,那会跳过所有图。
- `ImageOps.exif_transpose`(26 行)按 EXIF tag 物理旋转像素并删除 tag;COLMAP 与 nerfstudio 都不理(或错理)EXIF,只有烙进像素它们才看到人看到的竖图。
- 非 48MP 帧(如 12MP 模式的 IMG_1100-1104)收集、末尾报告、绝不写出。

**为什么这样设计**:下游 `single_camera 1` 的共享单相机模型要求全部照片同一拍摄模式,这个闸是它不被一张杂图毒死的唯一防线。quality=95 重存且丢弃全部元数据是故意的——内参由 3.2 注入,不靠 EXIF(代价:lab_upright 的图从此不能做 EXIF 自动内参)。

**参数速查**:argv[1] SRC(默认 `E:\Grasp\data\lab`)| argv[2] DST(默认 `lab_upright`,makedirs 自动建)| `EXPECTED_RAW_SIZE=(5712,4284)` 硬编码在 15 行——换手机或拍摄模式,先重标定内参再改它。

**坏了长什么样**:
- `converted: 0` → SRC 路径打错,或整场用 12MP 拍的。
- 下游 COLMAP 注册率暴跌 → 某些图没 EXIF tag(exif_transpose 无操作,横置图混进共享相机)。
- SRC/DST 文件数不等 → 过滤在干活;**下游按文件名配对,别按索引**。
- 画质缓慢劣化 → 有人把 DST 当 SRC 反复跑,多次有损重编码。

### 3.2 run_colmap_fixed_intrinsics.ps1 — 用标定内参锁死的 COLMAP SfM

**核心机制**:
- feature_extractor(24-30 行):`--ImageReader.single_camera 1` + `--ImageReader.camera_model OPENCV`,8 参在 29 行**硬编码**:fx=4024.8166 fy=4000.6211 cx=2159.5867 cy=2814.4041 k1=0.16665225 k2=-0.14699350 p1=-0.00598953 p2=0.00152398——是 `phone_camera_calibration.npz['upright_90cw']` 的手抄,**不读文件**,重标定后要手改这一行。
- mapper(35-42 行):`ba_refine_focal_length 0` + `ba_refine_principal_point 0` 冻结 f/c;`ba_refine_extra_params 1` 允许 BA 从标定种子起打磨畸变。
- 前三个原生调用(extractor/matcher/mapper)后手动 `if ($LASTEXITCODE -ne 0) { throw }`(PowerShell 对原生 exe 失败不自动抛;末尾 model_analyzer 无此检查);model_analyzer 只看 `sparse\0`。

**为什么这样设计**:平面偏多的场景自标定有焦距/深度 gauge 歧义,锁定标定值消除它。畸变可打磨意味着**最终权威畸变在 cameras.bin,不在 npz**——下游 undistort 读 npz 就是在用陈旧系数。cy>cx 是"这套是竖置内参"的指纹,数值看着反了就是有人喂了横置图。

**参数速查**:`-Images`(默认 lab_upright,**必须是烘焙后的集合**)| `-Out`(默认 lab_colmap)| `ba_refine_extra_params` 1↔0(0 = 全场 bit 级一致的标定畸变,重投影略高)。

**坏了长什么样**:
- 注册图极少 → 喂了原始横置图(内参轴向不符),或 12MP 帧漏网。
- 重投影误差大 → 8 参顺序抄错(k/p 调换是经典)。
- **旧 database.db 上重跑会追加出多余相机——重跑前整目录删除**(实测教训)。
- 下游部分相机凭空消失 → mapper 分裂出 sparse\1,脚本只看 sparse\0,静默丢。

### 3.3 align_to_charuco.py — 把 COLMAP 世界变换到板坐标系(米制)

**核心机制**:
- `build_board`(54-63 行):DICT_6X6_250、marker id 从 30 起、`setLegacyPattern` 默认开(板是 OpenCV 4.6 前的位型印的;legacy 错配的症状是 marker 检出但 ChArUco 角点插值为 0——看着像检测失败,实为位型不符)。
- `c2w_gl_to_w2c_cv`(76-79 行):§2#2 的列翻转,同一操作在 232/235 行(烘焙进出)再出现两次,**三处同改或都不改**。
- `triangulate_dlt`(106-114 行):角点先 `undistortPoints`(不传 P → 归一化坐标),投影矩阵就是裸 `[R|t]`;逐视图叠 `x·P[2]−P[0]` / `y·P[2]−P[1]`,SVD 最后一个右奇异向量为齐次解。代数误差最小,不是重投影误差。
- 三角化循环(169-195 行):每角点 ≥`--min-views`,一轮重投影 >`--max-reproj-px` 的视图剔除后重三角化。注意打印的像素中位数可能仍含最后一轮剔除的 outlier(188 行;更早轮次剔掉的不在内)——**判质量看板单位残差 RMS(219 行)**。
- `umeyama`(123-135 行):闭式相似拟合 `dst ≈ s·R·src + t`;`S=diag(1,1,det修正)` 强制 det(R)=+1——板角点共面(z=0),反射解拟合得一样好,必须禁。
- Z 翻转 + 烘焙(203-240 行):相机中心多数票定向板 Z 朝相机(翻转通过镜像参考角点再重拟合实现,Umeyama 禁反射所以 z 随 y 一起翻,是正规 180° 旋转;副作用:`--origin corner` 时原点跑到 Y 方向对角);烘焙时左乘 S4 后 `c2w[:3,:3] /= s` 恢复正交(仅因线性部分恰是 s·R 才合法),applied_transform 更新为 S4@A4 保住 COLMAP 帧的账。

**为什么这样设计**:全管线的米制世界系在此诞生。marker id 从 30 起是刻意的——**0-29 留给房间墙面 tag**(见 3.4)。训练必须 `--orientation-method none --center-method none --auto-scale-poses False` 三关齐下,否则 nerfstudio 悄悄重归一化,板坐标系丢失、tags_world.json 全部作废。

**参数速查**:`--square-size`(**必传 0.030 米**;默认 1.0 = square-units,只打印提示不失败,毒到 survey 的边长比检查才报警)| `--marker-id-start 30` | `--no-legacy`(默认 legacy 开)| `--origin corner|center`(center 在 Z 翻转下对称,更好推理)| `--min-views 3` | `--max-reproj-px 4.0`(两者联动:剔除会把角点视图数压破 min-views,整颗角点丢弃)。

**坏了长什么样**:
- `Only N corners triangulated`(N<6)→ 字典 / marker-id-start / legacy 三选一不匹配,或板在图里太小太糊。
- 残差 RMS 远超 mm 级 → 板拍摄途中被挪过、square-size 与实印不符,或 per-frame 内参被静默忽略(**本脚本只读全局单套内参**,多相机/变焦数据集悄悄劣化;k3 也从不读,鱼眼模型按错误模型去畸变且无警告)。
- 训练出的 splat 相对 tags_world 旋转/缩放 → 三关没关。
- `transform_ply_ascii` 崩 → sparse_pc.ply 是二进制 PLY(它只会文本解析;另注意法线不随变换旋转,只 xyz 变)。

### 3.4 survey_aruco_tags.py — 把每个墙面 tag 测绘进板世界(在线定位的地图)

**核心机制**:
- 检测循环(146-175 行):裸 ArUco(非 ChArUco)+ `CORNER_REFINE_SUBPIX`(139 行)——角点亚像素噪声直接变世界 mm。观测按 (tag_id, corner_idx) 收集,角点序绑定**印刷位型**而非图像方向,tag 侧装也得到一致的自身系。
- 三角化循环(177-194 行):与 3.3 同款 DLT + 一轮剔除,**完全尺寸无关**——`--tag-size(s)` 不进三角化。
- `kabsch`(109-119 行):刚体拟合(Umeyama 去掉尺度),det 修正防反射(4 个共面点同样招反射解);把已知边长的正方模板贴到 4 个三角化角点得 T_world_tag。
- fit+体检(196-231 行):要求 4/4 角点齐(3/4 报告并跳过);量 4 条边长;全 tag `measured/expected` 均值偏 1 超 5% 报警——**这是揪出 square-units 世界的唯一守门员**。

**为什么这样设计**:Kabsch 的旋转与质心平移对模板均匀尺度误差**不变**——`--tag-sizes` 填错不毁 T_world_tag,只吹大 fit_rms。这正是 99mm/240mm 混布免疫的原理:**在线定位对测绘 corners_world 联合 PnP,tag 物理尺寸根本不进在线链路**。240mm 大 tag 的动机:单 tag 基线 = 姿态噪声 ∝ 1/尺寸;旧 6 合 1 纸只有 A4 基线,远距噪声大,定位覆盖率 54-77% 曾是全系统头号瓶颈(v2 地图待评测验证收益)。

**参数速查**:`--transforms`(默认 transforms_aligned.json;指到原始 transforms.json 一样能跑,产出却在 COLMAP gauge 里,只有边长比会告诉你)| `--tag-ids 0-29`(**v2 墙面大 tag 在 74-249,必须覆写**;范围放宽过 30 会把建图板自己的 marker 当墙 tag 测)| `--tag-size 0.10` / `--tag-sizes '0-29:0.099,74-249:0.24'`(区间按书写序先到先得)| `--min-views 3 / --max-reproj-px 4.0`(联动同 3.3:剔除压破 min-views,一颗角点丢 = 整个 tag 丢)。

**坏了长什么样**:
- 比值 ≈0.03 或 ≈33 → align 用了 square-units,重跑 align(传米)再重测绘。
- 比值只在高 id ≈2.4 → 忘了 `--tag-sizes` 的 '74-249:0.24'。
- 单 tag fit_rms 远超同伴但边长正常 → 纸/板不平,或角点在眩光处误精化。
- 缺 tag **无警告地缺席输出**——拿控制台 `Tags seen:` 对布置清单;`n_views` 是"有检出的帧数"而非剔除后逐角点存活数,对边缘 tag 偏乐观。
- 在线定位全局恒偏一个刚体 → 定位端 objectPoints 模板与 §2#9 约定漂移(此约定只以人读字符串形式存在 JSON 里,无机器校验)。

### 3.5 calibrate_charuco.py — 离线手机内参标定(喂 COLMAP)

**核心机制**:
- `import_cv2`(49-63 行):函数内延迟导入,给出可操作的 'pip install opencv-contrib-python' 报错;但 hasattr(cv2,'aruco') 检查对 OpenCV≥4.7 已过时(主包也带 aruco),真正 contrib-only 的 `calibrateCameraCharuco` 会在**全部检测做完之后**(222 行)才 AttributeError。
- 新旧 API 双轨板构造/检测(85-110 行);逐图过闸(不可读 / 分辨率与**首图**不符 / 角点 <`--min-corners`)后单次 `calibrateCameraCharuco`(158-242 行),硬性下限 5 张(推荐 15-30)。
- `save_output`(119-155 行):按 `--out` 后缀写 JSON 或 OpenCV-YAML(带 `%YAML:1.0` 头 + `!!opencv-matrix`,普通 YAML 解析器直接拒收)。

**为什么这样设计**:`--fix-k3` 是本管线的**必选项**——align/survey/COLMAP OPENCV 模型都只读 4 项畸变,fit 出的非零 k3 会被下游静默截断,k1/k2 便不再自洽(2026-07-04 重标定即 FIX_K3,RMS 0.69px)。注意:**它默认 DICT_4X4_50、id 从 0、无 legacy 开关——标定板和建图板是两块板、两条流**,拿建图板喂它必失败(且没有可传的 flag 能救)。

**参数速查**:`--squares-x/y --square-length --marker-length`(必填;只有比值影响内参,rvecs/tvecs 反正被丢弃)| `--min-corners 8`(低了收斜视角、姿态多样但角点噪;高了全正面、fx/fy/畸变条件差)| `--fix-k3`(接本管线必开)| `--out`(.json 才通用)。

**坏了长什么样**:
- 全部 `[skip] 0 ChArUco corners` → 字典错,或想用建图板(legacy+id30 不可表达)。
- 收尾 `AttributeError: calibrateCameraCharuco` → 非 contrib 的 opencv-python≥4.7。
- 'Only N valid' 但文件夹很大 → 首图分辨率异常锁死 image_size,其余全被"尺寸不符"跳过。
- RMS>1.5 → 板规格比值错、运动糊,或 EXIF 混转(imread 对 EXIF 旋转跨版本不一致,混拍竖横标出的畸变是垃圾但 RMS 可能好看)。
- COLMAP 侧去畸变反而更糟 → 忘开 --fix-k3,消费端截断了 k3。

### 3.6 lift_sam_instances.py — 分割 v2:SAM mask 提升为高斯 ID 实例(核心资产)

**核心机制**(全在 main,§5.1 有算法级展开):
1. **加载与背景打标**(207-231 行):sigmoid(不透明度 logit)、exp(log 尺度);keep = 不透明度≥0.5 ∧ max_scale<0.5 ∧ 房间 1/99 分位界 +0.2m;背景规则(221-230 行)**抄自** segment_splat.py:75-84——是复制不是共享,两边阈值各自 argparse,改一处不动另一处;对保留子集建 cKDTree。
2. **逐帧提升**(263-334 行):GL→CV 列翻转(275-277 行)→ 同 K undistort(272 行)→ gsplat 渲 z-depth(RGB+ED)→ SAM 自动 mask → 针孔反投 `x=(u+0.5−cx)/fx·d`(306-307 行,+0.5 对齐 gsplat 像素中心)→ snap 到 ≤`match_eps`=0.04m 的最近高斯(309-310 行;scipy 未命中返回 idx=n 越界哨兵,**isfinite 过滤必须先于索引**)→ 三道 mask 闸:bg_frac>0.6 弃、剥背景后 ≥30 个物体高斯、robust 直径 `‖p98−p2‖>2.0m` 按视线向多物并集弃(319-322 行)。
3. **稀疏跨帧 IoU 图**(336-350 行):mask×高斯 CSR 关联矩阵 M,`(M@M.T).tocoo()` 一把算全部两两交集,`IoU≥0.3` **且异帧**才连边,连通分量 = 原始实例。
4. **部件→整体包含性合并**(352-395 行):containment≥0.85 ∧ 吸收方 ≥6 视角 ∧ 严格更大 ∧ robust 直径 ≤2.2m;remap 迭代到不动点(目标严格变大保证无环)。
5. **逐高斯多数票 + 拆焊**(404-437 行):每高斯归覆盖它 mask 数最多的分量(≥min_votes=2);每分量 10cm 体素 26 连通 `ndimage.label` 拆焊,≥80 高斯成实例,robust bbox 对角 ≥0.06m,id 从 OBJ0=10 起按高斯数降序。

**为什么这样设计**:四个闸门每个对应 rec002 一次实弹翻车(§5.1)。SAM 按颜色拆机器人(黄腿黑身)是它的世界观,包含性合并救不了的交 `names.json` 同名合并兜底。粒度刻意停在物体级:gaze 角精度 1-2°(≈5-15cm@3m)撑不起 part 级。重型 import(torch/cv2/SAM/gsplat/scipy)全在 main 内(188-196 行)——`--help` 在无 CUDA 环境可用,**别提升到模块级**。

**参数速查**:`--every 3`(SAM 是运行时瓶颈,近线性)| `--match-eps 0.04` | `--max-mask-extent 2.0` | `--edge-iou 0.3` | `--merge-containment 0.85` | `--attractor-min-views 6` / `--max-attractor-extent 2.2`(**恒保持 2.2 > 2.0**,反过来会出现任何合法吸收者都收不了的 mask,大物永久碎裂)| `--split-voxel 0.10` | `--min-votes 2` | `--bg-frac 0.6` | `--points-per-batch 16`(SAM 的 GPU 后处理按批×全分辨率:64 @2048px+vit_h 在 24G 上 OOM,16-32 安全、峰值 ~14G)| `--long-side 2048`(照片/undistort/渲染/SAM 的统一工作分辨率)。另:`min_mask_region_area=100`(261 行硬编码,mask 内去斑)与 `--min-mask-area 300`(291 行,整 mask 过滤)是**两个不同的闸**。

**坏了长什么样**:
- `render_check.jpg` 照片/渲染不重合 → transforms 与 ckpt 不配套,下游全废。**任何位姿路径改动后先看这张图,它是全文件的绊线**。
- `No masks lifted` → 位姿/undistort 断裂(先看 render_check)、alpha 闸(ckpt 在这些位姿渲出透明)、或 match_eps 对噪声重建太紧。
- 异物焊成一个实例 → edge_iou 太低 / 两个尺寸闸被抬 / split_voxel 对实际缝隙太粗。
- 单物碎成零件 → 物体超 max_mask_extent(整体 mask 全灭)叠加吸收闸拦路,或 edge_iou 太高。
- 低平物整个消失 → floor_z=0.08 吞其高斯进 FLOOR,bg_frac=0.6 再补刀杀 mask。
- 杯子级小物消失 → 自动选中 vit_b 权重(看启动行 `SAM: ... (vit_h)`)或 long-side 降过头。
- 换 ckpt 后一切"看起来能跑"但下游全灰/全 none → `points.npz` 存 keep 子集中已标注(label≥0)部分的**逐位相同 float32 坐标**,与 ckpt 坐标相等焊死;重训即静默失配。
- 拆焊出的两个实例 n_views/n_masks 一模一样地虚高 → 沿用拆前分量统计(449 行),已知账目瑕疵。

### 3.7 segment_splat.py — 分割 v1:纯几何体素连通域(已被 v2 取代,仍供 render_thumbs)

**核心机制**:
- 过滤+背景打标(49-84 行):v2 抄的就是这段;顺序承重——地板/天花板先认领,x 墙先于 y 墙,每级只碰 label==-1。
- 体素连通域(86-118 行):`floor((pts−(lo−0.2))/0.05)` 进稠密 bool 网格,26 连通 `ndimage.label`;≥150 高斯 ∧ **原始 min/max** bbox 对角 ≥0.12m 成实例(无分位鲁棒,一颗 floater 撑爆 bbox——v1 seg 目录上 export_seg_splat 的 bead 框肥大即此源)。
- `render_thumbs`(145-189 行):其余高斯不透明度×0.06,轨道相机手搓 CV 约定 look-at(z=normalize(c−eye), x=z×[0,0,1], y=z×x;0.5 仰角项躲开朝下看的退化极点,别清零);label 是 keep 子集索引,靠 `full_label[flatnonzero(keep)]` 回贴——传全长 label 数组进来会静默全错。**这是 v2 import 复用的唯一函数**。

**为什么这样设计**:占据连通 = 接触即焊接(桌上杯并进桌链),这是它让位给 v2 的根本原因——**别调 --voxel 指望修好,换 v2**。留着它因为背景规则参考实现和 thumbs 在这。

**参数速查**:`--voxel 0.05` | `--floor-z 0.08 / --ceiling-z 2.5` | `--wall-margin 0.15` | `--min-opacity 0.5` | `--min-gaussians 150 / --min-size 0.12`(比 v2 的 80/0.06 严,因体素 CC 碎渣多)。

**坏了长什么样**:
- 一个房间级巨实例 → 地板残斑当桥,微抬 floor-z 或 min-opacity。
- 启动 UnicodeDecodeError / 中文名乱码 → 127 行 `read_text()` 没写 encoding,按本机代码页读上次以 utf-8 写的 names.json(v2 的 468 行是对的)。
- 靠墙家具消失 / 边缘全标墙 → 房间分位界歪(场景没板对齐或 z-up 破坏),墙 margin 切进房间。
- thumbs 全黑 → 无 CUDA,或 ckpt 参数键不是 splatfacto 布局。

### 3.8 export_seg_splat.py — 审计出口:按分割标签重上色的 3DGS ply

**核心机制**:
- 标签回贴(81-92 行):points.npz 子集坐标建 cKDTree,上限 1e-4 查全量 means——**坐标相等连接假扮最近邻**,只在同一训练产物内成立;未命中 → label −1。
- 颜色(46-48, 94-114 行):`stable_color(seed)=default_rng(seed).integers(80,255,3)/255` 与 gaze_video **逐字节同式**;同名实例取该名下最小 id 的颜色池化;背景/未标注暗灰(×dim),未命名实例自有色 ×0.8——**抢票者刻意保持可见且可区分**。
- bead 框(51-62, 117-156 行):bbox 12 棱按 `--bead-step` 撒 6mm 各向同性高斯(scale=log(0.006)、opacity logit 6.0≈0.998,**常量已在 raw 空间,别再激活**);`--boxes named` 画同名 union 框,`--boxes all` 逐实例(1.6× 稀疏)。
- SH 重上色 + 写 ply(152-176 行):`f_dc=(rgb−0.5)/C0`,C0=0.28209479177387814(=1/(2√π));45 个 f_rest 全零但**占位承重**——viewer 靠属性数推 SH 度 3,删列即拒收;opacity/scale 原样 raw。

**为什么这样设计**:这是"物体后验到底在对什么投票"的肉眼审计面。颜色与视频框一致,组会/答辩两边指认同一物体;`--boxes all` 专抓未命名重复实例(抢票者)。注意颜色只在**一次分割产物内**稳定——重跑分割 id 重排,最小 id 换人,颜色跟着漂。

**参数速查**:`--boxes named|all|none` | `--dim 0.35` | `--bead-step 0.025` | `--box-min-diag 0.25`(杯子 ~0.15,别调高过它)| `--preview`(需 GPU;顶视 up=[0,1,0] 因视轴即世界 z,常规 up 会退化)/ `--preview-clip-z 2.6`(只影响 preview,不动 ply)。

**坏了长什么样**:
- 整团灰 → ckpt/seg 失配。看 `labels mapped: X/N`:健康时映射绝大多数,失配 ≈0;把 `--ckpt` 钉到 instances.json['ckpt']。
- 颜色与视频框对不上 → 两份 stable_color 公式或最小-id 种子逻辑漂了。
- 一个名字一个房间大的框 → 两个远隔实例同名(union bbox 如实工作),查错标的实例 id。
- viewer 里 bead 变巨球/隐形 → viewer 不做 exp/sigmoid,或有人把 raw 常量"修"成激活值。

### 3.9 export_splat_from_ckpt.py — 绕开 ns-export 直接从 ckpt 出 splat.ply

**核心机制**(main,23-66 行):
- CPU 加载 `['pipeline']`,抓 `_model.gauss_params.*`,**原样写出**:scale 留 log、opacity 留 logit、quat 留 wxyz、法线写零——原样即 3DGS ply 格式,先激活再写的 ply 在所有 viewer 里过曝。
- 唯一实际变换:features_rest 的 `transpose(0,2,1).reshape(N,-1)`(41 行)——splatfacto 存 (N,15,3) 系数主序,PLY 要通道主序(R 的 15 系数、再 G、再 B)。**删掉它 = 颜色迷幻而几何正常,极易误诊为 SH 度问题**。
- 末尾打 p5..p95 逐轴范围(分位数抗 floater):板坐标系模型应是房间级米数。

**为什么这样设计**:`ns-export gaussian-splat` 要训练集在盘上,且 Windows 训练的 config.yml 里 pickle 了 WindowsPath,Linux 上 ns-* 一碰就崩——**一切直接读 ckpt** 是全管线通则(SplatDepth、lift、seg 全如此)。

**参数速查**:`--ckpt`(必填,splatfacto 专用)| `--out`(默认 ckpt 上上级/splat.ply,假设 ckpt 还在 nerfstudio_models/ 里;拷走的 ckpt 必须显式传)。

**坏了长什么样**:
- 范围打出 ~[-1,1] 而非房间米级 → 训练没关三关;splat 不在板坐标系,下游视线全 miss(ply 里没有任何元数据说明帧,这行打印是唯一的告示)。
- torch≥2.6 UnpicklingError → weights_only 默认翻转,加 `weights_only=False`。
- KeyError gauss_params → 不是 splatfacto ckpt,或 nerfstudio 改了命名空间。

### 3.10 run_lift_sam.ps1 — Windows/4090 上跑 lift 的环境包装器

**核心机制**(5-12 行):
- `TORCH_EXTENSIONS_DIR=E:\Grasp\torch_extensions`(预编 JIT 缓存,秒级 import;wipe 后是数分钟重编且要求 nvcc+MSVC 同时健在)。
- `CUDA_HOME/CUDA_PATH` = conda env 根(cudatoolkit-dev 把 nvcc 装在 env\bin,torch cpp_extension 经 CUDA_HOME 找它)。
- PATH 前插顺序刻意:env\bin(nvcc)→ env\Library\bin(cudart/cublas DLL)→ **MSVC 14.38.33130** Hostx64\x64——torch 把工具链身份哈希进缓存键,cl 版本换了就是冷缓存重编,不是命中。
- `python -u` 直调 env 解释器(不 conda activate),SAM 逐视图进度实时流出;`$PSScriptRoot` 定位同目录的 lift_sam_instances.py。

**为什么这样设计**:gsplat 在 Windows 是 import 时 JIT。裸跑 python 的死相是 gsplat import 处一坨 ninja/cpp_extension 回溯,离真因十万八千里——实测症状链:"No CUDA toolkit found" → `CameraModelType AttributeError`(ns-train 同坑)。与记忆笔记 `win-4090-mapping-env.md` 互为镜像,改路径两边同步。

**参数速查**:`@args` 全透传;三个硬编码:env 根(5 行)、MSVC 路径(10 行)、缓存目录。

**坏了长什么样**:
- `cl.exe not found` / ninja 编译错 → VS 更新删了 14.38 目录(**本脚本最可能的腐烂方式**):改 10 行路径,认一次慢重编。
- import "卡死"几分钟后成功 → 缓存冷了在重编,不是挂了。
- DLL load failed → 系统级 CUDA 抢了 PATH 前排(9 行被改)。
- `can't open file ...lift_sam_instances.py` → ps1 被单独拷走,不再与目标脚本同目录。

### 3.11 环境坑速查(全部实测踩过)

| 坑 | 症状 | 解法 |
|---|---|---|
| gsplat Windows JIT | "No CUDA toolkit found" → CameraModelType AttributeError(ns-train 同中招) | 走 run_lift_sam.ps1:TORCH_EXTENSIONS_DIR + CUDA_HOME=conda env + MSVC 14.38 上 PATH |
| SAM 自动模式显存 | points_per_batch 64 @2048px+vit_h → OOM(24G) | 16-32 安全(峰值 ~14G);先降 batch 再考虑 long-side |
| nerfstudio emoji × GBK 控制台/重定向 | UnicodeEncodeError 中断训练/处理 | `PYTHONUTF8=1` 保命 |
| COLMAP 旧 database.db | 重跑 feature_extractor 多出相机 | 重跑前整个输出目录删除 |
| Windows config.yml 的 WindowsPath | Linux 上 ns-* 一碰就崩 | 一切直接读 ckpt(export_splat_from_ckpt / SplatDepth 均如此) |

---

## 4. 视线侧工具

> 本节工具**现居 `~/Project/Eye_Tracker/tools/`**(机器人侧 Linux 机),本节以 **2026-07-09 迁出版(本仓库 git `7b707a8`)为基准**,行号均指该提交。Eye_Tracker 侧已有演进(如 `gaze_live` 的滚动 bias 重估,§5.3),核对行号前先 diff。

### 4.1 pupil_localizer.py — 头戴相机在线/离线定位(tags → T_world_cam)

**核心机制**:
- 帧源(71-138 行):live 走 Pupil Remote 握手 + SUB 'frame.world'(需 Frame Publisher 插件),时间戳取 Pupil meta,与 gaze.pldata 同钟;离线读 world.mp4 + world_timestamps.npy 按索引配对。
- `solve_pose`(167-195 行):原始畸变图检测 ArUco → **仅角点** fisheye undistort 进归一化坐标 → **SOLVEPNP_ITERATIVE**(tag 全共面,SQPNP 对平面点云直接断言崩溃;≤4 点即单 tag 走裸 solvePnP,无 RANSAC 可言)→ >4 点 solvePnPRansac(归一化阈值,200 iter)+ solvePnPRefineLM;返回 `inv(T_cam_world)` = T_world_cam。
- 三重验收闸(285-303 行):① 位置在测绘 tag 范围 ±3m 内、z∈(0.15, 2.8)m **硬编码**(头戴相机 + z-up 房间假设,梯子/趴地录像会被静默拒);② mean_reproj_norm ≤`--max-mean-reproj`;③ 0.25s 内平移跳变 >`--max-jump` 拒,**连拒 5 次强制信任新位姿**(真实快速头动的恢复机制)。
- `ema_pose`(198-209 行):位置 lerp + 旋转测地线定比步进;**闸后**作用,日志/发布的是平滑后位姿。

**为什么这样设计**:两个重投影阈值刻意不等——RANSAC 内点门 0.01(≈8px)松、验收门 0.006(≈5px)紧,逐点过 RANSAC 不代表整体位姿被接受。定位对 tags_world.json 的 corners_world 联合解 PnP,tag 物理尺寸不参与(§3.4)。

**参数速查**:`--min-tags 1`(单 tag 有平面二义性且无 RANSAC;要可靠升 2,换覆盖率)| `--max-reproj-norm 0.01` | `--max-mean-reproj 0.006` | `--max-jump 1.0` | `--ema 0.0`(**alpha 是上一帧的权重**;0.7 起步,高了位姿滞后毒化下游射线原点)| `--dictionary DICT_6X6_250`(须与墙 tag 实印及 verify_pose_render 的硬编码一致)。

**坏了长什么样**:
- `ValueError: not enough values to unpack (expected 3, got 2)` → **已知 bug**:184 行 PnP 失败路径返回 2 元组而其余路径 3 元组;修法:改成 `(None, 0, None)`。
- 零定位帧但 `--detect-only` 有 tag → tags_world.json 的 id 与检出 id 不重叠,或字典错。
- live 卡 'waiting for world frames' → Pupil Capture 没开 Frame Publisher。
- 位姿米级离谱 → survey 非米制,或 calib npz 张冠李戴。
- 同一路段每次都空洞 → 只见一枚远 tag 时 mean-reproj 闸拒帧;放宽阈值或补测绘那面墙。
- 离线无 world_timestamps.npy → 合成 t=i/30(111 行),**与 Pupil 时钟对不上,下游所有时间连接静默断**。
- 位姿"橡皮筋"/冻结 → --ema 过高(记住 alpha 权重方向)。

### 4.2 gaze_precision.py — 盯 tag 协议 → bias/σ/drift 精度戳

**核心机制**:
- `window_stats`(54-93 行):窗内每个 gaze 样本 undistort 到 pn(与 gaze_to_world 同款 y 翻转 + fisheye undistort);把每个测绘 tag 中心投进相机 `x = Rᵀ(c − cam)`,x[2]<0.3m 剔除(太近/在背后);角偏 = `pn − x[:2]/x[2]`(**gaze 减 target,归一化平面**);按 tag 收 ‖偏差‖<on_tag_rad 的样本,≥min_samples 者以 ‖逐轴中位偏差‖ 最小胜出;bias = 逐轴中位数;σ = (n,2) 残差**逐轴池化 RMS**(90 行,"1D 等效")。
- main(96-167 行):`--on-tag-deg` 只在 I/O 处是度,内部一律 `tan(radians(·))`(116 行);头/尾双窗;合并 bias 按样本数加权、合并 σ=√(Σnσ²/n_tot);drift=‖bias_head−bias_tail‖(度向量 L2);`σ = hypot(σ, drift/4)`(145 行,前提是下游线性插值,§5.3);verdict:σ>2.5 poor-sigma / drift>4 re-record / 1.5<drift≤4 lerp 可治。
- 真正的产物是按盯视中位时间排序的 `stamps` 数组(163 行)——resolve_bias 插值穿过它;顶层 bias_deg/sigma_deg 只是给人看的加权汇总。

**为什么这样设计**:协议(片头/片尾各盯一枚 tag 2-3s)= 每段录像的**免费重标**——tag 位置 survey 到毫米即免费真值,视线层的系统偏差当场可测、可下发修正。它 import gaze_to_world 复用 PoseTrack/load_gaze(31 行)——之所以无 CUDA 也能跑,全靠 g2w 的 torch/gsplat 是惰性导入,**谁把那些 import 提到模块级,这个审计工具就得配 GPU 环境**。

**参数速查**:`--head-window 15` | `--tail-window 30`(**默认从 12 改 30:rec002 的片尾戳实际在结束前 12-30s,12s 窗漏掉了它**;代价:<~45s 的短录像头尾窗重叠,同一 stare 数两遍,drift 假 0 且合并 σ 双计)| `--on-tag-deg 4.0`(必须 >|真 bias|+几个 σ,否则截断分布低估 bias;rec001 的 7-18° 滑移对 4° 门**不可见**——你得到 SystemExit 而不是坏数字)| `--min-samples 20`(原始行与 on-tag 子集各查一遍)| `--min-confidence 0.6`(与 gaze_to_world 保持一致,σ 才描述真正被映射的样本)| `--max-gap 1.0`(录像头尾恰是定位可能还没锁的时刻)。

**坏了长什么样**:
- `No usable tag-stare window` → 没执行协议 / bias 超 on-tag 门 / 头尾窗内定位没锁 / **tags_world.json 是旧图的**(7b707a8 恰好重测绘过墙 tag,陈旧 tags 文件静默杀死此工具)。
- drift≈0 但肉眼可见滑移 → 录像太短,双窗抓到同一 stare。
- bias 很小、下游注视仍系统性偏 → 锁错了 tag(核对 stamps[i].tag 与 n,被试盯的是不是它)。
- verdict 是 elif 链:又吵(σ>2.5)又轻漂(1.5-4°)时只报 drift——**读 JSON 的 sigma_deg,别只看 verdict**。

### 4.3 gaze_to_world.py — 视线样本 → 板坐标系 3D 注视点(几何心脏)

**核心机制**:
- `PoseTrack.query`(99-115 行):searchsorted + 位置 lerp + 旋转 Slerp;括号间隔 >`--max-gap` 返回 None(→ no-pose);轨迹外只在 0.05s 内 snap 到首/尾位姿。
- `SplatDepth`(121-142 行):加载 ckpt 后激活(normalize quats / exp scales / sigmoid opacities),SH=dc+rest、degree 3;`_render` 把 w2c 直接当 gsplat viewmat(全 CV 约定,§2#3)。
- `depth_along_ray`(144-161 行):造一个 +z 轴 = 射线的合成相机(up=[0,0,1],|z_z|≥0.95 时换 [0,1,0] 躲极点),f=256 渲 33×33(半 FOV≈3.7°);取中心 3×3 中位 z-depth——中心处 z-depth≡射线长;中心 alpha 中位 <0.5 → None(no-surface:窗、天空、未建图体积)。**别在不加 tmul 的情况下加宽这个 3×3**。
- `patch_along_ray`(163-186 行):f=(S/2)/tan(half_angle) 让 patch 恰好张满锥;像素方向取半像素中心;返回 depth、alpha、单位世界方向、**tmul=‖[nx,ny,1]‖**(§2#5)——供 gaze_object --cone。
- `resolve_bias / bias_at`(193-223 行):读 gaze_precision.json 的 stamps,`tan(radians(deg))` 进归一化平面(与 undistortPoints 输出同空间;4° 内 tan≈rad 差 0.1%),`np.interp` 逐轴线性插值、两端常数外推;`--bias-deg` 强制常数、`--no-bias` 归零。
- 样本管线(默认注视模式 344-437 行 / `--continuous` 266-339 行):`u=x·W, v=(1−y)·H` → `cv2.fisheye.undistortPoints` → `pn −= bias_at(t)` → `ray_cam=[pn,1]/‖·‖` → `R·ray_cam` → 沿射线渲深度 → `point = origin + d·ray`。默认模式先对 fixations.pldata **按 id 保留最后一条**(77-79 行,在线检测器对同一 id 发增长更新,直接迭代原始流会重复计数)。
- continuous 聚类(233-263 行):贪心单程——样本距**运行质心** <`--cluster-radius` 入簇(质心逐样本更新),断即封簇;保留时长 ≥`--min-fix-dur` ∧ ≥4 样本;`spread_m`=到质心均距,`ang_spread_deg=atan2(spread,dist)`。顺序依赖:一个扫视样本或 floater 深度毛刺就能把一次注视劈成两簇,无回并。

**为什么这样设计**:continuous 模式的存在理由是 VOR——边走边盯时图像系检测器看到的是"扫视",世界系聚类才认得出注视。惰性 import 是承重结构(msgpack 67 行 / scipy 89 行 / torch+gsplat 122-123 行):gaze_precision 免 CUDA、首次 gsplat import 的 JIT 只有真用 SplatDepth 的人付。

**参数速查**:`--max-gap 1.0`(掉线恰与快头动相关,lerp 最烂的时刻,别贪大)| `--min-confidence 0.6` | `--continuous` | `--sample-hz 30`(等距抽样假设均匀采样率,重度置信过滤会偏斜)| `--cluster-radius 0.15`(3m 处 1° 抖动 ≈5cm,4m≈7cm,0.15 勉强兜 2σ;过大把邻物注视合成两物之间的假质心)| `--min-fix-dur 0.25` | `--bias-deg`(强制常数,**丢掉抗漂移插值**)/ `--no-bias`(仅 A/B)| `--ckpt`(默认 lab_result 下 mtime 最新——**重训过别的东西后会静默抓错图**)。

**坏了长什么样**:
- 全 no-surface → ckpt 抓错(mtime 陷阱)/ 视线出图 / 位姿表达在另一张图的板坐标系里。
- 全 no-pose → 时间戳不在 Pupil 钟上,或定位空洞 >max-gap;对照 358-359 行打印的 poses 时间范围。
- 点垂直镜像 → y 翻转被删或翻两次(280/393 行)。
- **恒定横偏且越到片尾越大** → bias 修正缺失或 gaze_precision.json 拿错佩戴——bias(t) 插值正是为 rec002 的 2.6° 慢漂而生。
- 首跑挂在 nvcc/ninja → gsplat JIT 缺环境变量(§3.11);之后从缓存加载。
- 无报错的垃圾射线 → calib npz 被换成 5 参针孔模型:`D_fish = dist[:4]`(354 行)把前 4 项误当鱼眼系数。
- continuous 注视碎成短簇 → cluster-radius 对该距离的 σ 太紧,或 floater 深度毛刺;拿 ang_spread_deg 对照 gaze_precision 的 σ。

### 4.4 gaze_object.py — 3D 注视点 → 命名物体(锥后验 / 球投票)

**核心机制**:
- `cone_votes`(62-81 行):均值射线 =(注视点 − 相机原点)归一化(<5cm 直接 ({},1.0));`patch_along_ray` 渲 SxS;逐像素权重 `w=exp(−arccos(dir·meandir)²/(2σ_rad²))`;z-depth∈(0.05, 12.0)m 的像素反投 `X=origin+(depth·tmul)·dirs`(72 行,tmul 见 §2#5);X snap 到 ≤`hit_eps`=0.05m 的最近**标注**高斯(isfinite 先滤再索引,越界哨兵永不解引用);票质量 = Σ(w·alpha) 按 label 累加;`p_none = 1 − Σvotes/Σw`,分母盖**全部** patch 像素——alpha 洞、>12m、无标注表面统统诚实地记作 none。
- main(84-192 行):注视点 = centroid_world 或 point_world;射线原点 = origin_world,退而 T_world_cam[:3,3];`--cone` 下缺原点的注视**逐条静默降级 sphere**,只有 per-entry 的 `mode='sphere-fallback'` 可辨(148-149 行)。sphere 模式:`query_ball_point(0.20m)` + 权重 `1/max(d,0.01)`。投票按 `name_of(label)` 池化(背景查 instances.json、再 names.json[str(id)]、兜底 'object#id')——同名实例合体投票;`object_centroid_world` 是**规范实例质心**(固定值,给机械臂的坐标),刻意不是视线命中点。
- σ 优先链(110-124 行):`--sigma-deg` > 注视文件同目录 gaze_precision.json > 1.5°;gaze_to_world 的 import 只在 --cone 分支内发生(CUDA JIT 只有锥用户付)。

**为什么这样设计**:见 §5.2。一句话:球是视角无关的(狗肚子底下的地板照样进票池),锥只让**可见表面**投票;意图层只消费锥输出。

**参数速查**:`--cone`(推荐;需 ckpt+CUDA,每注视一次 33×33 渲染)| `--sigma-deg`(必须是**bias 修正后**残差——一个注视 = 一次观测,证据不得随样本数锐化,docstring 14-16 行)| `--span-sigmas 2.5` | `--patch 33` | `--hit-eps 0.05`(**帮助文本原话 'do not raise casually'**:轮廓处 ED 深度混合前后景,反投点悬在两物之间,5cm 上限让它们落 none 桶而不是随机认亲)| `--radius 0.20`(仅 sphere)。

**坏了长什么样**:
- 全员 mode:'sphere-fallback' → 注视文件是旧格式缺 origin_world;用 --continuous 重生成。
- p_none≈1 全场 → seg 与 ckpt 不同图(标签在另一个坐标系),或原点错到锥渲空。
- 薄物输给地板/桌面 → 其实在跑 sphere(或悄悄 fallback 了),或 hit-eps 被抬、轮廓混合点替背后表面投票。
- 全是 'object#N' → names.json 缺失,或键写成 int(**键必须是 id 的字符串**,104 行)。
- ValueError max() 空序列 → lab_result 下没有 step-*.ckpt,显式传 --ckpt。

### 4.5 gaze_video.py — 从原始录像重建 gaze overlay 视频(不依赖 Pupil Player)

**核心机制**:
- `load_instances`(56-70 行):bbox 8 角点 + `default_rng(inst_id).integers(80,255,3)` 稳定色(与 export_seg_splat 同式,跨工具同色)。
- `draw_instances`(73-101 行):w2c=inv(T_world_cam);先手动变换 8 角做可见性测试——**8 角全部 z>0.15m 才画**(fisheye projectPoints 对相机后/极近点退化);幸存者一次批量 `cv2.fisheye.projectPoints`;再剔除飞出 (−W..2W, −H..2H) 的鱼眼环绕伪影。
- 主循环(166-235 行):cap.read() 与 world_timestamps.npy **按索引配对**(帧数不等 = 全片时移);y 翻转画轨迹(绿新→红旧,searchsorted 取近 `--trail` 秒)与当前点(50ms 内最近样本,conf≥0.8 绿否则橙);verdict 覆盖:注视活跃窗 t_start..t_end+0.15s,画框要求 `object_label ≥10`(219 行硬编码,须与 grasp_intent.OBJ0 同步),把胜名下**全部池化 id**(candidates[0].labels)都画:主 id 粗 3 带名,其余粗 2 同色无名。

**为什么这样设计**:这是全管线的"演示面"和排障入口(§7 从它出发)。`--boxes all` 用来一眼检视分割覆盖(每帧框出所有实例);PoseTrack 是惰性 import + sys.path.insert(tools/ 非包),不传 --poses 时它只需 cv2/msgpack。

**参数速查**:`--boxes target|named|both|all|off`(target 连未命名物体也能框)| `--box-min-diag 0.15` | `--min-confidence 0.4` | `--trail 0.4`(图像空间轨迹,快头动拖糊,非世界稳定)| `--start/--duration`(start 之前的帧仍要解码,只是不写)。

**坏了长什么样**:
- 走近物体框消失 → 8 角 z 剔除,**已知美学缺陷,不是位姿坏**。
- 框闪烁 → poses.jsonl 空洞 >1s(此处 max_gap **硬编码** 1.0,139 行),回 4.1 查定位闸。
- 框整体恒偏 → calib npz 不配这台相机,或录像分辨率非 1920×1080 的纯缩放。
- 有 verdict 文字没框 → object_label<10(背景)、objects 投票用的分割与 gaze_video 的 --seg-dir 不是同一份(id 不在 instances.json,218-219 行静默过滤),或没同时给 --poses;`--boxes target` 需要 --poses 与 --objects **同时**在场,缺一静默无框。
- 输出打不开/空 → mp4v 编码器平台性失败(158 行只查 isOpened,查不出编码质量)。

### 4.6 grasp_intent.py — 物体 verdict → 抓取意图事件(v0,纯 stdlib)

**核心机制**:
- 过滤(44-47 行):`object_label ≥ OBJ0=10` ∧ `vote_share ≥ --min-vote` ∧ 有 object_centroid_world,按 t_start 排序。
- visit 分组(52-64 行):同**名**(池化后)且瞥离间隔 ≤`--merge-gap` 并入当前 visit;dwell 只累计注视时长——**闸式累加器**,gap 不加不衰;换物或超 gap 即开新 visit。
- 意图判定(70-87 行):dwell≥`--dwell` 触发;`confidence = min(1, dwell/(2·阈值)) · mean_vote_share`(2 倍阈值处饱和,再按 verdict 干净度缩放);`target_world` = 最后一个注视的**实例质心**(静态、可直接喂臂);`t_intent = visit 起点 + dwell 阈值`——时基是第一个**过滤后**注视(t0_all),与 gaze_to_world/gaze_video 打印的时间不同基,同一事件跨工具读数不同。
- revisits(66-68 行):全片计数——**非因果,含意图之后的访问;在线移植必须改成因果**。

**为什么这样设计**:刻意简单:dwell 阈 + revisit 数,不做抓取点(等贝叶斯射线模型),只在有真值目标的录像上可验证。锥上游是它可用的前提——sphere 的 vote_share 噪声让 0.5 门静默丢掉大多数注视。纯 stdlib(argparse/json/pathlib),任何环境可跑。

**参数速查**:`--dwell 0.8`(低了长打量出假意图;与上游 --min-fix-dur/--cluster-radius 联动,那边决定注视怎么被切)| `--merge-gap 0.6`(须高于典型"扫视+确认瞥"0.3-0.5s,否则瞥一眼机械臂就劈断 visit 清零 dwell)| `--min-vote 0.5`(被丢的含糊注视**连桥都不搭**——中途一个坏 verdict 能撑开 >merge-gap 的洞,两阈值联动)。

**坏了长什么样**:
- 明明盯了却零意图 → 上游 sphere 噪声压 share(重跑 gaze_object --cone),或 JSON 是加质心字段之前的旧格式。
- 一次长盯拆成多个 visit → 上游聚类劈了注视,或中途含糊 verdict 被 min-vote 丢弃开洞。
- 物体挪走后 target_world 不动 → 预期行为:那是建图时的静态实例质心,不是实时位置。
- 墙地板成为目标 → 分割把背景标进了 ≥10;修分割的背景映射,别动 OBJ0。

### 4.7 verify_pose_render.py — 单帧全链体检(标定×测绘×定位×地图唯一同框处)

**核心机制**:
- 选帧(88-103 行):`score = 检出测绘 tag 数 + min(拉普拉斯方差/500, 2.0)`——锐度封顶 2.0,tag 数永远主导;`--frame` 可钉死。
- 虚拟针孔(126-131 行):手搓 `K_new = diag(fx·s, fy·s)`、主点强制画面中心(`estimateNewCameraMatrixForUndistortRectify` 返回垃圾,帮助文本 48-49 行);`initUndistortRectifyMap + remap`——**全管线唯一一次全图 remap**。
- 残差(139-150 行):绿十字 = 测绘角点经 PnP 位姿 `projectPoints(K_new, dist=None)`;红圈 = 归一化检测点解析映射 `pts_norm·diag(fx,fy)+(cx,cy)`;`角度 = degrees(err/fx)`、`mm@tag = err/fx·‖角点−相机‖·1000`——焦距归一,跨 run 可比;**raw px 随 --pinhole-scale 变,别拿它对比**。
- 渲染(133-137 行):同一 K_new 同一位姿喂 SplatDepth,链上无任何轴翻转,也不许加。blend 在打标签**之前**做(160 行),文字不重影。

**为什么这样设计**:blend 像一张清晰照片 = 标定、测绘、定位、地图四个子系统**同时**正确——"几何链 ~0.1°(0.93px)"验收数字即出于此。位姿故意取**单帧、无时序闸**——隔离单帧误差,预期略差于 gated 轨迹。它复用 pupil_localizer 与 gaze_to_world 的函数(32-34 行 sys.path.insert),继承了前者 solve_pose 的 2 元组 bug。

**参数速查**:`--pinhole-scale 0.7`(小 = 保 FOV 多 tag、黑边;角度/mm 指标不变)| `--frame`(钉帧才 apples-to-apples)| `--scan-step 5`。

**坏了长什么样**(§7 决策树的第一分叉):
- **残差小(<2px)+ blend 重影 → 地图问题**:splat 没对齐 tag 世界(v2 transforms 之前训的图),重训/重对齐,位姿没错。
- **残差大 → 标定/测绘/定位问题**,与地图无关。
- 每帧都 'no surveyed tags detected' → 录像根本没看向测绘墙,或 tags_world.json 是旧 survey(v2 240mm 重建前)的 id。
- 渲染全黑 → mtime 抓错 ckpt(控制台会打 ckpt 名,先核对)。
- 隐性依赖:ArUco 字典**硬编码** DICT_6X6_250(76 行),不吃 pupil_localizer 的 --dictionary;角点序必须全链一致(survey 按 TL,TR,BR,BL 存),否则 PnP 产出一个自信的错位姿、本工具再郑重其事地"验证"它。

### 4.8 process_recording.sh — 一条命令跑完一段录像(Linux 机)

**核心机制**:五步串行,`set -euo pipefail`:
1. pupil_localizer → `poses.jsonl`;
2. gaze_precision → `gaze_precision.json`(**唯一容错步**:失败仅打 `WARN: no usable tag stare; continuing without bias correction (sigma default)` 继续——整段退化为无 bias 修正 + σ 默认 1.5°);
3. gaze_to_world `--continuous --annotate-dir wfix` → `world_fixations.json` + `wfix/`;
4. gaze_object `--cone` → `world_fixations_objects.json`;
5. gaze_video → `gaze_objects_overlay.mp4`(`--skip-video` 跳过)。

头部导出 Linux 侧 gsplat JIT 环境:`ENV=$HOME/miniconda3/envs/nerfstudio`、`CUDA_HOME=$ENV`、conda 的 gcc/g++ 作 CC/CXX、`TORCH_CUDA_ARCH_LIST="5.2"`;tags 固定指仓库 `world_size/tags_world.json`。

**为什么这样设计**:它是"在线侧每帧"那半张数据流图的可执行版;所有产物落在录像目录内——**一段录像 = 一个自包含的证据包**,归档、复查、跨机拷贝都以目录为单位。

**坏了长什么样**:
- 后段注视系统性偏移 → **先翻日志找第 2 步的 WARN**,整段可能没有 bias 修正。
- 第 1 步产出稀疏 → 后面全部跟着稀疏(gap>1s 处注视变 no-pose、视频框闪烁),根因永远在定位层。
- 第 3 步起崩在 torch/gsplat → Linux 侧 JIT 环境变量段被改,或 conda env 路径变了。

---

## 5. 三个关键算法深讲(论文方法节素材)

### 5.1 mask-lift 共识聚类 + v2.1 四闸门

**实测故事一:旧 build_object_map 之死**。前身(`E:\Grasp\tools\build_object_map.py`,SAM+CLIP+bbox 关联)有两个结构性死因:(a) 跨视图关联用 axis-aligned bbox overlap + CLIP cosine——同一物体不同视角的点云根本不重叠,CLIP 对小 crop 不稳,part/whole 碎片跨帧 IoU 永远过不了阈,同一物体被框成多个;(b) 精度优先的过滤链**连乘**(pred_iou>0.86 × stability>0.92 × 面积窗 × depth-band × min_views≥4)把召回杀穿,一只机械狗整只漏掉(dedup 保大弃小、大 mask 又被 max_frac 丢 = 狗整帧消失)。教训一句话:**2D 只出证据,3D(高斯 ID)管关联,几何终审**。

**v2 的解**:每个 SAM mask 经渲染深度反投 + snap,变成一个高斯 ID 集合;关联从此发生在 3D:掩码-高斯关联矩阵 M(CSR),`(M@M.T)` 一次算出全部两两交集,`IoU = inter/(|A|+|B|−inter) ≥ 0.3` 且**异帧**才连边(`lift_sam_instances.py:336-350`)。0.3 之所以低:不同视角照到同一物体的不同表面,跨视 IoU 天然低。

**实测故事二:rec002 焊接实例与四闸门**。v2.0 在 rec002 出现 MERGED#11/12/14——2-5 米长的焊接怪物吞掉了两只机器狗。逐一验尸得到 v2.1 的四个闸门:

1. **max-mask-extent 2.0m**(319-322 行):焊接源头是杂物区 mask 被 bg-frac 剥掉地板后剩下"多物体并集",robust 直径 ‖p98−p2‖ 超 2m 的提升结果按视线方向多物并集丢弃。低于 ~1.9 会毁掉 1.6m 桌子的整体 mask,只剩零件实例——阈值卡在场景最大单物之上。
2. **只允许跨帧建边**(346 行):同帧内 SAM 的 part/whole 粒度阶梯(狗 → 狗+桌)会通过传递闭包焊接不同物体;异帧同物的两个 mask 才是"共识"。
3. **吸收方三重门**(377-381 行):containment≥0.85 ∧ 吸收方 ≥6 视角 ∧ robust 直径 ≤2.2m。视角共识单独用会失效——rec002 的"家具条带"分量有 60-80 个视角照样是多物体,**尺寸上限才是承重闸**;2.2 略高于 2.0,让合法的整桌分量仍能吸收自己的桌角。
4. **10cm 体素 26 连通拆焊收尾**(426-429 行):任何漏网的远隔并集最终按体素连通性拆开。= "SAM 提名,几何终审"。

**剩余问题与出口**:SAM 按颜色拆机器人(黄腿黑身)是它的世界观,包含性合并救不回;出口是 `names.json` 同名合并——命名即合并,下游 gaze_object 投票、gaze_video 画框、export_seg_splat 上色全按名字池化。粒度停在物体级是有意的:gaze 角精度 1-2° ≈ 5-15cm@3m,part 级分辨不出来。

### 5.2 视线锥后验(gaze_object --cone)

**病根:球投票是视角无关的**。0.2m 球以注视点为心收邻域高斯,1/d 加权——狗身子底下的**地板高斯**照样进票池,尽管从相机看它们根本不可见。注视本质是视角相关的行为。13cm 高的趴姿狗在 v1 下只有 51-82% 票面,失掉的票全被地板抢走。

**解:让可见表面投票**。33×33 深度块本来就在渲(旧代码只用中心 3×3 求深度),锥模式把整块用满 = **1089 条加权射线**,零额外渲染成本——且比"补一圈低权重离散射线"的采样方案严格更优(连续覆盖锥立体角,每像素自带 alpha)。数学(`gaze_object.py:62-81`):

- 权重:`w_i = exp(−θ_i²/(2σ_rad²))`,θ_i = arccos(dir_i·mean_dir);
- 票:label 的质量 = Σ w_i·alpha_i,遍历 z-depth∈(0.05, 12)m、乘 tmul 反投后 snap 进 ≤5cm 标注高斯的像素;
- `p_none = 1 − Σvotes/Σw`,分母是**全部** patch 像素——不可见、超距、无标注全部诚实地记作"不知道"。

**σ 的统计要点(方法节必写)**:一个注视 = **一次观测**。标定偏置在整段共享,不随样本数平均掉,所以

> σ_eff² = σ_bias² + σ_jitter²/N

——绝不能把 30Hz 样本当独立观测连乘似然,否则 N=30 时锥宽被砍到 1/√30,置信爆炸、锥退化成一根针。σ 取 gaze_precision 的 **bias 修正后**残差(其定义是逐轴池化 RMS,恰好匹配逐轴角高斯,别乘 √2"转径向"——会双计)。

**hit-eps 5cm 别放大**:ED 深度在物体轮廓处 alpha 混合前景/背景 z,反投点悬在两物之间的自由空间;5cm 上限让这些轮廓混合点落进 none 桶而不是随机认亲。

**实测故事:rec002 四方对比**。输入修好(v2.1 分割 + bias 插值)后,球的票面反而**更高**(87-96%)但出现两次自信错判(错误物体 84%/44%);锥票面略低(70-76%)但**零翻转** + p_none 诚实。结论写进意图层设计:**意图层只消费锥输出——"锥是诚实的估计器,球靠迟钝掩盖输入错误"**。票面高不是好事,校准好才是。

### 5.3 bias(t) 精度戳与佩戴漂移

**侦探故事:2.6° 慢漂**。rec002 早期出现"锥比球差"的反直觉结果。排查发现:片头 y 偏置 −0.14°、片尾 +2.44°——一次佩戴内头带慢滑 **2.6° ≈ 13cm@3m,恰好一只趴姿狗的身高**。只扣片头偏置时后半段欠修正,σ=1° 的锥被系统性带偏整只狗;20cm 的球反而误打误撞地宽容。真凶不是锥,是**常数 bias 假设**。

**解:头尾戳 + 线性插值**。协议:片头/片尾各盯一枚测绘 tag 2-3s(tag 位置 survey 到毫米 = 免费真值)。gaze_precision 产出按时间排序的 stamps,gaze_to_world 的 `resolve_bias/bias_at`(193-223 行)对每轴 `np.interp` 线性插值、两端常数外推,单位在 `tan(radians(deg))` 的归一化平面上与 undistortPoints 输出同空间。两处工程细节都是实测换的:

- **尾窗 12s → 30s**:用户的片尾戳实际落在结束前 12-30s,12s 默认窗直接漏掉(`gaze_precision.py:42-44` 注释即此案)。
- **σ 漂移罚项 drift/2 → drift/4**(145 行):常数 bias 模型下未修正残差 ~drift/2;下游线性插值吃掉一阶漂移后只剩**非线性残差**,罚 drift/4。若有人把下游改回常数 bias,这里的 σ 立刻变乐观——两个文件耦合。

**验证与演进**:重跑 rec002:三狗零错误翻转、趴狗中位 76%[61-90]。Eye_Tracker 新版 `gaze_live` 把它推广成**滚动重估**——视线扫过任意测绘 tag 即在线更新 bias,不再依赖头尾协议。

---

## 6. 参数速查总表

| 工具 | 参数 | 默认 | 调它管什么 | 什么症状该动它 |
|---|---|---|---|---|
| align_to_charuco | --square-size | 1.0 | 世界系是否米制 | survey 边长比 ≈0.03/33 → 忘传 0.030 |
| align_to_charuco | --min-views / --max-reproj-px | 3 / 4.0 | 三角化质量 vs 角点存活 | 角点少于 6 被中止;内参糙时放宽 px |
| survey_aruco_tags | --tag-ids | 0-29 | 测绘哪些 tag | v2 墙面大 tag 在 74-249,必须覆写 |
| survey_aruco_tags | --tag-sizes | 无 | 混布尺寸模板 | 高 id 比值 ≈2.4 → 补 '74-249:0.24' |
| calibrate_charuco | --fix-k3 | 关 | 畸变 4/5 参 | 下游只读 4 参,接本管线必开 |
| lift_sam | --every | 3 | 视角数 vs SAM 时长 | min-views 存活率低 → 降;太慢 → 升 |
| lift_sam | --match-eps | 0.04 | 反投点 snap 距离 | 薄物丢像素 → 微升;轮廓飞点认错主 → 降 |
| lift_sam | --max-mask-extent | 2.0 | 多物并集 mask 闸 | 大单物碎裂 → 微升;条带焊接 → 降 |
| lift_sam | --edge-iou | 0.3 | 跨帧 mask 连边 | 单物碎裂 → 降;异物焊接 → 升 |
| lift_sam | --merge-containment | 0.85 | 部件吸收门 | 腿/角残留独立实例 → 降;邻物被吸 → 升 |
| lift_sam | --attractor-min-views / --max-attractor-extent | 6 / 2.2 | 吸收方资格 | 家具条带吞物 → 查 extent(承重闸);须恒 > max-mask-extent |
| lift_sam | --split-voxel | 0.10 | 拆焊粒度 | 稀疏单物被劈半 → 升;10cm 缝焊接残留 → 降 |
| lift_sam | --min-votes | 2 | 高斯归属最低票 | 单 mask 毛边入侵 → 升;薄面丢失 → 降 |
| lift_sam | --bg-frac | 0.6 | 背景 mask 剔除 | 低平物消失 → 升;地板假实例 → 降 |
| lift_sam | --points-per-batch | 16 | SAM 显存 | OOM → 先降到 8 再考虑 long-side |
| lift_sam | --long-side | 2048 | 工作分辨率 | 杯子级小物丢失 → 别降;慢 → 降(有代价) |
| segment_splat | --floor-z / --ceiling-z | 0.08 / 2.5 | 背景绝对高度 | 房间级巨实例 → 微升 floor-z;矮物被吞 → 降 |
| pupil_localizer | --min-tags | 1 | 单 tag 位姿准入 | 位姿偶发翻面 → 升 2(牺牲覆盖率) |
| pupil_localizer | --max-mean-reproj | 0.006 | 位姿验收门(归一化) | 固定路段空洞 → 放宽或补测绘该墙 |
| pupil_localizer | --max-jump | 1.0 | 0.25s 内跳变门 | 快头动被连拒 → 靠 5 击自复位,别轻动 |
| pupil_localizer | --ema | 0.0 | 平滑(prev 权重) | 抖 → 0.7;下游射线原点滞后 → 降 |
| gaze_precision | --tail-window | 30 | 片尾戳搜索窗 | 尾戳漏检 → 加宽;短录像 drift 假 0 → 缩 |
| gaze_precision | --on-tag-deg | 4.0 | 盯 tag 认定门 | 大 bias 时 SystemExit → 临时放宽测漂移 |
| gaze_to_world | --max-gap | 1.0 | 位姿插值容忍 | no-pose 多 → 先修定位覆盖再考虑放宽 |
| gaze_to_world | --cluster-radius | 0.15 | 世界系聚类半径 | 长盯碎裂 → 升;邻物合簇 → 降 |
| gaze_to_world | --min-fix-dur | 0.25 | 注视时长门 | 短瞥丢失 → 降(噪声↑) |
| gaze_to_world | --bias-deg / --no-bias | 自动读 json | bias 来源 | 仅 A/B 实验;常数 bias 丢抗漂移能力 |
| gaze_object | --sigma-deg | json 否则 1.5 | 锥宽 | 过窄自信爆炸;过宽全糊进桌面 |
| gaze_object | --hit-eps | 0.05 | 反投点认领距离 | 轮廓误判 → **别升**;seg 点稀 → 才考虑微升 |
| gaze_object | --span-sigmas / --patch | 2.5 / 33 | 锥覆盖/采样密度 | p_none 校准差 → 升 span;慢 → 降 patch |
| gaze_object | --radius | 0.20 | 球模式半径 | 仅遗留对比用,意图层不消费球输出 |
| grasp_intent | --dwell | 0.8 | 意图触发累计 | 误触发 → 升;响应慢 → 降 |
| grasp_intent | --merge-gap | 0.6 | 瞥离容忍 | 瞥一眼臂就断 visit → 升(>0.5) |
| grasp_intent | --min-vote | 0.5 | 注视准入 | 锥上游正常时不必动;球上游会静默丢光 |
| gaze_video | --boxes | target | 画框策略 | 审计分割覆盖 → all;演示 → target |
| export_seg_splat | --boxes | named | bead 框策略 | 抓未命名抢票者 → all |

---

## 7. 排障决策树

入口症状:**"视频里框错了"**(gaze_overlay.mp4 的框/圈不对)。按层自上而下,每层用一个专属证物判定,**不确定哪层坏时永远先跑 verify_pose_render**。

```
第 0 层|定性症状(看 gaze_overlay.mp4)
├─ 框整体恒偏 / 与画面错位 ──────────────▶ 第 1 层(几何底座)
├─ 框闪烁、时有时无 ────────────────────▶ 定位覆盖:poses.jsonl 空洞>1s?
│    证物:poses.jsonl 时间戳间隔;pupil_localizer 的 mean_reproj_norm 门
│    (走近物体框消失 = 8 角 z>0.15 剔除,美学缺陷,不修)
├─ 圈不在被试实际看的地方 ──────────────▶ 第 2 层(gaze 标定)
└─ 圈对、名字/目标错 ──────────────────▶ 第 3/4 层

第 1 层|几何底座:verify_pose_render 单帧体检
证物:verify_f<N>_blend.jpg + 控制台残差(px/deg/mm@tag)
├─ blend 糊 + 残差小(<2px)──▶ 地图问题:splat 没对齐 tag 世界
│    查:训练是否用 transforms_aligned.json 且三关全关;
│    export_splat_from_ckpt 的 p5..p95 范围是否房间米级;
│    地图重建后 tags_world.json 是否同步重测绘
├─ blend 糊 + 残差大 ──▶ 标定/survey/定位问题(与地图无关)
│    查:tags_world.json 是否 v2(fit_rms_m 列、Tags seen 覆盖);
│    calib npz 是否本相机;survey 边长比警告
└─ blend 清晰(~1px)──▶ 底座无罪,上第 2 层

第 2 层|gaze 标定层:wfix/ 逐注视帧 + gaze_precision.json
证物:wfix/*.jpg(圈的位置)、gaze_precision.json 的
      bias_deg / sigma_deg / drift_deg / verdict / stamps[].tag,n
├─ json 不存在或 process_recording 第 2 步打了 WARN
│    ──▶ 整段无 bias 修正;协议没执行或 on-tag-deg 门挡住大 bias
├─ 偏移随录像后段变大 ──▶ 漂移:stamps 只有一枚?尾窗漏检(→30s)?
├─ drift_deg>4 ──▶ 线性插值救不了,重录
└─ stamps[].tag 不是被试实际盯的 tag ──▶ 锁错目标,bias 无意义

第 3 层|3D 点层:world_fixations.json
证物:每条的 status / point_world / distance_m
├─ status=no-surface 成片 ──▶ ckpt 抓错(mtime 陷阱)或视线出图
├─ status=no-pose 成片 ──▶ 时钟不一致(world_timestamps.npy 缺失
│    → 合成 i/30)或定位空洞;对照 poses.jsonl 时间范围
└─ point_world 落在空处/旧位置 ──▶ 物体建图后被挪过:
     target_world 是建图时的静态实例质心,地图不更新它不动;
     重跑 lift 或把物体放回去

第 4 层|语义层:world_fixations_objects.json + splat_seg.ply
证物:object / vote_share / p_none / mode / candidates[]
├─ mode=sphere-fallback ──▶ 注视文件缺 origin_world,重跑 --continuous
├─ p_none≈1 全场 ──▶ seg 与 ckpt 不同训练产物
│    (export_seg_splat 的 'labels mapped: X/N' ≈0 是同一病)
├─ 名字是 object#N ──▶ names.json 缺名或键不是字符串
└─ 名字张冠李戴 ──▶ SuperSplat 开 splat_seg.ply(--boxes all):
     找未命名重复实例(暗色自有色块)抢票 → 补命名(同名即合并);
     或 candidates[] 里第二名很近 → 分割焊接/碎裂,回 §5.1 闸门
```

---

## 8. 自测 20 问

### 坐标系(5)

**Q1. nerfstudio transforms.json 的 c2w 要进 OpenCV/gsplat,翻什么?为什么翻列不翻行?**
<details><summary>答案要点</summary>

`c2w[0:3,1:3] *= -1`——旋转块第 1、2 **列**取反。列是相机自身的 y/z 基向量(GL:+Y 上、−Z 前 → CV:+Y 下、+Z 前),这是相机轴的基变换;翻行改的是世界轴,渲出来上下颠倒/镜像。三处出现:`align_to_charuco.py:76-79,232,235`、`lift_sam_instances.py:275-277`。改完看 render_check.jpg。
</details>

**Q2. Pupil norm_pos = (0.3, 0.8),1920×1080 下像素坐标是?**
<details><summary>答案要点</summary>

u = 0.3×1920 = 576;v = (1−0.8)×1080 = 216。norm_pos 左下原点 y 向上,只翻 y 不翻 u(`gaze_to_world.py:280`)。
</details>

**Q3. gsplat 'RGB+ED' 的深度通道是什么?patch 边缘像素怎么换算射线长?**
<details><summary>答案要点</summary>

z-depth(沿相机 z 的期望深度),不是射线长。射线长 t = d·‖[nx,ny,1]‖(tmul,`gaze_to_world.py:184-186`)。`depth_along_ray` 免换算是因为只用中心 3×3(中心处两者相等);`cone_votes` 在 72 行乘 tmul。漏乘 = 离轴点系统性偏短。
</details>

**Q4. lift_sam_instances 的 undistort 为什么 newCameraMatrix 传原 K?**
<details><summary>答案要点</summary>

`cv2.undistort(arr, K, dist, None, K)`(272 行):照片去畸变后与 gsplat 渲染深度共享**同一个针孔框架**,mask 像素 (u,v) 才能直接查渲染深度做反投。换成新 K,照片与深度不同框,提升点全部错位。
</details>

**Q5. 在线定位链里哪一步做了全图 remap?**
<details><summary>答案要点</summary>

没有。定位在原始畸变图上检测 ArUco,只对**角点**做 `cv2.fisheye.undistortPoints`,PnP 用 K=I、归一化单位阈值(`pupil_localizer.py:282-283`)。全图 remap 只在 verify_pose_render 为可视化做一次(126-131 行,手搓 K_new)。省算力,且鱼眼 remap 的边缘信息损失不进定位。
</details>

### 分割(5)

**Q6. 为什么同帧 mask 之间永远不连边?**
<details><summary>答案要点</summary>

SAM 同帧输出 part/whole 粒度阶梯(狗、狗+桌),同帧连边会把阶梯变成传递闭包,把不同物体焊在一起(v2.1 闸门 b,`lift_sam_instances.py:346`)。跨帧同物共识才是关联证据。帧身份用的是循环下标 i(324 行),排序+一次跨步后成立,别在其后重排帧。
</details>

**Q7. max_mask_extent(2.0) 和 max_attractor_extent(2.2) 的大小关系为什么不能反?**
<details><summary>答案要点</summary>

2.2 略大于 2.0,是为了让合法的整桌分量(由 ≤2.0m 的 mask 组成)仍有资格吸收自己的桌角。若把 mask 闸抬到吸收闸之上,会存在"任何合法吸收者都收不了的 mask",大物体永久碎裂。
</details>

**Q8. 一个物体被拆成腿/角碎片,先查哪三个参数?**
<details><summary>答案要点</summary>

① 物体是否 > max-mask-extent(整体 mask 全被丢,只剩零件);② edge-iou 是否过高(跨视 IoU 天然低);③ 吸收三重门是否拦住合并(merge-containment / attractor-min-views / max-attractor-extent)。SAM 按颜色拆的(黄腿黑身机器人)参数救不了,走 names.json 同名合并。
</details>

**Q9. points.npz 和 ckpt 怎么耦合?换 ckpt 会怎样?**
<details><summary>答案要点</summary>

points.npz 存 keep 子集中已标注(label≥0)部分的**逐位相同 float32 坐标**;export_seg_splat 用 cKDTree 上限 1e-4 做"坐标相等连接"回贴(81-92 行)。重训/换 ckpt 后坐标不再相等 → 标签静默失配,大部分高斯落 unlabeled;唯一的告示是 'labels mapped: X/N' ≈0 与 gaze_object 的 p_none≈1。
</details>

**Q10. names.json 同名合并在下游哪些地方生效?**
<details><summary>答案要点</summary>

① gaze_object 投票按 `name_of(label)` 池化(165-179 行),多 id 一名合体投票;② gaze_video 把胜名下全部池化 id 都画框(candidates[0].labels,217 行);③ export_seg_splat 同名共享颜色和 union bead 框(101-102 行);④ grasp_intent 的 visit 按名分组。键必须是 id 的**字符串**。
</details>

### 视线链(5)

**Q11. bias 的符号与单位约定?**
<details><summary>答案要点</summary>

bias = gaze **减** target,在去畸变归一化平面上,存 JSON 时转度、用时 `tan(radians(deg))` 回平面;下游**减去**(`pn − bias_at(t)`)。任一侧翻符号 = 误差翻倍。出处:`gaze_precision.py:72,90,160`,`gaze_to_world.py:282,396`。
</details>

**Q12. 为什么 PnP 强制 SOLVEPNP_ITERATIVE?**
<details><summary>答案要点</summary>

墙面 tag 角点共面(早期全在地板更甚),SQPNP 对平面/退化点云直接断言崩溃;ITERATIVE 用基于单应的初始化,天生处理平面点集(`pupil_localizer.py:167-172`)。这是实测崩出来的选择,不是偏好。
</details>

**Q13. tag 的物理尺寸参与在线定位吗?**
<details><summary>答案要点</summary>

不参与。在线 PnP 直接对 tags_world.json 的测绘 corners_world 联合求解;`--tag-sizes` 只影响 survey 的 Kabsch 刚体模板与 fit_rms 体检数字(Kabsch 对模板均匀尺度误差不变)。所以 99mm/240mm 混布对定位免疫,尺寸填错只是体检数字吓人。
</details>

**Q14. no-surface 和 no-pose 各指什么、各查什么?**
<details><summary>答案要点</summary>

no-surface:射线处 3×3 中心 alpha 中位 <0.5(`gaze_to_world.py:159`)——splat 在那没有表面:ckpt 抓错(mtime 陷阱)、视线出图、位姿在别的图坐标系。no-pose:PoseTrack 在该时刻无 ≤max-gap 的位姿括号——定位空洞或时钟不一致。stdout 状态列先分清,别拿 splat 背定位的锅。
</details>

**Q15. --ema 0.7 平滑的是谁?副作用?**
<details><summary>答案要点</summary>

alpha 是**上一帧位姿**的权重(0=生数据),位置 lerp + 旋转测地线步进(`pupil_localizer.py:198-209`)。副作用:日志里就是平滑后位姿,恒定滞后于真实头动,下游 PoseTrack 插值无法撤销,gaze 射线原点跟着滞后。调太高的症状是位姿"橡皮筋"。
</details>

### 统计与参数(5)

**Q16. 写出 σ_eff 公式并解释为什么不能 σ/√N。**
<details><summary>答案要点</summary>

σ_eff² = σ_bias² + σ_jitter²/N。标定偏置 σ_bias 在一次注视的所有样本间**共享**,不随 N 平均;只有抖动项被 N 压。把 30Hz 样本当独立观测连乘似然,N=30 时锥宽被砍到 1/√30,置信爆炸、单物体自信错判。所以"一个注视 = 一次观测",σ 取 bias 修正后的残差精度。
</details>

**Q17. hit-eps=5cm 挡的是什么?什么情况才允许动?**
<details><summary>答案要点</summary>

物体轮廓处 ED 深度是前后景 alpha 混合值,反投点悬在两物之间的自由空间;5cm 上限让它们落 none 桶而不是随机 snap 到某个物体。唯一合法的放大理由:points.npz 被下采样导致真实表面点离最近标注高斯 >5cm(症状是 p_none 系统性虚高)。
</details>

**Q18. σ 罚项为什么是 drift/4 而不是 drift/2?前提是什么?**
<details><summary>答案要点</summary>

前提:gaze_to_world 对 stamps 做 bias(t) **线性插值**。常数 bias 模型下未修正残差 ~drift/2;线性插值吃掉一阶漂移后只剩非线性残差,经验记 drift/4(`gaze_precision.py:145`)。若下游改回常数 bias,这个 σ 立刻变乐观——两个文件耦合。
</details>

**Q19. rec002 上球 87-96%、锥 70-76%,为什么选票面更低的锥?**
<details><summary>答案要点</summary>

球的两次错判也带 84%/44% 的高票——它的高票面来自视角无关的迟钝(不可见高斯也投票),是**未校准的自信**。锥零错误翻转 + p_none 诚实(不知道就说不知道)。意图层要的是校准好的后验,不是高分。一句话:"锥是诚实的估计器,球靠迟钝掩盖输入错误。"
</details>

**Q20. 尾窗默认为什么是 30s?窗太宽的代价?**
<details><summary>答案要点</summary>

rec002 的片尾盯 tag 实际发生在结束前 12-30s,旧默认 12s 直接漏检 → 无尾戳 → 无漂移估计。代价:短录像(<~45s)上头尾窗重叠,同一 stare 数两遍,drift 假 0 且合并 σ 双计这批样本;head-window 也可能吞掉第一个任务注视当假 stare。
</details>

---

## 9. 主动内化路径

三步,顺序执行,每步有明确的完成判据。

**第一步|通读 + 亲手跑通一遍数据流**。读完本文档后,对照 `PIPELINE.md` 完整处理一段录像(`process_recording.sh <rec>`),然后**逐个打开每个产物**:poses.jsonl(挑几行,肉眼验证 T_world_cam 平移在房间尺度内、时间戳连续段与空洞的位置)、gaze_precision.json(找到 stamps 两枚,核对 tag id 是被试真盯的那枚)、world_fixations.json(数 no-pose/no-surface 比例)、wfix/ 帧(圈在不在目标上)、world_fixations_objects.json(vote_share 与 p_none 的分布)、最后 overlay 视频。完成判据:对每个文件能回答"这个字段是谁写的、上游是什么、下游谁在读"。

**第二步|反向讲解,被挑刺**。向 Claude(或组会)完整讲两个算法:锥后验(§5.2)和 mask-lift 共识(§5.1),**不看文档**,并要求对方专挑软肋提问——"为什么 σ 不随样本数收敛?""同帧为什么不连边?""hit-eps 放大到 15cm 会怎样?""球票面更高凭什么说它差?"。讲不圆的地方就是还没拥有的地方,回到对应源码行读到能圆为止。完成判据:能不查资料答出 §8 全部 20 问。

**第三步|预测-验证循环**。挑至少三个参数实验,**跑之前先白纸黑字写下预测**,再跑,再对账:

- `--edge-iou 0.3→0.5`:预测哪些物体会碎?(提示:跨视 IoU 天然低的大物体先碎)
- `--sigma-deg 1→3`:预测趴狗的 vote_share 和 p_none 各往哪走?
- `--no-bias`(或 `--bias-deg` 强制常数):预测 rec002 后半段注视点往哪个方向偏、偏多少 cm?(提示:2.6°@3m)

错误的预测 = 理解缺口,是这套流程里**最有价值的产出**——每个错误预测都指向一个该重读的章节。三个预测全对之后,这套代码就是你的了。
