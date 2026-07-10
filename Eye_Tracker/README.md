# Eye_Tracker — 视线追踪侧(Pupil Labs Core)

眼动仪相关的工具与数据资产。地图/标定/tag 测绘在兄弟目录 `../SceneRebuild`
(git 与 Windows 4090 同步);本目录是 Linux 本机独立 git 仓库,不与 Windows 同步。

## tools/ — 视线追踪工具链

所有默认路径已锚定 `../SceneRebuild`(标定 npz、tags_world.json、最新地图 ckpt),
两个目录必须保持同级。

| 工具 | 作用 |
|---|---|
| `gaze_live.py` | **实时模式**:单进程跑完 定位→gaze→世界→注视聚类→锥判定,cv2 窗口实时叠加(绿十字=原始视线、蓝点=修正后射线、物体框、判定+世界坐标);盯 tag 自动重估 bias/σ 且按戳龄衰减;注视 0.3s 出暂定判定(黄'~')、移开定案(红'->');`--publish` 发 ZMQ `gaze.intent`(带 provisional 标志);`--replay <录像>` 无硬件回放;`--dump-video` 导出演示视频 |
| `process_recording.sh` | **离线入口**:一条命令跑完 定位→精度戳→注视聚类→物体判定→叠加视频 |
| `pupil_localizer.py` | tag→T_world_cam(实时流/离线录像),鱼眼 PnP + 三道门限 |
| `gaze_precision.py` | 片头/片尾盯 tag 精度戳 → 本段 gaze 偏置/σ/漂移 |
| `gaze_to_world.py` | gaze→世界 3D 落点→注视聚类(--continuous),bias(t) 插值修正 |
| `gaze_object.py` | 注视→物体后验(--cone 视线锥积分,推荐) |
| `gaze_video.py` | 还原注视十字视频 + 命名实例 3D 包围盒(--boxes all 全实例) |
| `grasp_intent.py` | v0 抓取意图:注视驻留+回访 → 意图事件(目标物体+世界坐标) |
| `verify_pose_render.py` | 单帧交叉校验:真实帧 vs 同位姿 3DGS 渲染(验证地图/标定/定位一致) |

```bash
# 实时(Pupil Capture 开着 Frame Publisher、gaze 已标定)
# 用 .sh 包装:任意 shell/conda 环境直接跑(内置 nerfstudio env + gsplat 环境变量)
tools/gaze_live.sh                      # 弹窗 UI,q 退出
tools/gaze_live.sh --publish 5581       # 同时对外发 gaze.intent 事件
tools/gaze_live.sh --replay ~/recordings/<日期>/<编号>   # 无硬件回放

# 离线处理一段录像(产物全落录像目录)
tools/process_recording.sh ~/recordings/<日期>/<编号> [--skip-video]

# 仅实时位姿流
python tools/pupil_localizer.py --print [--ema 0.7] [--publish 5580]
```

## 数据资产

| 位置 | 内容 |
|---|---|
| `world_camera_calibration_imgs/` | 世界相机(1920x1080 鱼眼)标定原始照片,162MB,不进库 |

## 眼动相关但**不在**这里的东西

| 位置 | 是什么 | 为什么在那边 |
|---|---|---|
| `../SceneRebuild/Calibration_result/world_camera_calibration.npz` | 世界相机鱼眼标定结果(rms 0.56px) | 与手机标定同放、git 同步到 Windows;本目录工具的默认 `--calib` 指向它 |
| `~/recordings/<日期>/<编号>/` | Pupil Capture 录像 + process_recording.sh 产物 | Pupil Capture 默认录制目录 |
| `/opt/pupil_capture` | Pupil Capture 3.5.8 本体(Pupil Remote 端口 50020) | 系统安装位置 |

## 环境

工具跑在 `~/miniconda3/envs/nerfstudio`;gsplat JIT 编译环境变量
(PATH/CUDA_HOME/CC/CXX/TORCH_CUDA_ARCH_LIST=5.2)已内置在 process_recording.sh,
单独跑 gaze_to_world / gaze_object / verify_pose_render 时需自带(见 SceneRebuild/PIPELINE.md 已知坑)。
