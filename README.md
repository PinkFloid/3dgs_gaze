# SceneRebuild — 眼动仪注视意图估计管线

戴 Pupil Core 眼动仪在实验室内走动，实时回答"我在看哪个物体"：

手机拍摄 → COLMAP → ChArUco 板对齐（米制世界系）→ 3DGS（splatfacto）→
SAM 实例分割（mask 提升到高斯 + 跨视角共识，手标命名，命名即合并）→
ArUco tag 定位眼动仪 → tag 精度戳测每段 gaze 偏置/σ → 世界系注视聚类 →
**视线锥后验**（角度高斯 × 可见表面积分）→ p(物体|注视)。

**全部流程、命令、精度指标、已知坑见 [PIPELINE.md](PIPELINE.md)。**

快速入口：

```bash
# 处理一段 Pupil 录像（定位 → 注视点 → 物体判定 → 叠加视频）
tools/process_recording.sh ~/recordings/<日期>/<编号>
```

不进库的大文件（见 .gitignore）：模型 ckpt/splat.ply、视频、原始照片数据集、
COLMAP 发行版。换机器后按 PIPELINE.md 第二部分重新就位。
