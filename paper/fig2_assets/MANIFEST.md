# Fig.2 素材清单(槽位文件名约定,2026-08-24)

> AI 生成稿仅作版式 spec;进论文的每张子图必须是系统真实产物。
> 文件放本目录,fig2_compose.py 按名取用。分辨率:小图短边 ≥600 px,
> 中心大图宽 ≥1600 px。照片可裁但不可修饰内容。

| 文件名 | 内容要求 | 来源 |
|---|---|---|
| offline_capture.jpg | 建图手机照片任一张 | SceneRebuild 数据集原图 |
| offline_colmap.png | 稀疏点云+相机位姿截图 | COLMAP GUI 或 points3D 绘制脚本 |
| offline_3dgs.png | v9 渲染(与中心图不同视角) | 渲染管线 |
| offline_sam.png | SAM 实例掩码可视化 | lab_result/segmentation_sam 产物 |
| wearer_pnp.jpg | 鱼眼帧+ArUco 检测角点叠加(**墙 tag,不是标定板**) | Eye_Tracker 调试可视化 |
| wearer_cluster.png | 世界系注视样本散点+聚类圈(真数据) | replay 从 gaze.jsonl 绘制 |
| hub_map.png | v9 渲染大图+实例框,三球着色;**视角避开桌面 tag、含墙 tag** | 渲染管线+实例框叠加 |
| robot_nav.jpg | 狗实拍或地图叠导航轨迹 | 外机位照片/狗端日志轨迹 |
| lastmeter.jpg | 真抓取相机帧+检测框+**投影点标记落在被选框内**(被选实线/落选虚线) | 狗端日志(demo 或 08-25) |
| speech_wave.png | 真实指令音频的波形 | 录音 + matplotlib |
| delivery.jpg | 送达瞬间照片 | demo 或 08-25 teaser 摆拍 |

无需图的槽位(脚本直接排版):命名注册表(读 names.json)、LLM 解析文本框
(`fetch(object = ⟨that⟩)`,**不出现 ball_2**)、绑定时间轴(⟨that⟩→ball_2 在这里)、
Task Order 文本框(`object_hint: c(ball_2)`)。

规格:figure* 全宽 18 cm、目标高 7-8 cm、最小字号 ≥7 pt、输出矢量 PDF
(嵌入照片 300 dpi)。英文标签用 08-24 会话给的对照表。
