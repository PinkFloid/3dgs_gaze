# 一键重建:原始照片 -> 米制 3DGS 地图 + tags_world + SAM 分割(无人值守 ~2h)。
# 人只做两头:拍照(死亡清单见 PIPELINE/TESTPLAN)与收尾(黏连拆分/命名/verify)。
#
#   powershell -File tools\run_map_pipeline.ps1 -Raw E:\Grasp\data\<照片目录> -Out E:\Grasp\data\lab_colmap_v5
#   可选:-SquareSize 0.032  -TagIds "74-139"  -TagSizes "74-89:0.24,90-119:0.1033,120-139:0.2493"
#
# 各步门槛(跑完在日志里人工核对,基线=v4):
#   COLMAP  registered = 全部、误差 ~1.4px、Cameras=1(>1 见 PIPELINE 已知坑)
#   align   板 RMS 亚毫米、相机 Z=拍摄高度、scale 打印在日志
#   survey  每 tag fit_rms 毫米级、实测/预期 ~0.99
#   train   dataparser_transforms.json 的 scale 必须 1.0
#   export  extent ~ 实验室尺寸(7x9x3m)
#   SAM     render_check.jpg 三联严丝合缝;黏连用 split_instance.py 拆
param(
    [Parameter(Mandatory = $true)][string]$Raw,
    [Parameter(Mandatory = $true)][string]$Out,
    [double]$SquareSize = 0.032,
    [string]$TagIds = "74-139",
    [string]$TagSizes = "74-89:0.24,90-119:0.1033,120-139:0.2493"
)
# 注意:不能用 $ErrorActionPreference=Stop —— PS 5.1 下它会把原生程序的
# stderr(COLMAP/ns-train 的进度条)当错误抛;一律显式查 $LASTEXITCODE。
$ErrorActionPreference = "Continue"
$e = "C:\Users\18488\anaconda3\envs\nerfstudio"
$py = "$e\python.exe"
$tools = "E:\Grasp\tools"
$env:PYTHONUTF8 = "1"
$env:TORCH_EXTENSIONS_DIR = "E:\Grasp\torch_extensions"   # gsplat JIT 三件套
$env:CUDA_HOME = $e
$env:CUDA_PATH = $e
$env:PATH = "$e\bin;$e\Library\bin;$e\Scripts;" +
    "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.38.33130\bin\Hostx64\x64;" +
    $env:PATH
$upright = "${Raw}_upright"
$name = Split-Path $Out -Leaf
$log = Join-Path $Out "pipeline.log"
New-Item -ItemType Directory -Force $Out | Out-Null

function Step([string]$title, [scriptblock]$body) {
    $t0 = Get-Date
    "=== $title  [$($t0.ToString('HH:mm:ss'))] ===" | Tee-Object -Append $log
    & $body 2>&1 | Tee-Object -Append $log
    if ($LASTEXITCODE -ne 0) { throw "$title 失败(exit $LASTEXITCODE),看 $log" }
    "--- $title 完成,耗时 $([int]((Get-Date) - $t0).TotalMinutes) min ---" | Tee-Object -Append $log
}

Step "1/7 make_upright" { & $py $tools\make_upright.py $Raw $upright }
Step "2/7 COLMAP(固定内参)" {
    powershell -File $tools\run_colmap_fixed_intrinsics.ps1 -Images $upright -Out $Out
}
Step "3/7 ns-process-data" {
    & "$e\Scripts\ns-process-data.exe" images --data $upright --output-dir $Out `
        --skip-colmap --colmap-cmd C:\tmp\colmap_nerfstudio_compat.bat
}
Step "4/7 align_to_charuco" {
    & $py $tools\align_to_charuco.py --dataset $Out `
        --marker-id-start 0 --square-size $SquareSize --marker-size 0.024
}
Step "5/7 survey tags" {
    & $py $tools\survey_aruco_tags.py --dataset $Out --tag-ids $TagIds --tag-sizes $TagSizes
}
Step "6/7 ns-train splatfacto(30k)" {
    & "$e\Scripts\ns-train.exe" splatfacto --data $Out\transforms_aligned.json `
        --output-dir E:\Grasp\outputs --viewer.quit-on-train-completion True `
        nerfstudio-data --orientation-method none --center-method none --auto-scale-poses False
}
$run = Get-ChildItem "E:\Grasp\outputs\$name\splatfacto" -Directory |
    Sort-Object Name | Select-Object -Last 1
$ckpt = Get-ChildItem "$($run.FullName)\nerfstudio_models\*.ckpt" | Sort-Object Name | Select-Object -Last 1
"训练 run:$($run.FullName)" | Tee-Object -Append $log
Step "7/7 export splat + SAM 分割" {
    & $py $tools\export_splat_from_ckpt.py --ckpt $ckpt.FullName --out $Out\splat.ply
    if ($LASTEXITCODE -ne 0) { throw "export 失败" }
    powershell -File $PSScriptRoot\run_lift_sam.ps1 --dataset $Out --ckpt $ckpt.FullName `
        --every 3 --points-per-batch 32 --crop-points-downscale 2 --previews 12
}
@"

全部跑完。剩下的人工步骤(顺序):
 1. 核对日志里各步门槛(本文件头部清单;日志 $log)
 2. 分割黏连的话:python tools\split_instance.py --seg-dir lab_result\segmentation_sam --id <坨> [--dry-run]
 3. 翻 lab_result\segmentation_sam\preview\ 命名 names.json(每球不同名)
 4. world_size 换新:copy $Out\tags_world.json、transforms_aligned.json -> SceneRebuild\world_size\
 5. brain --frame 升版(Intension/brain.py 默认值),狗端同学对齐同一版本 + 发新 tags_world.json
 6. 拷 Linux 四样:$($run.FullName) / tags_world / transforms_aligned / segmentation_sam
 7. 戴眼镜录视频1 质检(overlay 名字全对才算过)
"@ | Tee-Object -Append $log
