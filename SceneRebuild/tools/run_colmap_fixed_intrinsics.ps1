# COLMAP with calibrated intrinsics for the lab_upright dataset.
#
# Intrinsics come from calibration_results\phone_camera_calibration.npz
# (2026-07-24 recalibration, A3 board id0-43, CALIB_FIX_K3, RMS 1.01 px
# over 45/110 frames after per-image pruning). The npz already stores the
# upright (90 deg CW) values under key `upright_90cw`:
#   fx=4037.3160 fy=4037.9549 cx=2155.5857 cy=2873.8241
#   k1=0.18320791 k2=-0.49267304 p1=-0.00009465 p2=-0.00114164
# Focal length and principal point are locked; distortion starts at the
# calibrated values and the mapper may refine it (set ba_refine_extra_params
# to 0 to lock distortion too).
param(
    [string]$Images = "E:\Grasp\data\lab_upright",
    [string]$Out    = "E:\Grasp\data\lab_colmap"
)
$ErrorActionPreference = "Stop"

$colmap = "E:\Grasp\tools\COLMAP\COLMAP.bat"
$images = $Images
$out    = $Out
$db     = "$out\colmap\database.db"

New-Item -ItemType Directory -Force "$out\colmap\sparse" | Out-Null

& $colmap feature_extractor `
    --database_path $db `
    --image_path $images `
    --ImageReader.single_camera 1 `
    --ImageReader.camera_model OPENCV `
    --ImageReader.camera_params "4037.3160,4037.9549,2155.5857,2873.8241,0.18320791,-0.49267304,-0.00009465,-0.00114164"
if ($LASTEXITCODE -ne 0) { throw "feature_extractor failed" }

& $colmap exhaustive_matcher --database_path $db
if ($LASTEXITCODE -ne 0) { throw "exhaustive_matcher failed" }

& $colmap mapper `
    --database_path $db `
    --image_path $images `
    --output_path "$out\colmap\sparse" `
    --Mapper.ba_refine_focal_length 0 `
    --Mapper.ba_refine_principal_point 0 `
    --Mapper.ba_refine_extra_params 1
if ($LASTEXITCODE -ne 0) { throw "mapper failed" }

& $colmap model_analyzer --path "$out\colmap\sparse\0"
