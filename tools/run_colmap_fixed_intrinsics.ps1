# COLMAP with calibrated intrinsics for the lab_upright dataset.
#
# Intrinsics come from calibration_results\phone_camera_calibration.npz
# (2026-07-04 recalibration, CALIB_FIX_K3, RMS 0.69 px). The npz already
# stores the upright (90 deg CW) values under key `upright_90cw`:
#   fx=4024.8166 fy=4000.6211 cx=2159.5867 cy=2814.4041
#   k1=0.16665225 k2=-0.14699350 p1=-0.00598953 p2=0.00152398
# Focal length and principal point are locked; distortion starts at the
# calibrated values and the mapper may refine it (set ba_refine_extra_params
# to 0 to lock distortion too).
$ErrorActionPreference = "Stop"

$colmap = "E:\Grasp\tools\COLMAP\COLMAP.bat"
$images = "E:\Grasp\data\lab_upright"
$out    = "E:\Grasp\data\lab_colmap"
$db     = "$out\colmap\database.db"

New-Item -ItemType Directory -Force "$out\colmap\sparse" | Out-Null

& $colmap feature_extractor `
    --database_path $db `
    --image_path $images `
    --ImageReader.single_camera 1 `
    --ImageReader.camera_model OPENCV `
    --ImageReader.camera_params "4024.8166,4000.6211,2159.5867,2814.4041,0.16665225,-0.14699350,-0.00598953,0.00152398"
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
