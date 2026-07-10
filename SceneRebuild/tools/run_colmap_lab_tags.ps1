# COLMAP for the new lab capture (with ArUco tags 0-29 in scene).
#
# fx fy cx cy from Calibration_result/phone_camera_calibration.npz
# (2026-07-04 recalibration: fx=4000.6211, fy=4024.8166, cx=2814.4041,
# cy=2123.4133 at raw landscape 5712x4284, rms 0.692px over 47 imgs),
# rotated 90 deg CW to match upright 4284x5712 images:
#   fx' = fy, fy' = fx, cx' = H-1-cy, cy' = cx
# Distortion starts at 0 and is refined by the mapper: the calibrated k1,k2
# pair is a see-saw fit (board never reached the frame corners) -- near-zero
# distortion inside the sampled region but unusable extrapolation outside it.
$ErrorActionPreference = "Stop"

$colmap = "E:\Grasp\tools\COLMAP\COLMAP.bat"
$images = "E:\Grasp\data\lab_tags_upright"   # <- EDIT: upright images of the new capture
$out    = "E:\Grasp\data\lab_tags_colmap"    # <- EDIT: output dir
$db     = "$out\colmap\database.db"

New-Item -ItemType Directory -Force "$out\colmap\sparse" | Out-Null

& $colmap feature_extractor `
    --database_path $db `
    --image_path $images `
    --ImageReader.single_camera 1 `
    --ImageReader.camera_model OPENCV `
    --ImageReader.camera_params "4024.8166,4000.6211,2159.5867,2814.4041,0,0,0,0"
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
