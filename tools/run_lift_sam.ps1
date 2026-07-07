# Env wrapper for lift_sam_instances.py on the Windows/4090 mapping machine.
# gsplat is JIT-compiled: it needs nvcc (conda env), the MSVC version the cache
# was built with, and TORCH_EXTENSIONS_DIR pointing at the prebuilt cache.
# Usage: powershell -File tools\run_lift_sam.ps1 [lift_sam_instances.py args...]
$e = 'C:\Users\18488\anaconda3\envs\nerfstudio'
$env:TORCH_EXTENSIONS_DIR = 'E:\Grasp\torch_extensions'
$env:CUDA_HOME = $e
$env:CUDA_PATH = $e
$env:PATH = "$e\bin;$e\Library\bin;$e\Scripts;" +
    "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.38.33130\bin\Hostx64\x64;" +
    $env:PATH
& "$e\python.exe" -u "$PSScriptRoot\lift_sam_instances.py" @args
