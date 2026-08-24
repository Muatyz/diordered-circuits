# 在指定 conda 环境中运行 Python（修复 matplotlib DLL 冲突）
#
# 背景：conda activate 在部分终端（尤其是 VS Code 集成终端 + 未 init 的
# PowerShell）不生效。直接用 <env>\python.exe 时，Windows DLL 搜索会先命中
# base 环境的 Library\bin（D:\miniconda\Library\bin），导致 matplotlib 的
# C 扩展（ft2font/_path/_image）加载到版本不匹配的 freetype/libpng/zlib，
# 进程崩溃（0xC06D007F / STATUS_ENTRYPOINT_NOT_FOUND）。
#
# 本脚本把目标环境的 Library\bin 插到 PATH 最前面再执行 python，彻底避免
# 该冲突。用法：
#
#     powershell -ExecutionPolicy Bypass -File work\run_env.ps1 dev  script.py arg1
#     powershell -ExecutionPolicy Bypass -File work\run_env.ps1 pro   -c "import matplotlib..."
#
# 支持的 env：dev / pro / origin（可自行扩展）。

param(
    [Parameter(Mandatory = $true)]
    [string]$EnvName,

    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$PythonArgs
)

$envRoot = "D:\miniconda\envs\$EnvName"
$pythonPath = Join-Path $envRoot "python.exe"

if (-not (Test-Path $pythonPath)) {
    Write-Error "找不到环境 $EnvName 的 python：$pythonPath"
    exit 1
}

$dllBin = Join-Path $envRoot "Library\bin"
if (Test-Path $dllBin) {
    $env:Path = "$dllBin;" + $env:Path
    Write-Host "[run_env] prepend DLL path: $dllBin"
}

Write-Host "[run_env] python: $pythonPath"
& $pythonPath @PythonArgs
exit $LASTEXITCODE
