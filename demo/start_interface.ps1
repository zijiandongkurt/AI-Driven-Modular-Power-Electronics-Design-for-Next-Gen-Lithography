# ======================================================================
#  start_interface.ps1
#  一键加载 venv 环境并启动 Gradio Web 界面（PowerShell 版本）
#  用法：在 PowerShell 中执行  .\start_interface.ps1
#  如遇执行策略限制，先运行：
#      Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
# ======================================================================

# ── 配置区 ─────────────────────────────────────────────────────────────
$VenvPy   = "D:\Document\Course\Team_intership\LLM\.venv-gpu\Scripts\python.exe"
$ModelDir = "D:\Document\Course\Team_intership\LLM\models\qwen25-coder-7b"
$ScriptDir = $PSScriptRoot
$Entry    = "demo_interface.py"
# ──────────────────────────────────────────────────────────────────────

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host " LLM Topology Generation - Web Interface" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host (" Python: {0}" -f $VenvPy)
Write-Host (" Model : {0}" -f $ModelDir)
Write-Host (" Script: {0}\{1}" -f $ScriptDir, $Entry)
Write-Host "======================================================================" -ForegroundColor Cyan

if (-not (Test-Path $VenvPy)) {
    Write-Host "[ERROR] venv Python not found: $VenvPy" -ForegroundColor Red
    Write-Host "        Please create the venv first or fix the `$VenvPy path above."
    Read-Host "Press Enter to exit"
    exit 1
}
if (-not (Test-Path $ModelDir)) {
    Write-Host "[WARN ] Model directory not found: $ModelDir" -ForegroundColor Yellow
    Write-Host "        Continuing anyway - the script may fail on first request."
}
$EntryPath = Join-Path $ScriptDir $Entry
if (-not (Test-Path $EntryPath)) {
    Write-Host "[ERROR] Entry script not found: $EntryPath" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

Set-Location $ScriptDir
Write-Host ""
Write-Host "Launching Gradio... (Ctrl+C to stop)" -ForegroundColor Green
Write-Host "Browser will open automatically at http://127.0.0.1:7860" -ForegroundColor Green
Write-Host ""

& $VenvPy $Entry

Write-Host ""
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host " Server stopped." -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan
Read-Host "Press Enter to exit"
