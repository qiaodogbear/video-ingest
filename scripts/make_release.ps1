# 打一个可分享的干净压缩包。
#
# 关键点：排除 .venv / output / 缓存 / 测试临时目录，并**主动扫描**
# 打包结果里是否残留凭证或绝对路径，避免把本机信息或版权材料发出去。
#
# 用法：
#   pwsh -File .\scripts\make_release.ps1
#   pwsh -File .\scripts\make_release.ps1 -SkillDir "$HOME\.codex\skills\video-ingest"

[CmdletBinding()]
param(
    [string]$OutRoot = (Join-Path $PSScriptRoot "..\dist-share"),
    [string]$SkillDir = "$HOME\.codex\skills\video-ingest",
    [switch]$SkipZip
)

$ErrorActionPreference = "Stop"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Write-Host "仓库根目录: $repo"

$name = "video-research-agent"
$stage = Join-Path $OutRoot $name

# 需要排除的路径片段（相对仓库）
$excludeDirs = @(".venv", "output", ".hf-cache", "__pycache__", ".pytest_cache",
                 "dist-share", "dist", "build", ".git")
$excludeFilePatterns = @("*.pyc", "*.cookies", "cookies.txt", "*_cookies.txt",
                         "*.m4a", "*.mp4", "*.webm")

function Test-Excluded([string]$relPath) {
    foreach ($d in $excludeDirs) {
        if ($relPath -eq $d -or $relPath.StartsWith("$d\") -or $relPath -match "\\$([regex]::Escape($d))\\") {
            return $true
        }
    }
    if ($relPath -match "pytest-of-") { return $true }
    foreach ($p in $excludeFilePatterns) {
        if ((Split-Path $relPath -Leaf) -like $p) { return $true }
    }
    return $false
}

# ---------- 1. 复制仓库 ----------
if (Test-Path $stage) { Remove-Item $stage -Recurse -Force }
New-Item -ItemType Directory -Force -Path $stage | Out-Null

# 待分享的文本与代码文件
$includeFiles = @("README.md", "SOP.md", "AGENTS.md", "VERIFICATION.md",
                  "pyproject.toml", "requirements.lock.txt", ".gitignore",
                  "corrections.example.json")
foreach ($f in $includeFiles) {
    $src = Join-Path $repo $f
    if (Test-Path $src) { Copy-Item $src (Join-Path $stage $f) } else { Write-Warning "缺少 $f" }
}

foreach ($d in @("video_ingest", "tests", "scripts")) {
    $src = Join-Path $repo $d
    if (-not (Test-Path $src)) { Write-Warning "缺少目录 $d"; continue }
    Get-ChildItem $src -Recurse -File | ForEach-Object {
        $rel = $_.FullName.Substring($repo.Length + 1)
        if (Test-Excluded $rel) { return }
        $dest = Join-Path $stage $rel
        New-Item -ItemType Directory -Force (Split-Path $dest) | Out-Null
        Copy-Item $_.FullName $dest
    }
}

# ---------- 2. 附带 Skill ----------
if (Test-Path $SkillDir) {
    $dest = Join-Path $stage "skill\video-ingest"
    New-Item -ItemType Directory -Force $dest | Out-Null
    Get-ChildItem $SkillDir -Recurse -File | ForEach-Object {
        $rel = $_.FullName.Substring($SkillDir.Length + 1)
        $out = Join-Path $dest $rel
        New-Item -ItemType Directory -Force (Split-Path $out) | Out-Null
        Copy-Item $_.FullName $out
    }
    Write-Host "已附带 Skill: $SkillDir"
} else {
    Write-Warning "未找到 Skill 目录，跳过: $SkillDir"
}

# ---------- 3. 安全扫描 ----------
Write-Host "`n=== 打包前安全扫描 ==="
$problems = 0

# 3a. 凭证**值**特征。
# 只匹配"名字 + 分隔符 + 看起来像真实值"的形式，以及高熵长串。
# 单纯出现 SESSDATA / bili_jct 这些**名字**是正常的（脱敏模块与文档都要提到它们），
# 报出来只会让人学会忽略告警，反而降低这个扫描的价值。
$credPatterns = @(
    '(?i)(SESSDATA|bili_jct|DedeUserID|buvid3|access_token|refresh_token|api[_-]?key)\s*[=:]\s*[A-Za-z0-9%._\-]{12,}',
    '(?i)Authorization\s*:\s*(Bearer|Basic)\s+\S{12,}'
)
# 这些文件是"定义/文档化凭证名"的地方，属正当内容
$credAllow = @("redact.py", "cookies.md", "test_workflow.py", "test_acquire.py")
$hits = Get-ChildItem $stage -Recurse -File |
    Where-Object { $_.Length -lt 2MB -and $credAllow -notcontains $_.Name } |
    Select-String -Pattern $credPatterns -ErrorAction SilentlyContinue

# 高熵串检测。仅"够长"会把 snake_case 标识符全部误报，因此改看信息熵：
# 真实 token（SESSDATA 等）是大小写数字混排，熵接近 4-6 bit/字符；
# 自然语言标识符（deterministic_pipeline_verification_test）熵只有 3 出头。
function Get-Entropy([string]$s) {
    if ([string]::IsNullOrEmpty($s)) { return 0.0 }
    $counts = @{}
    foreach ($ch in $s.ToCharArray()) {
        $counts[$ch] = 1 + $(if ($counts.ContainsKey($ch)) { $counts[$ch] } else { 0 })
    }
    $len = $s.Length
    $e = 0.0
    foreach ($k in $counts.Keys) {
        $p = $counts[$k] / $len
        $e -= $p * [math]::Log($p, 2)
    }
    return $e
}

$ENTROPY_MIN = 3.6
$LEN_MIN = 28
$entropyHits = @()
Get-ChildItem $stage -Recurse -File -Include *.md, *.py, *.toml, *.txt, *.json, *.yaml |
    Where-Object { $_.Length -lt 2MB -and $credAllow -notcontains $_.Name } |
    ForEach-Object {
        $lineNo = 0
        foreach ($line in (Get-Content $_.FullName -ErrorAction SilentlyContinue)) {
            $lineNo++
            if ($line.Length -gt 400) { continue }   # 跳过被压缩/生成的超长行
            foreach ($m in [regex]::Matches($line, '[A-Za-z0-9_\-]{' + $LEN_MIN + ',}')) {
                $tok = $m.Value
                # sha256 之类的十六进制摘要长度固定，单独排除（它们不是凭证，
                # 且本工具会把文件哈希写进 manifest）
                if ($tok -match '^[0-9a-f]{40,64}$') { continue }
                # 关键判据：真实 token 是**大小写与数字混排**。
                # 只用熵不够——snake_case 标识符（test_xxx_yyy）因为有下划线，
                # 熵也会超过 3.6，会被大量误报。
                $hasUpper = $tok -cmatch '[A-Z]'
                $hasLower = $tok -cmatch '[a-z]'
                $hasDigit = $tok -match '[0-9]'
                $identLike = $tok -match '^[a-z][a-z0-9_]*$'      # 纯小写+下划线 → 标识符
                if ($identLike) { continue }
                # 模型/包缓存目录名（models--Org--name）也是大小写与连字符混排，
                # 但不是凭证。
                if ($tok -match '^models--' -or $tok -match 'faster[-_]whisper' -or
                    $tok -match '^(Systran|openai|pyannote)[-_.]') { continue }
                if (-not (($hasUpper -and $hasLower -and $hasDigit) -or
                          ($hasUpper -and $hasLower -and $tok -match '[_\-]'))) { continue }
                if ((Get-Entropy $tok) -ge $ENTROPY_MIN) {
                    $entropyHits += [pscustomobject]@{
                        File = $_.Name; Line = $lineNo; Token = $tok.Substring(0, [Math]::Min(40, $tok.Length))
                    }
                }
            }
        }
    }

if ($hits -or $entropyHits) {
    $problems++
    Write-Warning "发现疑似凭证值（需人工确认）："
    $hits | Select-Object -First 8 | ForEach-Object { "   名字+值: $($_.Filename):$($_.LineNumber)" }
    $entropyHits | Select-Object -First 8 | ForEach-Object { "   高熵串: $($_.File):$($_.Line)  $($_.Token)..." }
} else { Write-Host "[OK] 未发现凭证值" }

# 3b. 本机绝对路径残留。文档里的"路径示例"是正常的，
# 但真正的本机路径会让接收者跑不起来，因此只放行明确标注为示例的行。
$absAll = Get-ChildItem $stage -Recurse -File -Include *.md, *.py, *.toml, *.txt |
    Select-String -Pattern "C:\\Users\\|C:\\DSH\\" -ErrorAction SilentlyContinue
$abs = $absAll | Where-Object { $_.Line -notmatch "例如|示例|比如|以 .* 表示" }
if ($abs) {
    $problems++
    Write-Warning "发现本机绝对路径（接收者可能跑不起来）："
    $abs | Select-Object -First 15 | ForEach-Object { "   $($_.Filename):$($_.LineNumber)  $($_.Line.Trim())" }
} else { Write-Host "[OK] 未发现本机绝对路径" }

# 3c. 不应出现的体积材料
$media = Get-ChildItem $stage -Recurse -File -Include *.m4a, *.mp4, *.webm, *.png |
    Where-Object { $_.DirectoryName -notmatch "fixtures" }
if ($media) {
    $problems++
    Write-Warning "发现媒体文件（可能含版权内容）："
    $media | Select-Object -First 10 | ForEach-Object { "   $($_.FullName)" }
} else { Write-Host "[OK] 未发现多余媒体文件" }

# ---------- 4. 统计与压缩 ----------
$files = Get-ChildItem $stage -Recurse -File
$sizeMB = [math]::Round(($files | Measure-Object Length -Sum).Sum / 1MB, 2)
Write-Host "`n打包内容: $($files.Count) 个文件, $sizeMB MB"

if ($SkipZip) {
    Write-Host "已跳过压缩（-SkipZip）。目录: $stage"
} else {
    $zip = Join-Path $OutRoot "$name.zip"
    if (Test-Path $zip) { Remove-Item $zip -Force }
    Compress-Archive -Path $stage -DestinationPath $zip -CompressionLevel Optimal
    $zipMB = [math]::Round((Get-Item $zip).Length / 1MB, 2)
    Write-Host "已生成: $zip ($zipMB MB)"
}

Write-Host "`n提醒："
Write-Host "  - 分享前确认没有 cookie 文件与任务产物（output/）"
Write-Host "  - 压缩包里**不含** output/，接收者需自行安装依赖并重跑"
Write-Host "  - 接收者第一步：pip install -e `".[dev]`"（否则只能在本目录运行）"
if ($problems -gt 0) { exit 2 } else { exit 0 }
