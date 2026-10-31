<#
.SYNOPSIS
    ghoul-cad 一键部署：把全部运行时装进当前文件夹（免安装、可整体搬走）。

.DESCRIPTION
    仓库里只有源码（约 6 MB）。本脚本负责下载并安置 6 个外部组件，
    全部落在仓库目录内部，装完即可双击「启动 Ghoo.bat」：

        python\          Python 3.13 运行时 + requirements.txt 里的依赖
        tools\node\      Node.js 22（Homotope 矢量化引擎要用）
        tools\homotope\  @homotope/engine（npm 安装）
        tools\libredwg\  LibreDWG 0.14（DXF -> DWG 离线转换）
        ollama\          Ollama 运行时 + 两个模型（约 10 GB，可用 -SkipModels 跳过）
        LibreCAD\        LibreCAD 2.2.2（打开/校验图纸，可选）

    下载源是「国内镜像优先、官方兜底」，每个地址失败会自动换下一个；
    中断的下载支持断点续传，重跑即可接着下。

.PARAMETER Force
    已存在的组件也重新下载 / 重装。

.PARAMETER SkipModels
    跳过约 10 GB 的模型下载（qwen2.5-coder:7b / qwen2.5vl:7b）。

.PARAMETER SkipLibreCAD
    跳过 LibreCAD（不装也能生成 DXF，只是不能自动打开预览）。

.PARAMETER Only
    只处理指定组件，逗号分隔：python,node,homotope,libredwg,ollama,models,librecad

.PARAMETER PypiMirror
    pip 镜像地址。默认依次尝试清华源、阿里源、官方源。

.PARAMETER OfficialFirst
    先试官方源再试镜像（海外网络或镜像不同步时用）。

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\setup_runtime.ps1

.EXAMPLE
    # 只装 Python 依赖，先看界面能不能开
    powershell -ExecutionPolicy Bypass -File scripts\setup_runtime.ps1 -Only python

.EXAMPLE
    # 跳过 10 GB 模型
    powershell -ExecutionPolicy Bypass -File scripts\setup_runtime.ps1 -SkipModels
#>

[CmdletBinding()]
param(
    [switch]$Force,
    [switch]$SkipModels,
    [switch]$SkipLibreCAD,
    [string[]]$Only = @(),
    [string]$PypiMirror = "",
    [switch]$OfficialFirst
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

# 让中文在管道 / 重定向下也不乱码（Windows PowerShell 5.1 默认按 GBK 解码）
try {
    [Console]::OutputEncoding = [Text.Encoding]::UTF8
    $OutputEncoding = [Text.Encoding]::UTF8
} catch { }

# ---------------------------------------------------------------- 路径与常量
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Repo      = Split-Path -Parent $ScriptDir
$PyDir     = Join-Path $Repo "python"
$NodeDir   = Join-Path $Repo "tools\node"
$HtDir     = Join-Path $Repo "tools\homotope"
$LdDir     = Join-Path $Repo "tools\libredwg"
$OllamaDir = Join-Path $Repo "ollama"
$LcDir     = Join-Path $Repo "LibreCAD"
$DlDir     = Join-Path $Repo "DOWNLOADS"

#: 版本号与作者实测通过的组合一致，改动前先跑 app\_tests
$PY_VER   = "3.13.14"
$NODE_VER = "22.22.2"
$OLLAMA   = "v0.40.10"
$LIBREDWG = "0.14"
$LIBRECAD_FILE = "LibreCAD-2.2.2_alpha1-839-gd66b7e137-win-x64.exe"

$REQ = Join-Path $Repo "requirements.txt"

$Fail = @()
$Done = @()

# ---------------------------------------------------------------- 工具函数
function Say($msg, $color = "White") {
    Write-Host $msg -ForegroundColor $color
}

function Step($n, $total, $title) {
    Write-Host ""
    Write-Host ("[" + $n + "/" + $total + "] " + $title) -ForegroundColor Cyan
}

#: -Only 既支持 "node,homotope" 也支持 node,homotope（PowerShell 会把它变成数组）
function Want($name) {
    if (-not $Only -or @($Only).Count -eq 0) { return $true }
    $list = @()
    foreach ($o in @($Only)) {
        foreach ($p in ("$o" -split "[,\s]+")) { if ($p) { $list += $p.ToLower() } }
    }
    return ($list -contains $name.ToLower())
}

function Get-Curl {
    $c = Get-Command "curl.exe" -ErrorAction SilentlyContinue
    if ($c) { return $c.Source }
    return $null
}

#: 取远端文件大小（HEAD）；拿不到返回 0
function Get-RemoteSize($url) {
    $curl = Get-Curl
    if (-not $curl) { return 0 }
    $len = 0
    try {
        $h = & $curl -sIL --connect-timeout 15 --max-time 40 $url 2>$null
        foreach ($line in $h) {
            if ($line -match "^\s*content-length:\s*(\d+)\s*$") { $len = [int64]$Matches[1] }
        }
    } catch { $len = 0 }
    return $len
}

#: 从 curl 输出里挑出真正的错误行（进度条之类的噪声丢掉）
function Pick-CurlError($lines) {
    if (-not $lines) { return "未知错误" }
    foreach ($l in $lines) {
        $s = "$l"
        if ($s -match "curl:\s*\(\d+\)\s*(.+)$") { return $Matches[1].Trim() }
    }
    foreach ($l in $lines) {
        $s = "$l".Trim()
        if ($s -and $s -notmatch "^%|^\d|^Dload|^Current") { return $s }
    }
    return "未知错误"
}

#: 多源 + 双网络模式 + 断点续传的下载。
#:   $urls 按优先级排列；GitHub 链接会自动套镜像前缀。
#:   注意 $urls 必须标注 [string[]]：PowerShell 在命令模式下会把未标注类型的
#:   数组参数用空格拼成一个字符串，curl 就会收到 "url1 url2 url3" 并报
#:   "URL rejected: Malformed input to a URL function"。
function Download([string[]]$urls, [string]$dest, [string]$what) {
    $curl = Get-Curl
    if (-not $curl) { throw "系统里没有 curl.exe（Windows 10 1803+ 自带），无法下载" }

    # PowerShell 命令模式下数组参数偶尔会被包成嵌套数组（@( @(a,b) )），
    # 那样一次 curl 会收到好几个 URL → "URL rejected: Malformed input"。
    # 入口处无条件压平，后面就只会按"一个地址一次"来。
    $flat = @()
    foreach ($x in @($urls)) { foreach ($y in @($x)) { if ($y) { $flat += [string]$y } } }
    $urls = $flat
    if ($urls.Count -eq 0) { throw "没有可用的下载地址" }
    if ($env:J022_DEBUG_ARGS -eq "1") {
        Say ("      [dbg] urls=" + $urls.Count + " types=" + (@($urls | ForEach-Object { $_.GetType().Name }) -join ",") + " first=" + $urls[0]) "DarkGray"
    }

    # 已经下完就不再重下（比对远端大小，避免半截文件被当成完成）
    if ((Test-Path $dest) -and -not $Force) {
        $local = (Get-Item $dest).Length
        $remote = Get-RemoteSize $urls[0]
        if ($remote -gt 0 -and $local -eq $remote) {
            Say ("    已下载过，跳过（" + [math]::Round($local / 1MB, 1) + " MB）") "Green"
            return $dest
        }
    }

    $dir = Split-Path -Parent $dest
    if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Path $dir -Force | Out-Null }

    $modes = @(
        @{ name = "默认网络"; extra = @() },
        @{ name = "直连";     extra = @("--noproxy", "*") }
    )

    foreach ($raw in $urls) {
        $cands = @()
        if ($raw -match "^https://github\.com/") {
            $cands += ("https://ghfast.top/" + $raw)
            $cands += ("https://gh-proxy.com/" + $raw)
            if ($OfficialFirst) { $cands = @($raw) + $cands } else { $cands += $raw }
        } else {
            $cands += $raw
        }
        foreach ($u in $cands) {
            $label = $u
            try { $label = ([Uri]$u).Host } catch { }
            foreach ($mode in $modes) {
                for ($attempt = 1; $attempt -le 2; $attempt++) {
                    Say ("    下载 " + $what + " ← " + $label + "（" + $mode.name + "，第 " + $attempt + " 次）") "DarkGray"
                    $ca = @("-L", "--fail", "--silent", "--show-error",
                            "--connect-timeout", "20", "--max-time", "7200",
                            "--continue-at", "-", "-o", $dest)
                    # 逐项追加：空数组直接展开会变成 $null，curl 会收到一个空参数并报
                    # "URL rejected: Malformed input to a URL function"
                    foreach ($e in $mode.extra) { if ($e) { $ca += $e } }
                    $ca += [string]$u
                    if ($env:J022_DEBUG_ARGS -eq "1") { Say ("      [curl] " + ($ca -join " ")) "DarkGray" }
                    # 原生命令往 stderr 写东西时，Stop 模式会抛 NativeCommandError，
                    # 得临时放宽，否则一次失败就把整个步骤炸掉、连重试都走不到
                    $prevEap = $ErrorActionPreference
                    $ErrorActionPreference = "Continue"
                    try {
                        $out = & $curl @ca 2>&1
                        $rc = $LASTEXITCODE
                    } finally {
                        $ErrorActionPreference = $prevEap
                    }
                    if ($rc -eq 0 -and (Test-Path $dest) -and ((Get-Item $dest).Length -gt 0)) {
                        Say ("      完成：" + [math]::Round((Get-Item $dest).Length / 1MB, 1) + " MB") "Green"
                        return $dest
                    }
                    $err = Pick-CurlError $out
                    Say ("      失败：" + $err) "DarkGray"
                    # 404 / 403 是地址本身不对，重试没意义，直接换下一个源
                    if ($rc -eq 22 -and ($err -match "404|403|410")) { break }
                    Start-Sleep -Seconds 1
                }
            }
        }
        Say "    这个源不通，换下一个…" "Yellow"
    }
    throw ("下载失败（已下载的部分会保留，重跑可续传）：" + $urls[0])
}

function Unzip($zip, $dest) {
    if (Test-Path $dest) { Remove-Item $dest -Recurse -Force }
    New-Item -ItemType Directory -Path $dest -Force | Out-Null
    Expand-Archive -LiteralPath $zip -DestinationPath $dest -Force
}

function Find-In($root, $pattern) {
    return Get-ChildItem -LiteralPath $root -Recurse -File -Filter $pattern -ErrorAction SilentlyContinue |
           Select-Object -First 1
}

#: pip 安装：按镜像顺序逐个试，最后兜底官方源
#:   顺序是「阿里 → 清华 → 官方」：阿里对新 wheel 的同步最快，
#:   清华偶尔缺某个 cpXXX wheel（踩过：numpy 2.5.4 的 cp313 就缺）。
function Install-Pip($pyExe) {
    $mirrors = @()
    if ($PypiMirror) { $mirrors += $PypiMirror }
    $mirrors += "https://mirrors.aliyun.com/pypi/simple/"
    $mirrors += "https://pypi.tuna.tsinghua.edu.cn/simple"
    $mirrors += ""
    foreach ($m in $mirrors) {
        $label = "官方 PyPI"
        if ($m) { $label = $m }
        $a = @("-m", "pip", "install", "--no-warn-script-location", "--disable-pip-version-check", "-r", $REQ)
        if ($m) { $a += @("-i", $m) }
        Say ("    使用源：" + $label) "DarkGray"
        # pip 往 stderr 写日志时，Stop 模式会抛 NativeCommandError 直接把循环炸掉，
        # 得临时放宽，否则一个源失败就再也试不到下一个
        $prevEap = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        & $pyExe @a
        $rc = $LASTEXITCODE
        $ErrorActionPreference = $prevEap
        if ($rc -eq 0) { return $true }
        Say "    这个源不行，换下一个…" "Yellow"
    }
    return $false
}

# ---------------------------------------------------------------- 0. 前置检查
Say "============================================================" "Cyan"
Say "  ghoul-cad · 运行时一键部署" "Cyan"
Say "============================================================" "Cyan"
Say ("  仓库位置 : " + $Repo)
Say ("  下载缓存 : " + $DlDir)

if (-not (Test-Path (Join-Path $Repo "app\launch.py"))) {
    Say "[错误] 这个脚本要放在仓库的 scripts\ 目录下运行。" "Red"
    exit 1
}
if ($Repo -match "\s") {
    Say "[警告] 仓库路径里有空格。LibreCAD 的静默安装（NSIS /D=）不支持带空格的路径。" "Yellow"
    Say "       建议移到无空格路径（如 D:\ghoul-cad）再装 LibreCAD。" "Yellow"
}
if (-not (Test-Path $DlDir)) { New-Item -ItemType Directory -Path $DlDir -Force | Out-Null }

# ---------------------------------------------------------------- 1. Python
if (Want "python") {
    Step 1 6 ("Python " + $PY_VER + " 运行时")
    $pyExe = Join-Path $PyDir "python.exe"
    if ((Test-Path $pyExe) -and -not $Force) {
        Say ("    已存在，跳过：" + $pyExe) "Green"
        $Done += "python(已存在)"
    } else {
        $inst = Join-Path $DlDir ("python-" + $PY_VER + "-amd64.exe")
        # 每个元素都要单独加括号：PowerShell 里逗号的优先级高于 +，
        # 写成 @("a" + $v, "b" + $v) 会被拼成一个字符串而不是两个元素
        $urls = @(
            ("https://mirrors.huaweicloud.com/python/" + $PY_VER + "/python-" + $PY_VER + "-amd64.exe"),
            ("https://mirrors.aliyun.com/python-release/windows/python-" + $PY_VER + "-amd64.exe"),
            ("https://www.python.org/ftp/python/" + $PY_VER + "/python-" + $PY_VER + "-amd64.exe")
        )
        if ($OfficialFirst) { $urls = @($urls[2], $urls[0], $urls[1]) }
        $ok = $false
        try {
            Download $urls $inst ("Python " + $PY_VER)
            Say "    静默安装到 python\ …" "DarkGray"
            # TargetDir 用引号包住，路径万一有空格也不至于装错位置
            $instArgs = @(
                "/quiet", "InstallAllUsers=0",
                ('TargetDir="' + $PyDir + '"'),
                "PrependPath=0", "Include_pip=1", "Include_test=0",
                "Include_launcher=0", "Include_doc=0", "Shortcuts=0"
            )
            $p = Start-Process -FilePath $inst -ArgumentList $instArgs -Wait -PassThru
            if (-not (Test-Path $pyExe)) { throw ("安装器返回 " + $p.ExitCode + "，但 python.exe 不在预期位置") }
            Say ("    OK  " + (& $pyExe -V 2>&1)) "Green"
            $Done += "python"
            $ok = $true
        } catch {
            Say ("    [失败] " + $_.Exception.Message) "Red"
        }
        if (-not $ok) {
            Say "    改用官方 Embeddable 包 + get-pip 的备用方案…" "Yellow"
            try {
                $zip = Join-Path $DlDir ("python-" + $PY_VER + "-embed-amd64.zip")
                $eurls = @(
                    ("https://mirrors.huaweicloud.com/python/" + $PY_VER + "/python-" + $PY_VER + "-embed-amd64.zip"),
                    ("https://www.python.org/ftp/python/" + $PY_VER + "/python-" + $PY_VER + "-embed-amd64.zip")
                )
                Download $eurls $zip "Python embeddable"
                Unzip $zip $PyDir
                # 解开 import site，否则 pip 装的包对解释器不可见
                $pth = Get-ChildItem $PyDir -Filter "python*._pth" | Select-Object -First 1
                if ($pth) {
                    $txt = Get-Content $pth.FullName
                    $txt = $txt -replace "^#\s*import site", "import site"
                    if ($txt -notcontains "import site") { $txt += "import site" }
                    Set-Content -Path $pth.FullName -Value $txt -Encoding ASCII
                }
                $gp = Join-Path $DlDir "get-pip.py"
                $gpUrls = @("https://bootstrap.pypa.io/get-pip.py", "https://mirrors.aliyun.com/pypi/get-pip.py")
                Download $gpUrls $gp "get-pip"
                & $pyExe $gp --no-warn-script-location | Out-Null
                Say ("    OK  " + (& $pyExe -V 2>&1) + "（embeddable）") "Green"
                $Done += "python(embeddable)"
            } catch {
                Say ("    [失败] 备用方案也不行：" + $_.Exception.Message) "Red"
                $Fail += "python"
            }
        }
    }

    # 依赖
    $pyExe = Join-Path $PyDir "python.exe"
    if (Test-Path $pyExe) {
        Say "    安装 requirements.txt …" "DarkGray"
        if (Install-Pip $pyExe) {
            & $pyExe -c "import numpy, cv2, PIL, ezdxf; print('    OK  numpy', numpy.__version__, '| cv2', cv2.__version__)"
            $Done += "python-deps"
        } else {
            Say "    [失败] 依赖没装全，可手动：python\python.exe -m pip install -r requirements.txt" "Red"
            $Fail += "python-deps"
        }
    }
}

# ---------------------------------------------------------------- 2. Node
if (Want "node") {
    Step 2 6 ("Node.js " + $NODE_VER + "（Homotope 引擎需要 ≥22）")
    $nodeExe = Join-Path $NodeDir "node.exe"
    if ((Test-Path $nodeExe) -and -not $Force) {
        Say ("    已存在，跳过：" + (& $nodeExe --version 2>&1)) "Green"
        $Done += "node(已存在)"
    } else {
        try {
            $zip = Join-Path $DlDir ("node-v" + $NODE_VER + "-win-x64.zip")
            $urls = @(
                ("https://registry.npmmirror.com/-/binary/node/v" + $NODE_VER + "/node-v" + $NODE_VER + "-win-x64.zip"),
                ("https://mirrors.huaweicloud.com/nodejs/v" + $NODE_VER + "/node-v" + $NODE_VER + "-win-x64.zip"),
                ("https://nodejs.org/dist/v" + $NODE_VER + "/node-v" + $NODE_VER + "-win-x64.zip")
            )
            if ($OfficialFirst) { $urls = @($urls[2], $urls[0], $urls[1]) }
            Download $urls $zip ("Node.js " + $NODE_VER)
            $tmp = Join-Path $DlDir "_node_tmp"
            Unzip $zip $tmp
            $inner = Get-ChildItem $tmp -Directory | Select-Object -First 1
            if (-not $inner) { throw "zip 里没有找到解压目录" }
            if (Test-Path $NodeDir) { Remove-Item $NodeDir -Recurse -Force }
            New-Item -ItemType Directory -Path (Split-Path -Parent $NodeDir) -Force | Out-Null
            Move-Item $inner.FullName $NodeDir
            Remove-Item $tmp -Recurse -Force -ErrorAction SilentlyContinue
            Say ("    OK  " + (& $nodeExe --version 2>&1)) "Green"
            $Done += "node"
        } catch {
            Say ("    [失败] " + $_.Exception.Message) "Red"
            $Fail += "node"
        }
    }
}

# ---------------------------------------------------------------- 3. Homotope
if (Want "homotope") {
    Step 3 6 "Homotope 矢量化引擎（npm）"
    $npmCmd = Join-Path $NodeDir "npm.cmd"
    $engDir = Join-Path $HtDir "node_modules\@homotope\engine"
    if (-not (Test-Path $npmCmd)) {
        Say "    先装 Node 才能装它，跳过。" "Yellow"
        $Fail += "homotope(缺 Node)"
    } elseif ((Test-Path $engDir) -and -not $Force) {
        Say "    已存在，跳过。" "Green"
        $Done += "homotope(已存在)"
    } else {
        try {
            Push-Location $HtDir
            $env:Path = $NodeDir + ";" + $env:Path
            & $npmCmd install --no-audit --no-fund
            $rc = $LASTEXITCODE
            Pop-Location
            if ($rc -ne 0) { throw ("npm 返回 " + $rc) }
            if (-not (Test-Path $engDir)) { throw "npm 装完但找不到 @homotope/engine" }
            Say "    OK  @homotope/engine 已就绪" "Green"
            $Done += "homotope"
        } catch {
            Pop-Location -ErrorAction SilentlyContinue
            Say ("    [失败] " + $_.Exception.Message) "Red"
            $Fail += "homotope"
        }
    }
}

# ---------------------------------------------------------------- 4. LibreDWG
if (Want "libredwg") {
    Step 4 6 ("LibreDWG " + $LIBREDWG + "（DXF → DWG 离线转换）")
    $dxf2dwg = Join-Path $LdDir "dxf2dwg.exe"
    if ((Test-Path $dxf2dwg) -and -not $Force) {
        Say "    已存在，跳过。" "Green"
        $Done += "libredwg(已存在)"
    } else {
        try {
            $zip = Join-Path $DlDir ("libredwg-" + $LIBREDWG + "-win64.zip")
            $urls = @("https://github.com/LibreDWG/libredwg/releases/download/" + $LIBREDWG + "/libredwg-" + $LIBREDWG + "-win64.zip")
            Download $urls $zip ("LibreDWG " + $LIBREDWG)
            $tmp = Join-Path $DlDir "_ld_tmp"
            Unzip $zip $tmp
            New-Item -ItemType Directory -Path $LdDir -Force | Out-Null
            foreach ($n in @("dxf2dwg.exe", "dwg2dxf.exe", "libredwg-0.dll", "libiconv-2.dll")) {
                $f = Find-In $tmp $n
                if ($f) { Copy-Item $f.FullName (Join-Path $LdDir $n) -Force }
            }
            foreach ($n in @("README-libredwg.txt", "NOTICE.txt", "COPYING")) {
                $f = Find-In $tmp $n
                if ($f) { Copy-Item $f.FullName (Join-Path $LdDir $n) -Force }
            }
            Remove-Item $tmp -Recurse -Force -ErrorAction SilentlyContinue
            if (-not (Test-Path $dxf2dwg)) { throw "解压后没找到 dxf2dwg.exe" }
            Say "    OK" "Green"
            $Done += "libredwg"
        } catch {
            Say ("    [失败] " + $_.Exception.Message) "Red"
            Say "    没有它也能出 DXF，只是不能导出 DWG。" "Yellow"
            $Fail += "libredwg"
        }
    }
}

# ---------------------------------------------------------------- 5. Ollama
if (Want "ollama") {
    Step 5 6 ("Ollama " + $OLLAMA + " 运行时")
    $olExe = Join-Path $OllamaDir "ollama.exe"
    if ((Test-Path $olExe) -and -not $Force) {
        Say "    已存在，跳过。" "Green"
        $Done += "ollama(已存在)"
    } else {
        try {
            Say "    这个包约 1.6 GB（含 CPU/GPU 全套后端），要一会儿。" "Yellow"
            $zip = Join-Path $DlDir "ollama-windows-amd64.zip"
            $urls = @("https://github.com/ollama/ollama/releases/download/" + $OLLAMA + "/ollama-windows-amd64.zip")
            Download $urls $zip "Ollama 运行时"
            $tmp = Join-Path $DlDir "_ol_tmp"
            Unzip $zip $tmp
            if (Test-Path $OllamaDir) { Remove-Item $OllamaDir -Recurse -Force }
            New-Item -ItemType Directory -Path $OllamaDir -Force | Out-Null
            $inner = Get-ChildItem $tmp -Directory | Select-Object -First 1
            $copied = $false
            if ($inner) {
                if (Find-In $inner.FullName "ollama.exe") {
                    Copy-Item (Join-Path $inner.FullName "*") $OllamaDir -Recurse -Force
                    $copied = $true
                }
            }
            if (-not $copied) { Copy-Item (Join-Path $tmp "*") $OllamaDir -Recurse -Force }
            Remove-Item $tmp -Recurse -Force -ErrorAction SilentlyContinue
            if (-not (Test-Path $olExe)) { throw "解压后没找到 ollama.exe" }
            New-Item -ItemType Directory -Path (Join-Path $OllamaDir "models") -Force | Out-Null
            Say "    OK" "Green"
            $Done += "ollama"
        } catch {
            Say ("    [失败] " + $_.Exception.Message) "Red"
            $Fail += "ollama"
        }
    }
}

# ---------------------------------------------------------------- 6. 模型
if (-not $SkipModels -and (Want "models")) {
    Step "5b" 6 "Ollama 模型（约 10 GB，可 -SkipModels 跳过）"
    $olExe = Join-Path $OllamaDir "ollama.exe"
    if (-not (Test-Path $olExe)) {
        Say "    没有 ollama.exe，跳过。" "Yellow"
        $Fail += "models(缺 Ollama)"
    } else {
        $env:OLLAMA_MODELS = Join-Path $OllamaDir "models"
        $env:OLLAMA_HOST = "127.0.0.1:11434"
        $serve = $null
        try {
            $up = $false
            try {
                $r = Invoke-WebRequest -Uri "http://127.0.0.1:11434/api/tags" -UseBasicParsing -TimeoutSec 3
                if ($r.StatusCode -eq 200) { $up = $true }
            } catch { $up = $false }
            if (-not $up) {
                Say "    启动本地推理服务…" "DarkGray"
                $serve = Start-Process -FilePath $olExe -ArgumentList "serve" -PassThru -WindowStyle Hidden
                for ($i = 0; $i -lt 40; $i++) {
                    Start-Sleep -Milliseconds 800
                    try {
                        $r = Invoke-WebRequest -Uri "http://127.0.0.1:11434/api/tags" -UseBasicParsing -TimeoutSec 2
                        if ($r.StatusCode -eq 200) { break }
                    } catch { }
                }
            }
            foreach ($m in @("qwen2.5-coder:7b", "qwen2.5vl:7b")) {
                Say ("    拉取 " + $m + " …（约 5 GB，取决于网速）") "DarkGray"
                & $olExe pull $m
                if ($LASTEXITCODE -ne 0) { throw ("pull " + $m + " 失败") }
            }
            Say "    OK  两个模型已就位" "Green"
            $Done += "models"
        } catch {
            Say ("    [失败] " + $_.Exception.Message) "Red"
            Say "    可稍后手动执行：ollama\ollama.exe pull qwen2.5-coder:7b" "Yellow"
            $Fail += "models"
        } finally {
            if ($serve) {
                try { Stop-Process -Id $serve.Id -Force -ErrorAction SilentlyContinue } catch { }
            }
        }
    }
}

# ---------------------------------------------------------------- 7. LibreCAD
if (-not $SkipLibreCAD -and (Want "librecad")) {
    Step 6 6 "LibreCAD 2.2.2（打开图纸预览，可选）"
    $lcExe = Join-Path $LcDir "librecad.exe"
    if ((Test-Path $lcExe) -and -not $Force) {
        Say "    已存在，跳过。" "Green"
        $Done += "librecad(已存在)"
    } else {
        try {
            $exe = Join-Path $DlDir $LIBRECAD_FILE
            $urls = @("https://github.com/LibreCAD/LibreCAD/releases/download/continuous/" + $LIBRECAD_FILE)
            Download $urls $exe "LibreCAD 2.2.2"
            Say "    静默安装到 LibreCAD\ …（80 MB，约 1 分钟）" "DarkGray"
            # NSIS：/S 静默，/D= 必须是最后一个参数且不能加引号
            Start-Process -FilePath $exe -ArgumentList @("/S", ("/D=" + $LcDir)) -Wait | Out-Null
            if (-not (Test-Path $lcExe)) {
                $alt = Join-Path $LcDir "LibreCAD.exe"
                if (Test-Path $alt) { $lcExe = $alt }
            }
            if (-not (Test-Path $lcExe)) { throw "安装后没找到 librecad.exe（路径带空格会导致 NSIS /D 失败）" }
            Say "    OK" "Green"
            $Done += "librecad"
        } catch {
            Say ("    [失败] " + $_.Exception.Message) "Red"
            Say "    不装也能用：生成 DXF 后手动用任意 CAD 软件打开。" "Yellow"
            $Fail += "librecad"
        }
    }
}

# ---------------------------------------------------------------- 汇总
Say ""
Say "============================================================" "Cyan"
if ($Done.Count -gt 0) { Say ("  已就绪：" + ($Done -join "、")) "Green" }
if ($Fail.Count -gt 0) {
    Say ("  未完成：" + ($Fail -join "、")) "Red"
    Say "  可以只重跑失败的组件，例如：" "Yellow"
    Say "    powershell -ExecutionPolicy Bypass -File scripts\setup_runtime.ps1 -Only librecad -Force" "Yellow"
    Say "  已下载的部分保留在 DOWNLOADS\，重跑会续传，不会从头再来。" "DarkGray"
} else {
    Say "  全部完成。下一步：双击仓库根目录的「启动 Ghoo.bat」。" "Green"
}
Say "  自检：双击「诊断环境.bat」，会逐项打出路径与版本。" "DarkGray"
Say "============================================================" "Cyan"

if ($Fail.Count -gt 0) { exit 1 }
exit 0
