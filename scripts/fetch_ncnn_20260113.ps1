param(
  [string]$OutDir = "deps/ncnn-prebuilt",
  [string]$Proxy = "",
  [ValidateSet("x64","arm64")]
  [string]$Arch = "x64"
)

$ErrorActionPreference = "Stop"
$Tag = "20260113"

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$ZipPath = Join-Path $OutDir "ncnn-$Tag-windows-vs2022.zip"
$ExtractDir = Join-Path $OutDir "extracted"

$Url = "https://github.com/Tencent/ncnn/releases/download/$Tag/ncnn-$Tag-windows-vs2022.zip"
Write-Host "Downloading: $Url"

$wc = New-Object System.Net.WebClient
if ($Proxy -eq "" -and $env:NCNN_PROXY) {
  $Proxy = $env:NCNN_PROXY
}
if ($Proxy -eq "" -and $env:HTTPS_PROXY) { $Proxy = $env:HTTPS_PROXY }
if ($Proxy -eq "" -and $env:https_proxy) { $Proxy = $env:https_proxy }
if ($Proxy -eq "" -and $env:HTTP_PROXY) { $Proxy = $env:HTTP_PROXY }
if ($Proxy -eq "" -and $env:http_proxy) { $Proxy = $env:http_proxy }
if ($Proxy -ne "") {
  if ($Proxy -notmatch "^http") { $Proxy = "http://$Proxy" }
  $wc.Proxy = New-Object System.Net.WebProxy($Proxy, $true)
}
$wc.DownloadFile($Url, $ZipPath)

if (Test-Path $ExtractDir) { Remove-Item -Recurse -Force $ExtractDir }
New-Item -ItemType Directory -Force -Path $ExtractDir | Out-Null
Expand-Archive -Path $ZipPath -DestinationPath $ExtractDir -Force

$root = Join-Path $ExtractDir "ncnn-$Tag-windows-vs2022"
$archRoot = Join-Path $root $Arch
if (Test-Path $archRoot) {
  $prefix = (Resolve-Path $archRoot | Select-Object -ExpandProperty Path) -replace '\\','/'
  Write-Host "NCNN arch prefix: $prefix"
  Set-Content -Path (Join-Path $OutDir "NCNN_PREFIX.txt") -Value $prefix -Encoding ascii
}

# Prefer arch-specific CMake package files (avoid accidentally picking arm64 on x64 runners).
$cfg = $null
if (Test-Path $archRoot) {
  $cfg = Get-ChildItem -Path $archRoot -Recurse -Filter "ncnnConfig.cmake" -ErrorAction SilentlyContinue | Select-Object -First 1
}
if ($cfg) {
  $prefix = Resolve-Path (Join-Path $cfg.Directory.FullName "..\\..\\..") | Select-Object -ExpandProperty Path
  $prefixCmake = $prefix -replace '\\','/'
  Write-Host "NCNN install prefix (arch): $prefixCmake"
  Set-Content -Path (Join-Path $OutDir "NCNN_PREFIX.txt") -Value $prefixCmake -Encoding ascii
} else {
  throw "ncnnConfig.cmake not found under arch root: $archRoot. This project uses find_package(ncnn CONFIG REQUIRED)."
}

$mat = $null
if (Test-Path (Join-Path $archRoot "include\\ncnn\\mat.h")) {
  $mat = Get-Item (Join-Path $archRoot "include\\ncnn\\mat.h")
} else {
  $mat = Get-ChildItem -Path $ExtractDir -Recurse -Filter "mat.h" | Where-Object { $_.FullName -match "\\\\include\\\\ncnn\\\\mat\\.h$" } | Select-Object -First 1
}
if (-not $mat) {
  throw "include\\ncnn\\mat.h not found under $ExtractDir"
}
$includeDir = Resolve-Path (Split-Path (Split-Path $mat.FullName -Parent) -Parent) | Select-Object -ExpandProperty Path
$includeDirCmake = $includeDir -replace '\\','/'
Write-Host "NCNN include dir: $includeDirCmake"
Set-Content -Path (Join-Path $OutDir "NCNN_INCLUDE_DIR.txt") -Value $includeDirCmake -Encoding ascii

function Pick-NcnnLib([string]$arch) {
  $root = Join-Path $ExtractDir "ncnn-$Tag-windows-vs2022"
  $archRoot = Join-Path $root $arch
  $direct = Join-Path $archRoot "lib\\ncnn.lib"
  if (Test-Path $direct) { return (Get-Item $direct) }

  $libs = Get-ChildItem -Path $ExtractDir -Recurse -Filter "ncnn.lib" -ErrorAction SilentlyContinue
  if (-not $libs) { return $null }
  if ($arch -eq "x64") {
    $picked = $libs | Where-Object { $_.FullName -match "\\\\x64\\\\lib\\\\ncnn\\.lib$" } | Select-Object -First 1
    if ($picked) { return $picked }
    $picked = $libs | Where-Object { $_.FullName -match "\\\\amd64\\\\lib\\\\ncnn\\.lib$" } | Select-Object -First 1
    if ($picked) { return $picked }
  } elseif ($arch -eq "arm64") {
    $picked = $libs | Where-Object { $_.FullName -match "\\\\arm64\\\\lib\\\\ncnn\\.lib$" } | Select-Object -First 1
    if ($picked) { return $picked }
  }
  return $null
}

$lib = Pick-NcnnLib $Arch
if (-not $lib) {
  throw "ncnn.lib for arch '$Arch' not found under $ExtractDir"
}
$libPath = Resolve-Path $lib.FullName | Select-Object -ExpandProperty Path
$libPathCmake = $libPath -replace '\\','/'
Write-Host "NCNN library: $libPathCmake"
Set-Content -Path (Join-Path $OutDir "NCNN_LIBRARY.txt") -Value $libPathCmake -Encoding ascii

$libDir = Split-Path $lib.FullName -Parent
$extraLibs = @()
if (Test-Path $libDir) {
  $extraLibs = Get-ChildItem -Path $libDir -Filter "*.lib" -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -ne "ncnn.lib" } |
    Sort-Object -Property Name
}
if ($extraLibs -and $extraLibs.Count -gt 0) {
  $extraPaths = $extraLibs | ForEach-Object { (Resolve-Path $_.FullName | Select-Object -ExpandProperty Path) -replace '\\','/' }
  $extraJoined = ($extraPaths -join ";")
  Write-Host "NCNN extra libs: $extraJoined"
  Set-Content -Path (Join-Path $OutDir "NCNN_EXTRA_LIBRARIES.txt") -Value $extraJoined -Encoding ascii
}

$openmp = Get-ChildItem -Path $ExtractDir -Recurse -Filter "openmp.lib" -ErrorAction SilentlyContinue | Select-Object -First 1
if ($openmp) {
  $openmpPath = Resolve-Path $openmp.FullName | Select-Object -ExpandProperty Path
  $openmpPathCmake = $openmpPath -replace '\\','/'
  Write-Host "OpenMP runtime: $openmpPathCmake"
  Set-Content -Path (Join-Path $OutDir "NCNN_OPENMP_LIBRARY.txt") -Value $openmpPathCmake -Encoding ascii
}
