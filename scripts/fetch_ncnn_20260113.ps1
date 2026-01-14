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
if ($Proxy -ne "") {
  if ($Proxy -notmatch "^http") { $Proxy = "http://$Proxy" }
  $wc.Proxy = New-Object System.Net.WebProxy($Proxy, $true)
}
$wc.DownloadFile($Url, $ZipPath)

if (Test-Path $ExtractDir) { Remove-Item -Recurse -Force $ExtractDir }
New-Item -ItemType Directory -Force -Path $ExtractDir | Out-Null
Expand-Archive -Path $ZipPath -DestinationPath $ExtractDir -Force

$cfg = Get-ChildItem -Path $ExtractDir -Recurse -Filter "ncnnConfig.cmake" | Select-Object -First 1
if (-not $cfg) {
  Write-Warning "ncnnConfig.cmake not found under $ExtractDir"
}

if ($cfg) {
  $prefix = Resolve-Path (Join-Path $cfg.Directory.FullName "..\\..\\..") | Select-Object -ExpandProperty Path
  Write-Host "NCNN install prefix: $prefix"
  Set-Content -Path (Join-Path $OutDir "NCNN_PREFIX.txt") -Value $prefix -Encoding ascii
}

$mat = Get-ChildItem -Path $ExtractDir -Recurse -Filter "mat.h" | Where-Object { $_.FullName -match "\\include\\ncnn\\mat.h$" } | Select-Object -First 1
if (-not $mat) {
  throw "include\\ncnn\\mat.h not found under $ExtractDir"
}
$includeDir = Resolve-Path (Split-Path (Split-Path $mat.FullName -Parent) -Parent) | Select-Object -ExpandProperty Path
Write-Host "NCNN include dir: $includeDir"
Set-Content -Path (Join-Path $OutDir "NCNN_INCLUDE_DIR.txt") -Value $includeDir -Encoding ascii

$lib = Get-ChildItem -Path $ExtractDir -Recurse -Filter "ncnn.lib" |
  Where-Object { $_.FullName -match "\\\\$Arch\\\\lib\\\\ncnn\\.lib$" } |
  Select-Object -First 1
if (-not $lib) {
  $lib = Get-ChildItem -Path $ExtractDir -Recurse -Filter "ncnn.lib" | Select-Object -First 1
}
if (-not $lib) {
  throw "ncnn.lib not found under $ExtractDir"
}
$libPath = Resolve-Path $lib.FullName | Select-Object -ExpandProperty Path
Write-Host "NCNN library: $libPath"
Set-Content -Path (Join-Path $OutDir "NCNN_LIBRARY.txt") -Value $libPath -Encoding ascii

$openmp = Get-ChildItem -Path $ExtractDir -Recurse -Filter "openmp.lib" -ErrorAction SilentlyContinue | Select-Object -First 1
if ($openmp) {
  $openmpPath = Resolve-Path $openmp.FullName | Select-Object -ExpandProperty Path
  Write-Host "OpenMP runtime: $openmpPath"
  Set-Content -Path (Join-Path $OutDir "NCNN_OPENMP_LIBRARY.txt") -Value $openmpPath -Encoding ascii
}
