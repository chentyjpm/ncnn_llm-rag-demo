param(
  [string]$OutDir = "deps/ncnn-prebuilt",
  [string]$Proxy = ""
)

$ErrorActionPreference = "Stop"
$Tag = "20260113"

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$ZipPath = Join-Path $OutDir "ncnn-$Tag-windows-vs2022.zip"
$ExtractDir = Join-Path $OutDir "extracted"

$Url = "https://github.com/Tencent/ncnn/releases/download/$Tag/ncnn-$Tag-windows-vs2022.zip"
Write-Host "Downloading: $Url"

$wc = New-Object System.Net.WebClient
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
  throw "ncnnConfig.cmake not found under $ExtractDir"
}

$prefix = Resolve-Path (Join-Path $cfg.Directory.FullName "..\\..\\..") | Select-Object -ExpandProperty Path
Write-Host "NCNN install prefix: $prefix"
Set-Content -Path (Join-Path $OutDir "NCNN_PREFIX.txt") -Value $prefix -Encoding ascii

