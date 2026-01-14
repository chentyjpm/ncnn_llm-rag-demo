#!/usr/bin/env bash
set -euo pipefail

TAG="20260113"
OUT_DIR="deps/ncnn-prebuilt"
PROXY=""
OS=""

usage() {
  cat <<'EOF'
Usage: scripts/fetch_ncnn_20260113.sh [--os linux|macos] [--out DIR] [--proxy HOST:PORT|URL]

Downloads Tencent/ncnn prebuilt library release 20260113 and extracts it.

Examples:
  scripts/fetch_ncnn_20260113.sh --out deps/ncnn-prebuilt
  scripts/fetch_ncnn_20260113.sh --proxy 192.168.8.3:18080
  scripts/fetch_ncnn_20260113.sh --proxy http://192.168.8.3:18080 --out /tmp/ncnn

Notes:
  - If --proxy is provided, it is used for both HTTP and HTTPS.
  - On Linux this tries ubuntu-2404 first, then ubuntu-2204.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --os) OS="${2:-}"; shift 2 ;;
    --out) OUT_DIR="${2:-}"; shift 2 ;;
    --proxy) PROXY="${2:-}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ -z "$PROXY" ]]; then
  PROXY="${NCNN_PROXY:-}"
fi

if [[ -z "$OS" ]]; then
  uname_s="$(uname -s | tr '[:upper:]' '[:lower:]')"
  case "$uname_s" in
    linux*) OS="linux" ;;
    darwin*) OS="macos" ;;
    *) echo "Unsupported OS for this script: $uname_s" >&2; exit 2 ;;
  esac
fi

mkdir -p "$OUT_DIR"

curl_args=(-fsSL -L)
if [[ -n "$PROXY" ]]; then
  if [[ "$PROXY" != http*://* ]]; then
    PROXY="http://$PROXY"
  fi
  curl_args+=(-x "$PROXY")
fi

download_and_extract() {
  local url="$1"
  local zip="$2"
  echo "Downloading: $url"
  curl "${curl_args[@]}" -o "$zip" "$url"
  rm -rf "$OUT_DIR/extracted"
  mkdir -p "$OUT_DIR/extracted"
  unzip -q -o "$zip" -d "$OUT_DIR/extracted"
}

if [[ "$OS" == "macos" ]]; then
  url="https://github.com/Tencent/ncnn/releases/download/${TAG}/ncnn-${TAG}-macos.zip"
  download_and_extract "$url" "$OUT_DIR/ncnn-${TAG}-${OS}.zip"
elif [[ "$OS" == "linux" ]]; then
  # Prefer newer ubuntu toolchain if available.
  url2404="https://github.com/Tencent/ncnn/releases/download/${TAG}/ncnn-${TAG}-ubuntu-2404.zip"
  url2204="https://github.com/Tencent/ncnn/releases/download/${TAG}/ncnn-${TAG}-ubuntu-2204.zip"
  if curl "${curl_args[@]}" -I "$url2404" >/dev/null 2>&1; then
    download_and_extract "$url2404" "$OUT_DIR/ncnn-${TAG}-ubuntu-2404.zip"
  else
    download_and_extract "$url2204" "$OUT_DIR/ncnn-${TAG}-ubuntu-2204.zip"
  fi
else
  echo "Unsupported --os: $OS" >&2
  exit 2
fi

cfg="$(find "$OUT_DIR/extracted" -name 'ncnnConfig.cmake' -print -quit || true)"
if [[ -z "$cfg" ]]; then
  echo "WARN: ncnnConfig.cmake not found under $OUT_DIR/extracted" >&2
fi

if [[ -n "$cfg" ]]; then
  prefix="$(cd "$(dirname "$cfg")/../../.." && pwd)"
  echo "NCNN install prefix: $prefix"
  echo "$prefix" > "$OUT_DIR/NCNN_PREFIX.txt"
fi

mat_h="$(find "$OUT_DIR/extracted" \( -path '*/include/ncnn/mat.h' -o -path '*/Headers/ncnn/mat.h' \) -print -quit || true)"
if [[ -z "$mat_h" ]]; then
  echo "ERROR: include/ncnn/mat.h not found under $OUT_DIR/extracted" >&2
  exit 1
fi
include_dir="$(cd "$(dirname "$mat_h")/.." && pwd)"
echo "NCNN include dir: $include_dir"
echo "$include_dir" > "$OUT_DIR/NCNN_INCLUDE_DIR.txt"

lib_path="$(find "$OUT_DIR/extracted" -name 'libncnn.a' -print -quit || true)"
if [[ -z "$lib_path" ]]; then
  lib_path="$(find "$OUT_DIR/extracted" -name 'libncnn.so' -print -quit || true)"
fi
if [[ -z "$lib_path" ]]; then
  lib_path="$(find "$OUT_DIR/extracted" -name 'libncnn.dylib' -print -quit || true)"
fi
if [[ -z "$lib_path" ]]; then
  lib_path="$(find "$OUT_DIR/extracted" -path '*/ncnn.framework/Versions/*/ncnn' -print -quit || true)"
fi
if [[ -z "$lib_path" ]]; then
  lib_path="$(find "$OUT_DIR/extracted" -path '*/ncnn.framework/ncnn' -print -quit || true)"
fi
if [[ -z "$lib_path" ]]; then
  echo "ERROR: libncnn not found under $OUT_DIR/extracted" >&2
  exit 1
fi
lib_path="$(cd "$(dirname "$lib_path")" && pwd)/$(basename "$lib_path")"
echo "NCNN library: $lib_path"
echo "$lib_path" > "$OUT_DIR/NCNN_LIBRARY.txt"
echo "Done."
