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

# Fall back to standard proxy env vars if NCNN_PROXY is not set.
if [[ -z "$PROXY" ]]; then
  PROXY="${HTTPS_PROXY:-${https_proxy:-${HTTP_PROXY:-${http_proxy:-}}}}"
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
  # Prefer Vulkan-enabled build if available, but only when it ships a Vulkan loader.
  # GitHub macOS runners typically do not have Vulkan/MoltenVK installed by default, so
  # a Vulkan-enabled ncnn that expects vkGetInstanceProcAddr at link time will fail.
  url_vulkan="https://github.com/Tencent/ncnn/releases/download/${TAG}/ncnn-${TAG}-macos-vulkan.zip"
  url_plain="https://github.com/Tencent/ncnn/releases/download/${TAG}/ncnn-${TAG}-macos.zip"

  picked_plain=0
  if curl "${curl_args[@]}" -I "$url_vulkan" >/dev/null 2>&1; then
    download_and_extract "$url_vulkan" "$OUT_DIR/ncnn-${TAG}-${OS}-vulkan.zip"
    vulkan_loader="$(find "$OUT_DIR/extracted" -type f \( -name 'libvulkan*.dylib' -o -path '*/vulkan.framework/Versions/*/vulkan' -o -path '*/vulkan.framework/vulkan' \) -print -quit 2>/dev/null || true)"
    if [[ -z "$vulkan_loader" ]]; then
      echo "WARN: macos-vulkan prebuilt found but no Vulkan loader shipped; falling back to plain macos prebuilt." >&2
      picked_plain=1
    fi
  else
    picked_plain=1
  fi

  if [[ "$picked_plain" == "1" ]]; then
    download_and_extract "$url_plain" "$OUT_DIR/ncnn-${TAG}-${OS}.zip"
  fi
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
  if [[ "$OS" != "macos" ]]; then
    echo "ERROR: ncnnConfig.cmake not found under $OUT_DIR/extracted" >&2
    echo "This project uses find_package(ncnn CONFIG REQUIRED); please use a prebuilt that ships CMake package files." >&2
    exit 1
  fi

  # macOS prebuilt may ship as frameworks without CMake package files.
  # Generate a minimal ncnnConfig.cmake so the project can still use find_package(ncnn CONFIG REQUIRED).
  ncnn_bin="$(find "$OUT_DIR/extracted" -type f -path '*/ncnn.framework/Versions/*/ncnn' -print -quit || true)"
  if [[ -z "$ncnn_bin" ]]; then
    ncnn_bin="$(find "$OUT_DIR/extracted" -type f -path '*/ncnn.framework/ncnn' -print -quit || true)"
  fi
  ncnn_hdr_dir="$(find "$OUT_DIR/extracted" -type d -path '*/ncnn.framework/Versions/*/Headers' -print -quit || true)"
  if [[ -z "$ncnn_hdr_dir" ]]; then
    ncnn_hdr_dir="$(find "$OUT_DIR/extracted" -type d -path '*/ncnn.framework/Headers' -print -quit || true)"
  fi
  openmp_bin="$(find "$OUT_DIR/extracted" -type f -path '*/openmp.framework/Versions/*/openmp' -print -quit || true)"
  if [[ -z "$openmp_bin" ]]; then
    openmp_bin="$(find "$OUT_DIR/extracted" -type f -path '*/openmp.framework/openmp' -print -quit || true)"
  fi

  # Optional deps for Vulkan-enabled builds.
  # - glslang/SPIRV may be shipped as static libs or frameworks.
  # - Vulkan loader may be shipped as a dylib or framework.
  glslang_libs=()
  while IFS= read -r p; do glslang_libs+=("$p"); done < <(find "$OUT_DIR/extracted" -type f \
    \( -name 'libglslang.a' -o -name 'libSPIRV.a' -o -name 'libSPIRV-Tools.a' -o -name 'libSPIRV-Tools-opt.a' \
       -o -name 'libOGLCompiler.a' -o -name 'libOSDependent.a' -o -name 'libHLSL.a' \) 2>/dev/null | sort || true)
  # Framework-style glslang (rare).
  glslang_framework_bin="$(find "$OUT_DIR/extracted" -type f -path '*/glslang.framework/Versions/*/glslang' -print -quit 2>/dev/null || true)"
  if [[ -z "$glslang_framework_bin" ]]; then
    glslang_framework_bin="$(find "$OUT_DIR/extracted" -type f -path '*/glslang.framework/glslang' -print -quit 2>/dev/null || true)"
  fi

  vulkan_loader="$(find "$OUT_DIR/extracted" -type f \( -name 'libvulkan*.dylib' -o -path '*/vulkan.framework/Versions/*/vulkan' -o -path '*/vulkan.framework/vulkan' \) -print -quit 2>/dev/null || true)"

  if [[ -z "$ncnn_bin" || -z "$ncnn_hdr_dir" ]]; then
    echo "ERROR: macOS framework prebuilt detected but ncnn.framework is incomplete (missing binary or headers)" >&2
    exit 1
  fi

  # Try to detect whether the headers expose Vulkan GPU instance API.
  _has_gpu_api=0
  if [[ -f "$ncnn_hdr_dir/ncnn/gpu.h" ]] && rg -n "create_gpu_instance\\s*\\(" "$ncnn_hdr_dir/ncnn/gpu.h" >/dev/null 2>&1; then
    _has_gpu_api=1
  fi

  gen_prefix="$OUT_DIR/prefix"
  gen_cfg_dir="$gen_prefix/lib/cmake/ncnn"
  mkdir -p "$gen_cfg_dir"

  # Resolve to absolute paths for robustness.
  ncnn_bin="$(cd "$(dirname "$ncnn_bin")" && pwd)/$(basename "$ncnn_bin")"
  ncnn_hdr_dir="$(cd "$ncnn_hdr_dir" && pwd)"
  if [[ -n "$openmp_bin" ]]; then
    openmp_bin="$(cd "$(dirname "$openmp_bin")" && pwd)/$(basename "$openmp_bin")"
  fi
  if [[ -n "$glslang_framework_bin" ]]; then
    glslang_framework_bin="$(cd "$(dirname "$glslang_framework_bin")" && pwd)/$(basename "$glslang_framework_bin")"
  fi
  if [[ -n "$vulkan_loader" ]]; then
    vulkan_loader="$(cd "$(dirname "$vulkan_loader")" && pwd)/$(basename "$vulkan_loader")"
  fi

  cat > "$gen_cfg_dir/ncnnConfig.cmake" <<EOF
set(NCNN_VERSION ${TAG})
set(NCNN_OPENMP ON)
set(NCNN_THREADS ON)
set(NCNN_VULKAN ${_has_gpu_api})
set(NCNN_SHARED_LIB ON)
set(NCNN_SIMPLEVK ON)

add_library(ncnn SHARED IMPORTED)
set_target_properties(ncnn PROPERTIES
  IMPORTED_LOCATION "${ncnn_bin}"
  INTERFACE_INCLUDE_DIRECTORIES "${ncnn_hdr_dir}"
)

if(EXISTS "${openmp_bin}")
  add_library(ncnn_openmp SHARED IMPORTED)
  set_target_properties(ncnn_openmp PROPERTIES IMPORTED_LOCATION "${openmp_bin}")
  set_property(TARGET ncnn APPEND PROPERTY INTERFACE_LINK_LIBRARIES ncnn_openmp)
endif()

if(EXISTS "${vulkan_loader}")
  add_library(ncnn_vulkan_loader SHARED IMPORTED)
  set_target_properties(ncnn_vulkan_loader PROPERTIES IMPORTED_LOCATION "${vulkan_loader}")
  set_property(TARGET ncnn APPEND PROPERTY INTERFACE_LINK_LIBRARIES ncnn_vulkan_loader)
endif()

if(EXISTS "${glslang_framework_bin}")
  add_library(ncnn_glslang SHARED IMPORTED)
  set_target_properties(ncnn_glslang PROPERTIES IMPORTED_LOCATION "${glslang_framework_bin}")
  set_property(TARGET ncnn APPEND PROPERTY INTERFACE_LINK_LIBRARIES ncnn_glslang)
endif()

find_package(Threads REQUIRED)
set_property(TARGET ncnn APPEND PROPERTY INTERFACE_LINK_LIBRARIES Threads::Threads)

set(ncnn_FOUND TRUE)
if(NOT ncnn_FIND_QUIETLY)
  message(STATUS "Found ncnn: \${NCNN_VERSION} (generated config for macOS frameworks)")
endif()
EOF

  if [[ ${#glslang_libs[@]} -gt 0 ]]; then
    {
      echo ""
      echo "# Link bundled glslang/SPIRV static libraries (required by Vulkan builds)."
      for p in "${glslang_libs[@]}"; do
        abs="$(cd "$(dirname "$p")" && pwd)/$(basename "$p")"
        echo "if(EXISTS \"${abs}\")"
        echo "  set_property(TARGET ncnn APPEND PROPERTY INTERFACE_LINK_LIBRARIES \"${abs}\")"
        echo "endif()"
      done
    } >> "$gen_cfg_dir/ncnnConfig.cmake"
  fi

  cfg="$gen_cfg_dir/ncnnConfig.cmake"
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

lib_path="$(find "$OUT_DIR/extracted" -type f -name 'libncnn.a' -print -quit || true)"
if [[ -z "$lib_path" ]]; then
  # Some zips may only contain versioned .so (e.g. libncnn.so.1.0.x), especially if symlinks are not preserved.
  lib_path="$(find "$OUT_DIR/extracted" -type f -name 'libncnn.so*' -print -quit || true)"
fi
if [[ -z "$lib_path" ]]; then
  lib_path="$(find "$OUT_DIR/extracted" -type f -name 'libncnn.dylib' -print -quit || true)"
fi
if [[ -z "$lib_path" ]]; then
  lib_path="$(find "$OUT_DIR/extracted" -type f -path '*/ncnn.framework/Versions/*/ncnn' -print -quit || true)"
fi
if [[ -z "$lib_path" ]]; then
  lib_path="$(find "$OUT_DIR/extracted" -type f -path '*/ncnn.framework/ncnn' -print -quit || true)"
fi
if [[ -z "$lib_path" ]]; then
  echo "ERROR: libncnn not found under $OUT_DIR/extracted" >&2
  exit 1
fi
lib_path="$(cd "$(dirname "$lib_path")" && pwd)/$(basename "$lib_path")"
echo "NCNN library: $lib_path"
echo "$lib_path" > "$OUT_DIR/NCNN_LIBRARY.txt"

openmp_bin="$(find "$OUT_DIR/extracted" -type f -path '*/openmp.framework/Versions/*/openmp' -print -quit || true)"
if [[ -z "$openmp_bin" ]]; then
  openmp_bin="$(find "$OUT_DIR/extracted" -type f -path '*/openmp.framework/openmp' -print -quit || true)"
fi
if [[ -n "$openmp_bin" ]]; then
  openmp_bin="$(cd "$(dirname "$openmp_bin")" && pwd)/$(basename "$openmp_bin")"
  echo "OpenMP runtime: $openmp_bin"
  echo "$openmp_bin" > "$OUT_DIR/NCNN_OPENMP_LIBRARY.txt"
fi
echo "Done."
