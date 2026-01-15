#!/usr/bin/env python3
import argparse
import os
from pathlib import Path


MIME_BY_EXT = {
    ".html": "text/html; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".ico": "image/x-icon",
    ".json": "application/json; charset=utf-8",
    ".txt": "text/plain; charset=utf-8",
}


def c_ident(s: str) -> str:
    out = []
    for ch in s:
        if ch.isalnum():
            out.append(ch)
        else:
            out.append("_")
    ident = "".join(out)
    if not ident or ident[0].isdigit():
        ident = "_" + ident
    return ident


def bytes_to_cpp_array(data: bytes, indent: str = "    ") -> str:
    if not data:
        return "{}"
    parts = []
    line = []
    for i, b in enumerate(data):
        line.append(str(b))
        if len(line) >= 16:
            parts.append(indent + ", ".join(line) + ",")
            line = []
    if line:
        parts.append(indent + ", ".join(line) + ",")
    return "{\n" + "\n".join(parts) + "\n}"


def main() -> int:
    ap = argparse.ArgumentParser(description="Embed src/web assets into C++ source.")
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--out-cpp", required=True)
    ap.add_argument("--out-h", required=True)
    args = ap.parse_args()

    input_dir = Path(args.input_dir).resolve()
    out_cpp = Path(args.out_cpp).resolve()
    out_h = Path(args.out_h).resolve()

    if not input_dir.is_dir():
        raise SystemExit(f"input dir not found: {input_dir}")

    files = []
    for p in sorted(input_dir.rglob("*")):
        if p.is_file():
            rel = p.relative_to(input_dir).as_posix()
            files.append((rel, p))

    ns = "ncnn_llm_rag_demo_web"

    out_h.parent.mkdir(parents=True, exist_ok=True)
    out_cpp.parent.mkdir(parents=True, exist_ok=True)

    h = []
    h.append("#pragma once")
    h.append("#include <cstddef>")
    h.append("#include <string_view>")
    h.append("")
    h.append(f"namespace {ns} {{")
    h.append("struct AssetView {")
    h.append("    const unsigned char* data;")
    h.append("    size_t size;")
    h.append("    const char* mime;")
    h.append("};")
    h.append("")
    h.append("bool get(std::string_view path, AssetView* out);")
    h.append(f"}} // namespace {ns}")
    h.append("")
    out_h.write_text("\n".join(h), encoding="utf-8")

    cpp = []
    cpp.append('#include "web_assets_embedded.h"')
    cpp.append("")
    cpp.append("#include <string_view>")
    cpp.append("")
    cpp.append(f"namespace {ns} {{")
    cpp.append("namespace {")
    cpp.append("struct Entry { std::string_view path; AssetView view; };")
    cpp.append("")

    entries = []
    for rel, full in files:
        data = full.read_bytes()
        ext = full.suffix.lower()
        mime = MIME_BY_EXT.get(ext, "application/octet-stream")
        var = c_ident(f"asset_{rel}")
        cpp.append(f"static const unsigned char {var}[] = {bytes_to_cpp_array(data)};")
        cpp.append("")
        entries.append((f'"/{rel}"', var, len(data), mime))

    cpp.append("static const Entry kEntries[] = {")
    for path_lit, var, size, mime in entries:
        cpp.append(f'    {{{path_lit}, AssetView{{{var}, {size}u, "{mime}"}}}},')
    cpp.append("};")
    cpp.append("")
    cpp.append("} // namespace")
    cpp.append("")
    cpp.append("bool get(std::string_view path, AssetView* out) {")
    cpp.append("    if (!out) return false;")
    cpp.append("    if (path.empty() || path == \"/\") path = \"/index.html\";")
    cpp.append("    for (const auto& e : kEntries) {")
    cpp.append("        if (e.path == path) {")
    cpp.append("            *out = e.view;")
    cpp.append("            return true;")
    cpp.append("        }")
    cpp.append("    }")
    cpp.append("    return false;")
    cpp.append("}")
    cpp.append(f"}} // namespace {ns}")
    cpp.append("")

    out_cpp.write_text("\n".join(cpp), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

