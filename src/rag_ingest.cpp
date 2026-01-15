#include "rag_ingest.h"

#include "rag_text.h"

#include <cstdlib>
#include <cstdio>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

#if defined(_WIN32)
#include <windows.h>
#else
#include <iconv.h>
#include <cerrno>
#endif

namespace fs = std::filesystem;

namespace {

#ifdef _WIN32
#define popen _popen
#define pclose _pclose
#endif

bool is_valid_utf8(const std::string& s) {
    size_t i = 0;
    const size_t n = s.size();
    while (i < n) {
        unsigned char c = static_cast<unsigned char>(s[i]);
        if (c < 0x80) {
            ++i;
            continue;
        }

        // 2-byte
        if (c >= 0xC2 && c <= 0xDF) {
            if (i + 1 >= n) return false;
            unsigned char c1 = static_cast<unsigned char>(s[i + 1]);
            if ((c1 & 0xC0) != 0x80) return false;
            i += 2;
            continue;
        }

        // 3-byte
        if (c == 0xE0) {
            if (i + 2 >= n) return false;
            unsigned char c1 = static_cast<unsigned char>(s[i + 1]);
            unsigned char c2 = static_cast<unsigned char>(s[i + 2]);
            if (!(c1 >= 0xA0 && c1 <= 0xBF)) return false;
            if ((c2 & 0xC0) != 0x80) return false;
            i += 3;
            continue;
        }
        if (c >= 0xE1 && c <= 0xEC) {
            if (i + 2 >= n) return false;
            unsigned char c1 = static_cast<unsigned char>(s[i + 1]);
            unsigned char c2 = static_cast<unsigned char>(s[i + 2]);
            if ((c1 & 0xC0) != 0x80 || (c2 & 0xC0) != 0x80) return false;
            i += 3;
            continue;
        }
        if (c == 0xED) { // exclude surrogates
            if (i + 2 >= n) return false;
            unsigned char c1 = static_cast<unsigned char>(s[i + 1]);
            unsigned char c2 = static_cast<unsigned char>(s[i + 2]);
            if (!(c1 >= 0x80 && c1 <= 0x9F)) return false;
            if ((c2 & 0xC0) != 0x80) return false;
            i += 3;
            continue;
        }
        if (c >= 0xEE && c <= 0xEF) {
            if (i + 2 >= n) return false;
            unsigned char c1 = static_cast<unsigned char>(s[i + 1]);
            unsigned char c2 = static_cast<unsigned char>(s[i + 2]);
            if ((c1 & 0xC0) != 0x80 || (c2 & 0xC0) != 0x80) return false;
            i += 3;
            continue;
        }

        // 4-byte
        if (c == 0xF0) {
            if (i + 3 >= n) return false;
            unsigned char c1 = static_cast<unsigned char>(s[i + 1]);
            unsigned char c2 = static_cast<unsigned char>(s[i + 2]);
            unsigned char c3 = static_cast<unsigned char>(s[i + 3]);
            if (!(c1 >= 0x90 && c1 <= 0xBF)) return false;
            if ((c2 & 0xC0) != 0x80 || (c3 & 0xC0) != 0x80) return false;
            i += 4;
            continue;
        }
        if (c >= 0xF1 && c <= 0xF3) {
            if (i + 3 >= n) return false;
            unsigned char c1 = static_cast<unsigned char>(s[i + 1]);
            unsigned char c2 = static_cast<unsigned char>(s[i + 2]);
            unsigned char c3 = static_cast<unsigned char>(s[i + 3]);
            if ((c1 & 0xC0) != 0x80 || (c2 & 0xC0) != 0x80 || (c3 & 0xC0) != 0x80) return false;
            i += 4;
            continue;
        }
        if (c == 0xF4) {
            if (i + 3 >= n) return false;
            unsigned char c1 = static_cast<unsigned char>(s[i + 1]);
            unsigned char c2 = static_cast<unsigned char>(s[i + 2]);
            unsigned char c3 = static_cast<unsigned char>(s[i + 3]);
            if (!(c1 >= 0x80 && c1 <= 0x8F)) return false;
            if ((c2 & 0xC0) != 0x80 || (c3 & 0xC0) != 0x80) return false;
            i += 4;
            continue;
        }

        return false;
    }
    return true;
}

void strip_utf8_bom(std::string* s) {
    if (!s || s->size() < 3) return;
    const unsigned char b0 = static_cast<unsigned char>((*s)[0]);
    const unsigned char b1 = static_cast<unsigned char>((*s)[1]);
    const unsigned char b2 = static_cast<unsigned char>((*s)[2]);
    if (b0 == 0xEF && b1 == 0xBB && b2 == 0xBF) {
        s->erase(0, 3);
    }
}

bool utf16_to_utf8(const uint16_t* data, size_t len, bool big_endian, std::string* out) {
    if (!out) return false;
    out->clear();
    out->reserve(len * 2);

    auto read_u16 = [&](size_t i) -> uint16_t {
        uint16_t v = data[i];
        if (!big_endian) return v;
        return static_cast<uint16_t>((v >> 8) | (v << 8));
    };

    size_t i = 0;
    while (i < len) {
        uint32_t cp = read_u16(i++);
        if (cp >= 0xD800 && cp <= 0xDBFF) { // high surrogate
            if (i >= len) return false;
            uint32_t lo = read_u16(i++);
            if (!(lo >= 0xDC00 && lo <= 0xDFFF)) return false;
            cp = 0x10000 + (((cp - 0xD800) << 10) | (lo - 0xDC00));
        } else if (cp >= 0xDC00 && cp <= 0xDFFF) {
            return false;
        }

        if (cp <= 0x7F) {
            out->push_back(static_cast<char>(cp));
        } else if (cp <= 0x7FF) {
            out->push_back(static_cast<char>(0xC0 | (cp >> 6)));
            out->push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        } else if (cp <= 0xFFFF) {
            out->push_back(static_cast<char>(0xE0 | (cp >> 12)));
            out->push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
            out->push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        } else if (cp <= 0x10FFFF) {
            out->push_back(static_cast<char>(0xF0 | (cp >> 18)));
            out->push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
            out->push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
            out->push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        } else {
            return false;
        }
    }
    return true;
}

#if defined(_WIN32)
bool convert_codepage_to_utf8(const std::string& in, UINT codepage, std::string* out) {
    if (!out) return false;
    out->clear();
    if (in.empty()) return true;

    int wide_len = MultiByteToWideChar(codepage, MB_ERR_INVALID_CHARS, in.data(),
                                       static_cast<int>(in.size()), nullptr, 0);
    if (wide_len <= 0) return false;
    std::wstring wide(static_cast<size_t>(wide_len), L'\0');
    if (MultiByteToWideChar(codepage, MB_ERR_INVALID_CHARS, in.data(),
                            static_cast<int>(in.size()), wide.data(), wide_len) != wide_len) {
        return false;
    }

    int utf8_len = WideCharToMultiByte(CP_UTF8, 0, wide.data(), wide_len, nullptr, 0, nullptr, nullptr);
    if (utf8_len <= 0) return false;
    out->resize(static_cast<size_t>(utf8_len));
    if (WideCharToMultiByte(CP_UTF8, 0, wide.data(), wide_len, out->data(), utf8_len, nullptr, nullptr) != utf8_len) {
        out->clear();
        return false;
    }
    return true;
}
#else
bool iconv_convert(const char* from_charset, const std::string& in, std::string* out) {
    if (!out) return false;
    out->clear();
    if (in.empty()) return true;

    iconv_t cd = iconv_open("UTF-8", from_charset);
    if (cd == (iconv_t)-1) return false;

    size_t in_left = in.size();
    const char* in_buf = in.data();
    size_t out_cap = in.size() * 4 + 32;
    std::string result;
    result.resize(out_cap);
    char* out_buf = result.data();
    size_t out_left = out_cap;

    while (in_left > 0) {
        size_t rc = iconv(cd, const_cast<char**>(&in_buf), &in_left, &out_buf, &out_left);
        if (rc != (size_t)-1) break;
        if (errno == E2BIG) {
            size_t used = out_cap - out_left;
            out_cap *= 2;
            result.resize(out_cap);
            out_buf = result.data() + used;
            out_left = out_cap - used;
            continue;
        }
        iconv_close(cd);
        return false;
    }

    iconv_close(cd);
    result.resize(out_cap - out_left);
    *out = std::move(result);
    return true;
}
#endif

std::string shell_escape(const std::string& s) {
#ifdef _WIN32
    std::string out = "\"";
    for (char c : s) {
        if (c == '"') out += "\"\"";
        else out.push_back(c);
    }
    out.push_back('"');
    return out;
#else
    std::string out = "'";
    for (char c : s) {
        if (c == '\'') {
            out += "'\\''";
        } else {
            out.push_back(c);
        }
    }
    out.push_back('\'');
    return out;
#endif
}

bool command_exists(const std::string& name) {
#ifdef _WIN32
    std::string cmd = "where " + name + " >nul 2>nul";
    return std::system(cmd.c_str()) == 0;
#else
    std::string cmd = "command -v " + name + " >/dev/null 2>&1";
    return std::system(cmd.c_str()) == 0;
#endif
}

} // namespace

bool normalize_utf8(std::string* s, std::string* err) {
    if (!s) {
        if (err) *err = "invalid string pointer";
        return false;
    }

    strip_utf8_bom(s);
    if (is_valid_utf8(*s)) return true;

    // UTF-16 BOM handling.
    if (s->size() >= 2) {
        const unsigned char b0 = static_cast<unsigned char>((*s)[0]);
        const unsigned char b1 = static_cast<unsigned char>((*s)[1]);
        if (b0 == 0xFF && b1 == 0xFE) { // UTF-16LE
            const size_t bytes = s->size() - 2;
            if (bytes % 2 != 0) {
                if (err) *err = "invalid UTF-16LE byte length";
                return false;
            }
            std::string out;
            const uint16_t* data = reinterpret_cast<const uint16_t*>(s->data() + 2);
            if (!utf16_to_utf8(data, bytes / 2, false, &out) || !is_valid_utf8(out)) {
                if (err) *err = "failed to decode UTF-16LE";
                return false;
            }
            *s = std::move(out);
            return true;
        }
        if (b0 == 0xFE && b1 == 0xFF) { // UTF-16BE
            const size_t bytes = s->size() - 2;
            if (bytes % 2 != 0) {
                if (err) *err = "invalid UTF-16BE byte length";
                return false;
            }
            std::string out;
            const uint16_t* data = reinterpret_cast<const uint16_t*>(s->data() + 2);
            if (!utf16_to_utf8(data, bytes / 2, true, &out) || !is_valid_utf8(out)) {
                if (err) *err = "failed to decode UTF-16BE";
                return false;
            }
            *s = std::move(out);
            return true;
        }
    }

    // Best-effort conversion for common legacy encodings (e.g., GBK/GB18030).
    std::string converted;
#if defined(_WIN32)
    // Try GB18030 then GBK, then system ANSI codepage.
    if (convert_codepage_to_utf8(*s, 54936 /*GB18030*/, &converted) && is_valid_utf8(converted)) {
        *s = std::move(converted);
        return true;
    }
    if (convert_codepage_to_utf8(*s, 936 /*GBK*/, &converted) && is_valid_utf8(converted)) {
        *s = std::move(converted);
        return true;
    }
    if (convert_codepage_to_utf8(*s, CP_ACP, &converted) && is_valid_utf8(converted)) {
        *s = std::move(converted);
        return true;
    }
#else
    if (iconv_convert("GB18030", *s, &converted) && is_valid_utf8(converted)) {
        *s = std::move(converted);
        return true;
    }
    if (iconv_convert("GBK", *s, &converted) && is_valid_utf8(converted)) {
        *s = std::move(converted);
        return true;
    }
    if (iconv_convert("CP936", *s, &converted) && is_valid_utf8(converted)) {
        *s = std::move(converted);
        return true;
    }
#endif

    if (err) *err = "text is not valid UTF-8 (try saving as UTF-8/UTF-8 BOM, or GB18030/GBK)";
    return false;
}

bool read_text_file(const fs::path& path, std::string* out, std::string* err) {
    std::ifstream ifs(path, std::ios::in | std::ios::binary);
    if (!ifs) {
        if (err) *err = "failed to open file";
        return false;
    }
    std::ostringstream oss;
    oss << ifs.rdbuf();
    *out = oss.str();
    std::string norm_err;
    if (!normalize_utf8(out, &norm_err)) {
        if (err) *err = norm_err;
        return false;
    }
    *out = trim_text(*out);
    if (out->empty()) {
        if (err) *err = "empty text file";
        return false;
    }
    return true;
}

bool extract_pdf_text(const fs::path& path, std::string* out, std::string* err) {
    if (!command_exists("pdftotext")) {
        if (err) *err = "pdftotext not found; please install poppler-utils";
        return false;
    }
    std::string cmd = "pdftotext -layout -q -enc UTF-8 " + shell_escape(path.string()) + " -";
    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) {
        if (err) *err = "failed to execute pdftotext";
        return false;
    }

    std::string result;
    char buf[4096];
    while (true) {
        size_t n = fread(buf, 1, sizeof(buf), pipe);
        if (n > 0) result.append(buf, n);
        if (n < sizeof(buf)) break;
    }
    int rc = pclose(pipe);
    if (rc != 0) {
        if (err) *err = "pdftotext failed";
        return false;
    }

    *out = result;
    std::string norm_err;
    if (!normalize_utf8(out, &norm_err)) {
        if (err) *err = norm_err;
        return false;
    }
    *out = trim_text(*out);
    if (out->empty()) {
        if (err) *err = "pdf contains no extractable text";
        return false;
    }
    return true;
}
