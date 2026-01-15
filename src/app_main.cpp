#include "rag_ingest.h"
#include "rag_text.h"
#include "rag_vector_db.h"

#include "json_utils.h"
#include "ncnn_llm_gpt.h"
#include "util.h"
#include "utils/prompt.h"
#include "web_assets_embedded.h"

#include <httplib.h>
#include <nlohmann/json.hpp>

#if defined(NCNN_RAG_HAS_VULKAN_API) && NCNN_RAG_HAS_VULKAN_API
#include <ncnn/gpu.h>
#endif

#if defined(__GLIBC__)
#include <malloc.h>
#endif

#include <cstdlib>
#include <algorithm>
#include <chrono>
#include <cctype>
#include <cinttypes>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_set>

#if defined(_WIN32)
#include <io.h>
#else
#include <unistd.h>
#endif

using nlohmann::json;

namespace {

bool is_embedded_web_root(const std::string& web_root) {
    if (web_root.empty()) return true;
    if (web_root == "embedded") return true;
    if (web_root == ":embedded:") return true;
    return false;
}

struct HttpUrlParts {
    std::string scheme;
    std::string host;
    int port = 80;
    std::string base_path;
};

struct MemSnapshot {
    size_t rss_bytes = 0;
    size_t hwm_bytes = 0;
};

bool getenv_int(const char* name, int* out) {
    if (!out) return false;
    const char* v = std::getenv(name);
    if (!v || !*v) return false;
    if (auto parsed = parse_int(v)) {
        *out = *parsed;
        return true;
    }
    return false;
}

void configure_glibc_malloc_from_env() {
#if defined(__GLIBC__)
    int arena_max = 0;
    if (getenv_int("NCNN_RAG_MALLOC_ARENA_MAX", &arena_max) && arena_max > 0) {
        mallopt(M_ARENA_MAX, arena_max);
    }

    int trim_threshold = 0;
    if (getenv_int("NCNN_RAG_MALLOC_TRIM_THRESHOLD", &trim_threshold) && trim_threshold >= 0) {
        mallopt(M_TRIM_THRESHOLD, trim_threshold);
    }

    int mmap_threshold = 0;
    if (getenv_int("NCNN_RAG_MALLOC_MMAP_THRESHOLD", &mmap_threshold) && mmap_threshold >= 0) {
        mallopt(M_MMAP_THRESHOLD, mmap_threshold);
    }
#endif
}

void maybe_malloc_trim(bool enabled) {
#if defined(__GLIBC__)
    if (enabled) {
        malloc_trim(0);
    }
#else
    (void)enabled;
#endif
}

size_t parse_proc_status_kb_line(const std::string& line) {
    // Example: "VmRSS:\t  12345 kB"
    size_t i = 0;
    while (i < line.size() && (line[i] < '0' || line[i] > '9')) ++i;
    size_t value = 0;
    for (; i < line.size() && line[i] >= '0' && line[i] <= '9'; ++i) {
        value = value * 10 + static_cast<size_t>(line[i] - '0');
    }
    return value * 1024;
}

MemSnapshot read_self_mem_snapshot() {
    MemSnapshot out;
    std::ifstream f("/proc/self/status");
    if (!f) return out;
    std::string line;
    while (std::getline(f, line)) {
        if (line.rfind("VmRSS:", 0) == 0) {
            out.rss_bytes = parse_proc_status_kb_line(line);
        } else if (line.rfind("VmHWM:", 0) == 0) {
            out.hwm_bytes = parse_proc_status_kb_line(line);
        }
    }
    return out;
}

size_t kv_cache_bytes(const std::shared_ptr<ncnn_llm_gpt_ctx>& ctx) {
    if (!ctx) return 0;
    size_t total = 0;
    for (const auto& kv : ctx->kv_cache) {
        const ncnn::Mat& k = kv.first;
        const ncnn::Mat& v = kv.second;
        total += static_cast<size_t>(k.total()) * static_cast<size_t>(k.elemsize);
        total += static_cast<size_t>(v.total()) * static_cast<size_t>(v.elemsize);
    }
    return total;
}

bool parse_url_base(const std::string& url_in, HttpUrlParts* out, std::string* err) {
    if (!out) return false;
    auto pos = url_in.find("://");
    if (pos == std::string::npos) {
        if (err) *err = "missing scheme";
        return false;
    }
    std::string scheme = url_in.substr(0, pos);
    std::string rest = url_in.substr(pos + 3);
    if (scheme != "http" && scheme != "https") {
        if (err) *err = "unsupported scheme: " + scheme;
        return false;
    }

    std::string host_port;
    std::string path = "/";
    auto slash = rest.find('/');
    if (slash == std::string::npos) {
        host_port = rest;
    } else {
        host_port = rest.substr(0, slash);
        path = rest.substr(slash);
    }

    std::string host = host_port;
    int port = (scheme == "https") ? 443 : 80;
    auto colon = host_port.rfind(':');
    if (colon != std::string::npos && colon + 1 < host_port.size()) {
        bool all_digits = true;
        for (size_t i = colon + 1; i < host_port.size(); ++i) {
            if (!std::isdigit(static_cast<unsigned char>(host_port[i]))) {
                all_digits = false;
                break;
            }
        }
        if (all_digits) {
            host = host_port.substr(0, colon);
            port = std::atoi(host_port.substr(colon + 1).c_str());
            if (port <= 0) port = (scheme == "https") ? 443 : 80;
        }
    }

    if (host.empty()) {
        if (err) *err = "missing host";
        return false;
    }
    if (path.empty()) path = "/";
    if (path.back() != '/') path.push_back('/');

    out->scheme = scheme;
    out->host = host;
    out->port = port;
    out->base_path = path;
    return true;
}

bool parse_host_port(const std::string& in, std::string* host, int* port, std::string* err) {
    if (!host || !port) return false;
    auto pos = in.rfind(':');
    if (pos == std::string::npos || pos == 0 || pos + 1 >= in.size()) {
        if (err) *err = "expected HOST:PORT";
        return false;
    }
    std::string h = in.substr(0, pos);
    std::string p = in.substr(pos + 1);
    for (char c : p) {
        if (!std::isdigit(static_cast<unsigned char>(c))) {
            if (err) *err = "invalid port";
            return false;
        }
    }
    int v = std::atoi(p.c_str());
    if (v <= 0 || v > 65535) {
        if (err) *err = "port out of range";
        return false;
    }
    *host = h;
    *port = v;
    return true;
}

bool file_exists_nonempty(const std::filesystem::path& p) {
    std::error_code ec;
    if (!std::filesystem::exists(p, ec)) return false;
    auto sz = std::filesystem::file_size(p, ec);
    if (ec) return false;
    return sz > 0;
}

std::vector<std::string> expected_model_files_from_config(const std::filesystem::path& model_dir, std::string* err) {
    std::ifstream ifs((model_dir / "model.json").string());
    if (!ifs) {
        if (err) *err = "missing model.json";
        return {};
    }
    json config;
    try {
        ifs >> config;
    } catch (const std::exception& e) {
        if (err) *err = std::string("parse model.json: ") + e.what();
        return {};
    }

    std::unordered_set<std::string> uniq;
    uniq.insert("model.json");
    try {
        auto params = config.at("params");
        uniq.insert(params.at("decoder_param").get<std::string>());
        uniq.insert(params.at("decoder_bin").get<std::string>());
        uniq.insert(params.at("embed_token_param").get<std::string>());
        uniq.insert(params.at("embed_token_bin").get<std::string>());
        uniq.insert(params.at("proj_out_param").get<std::string>());
        uniq.insert(params.at("proj_out_bin").get<std::string>());
        auto tok = config.at("tokenizer");
        uniq.insert(tok.at("vocab_file").get<std::string>());
        uniq.insert(tok.at("merges_file").get<std::string>());
    } catch (const std::exception& e) {
        if (err) *err = std::string("model.json missing fields: ") + e.what();
        return {};
    }

    std::vector<std::string> files;
    files.reserve(uniq.size());
    for (const auto& s : uniq) files.push_back(s);
    std::sort(files.begin(), files.end());
    return files;
}

bool is_model_complete(const std::filesystem::path& model_dir, std::vector<std::string>* missing_files, std::string* err) {
    if (!std::filesystem::is_directory(model_dir)) {
        if (missing_files) missing_files->push_back("model.json");
        if (err) *err = "model dir not found: " + model_dir.string();
        return false;
    }
    if (!file_exists_nonempty(model_dir / "model.json")) {
        if (missing_files) missing_files->push_back("model.json");
        if (err) *err = "model.json missing";
        return false;
    }

    std::string parse_err;
    auto expected = expected_model_files_from_config(model_dir, &parse_err);
    if (expected.empty()) {
        if (missing_files) missing_files->push_back("model.json");
        if (err) *err = parse_err.empty() ? "invalid model.json" : parse_err;
        return false;
    }

    bool ok = true;
    if (missing_files) missing_files->clear();
    for (const auto& rel : expected) {
        auto p = model_dir / rel;
        if (!file_exists_nonempty(p)) {
            ok = false;
            if (missing_files) missing_files->push_back(rel);
        }
    }
    if (!ok && err) *err = "missing or empty model files";
    return ok;
}

struct CurlDownloadOptions {
    long connect_timeout_sec = 15;
    long stall_timeout_sec = 60;
    long stall_min_bytes_per_sec = 1;
    long total_timeout_sec = 0; // 0 = no overall timeout (avoid breaking large model downloads)
    std::string proxy_host;
    int proxy_port = 0;
    bool show_progress = true; // TTY-only; ignored in non-interactive output.
};

bool is_tty_stderr() {
#if defined(_WIN32)
    return _isatty(_fileno(stderr)) != 0;
#else
    return isatty(fileno(stderr)) != 0;
#endif
}

std::string human_bytes(uint64_t bytes) {
    static const char* units[] = {"B", "KiB", "MiB", "GiB", "TiB"};
    double v = static_cast<double>(bytes);
    int idx = 0;
    while (v >= 1024.0 && idx < 4) {
        v /= 1024.0;
        ++idx;
    }
    char buf[64];
    if (idx == 0) {
        std::snprintf(buf, sizeof(buf), "%" PRIu64 " %s", bytes, units[idx]);
    } else {
        std::snprintf(buf, sizeof(buf), "%.1f %s", v, units[idx]);
    }
    return buf;
}

struct DownloadProgressPrinter {
    std::string label;
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    std::chrono::steady_clock::time_point last_print = start;
    uint64_t last_bytes = 0;
    bool enabled = false;

    explicit DownloadProgressPrinter(std::string label_, bool enabled_)
        : label(std::move(label_)), enabled(enabled_) {}

    void finish_line() {
        if (!enabled) return;
        std::cerr << "\n";
    }

    bool update(uint64_t current, uint64_t total) {
        if (!enabled) return true;
        auto now = std::chrono::steady_clock::now();
        if (now - last_print < std::chrono::milliseconds(120) && current != total) return true;

        double seconds = std::chrono::duration_cast<std::chrono::duration<double>>(now - start).count();
        double speed = seconds > 0.0 ? static_cast<double>(current) / seconds : 0.0;
        int width = 26;
        std::string bar;
        bar.reserve(static_cast<size_t>(width));
        if (total > 0) {
            double frac = static_cast<double>(current) / static_cast<double>(total);
            if (frac < 0.0) frac = 0.0;
            if (frac > 1.0) frac = 1.0;
            int filled = static_cast<int>(frac * width + 0.5);
            for (int i = 0; i < width; ++i) bar.push_back(i < filled ? '#' : '.');
            int pct = static_cast<int>(frac * 100.0 + 0.5);
            std::cerr << "\rDownloading " << label << " [" << bar << "] " << pct << "% "
                      << human_bytes(current) << "/" << human_bytes(total)
                      << " (" << human_bytes(static_cast<uint64_t>(speed)) << "/s)";
        } else {
            for (int i = 0; i < width; ++i) bar.push_back('.');
            std::cerr << "\rDownloading " << label << " [" << bar << "] "
                      << human_bytes(current)
                      << " (" << human_bytes(static_cast<uint64_t>(speed)) << "/s)";
        }
        std::cerr.flush();
        last_print = now;
        last_bytes = current;
        return true;
    }
};

template <typename ClientT>
bool download_file_with_httplib(ClientT& cli,
                                const std::string& remote_path,
                                const std::filesystem::path& local_path,
                                const std::string& progress_label,
                                const CurlDownloadOptions& opt,
                                std::string* err) {
    std::filesystem::create_directories(local_path.parent_path());
    std::filesystem::path tmp = local_path;
    tmp += ".part";

    std::ofstream ofs(tmp, std::ios::binary);
    if (!ofs) {
        if (err) *err = "open temp file failed: " + tmp.string();
        return false;
    }

    bool write_ok = true;
    const bool progress_enabled = opt.show_progress && is_tty_stderr();
    DownloadProgressPrinter printer(progress_label, progress_enabled);
    uint64_t downloaded = 0;
    auto res = cli.Get(
        remote_path,
        [&](const char* data, size_t data_length) {
        if (!write_ok) return false;
        ofs.write(data, static_cast<std::streamsize>(data_length));
        if (!ofs) {
            write_ok = false;
            return false;
        }
        downloaded += static_cast<uint64_t>(data_length);
        return true;
        },
        [&](uint64_t current, uint64_t total) {
            // Prefer httplib's current/total when available; otherwise fall back to bytes written.
            uint64_t cur = current ? current : downloaded;
            return printer.update(cur, total);
        });

    ofs.close();
    printer.update(downloaded, downloaded);
    printer.finish_line();
    if (!res) {
        std::error_code ec;
        std::filesystem::remove(tmp, ec);
        if (err) *err = "http request failed: " + remote_path;
        return false;
    }
    if (res->status != 200) {
        std::error_code ec;
        std::filesystem::remove(tmp, ec);
        if (err) *err = "http status " + std::to_string(res->status) + " for " + remote_path;
        return false;
    }
    if (!write_ok) {
        std::error_code ec;
        std::filesystem::remove(tmp, ec);
        if (err) *err = "write failed: " + tmp.string();
        return false;
    }

    std::error_code ec;
    std::filesystem::rename(tmp, local_path, ec);
    if (ec) {
        std::filesystem::remove(local_path, ec);
        ec.clear();
        std::filesystem::rename(tmp, local_path, ec);
    }
    if (ec) {
        if (err) *err = "rename failed: " + ec.message();
        return false;
    }
    return true;
}

bool download_url_to_file(const HttpUrlParts& base,
                          const std::string& rel,
                          const std::filesystem::path& local_path,
                          const CurlDownloadOptions& opt,
                          const std::string& progress_label,
                          std::string* err) {
    std::string remote_path = base.base_path + rel;
    if (base.scheme == "https") {
        httplib::SSLClient cli(base.host, base.port);
        cli.set_follow_location(true);
        cli.set_connection_timeout(static_cast<int>(opt.connect_timeout_sec));
        cli.set_read_timeout(static_cast<int>(opt.stall_timeout_sec));
        if (!opt.proxy_host.empty() && opt.proxy_port > 0) {
            cli.set_proxy(opt.proxy_host, opt.proxy_port);
        }
        // If you are behind a proxy doing TLS interception, you may need to disable verification or set custom CA.
        cli.enable_server_certificate_verification(true);
        return download_file_with_httplib(cli, remote_path, local_path, progress_label, opt, err);
    } else {
        httplib::Client cli(base.host, base.port);
        cli.set_follow_location(true);
        cli.set_connection_timeout(static_cast<int>(opt.connect_timeout_sec));
        cli.set_read_timeout(static_cast<int>(opt.stall_timeout_sec));
        if (!opt.proxy_host.empty() && opt.proxy_port > 0) {
            cli.set_proxy(opt.proxy_host, opt.proxy_port);
        }
        return download_file_with_httplib(cli, remote_path, local_path, progress_label, opt, err);
    }
}

bool ensure_model_downloaded(const std::filesystem::path& model_dir,
                             const std::string& model_url,
                             const CurlDownloadOptions& dlopt,
                             std::string* err) {
    HttpUrlParts base;
    std::string parse_err;
    if (!parse_url_base(model_url, &base, &parse_err)) {
        if (err) *err = "invalid model url: " + parse_err;
        return false;
    }

    // Ensure model.json exists first (so we can infer the rest of required files).
    if (!file_exists_nonempty(model_dir / "model.json")) {
        std::string dl_err;
        if (!download_url_to_file(base, "model.json", model_dir / "model.json", dlopt, "model.json", &dl_err)) {
            if (err) *err = "download model.json failed: " + dl_err;
            return false;
        }
    }

    std::string cfg_err;
    auto expected = expected_model_files_from_config(model_dir, &cfg_err);
    if (expected.empty()) {
        if (err) *err = "invalid downloaded model.json: " + cfg_err;
        return false;
    }

    size_t idx = 0;
    const size_t total_files = expected.size();
    for (const auto& rel : expected) {
        ++idx;
        if (rel == "model.json") continue;
        auto local = model_dir / rel;
        if (file_exists_nonempty(local)) continue;
        std::string dl_err;
        std::ostringstream label;
        label << idx << "/" << total_files << " " << rel;
        if (!download_url_to_file(base, rel, local, dlopt, label.str(), &dl_err)) {
            if (err) *err = "download failed: " + rel + " (" + dl_err + ")";
            return false;
        }
    }
    return true;
}

std::string sanitize_for_log(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (char c : s) {
        if (c == '\n' || c == '\r' || c == '\t') {
            out.push_back(' ');
        } else {
            out.push_back(c);
        }
    }
    return out;
}

std::string truncate_for_log(const std::string& s, size_t max_len) {
    std::string cleaned = sanitize_for_log(s);
    if (cleaned.size() <= max_len) return cleaned;
    return cleaned.substr(0, max_len) + "...(" + std::to_string(cleaned.size()) + " bytes)";
}

std::string sanitize_utf8_strict(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    auto is_cont = [&](unsigned char c) { return (c & 0xC0) == 0x80; };
    for (size_t i = 0; i < s.size();) {
        unsigned char c = static_cast<unsigned char>(s[i]);
        if (c < 0x80) {
            out.push_back(static_cast<char>(c));
            ++i;
            continue;
        }

        // 2-byte: U+0080..U+07FF
        if (c >= 0xC2 && c <= 0xDF) {
            if (i + 1 < s.size() && is_cont(static_cast<unsigned char>(s[i + 1]))) {
                out.append(s, i, 2);
                i += 2;
                continue;
            }
            out.push_back('?');
            ++i;
            continue;
        }

        // 3-byte: U+0800..U+FFFF (excluding surrogates)
        if (c == 0xE0) {
            if (i + 2 < s.size()) {
                unsigned char c1 = static_cast<unsigned char>(s[i + 1]);
                unsigned char c2 = static_cast<unsigned char>(s[i + 2]);
                if (c1 >= 0xA0 && c1 <= 0xBF && is_cont(c2)) {
                    out.append(s, i, 3);
                    i += 3;
                    continue;
                }
            }
            out.push_back('?');
            ++i;
            continue;
        }
        if (c >= 0xE1 && c <= 0xEC) {
            if (i + 2 < s.size() && is_cont(static_cast<unsigned char>(s[i + 1])) &&
                is_cont(static_cast<unsigned char>(s[i + 2]))) {
                out.append(s, i, 3);
                i += 3;
                continue;
            }
            out.push_back('?');
            ++i;
            continue;
        }
        if (c == 0xED) {
            if (i + 2 < s.size()) {
                unsigned char c1 = static_cast<unsigned char>(s[i + 1]);
                unsigned char c2 = static_cast<unsigned char>(s[i + 2]);
                if (c1 >= 0x80 && c1 <= 0x9F && is_cont(c2)) {
                    out.append(s, i, 3);
                    i += 3;
                    continue;
                }
            }
            out.push_back('?');
            ++i;
            continue;
        }
        if (c >= 0xEE && c <= 0xEF) {
            if (i + 2 < s.size() && is_cont(static_cast<unsigned char>(s[i + 1])) &&
                is_cont(static_cast<unsigned char>(s[i + 2]))) {
                out.append(s, i, 3);
                i += 3;
                continue;
            }
            out.push_back('?');
            ++i;
            continue;
        }

        // 4-byte: U+10000..U+10FFFF
        if (c == 0xF0) {
            if (i + 3 < s.size()) {
                unsigned char c1 = static_cast<unsigned char>(s[i + 1]);
                unsigned char c2 = static_cast<unsigned char>(s[i + 2]);
                unsigned char c3 = static_cast<unsigned char>(s[i + 3]);
                if (c1 >= 0x90 && c1 <= 0xBF && is_cont(c2) && is_cont(c3)) {
                    out.append(s, i, 4);
                    i += 4;
                    continue;
                }
            }
            out.push_back('?');
            ++i;
            continue;
        }
        if (c >= 0xF1 && c <= 0xF3) {
            if (i + 3 < s.size() && is_cont(static_cast<unsigned char>(s[i + 1])) &&
                is_cont(static_cast<unsigned char>(s[i + 2])) &&
                is_cont(static_cast<unsigned char>(s[i + 3]))) {
                out.append(s, i, 4);
                i += 4;
                continue;
            }
            out.push_back('?');
            ++i;
            continue;
        }
        if (c == 0xF4) {
            if (i + 3 < s.size()) {
                unsigned char c1 = static_cast<unsigned char>(s[i + 1]);
                unsigned char c2 = static_cast<unsigned char>(s[i + 2]);
                unsigned char c3 = static_cast<unsigned char>(s[i + 3]);
                if (c1 >= 0x80 && c1 <= 0x8F && is_cont(c2) && is_cont(c3)) {
                    out.append(s, i, 4);
                    i += 4;
                    continue;
                }
            }
            out.push_back('?');
            ++i;
            continue;
        }

        out.push_back('?');
        ++i;
    }
    return out;
}

void log_event(const std::string& tag, const std::string& msg) {
    std::cerr << "[" << now_ms_epoch() << "] " << tag << " " << msg << "\n";
}

std::string dump_json_safe(const json& j) {
    return j.dump(-1, ' ', false, json::error_handler_t::replace);
}

size_t utf8_safe_cut_pos(const std::string& s, size_t pos) {
    if (pos >= s.size()) return s.size();
    while (pos > 0) {
        unsigned char c = static_cast<unsigned char>(s[pos]);
        if ((c & 0xC0) != 0x80) break; // not a continuation byte
        --pos;
    }
    return pos;
}

std::vector<std::string> split_prompt_chunks(const std::string& prompt, size_t chunk_bytes) {
    std::vector<std::string> out;
    if (chunk_bytes == 0 || prompt.size() <= chunk_bytes) {
        out.push_back(prompt);
        return out;
    }

    size_t pos = 0;
    while (pos < prompt.size()) {
        size_t remaining = prompt.size() - pos;
        size_t want = std::min(chunk_bytes, remaining);
        size_t end = pos + want;
        if (end < prompt.size()) {
            size_t best = std::string::npos;
            size_t window_start = (end > 256) ? (end - 256) : pos;
            for (size_t i = end; i > window_start; --i) {
                char c = prompt[i - 1];
                if (c == '\n' || c == ' ' || c == '\t') {
                    best = i;
                    break;
                }
            }
            if (best != std::string::npos && best > pos) {
                end = best;
            }
            end = utf8_safe_cut_pos(prompt, end);
            if (end <= pos) {
                end = utf8_safe_cut_pos(prompt, pos + want);
                if (end <= pos) end = std::min(pos + want, prompt.size());
            }
        }
        out.push_back(prompt.substr(pos, end - pos));
        pos = end;
    }
    return out;
}

std::shared_ptr<ncnn_llm_gpt_ctx> prefill_chunked(const ncnn_llm_gpt& model,
                                                  const std::string& prompt,
                                                  size_t chunk_bytes,
                                                  const std::string& req_id) {
    auto chunks = split_prompt_chunks(prompt, chunk_bytes);
    if (chunks.empty()) return nullptr;
    if (chunks.size() == 1) return model.prefill(prompt);

    std::shared_ptr<ncnn_llm_gpt_ctx> ctx;
    for (size_t i = 0; i < chunks.size(); ++i) {
        log_event("chat.prefill.chunk", "id=" + req_id +
                                        " idx=" + std::to_string(i) +
                                        " bytes=" + std::to_string(chunks[i].size()) +
                                        " total_chunks=" + std::to_string(chunks.size()));
        if (ctx) ctx = model.prefill(chunks[i], ctx);
        else ctx = model.prefill(chunks[i]);
    }
    return ctx;
}

std::string summarize_messages(const std::vector<Message>& messages, const std::string& last_user) {
    size_t system = 0;
    size_t user = 0;
    size_t assistant = 0;
    size_t tool = 0;
    size_t other = 0;
    for (const auto& msg : messages) {
        if (msg.role == "system") {
            ++system;
        } else if (msg.role == "user") {
            ++user;
        } else if (msg.role == "assistant") {
            ++assistant;
        } else if (msg.role == "tool") {
            ++tool;
        } else {
            ++other;
        }
    }
    std::ostringstream oss;
    oss << "messages=" << messages.size()
        << " roles(system=" << system
        << ",user=" << user
        << ",assistant=" << assistant
        << ",tool=" << tool
        << ",other=" << other << ")"
        << " last_user_len=" << last_user.size();
    if (!last_user.empty()) {
        oss << " last_user=\"" << truncate_for_log(last_user, 200) << "\"";
    }
    return oss.str();
}

std::string summarize_hits(const std::vector<RagSearchHit>& hits, size_t max_items = 3) {
    std::ostringstream oss;
    oss << "hits=" << hits.size();
    if (!hits.empty()) {
        oss << " top=[";
        size_t limit = std::min(max_items, hits.size());
        for (size_t i = 0; i < limit; ++i) {
            if (i > 0) oss << ",";
            oss << hits[i].source << ":" << hits[i].score;
        }
        oss << "]";
    }
    return oss.str();
}

std::string doc_chunk_url(size_t doc_id, int chunk_index) {
    return "/rag/doc/" + std::to_string(doc_id) + "#chunk-" + std::to_string(chunk_index);
}

std::string escape_html(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (char c : s) {
        switch (c) {
            case '&': out += "&amp;"; break;
            case '<': out += "&lt;"; break;
            case '>': out += "&gt;"; break;
            case '"': out += "&quot;"; break;
            default: out.push_back(c); break;
        }
    }
    return out;
}

struct AppOptions {
    std::string model_path = "assets/qwen3_0.6b";
    std::string model_url = "https://mirrors.sdu.edu.cn/ncnn_modelzoo/qwen3_0.6b/";
    std::string web_root = ":embedded:";
    std::string docs_path = "assets/rag";
    std::string data_dir = "data";
    std::string db_path = "data/rag.sqlite";
    std::string pdf_txt_dir = "data/pdf_txt";
    size_t chunk_size = 600;
    int embed_dim = 256;
    int port = 8080;
    bool use_vulkan = false;
    bool rag_enabled = true;
    size_t rag_top_k = 10;
    int rag_neighbor_chunks = 1;
    size_t rag_chunk_max_chars = 1800;
    size_t llm_prefill_chunk_bytes = 2048;
    bool save_pdf_txt = true;
    bool auto_download_model = true;
    int model_download_connect_timeout_sec = 15;
    int model_download_stall_timeout_sec = 60;
    int model_download_total_timeout_sec = 0; // 0 = no overall timeout
    bool model_download_use_proxy = false;
    std::string model_download_proxy = "";
    bool malloc_trim = false;
};

void print_usage(const char* argv0) {
    std::cout << "Usage: " << argv0 << " [options]\n"
              << "  --model PATH      Model directory (default: assets/qwen3_0.6b)\n"
              << "  --model-url URL   Model download base URL (default: https://mirrors.sdu.edu.cn/ncnn_modelzoo/qwen3_0.6b/)\n"
              << "  --model-dl-connect-timeout N  Connect timeout in seconds (default: 15)\n"
              << "  --model-dl-stall-timeout N    Abort if transfer stalls for N seconds (default: 60)\n"
              << "  --model-dl-timeout N          Overall timeout per file (0=disable, default: 0)\n"
              << "  --model-dl-proxy HOST:PORT    Use HTTP proxy for downloads (default: none)\n"
              << "  --no-model-dl-proxy           Disable download proxy\n"
              << "  --docs PATH       Seed docs directory (default: assets/rag)\n"
              << "  --web PATH        Web root to serve (default: :embedded:)\n"
              << "  --data PATH       Data directory (default: data)\n"
              << "  --db PATH         SQLite database path (default: data/rag.sqlite)\n"
              << "  --pdf-txt PATH    Exported PDF text directory (default: data/pdf_txt)\n"
              << "  --chunk-size N    Chunk size for indexing (default: 600)\n"
              << "  --embed-dim N     Embedding dimension (default: 256)\n"
              << "  --port N          HTTP port (default: 8080)\n"
              << "  --rag-top-k N     Retrieved chunks (default: 10)\n"
              << "  --rag-neighbors N Include neighbor chunks around each hit (default: 1)\n"
              << "  --rag-chunk-max N Max chars per returned chunk after expansion (default: 1800)\n"
              << "  --prefill-chunk-bytes N Chunk prompt for prefill to reduce memory (default: 2048)\n"
              << "  --no-model-download Disable automatic model download\n"
              << "  --no-rag          Disable retrieval\n"
              << "  --no-pdf-txt      Disable exporting extracted PDF text\n"
              << "  --vulkan          Enable Vulkan compute\n"
              << "  --malloc-trim     Call malloc_trim(0) after each request (glibc)\n"
              << "  --help            Show this help\n";
}

AppOptions parse_options(int argc, char** argv) {
    AppOptions opt;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help") {
            print_usage(argv[0]);
            std::exit(0);
        } else if (arg == "--model" && i + 1 < argc) {
            opt.model_path = argv[++i];
        } else if (arg == "--model-url" && i + 1 < argc) {
            opt.model_url = argv[++i];
        } else if (arg == "--model-dl-connect-timeout" && i + 1 < argc) {
            if (auto v = parse_int(argv[++i])) opt.model_download_connect_timeout_sec = std::max(1, *v);
        } else if (arg == "--model-dl-stall-timeout" && i + 1 < argc) {
            if (auto v = parse_int(argv[++i])) opt.model_download_stall_timeout_sec = std::max(1, *v);
        } else if (arg == "--model-dl-timeout" && i + 1 < argc) {
            if (auto v = parse_int(argv[++i])) opt.model_download_total_timeout_sec = std::max(0, *v);
        } else if (arg == "--model-dl-proxy" && i + 1 < argc) {
            opt.model_download_proxy = argv[++i];
            opt.model_download_use_proxy = true;
        } else if (arg == "--no-model-dl-proxy") {
            opt.model_download_use_proxy = false;
        } else if (arg == "--docs" && i + 1 < argc) {
            opt.docs_path = argv[++i];
        } else if (arg == "--web" && i + 1 < argc) {
            opt.web_root = argv[++i];
        } else if (arg == "--data" && i + 1 < argc) {
            opt.data_dir = argv[++i];
        } else if (arg == "--db" && i + 1 < argc) {
            opt.db_path = argv[++i];
        } else if (arg == "--pdf-txt" && i + 1 < argc) {
            opt.pdf_txt_dir = argv[++i];
        } else if (arg == "--chunk-size" && i + 1 < argc) {
            if (auto v = parse_int(argv[++i])) opt.chunk_size = static_cast<size_t>(*v);
        } else if (arg == "--embed-dim" && i + 1 < argc) {
            if (auto v = parse_int(argv[++i])) opt.embed_dim = *v;
        } else if (arg == "--port" && i + 1 < argc) {
            if (auto v = parse_int(argv[++i])) opt.port = *v;
        } else if (arg == "--rag-top-k" && i + 1 < argc) {
            if (auto v = parse_int(argv[++i])) opt.rag_top_k = static_cast<size_t>(*v);
        } else if (arg == "--rag-neighbors" && i + 1 < argc) {
            if (auto v = parse_int(argv[++i])) opt.rag_neighbor_chunks = *v;
        } else if (arg == "--rag-chunk-max" && i + 1 < argc) {
            if (auto v = parse_int(argv[++i])) opt.rag_chunk_max_chars = static_cast<size_t>(*v);
        } else if (arg == "--prefill-chunk-bytes" && i + 1 < argc) {
            if (auto v = parse_int(argv[++i])) {
                if (*v > 0) opt.llm_prefill_chunk_bytes = static_cast<size_t>(*v);
            }
        } else if (arg == "--no-model-download") {
            opt.auto_download_model = false;
        } else if (arg == "--no-rag") {
            opt.rag_enabled = false;
        } else if (arg == "--no-pdf-txt") {
            opt.save_pdf_txt = false;
        } else if (arg == "--vulkan") {
            opt.use_vulkan = true;
        } else if (arg == "--malloc-trim") {
            opt.malloc_trim = true;
        }
    }
    return opt;
}

std::string normalize_path(const std::string& path, const std::string& base) {
    std::filesystem::path p(path);
    if (p.is_absolute()) return path;
    if (!p.has_parent_path()) {
        return (std::filesystem::path(base) / p).string();
    }
    return path;
}

std::string build_rag_context(const std::vector<RagSearchHit>& hits) {
    if (hits.empty()) return {};
    std::string ctx;
    for (size_t i = 0; i < hits.size(); ++i) {
        ctx += "[" + std::to_string(i + 1) + "] Source: " + sanitize_utf8_strict(hits[i].source) + "\n";
        ctx += sanitize_utf8_strict(hits[i].text) + "\n\n";
    }
    return ctx;
}

void expand_hits_with_neighbors(const RagVectorDb& rag,
                                std::vector<RagSearchHit>& hits,
                                int neighbor_chunks,
                                size_t max_chunk_chars) {
    if (neighbor_chunks <= 0 || hits.empty()) return;

    struct RangeHit {
        size_t doc_id = 0;
        int start = 0;
        int end = 0;
        double best_score = 0.0;
        int center_chunk_index = 0;
        std::string source;
    };

    std::vector<RangeHit> ranges;
    ranges.reserve(hits.size());
    for (const auto& hit : hits) {
        int start = hit.chunk_index - neighbor_chunks;
        if (start < 0) start = 0;
        int end = hit.chunk_index + neighbor_chunks;
        RangeHit r;
        r.doc_id = hit.doc_id;
        r.start = start;
        r.end = end;
        r.best_score = hit.score;
        r.center_chunk_index = hit.chunk_index;
        r.source = hit.source;
        ranges.push_back(std::move(r));
    }

    std::sort(ranges.begin(), ranges.end(), [](const RangeHit& a, const RangeHit& b) {
        if (a.doc_id != b.doc_id) return a.doc_id < b.doc_id;
        if (a.start != b.start) return a.start < b.start;
        return a.end < b.end;
    });

    std::vector<RangeHit> merged;
    merged.reserve(ranges.size());
    for (const auto& r : ranges) {
        if (merged.empty() || merged.back().doc_id != r.doc_id || r.start > merged.back().end) {
            merged.push_back(r);
            continue;
        }
        auto& cur = merged.back();
        cur.end = std::max(cur.end, r.end);
        if (r.best_score > cur.best_score) {
            cur.best_score = r.best_score;
            cur.center_chunk_index = r.center_chunk_index;
            cur.source = r.source;
        }
    }

    hits.clear();
    hits.reserve(merged.size());
    for (const auto& r : merged) {
        RagSearchHit out;
        out.doc_id = r.doc_id;
        out.chunk_index = r.center_chunk_index;
        out.source = r.source;
        out.score = r.best_score;
        std::string expanded = rag.expand_range(r.doc_id, r.start, r.end, r.center_chunk_index);
        if (!expanded.empty()) {
            out.text = shorten_text(sanitize_utf8_strict(expanded), max_chunk_chars);
        }
        hits.push_back(std::move(out));
    }
}

std::string build_system_prompt(const std::string& rag_context, bool rag_enabled) {
    std::string prompt =
        "You are a helpful assistant. Answer using the provided context. "
        "If the context does not contain the answer, say you do not know. "
        "Keep responses concise and cite sources by their bracketed ids.";
    if (rag_enabled && !rag_context.empty()) {
        prompt += "\n\nContext:\n" + rag_context;
    } else if (rag_enabled) {
        prompt += "\n\nContext:\n(No relevant sources found.)";
    }
    return prompt;
}

json build_rag_payload(const std::vector<RagSearchHit>& hits,
                       bool rag_enabled,
                       size_t top_k,
                       size_t doc_count,
                       size_t chunk_count,
                       const std::vector<std::string>* trace = nullptr,
                       const std::string* error = nullptr) {
    json rag = {
        {"enabled", rag_enabled},
        {"top_k", top_k},
        {"doc_count", doc_count},
        {"chunk_count", chunk_count},
        {"chunks", json::array()}
    };
    if (trace) rag["trace"] = *trace;
    if (error && !error->empty()) rag["error"] = *error;
    for (const auto& hit : hits) {
        rag["chunks"].push_back({
            {"source", sanitize_utf8_strict(hit.source)},
            {"score", hit.score},
            {"text", sanitize_utf8_strict(hit.text)},
            {"doc_id", hit.doc_id},
            {"chunk_index", hit.chunk_index},
            {"url", doc_chunk_url(hit.doc_id, hit.chunk_index)}
        });
    }
    return rag;
}

json rag_tool_schema() {
    return {
        {"name", "rag_search"},
        {"description", "Search local documents and return relevant chunks."},
        {"inputSchema", {
            {"type", "object"},
            {"properties", {
                {"query", {{"type", "string"}, {"description", "User query"}}},
                {"top_k", {{"type", "integer"}, {"minimum", 1}, {"maximum", 10}}}
            }},
            {"required", json::array({"query"})}
        }}
    };
}

json rag_tool_call(const json& args,
                   const RagVectorDb& rag,
                   const RagEmbedder& embedder,
                   size_t default_top_k,
                   int neighbor_chunks,
                   size_t max_chunk_chars) {
    const std::string query = args.value("query", std::string());
    size_t top_k = default_top_k;
    if (args.contains("top_k") && args["top_k"].is_number_integer()) {
        int v = args["top_k"].get<int>();
        if (v > 0) top_k = static_cast<size_t>(v);
    }

    auto t0 = std::chrono::steady_clock::now();
    std::vector<std::string> trace;
    std::vector<float> query_vec;
    std::vector<RagSearchHit> hits;
    if (!query.empty()) {
        trace.push_back("tokenize+embed");
        query_vec = embedder.embed(query);
        trace.push_back("vector search");
        hits = rag.search(query_vec, top_k);
        if (neighbor_chunks > 0 && !hits.empty()) {
            trace.push_back("expand neighbors");
            trace.push_back("dedupe overlaps");
            expand_hits_with_neighbors(rag, hits, neighbor_chunks, max_chunk_chars);
        }
    }
    auto t1 = std::chrono::steady_clock::now();
    int64_t elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

    json result = {
        {"query", query},
        {"top_k", top_k},
        {"elapsed_ms", elapsed_ms},
        {"trace", trace},
        {"chunks", json::array()},
        {"context", build_rag_context(hits)}
    };
    for (const auto& hit : hits) {
        result["chunks"].push_back({
            {"source", sanitize_utf8_strict(hit.source)},
            {"score", hit.score},
            {"text", sanitize_utf8_strict(hit.text)},
            {"doc_id", hit.doc_id},
            {"chunk_index", hit.chunk_index},
            {"url", doc_chunk_url(hit.doc_id, hit.chunk_index)}
        });
    }
    return result;
}

std::string to_lower_copy(std::string s) {
    for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

std::string file_ext_lower(const std::string& name) {
    std::filesystem::path p(name);
    std::string ext = p.extension().string();
    return to_lower_copy(ext);
}

std::string sanitize_filename(std::string s) {
    for (char& c : s) {
        if (c == '/' || c == '\\' || c == ':' || c == '\0' ||
            c == '*' || c == '?' || c == '"' || c == '<' || c == '>' || c == '|') {
            c = '_';
        }
    }
    return s;
}

std::filesystem::path path_from_utf8(const std::string& u8) {
#if defined(_WIN32)
    return std::filesystem::u8path(u8);
#else
    return std::filesystem::path(u8);
#endif
}

bool write_text_file_utf8(const std::filesystem::path& path, const std::string& text, std::string* err) {
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs) {
        if (err) *err = "failed to write file";
        return false;
    }
    ofs.write(text.data(), static_cast<std::streamsize>(text.size()));
    if (!ofs) {
        if (err) *err = "failed to write file";
        return false;
    }
    return true;
}

bool write_file(const std::filesystem::path& path, const std::string& data, std::string* err) {
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs) {
        if (err) *err = "failed to write file";
        return false;
    }
    ofs.write(data.data(), static_cast<std::streamsize>(data.size()));
    if (!ofs) {
        if (err) *err = "failed to write file";
        return false;
    }
    return true;
}

bool ingest_document(const std::string& filename,
                     const std::string& mime,
                     const std::filesystem::path& path,
                     RagVectorDb& rag,
                     const AppOptions& opt,
                     std::vector<std::string>* trace,
                     size_t* out_doc_id,
                     size_t* out_chunks,
                     std::string* err) {
    std::string text;
    std::string local_err;
    std::string normalized_filename = filename;
    std::string ext = file_ext_lower(filename);

    if (trace) trace->push_back("read content");
    if (ext == ".txt") {
        if (!read_text_file(path, &text, &local_err)) {
            if (err) *err = local_err;
            return false;
        }
    } else if (ext == ".pdf") {
        if (!extract_pdf_text(path, &text, &local_err)) {
            if (err) *err = local_err;
            return false;
        }
        if (opt.save_pdf_txt) {
            std::error_code dir_ec;
            std::filesystem::create_directories(opt.pdf_txt_dir, dir_ec);
            if (dir_ec) {
                if (trace) trace->push_back("pdf txt export skipped: " + dir_ec.message());
            } else {
                std::filesystem::path outdir(opt.pdf_txt_dir);
                std::string base = sanitize_filename(std::filesystem::path(filename).stem().string());
                if (base.empty()) base = "pdf";
                std::filesystem::path outpath = outdir / (base + ".txt");
                for (int i = 1; std::filesystem::exists(outpath, dir_ec) && i < 1000; ++i) {
                    outpath = outdir / (base + "_" + std::to_string(i) + ".txt");
                }
                if (!dir_ec) {
                    if (write_text_file_utf8(outpath, text, &local_err)) {
                        if (trace) trace->push_back("export pdf txt to " + outpath.string());
                    } else if (trace) {
                        trace->push_back("pdf txt export failed: " + local_err);
                    }
                }
            }
        }
    } else {
        if (err) *err = "unsupported file type";
        return false;
    }

    // Ensure source/metadata is valid UTF-8 for web/UI output.
    {
        std::string name_err;
        if (!normalize_utf8(&normalized_filename, &name_err)) {
            if (trace) trace->push_back("warn: filename not utf8 (" + name_err + ")");
            normalized_filename = filename;
        }
    }

    if (trace) trace->push_back("chunk+embed+store");
    size_t doc_id = 0;
    size_t chunk_count = 0;
    if (!rag.add_document(normalized_filename, mime, text, opt.chunk_size, &local_err, &doc_id, &chunk_count)) {
        if (err) *err = local_err;
        return false;
    }
    if (out_doc_id) *out_doc_id = doc_id;
    if (out_chunks) *out_chunks = chunk_count;
    return true;
}

size_t ingest_directory(const std::string& dir,
                        RagVectorDb& rag,
                        const AppOptions& opt,
                        std::vector<std::string>* trace) {
    std::error_code ec;
    std::filesystem::path root(dir);
    if (!std::filesystem::exists(root, ec)) return 0;

    size_t count = 0;
    for (auto it = std::filesystem::recursive_directory_iterator(root, ec);
         it != std::filesystem::recursive_directory_iterator();
         it.increment(ec)) {
        if (ec) break;
        if (!it->is_regular_file()) continue;
        const auto path = it->path();
        std::string ext = file_ext_lower(path.string());
        if (ext != ".txt" && ext != ".pdf") continue;

        std::string filename = path.filename().string();
        std::string err;
        size_t doc_id = 0;
        size_t chunks = 0;
        if (ingest_document(filename, ext == ".pdf" ? "application/pdf" : "text/plain", path, rag, opt, trace, &doc_id, &chunks, &err)) {
            ++count;
        } else if (trace) {
            trace->push_back("skip " + filename + ": " + err);
        }
    }
    return count;
}

} // namespace

int main(int argc, char** argv) {
    AppOptions opt = parse_options(argc, argv);
    configure_glibc_malloc_from_env();
    {
        int trim_enabled = 0;
        if (getenv_int("NCNN_RAG_MALLOC_TRIM", &trim_enabled)) {
            opt.malloc_trim = (trim_enabled != 0);
        }
    }
    opt.model_path = normalize_path(opt.model_path, "./assets");
    opt.docs_path = normalize_path(opt.docs_path, ".");
    opt.data_dir = normalize_path(opt.data_dir, ".");
    opt.db_path = normalize_path(opt.db_path, opt.data_dir);
    opt.pdf_txt_dir = normalize_path(opt.pdf_txt_dir, opt.data_dir);

    {
        std::vector<std::string> missing;
        std::string model_err;
        if (!is_model_complete(opt.model_path, &missing, &model_err)) {
            if (opt.auto_download_model) {
                log_event("model.download", "needed=1 model_path=" + opt.model_path +
                                               " url=" + opt.model_url +
                                               " missing_files=" + std::to_string(missing.size()));
                std::string dl_err;
                CurlDownloadOptions dlopt;
                dlopt.connect_timeout_sec = std::max(1, opt.model_download_connect_timeout_sec);
                dlopt.stall_timeout_sec = std::max(1, opt.model_download_stall_timeout_sec);
                dlopt.total_timeout_sec = std::max(0, opt.model_download_total_timeout_sec);
                {
                    int progress = 1;
                    if (getenv_int("NCNN_RAG_MODEL_DL_PROGRESS", &progress)) {
                        dlopt.show_progress = (progress != 0);
                    }
                }
                if (opt.model_download_use_proxy) {
                    std::string proxy_host;
                    int proxy_port = 0;
                    std::string proxy_err;
                    if (!parse_host_port(opt.model_download_proxy, &proxy_host, &proxy_port, &proxy_err)) {
                        std::cerr << "Invalid proxy '" << opt.model_download_proxy << "': " << proxy_err << "\n";
                        return 2;
                    }
                    dlopt.proxy_host = proxy_host;
                    dlopt.proxy_port = proxy_port;
                }
                if (!ensure_model_downloaded(opt.model_path, opt.model_url, dlopt, &dl_err)) {
                    std::cerr << "Model download failed: " << dl_err << "\n";
                    log_event("model.download.error", dl_err);
                    return 2;
                }
                missing.clear();
                model_err.clear();
                if (!is_model_complete(opt.model_path, &missing, &model_err)) {
                    std::cerr << "Model is still incomplete after download: " << model_err << "\n";
                    log_event("model.download.error", "incomplete_after_download missing=" + std::to_string(missing.size()));
                    return 2;
                }
                log_event("model.download.done", "ok=1 model_path=" + opt.model_path);
            } else {
                std::cerr << "Model not found or incomplete at " << opt.model_path << " (" << model_err << ")\n";
                std::cerr << "Tip: remove --no-model-download, or pass --model to a complete model dir.\n";
                return 2;
            }
        }
    }

    bool use_vulkan_runtime = opt.use_vulkan;
#if defined(NCNN_RAG_HAS_VULKAN_API) && NCNN_RAG_HAS_VULKAN_API
    if (opt.use_vulkan) {
        ncnn::create_gpu_instance();
        int gpu_count = ncnn::get_gpu_count();
        if (gpu_count <= 0) {
            use_vulkan_runtime = false;
            log_event("vulkan", "requested=1 gpu_count=0 (fallback to cpu)");
        } else {
            log_event("vulkan", "requested=1 gpu_count=" + std::to_string(gpu_count));
        }
    }
#else
    if (opt.use_vulkan) {
        use_vulkan_runtime = false;
        log_event("vulkan", "requested=1 but ncnn gpu api unavailable (fallback to cpu)");
    }
#endif

    log_event("startup", "model_path=" + opt.model_path +
                         " model_url=" + opt.model_url +
                         " auto_model_download=" + std::string(opt.auto_download_model ? "1" : "0") +
                         " model_dl_connect_timeout=" + std::to_string(opt.model_download_connect_timeout_sec) +
                         " model_dl_stall_timeout=" + std::to_string(opt.model_download_stall_timeout_sec) +
                         " model_dl_timeout=" + std::to_string(opt.model_download_total_timeout_sec) +
                         " model_dl_proxy=" + std::string(opt.model_download_use_proxy ? opt.model_download_proxy : "disabled") +
                         " docs_path=" + opt.docs_path +
                         " web_root=" + opt.web_root +
                         " data_dir=" + opt.data_dir +
                         " db_path=" + opt.db_path +
                         " pdf_txt_dir=" + opt.pdf_txt_dir +
                         " chunk_size=" + std::to_string(opt.chunk_size) +
                         " embed_dim=" + std::to_string(opt.embed_dim) +
                         " rag_top_k=" + std::to_string(opt.rag_top_k) +
                         " rag_neighbor_chunks=" + std::to_string(opt.rag_neighbor_chunks) +
                         " rag_chunk_max_chars=" + std::to_string(opt.rag_chunk_max_chars) +
                         " prefill_chunk_bytes=" + std::to_string(opt.llm_prefill_chunk_bytes) +
                         " rag_enabled=" + std::string(opt.rag_enabled ? "1" : "0") +
                         " save_pdf_txt=" + std::string(opt.save_pdf_txt ? "1" : "0") +
                         " vulkan=" + std::string(opt.use_vulkan ? "1" : "0") +
                         " vulkan_runtime=" + std::string(use_vulkan_runtime ? "1" : "0"));

    std::filesystem::path data_root(opt.data_dir);
    std::filesystem::path upload_dir = data_root / "uploads";
    std::filesystem::path pdf_txt_dir(opt.pdf_txt_dir);
    std::error_code ec;
    std::filesystem::create_directories(upload_dir, ec);
    if (ec) {
        std::cerr << "Warning: failed to create data dir: " << ec.message() << "\n";
    }
    if (opt.save_pdf_txt) {
        ec.clear();
        std::filesystem::create_directories(pdf_txt_dir, ec);
        if (ec) {
            std::cerr << "Warning: failed to create pdf txt dir: " << ec.message() << "\n";
        }
    }

    RagVectorDb rag;
    RagEmbedder embedder(opt.embed_dim);
    std::mutex rag_mutex;
    std::string rag_err;
    bool rag_ready = rag.open(opt.db_path, opt.embed_dim, &rag_err);
    std::string rag_open_err = rag_err;
    if (!rag_ready) {
        std::cerr << "RAG db warning: " << rag_err << "\n";
        log_event("rag.db", "ready=0 err=" + rag_err);
    } else if (rag.chunk_count() == 0) {
        std::vector<std::string> seed_trace;
        size_t ingested = ingest_directory(opt.docs_path, rag, opt, &seed_trace);
        if (ingested > 0) {
            std::cerr << "Seeded " << ingested << " document(s) from " << opt.docs_path << "\n";
        }
        log_event("rag.db", "ready=1 doc_count=" + std::to_string(rag.doc_count()) +
                              " chunk_count=" + std::to_string(rag.chunk_count()) +
                              " seeded=" + std::to_string(ingested));
    } else {
        log_event("rag.db", "ready=1 doc_count=" + std::to_string(rag.doc_count()) +
                              " chunk_count=" + std::to_string(rag.chunk_count()));
    }

    ncnn_llm_gpt model(opt.model_path, use_vulkan_runtime);
    std::mutex model_mutex;

    httplib::Server server;
    // Avoid silent hangs when clients stall (common on Windows with AV/proxy).
    // Note: handlers are invoked only after the full request body is received.
    server.set_read_timeout(std::chrono::seconds(120));
    server.set_payload_max_length(256ULL * 1024 * 1024);
    server.set_exception_handler([&](const httplib::Request& req, httplib::Response& res, std::exception_ptr ep) {
        try {
            if (ep) std::rethrow_exception(ep);
            throw std::runtime_error("unknown exception");
        } catch (const std::exception& e) {
            log_event("http.exception", "path=" + req.path + " what=" + std::string(e.what()));
            res.status = 500;
            res.set_content(dump_json_safe(make_error(500, std::string("Internal error: ") + e.what())), "application/json");
        }
    });
    server.set_error_handler([&](const httplib::Request& req, httplib::Response& res) {
        // Prefer JSON errors for API endpoints so the frontend can surface details.
        if (req.path.rfind("/rag/", 0) == 0 || req.path.rfind("/v1/", 0) == 0 || req.path.rfind("/mcp/", 0) == 0) {
            if (res.body.empty()) {
                res.set_content(dump_json_safe(make_error(res.status ? res.status : 500, "Internal Server Error")), "application/json");
            }
        }
    });
    server.set_logger([&](const httplib::Request& req, const httplib::Response& res) {
        log_event("http", req.method + " " + req.path + " status=" + std::to_string(res.status));
    });
    bool mounted_web_root = false;
    if (!is_embedded_web_root(opt.web_root)) {
        mounted_web_root = server.set_mount_point("/", opt.web_root.c_str());
        if (!mounted_web_root) {
            std::cerr << "Warning: failed to mount web root at " << opt.web_root << " (fallback to embedded)\n";
        }
    }
    if (!mounted_web_root) {
        auto serve_embedded = [&](const httplib::Request& req, httplib::Response& res) {
            ncnn_llm_rag_demo_web::AssetView asset;
            if (!ncnn_llm_rag_demo_web::get(req.path, &asset)) {
                res.status = 404;
                res.set_content("Not Found", "text/plain");
                return;
            }
            res.set_content(std::string(reinterpret_cast<const char*>(asset.data), asset.size), asset.mime);
        };
        server.Get("/", serve_embedded);
        server.Get("/index.html", serve_embedded);
        server.Get("/app.js", serve_embedded);
        server.Get("/styles.css", serve_embedded);
    }

    server.Get("/mcp/tools/list", [&](const httplib::Request&, httplib::Response& res) {
        json tools = json::array();
        tools.push_back(rag_tool_schema());
        res.set_content(dump_json_safe(tools), "application/json");
    });

    server.Post("/mcp/tools/call", [&](const httplib::Request& req, httplib::Response& res) {
        std::vector<std::string> err_trace;
        try {
            json body;
            try {
                body = json::parse(req.body);
            } catch (const std::exception& e) {
                res.status = 400;
                err_trace.push_back("parse json");
                json errj = make_error(400, std::string("Invalid JSON: ") + e.what());
                errj["trace"] = err_trace;
                res.set_content(dump_json_safe(errj), "application/json");
                log_event("mcp.call.error", std::string("invalid_json=") + e.what());
                return;
            }

            const std::string name = body.value("name", "");
            json args = body.value("arguments", json::object());
            if (name != "rag_search") {
                res.status = 400;
                err_trace.push_back("validate tool name");
                json errj = make_error(400, "Unknown tool: " + name);
                errj["trace"] = err_trace;
                res.set_content(dump_json_safe(errj), "application/json");
                log_event("mcp.call.error", "unknown_tool name=" + name);
                return;
            }
            if (!rag_ready) {
                res.status = 500;
                std::string msg = "RAG database not ready";
                if (!rag_open_err.empty()) msg += ": " + rag_open_err;
                err_trace.push_back("open db");
                json errj = make_error(500, msg);
                errj["trace"] = err_trace;
                res.set_content(dump_json_safe(errj), "application/json");
                log_event("mcp.call.error", "rag_not_ready");
                return;
            }

            const std::string query = args.value("query", std::string());
            size_t top_k = opt.rag_top_k;
            if (args.contains("top_k") && args["top_k"].is_number_integer()) {
                int v = args["top_k"].get<int>();
                if (v > 0) top_k = static_cast<size_t>(v);
            }
            log_event("mcp.call", "name=" + name +
                                  " query_len=" + std::to_string(query.size()) +
                                  " query=\"" + truncate_for_log(query, 200) + "\"" +
                                  " top_k=" + std::to_string(top_k));

            json result;
            {
                std::lock_guard<std::mutex> lock(rag_mutex);
                result = rag_tool_call(args, rag, embedder, opt.rag_top_k, opt.rag_neighbor_chunks, opt.rag_chunk_max_chars);
            }
            size_t hit_count = 0;
            if (result.contains("chunks") && result["chunks"].is_array()) {
                hit_count = result["chunks"].size();
            }
            log_event("mcp.call.done", "name=" + name +
                                      " hits=" + std::to_string(hit_count) +
                                      " elapsed_ms=" + std::to_string(result.value("elapsed_ms", 0)));
            json resp = {{"name", name}, {"result", result}};
            res.set_content(dump_json_safe(resp), "application/json");
        } catch (const std::exception& e) {
            res.status = 500;
            err_trace.push_back("exception");
            json errj = make_error(500, std::string("Internal error: ") + e.what());
            errj["trace"] = err_trace;
            res.set_content(dump_json_safe(errj), "application/json");
            log_event("mcp.call.exception", e.what());
        }
    });

    server.Post("/rag/upload", [&](const httplib::Request& req, httplib::Response& res) {
        if (!req.is_multipart_form_data() || !req.has_file("file")) {
            res.status = 400;
            res.set_content(dump_json_safe(make_error(400, "multipart file field 'file' required")), "application/json");
            log_event("rag.upload.error", "invalid_form");
            return;
        }
        if (!rag_ready) {
            res.status = 500;
            std::string msg = "RAG database not ready";
            if (!rag_open_err.empty()) msg += ": " + rag_open_err;
            res.set_content(dump_json_safe(make_error(500, msg)), "application/json");
            log_event("rag.upload.error", "rag_not_ready");
            return;
        }

        const auto file = req.get_file_value("file");
        std::string filename = file.filename.empty() ? "upload.txt" : file.filename;
        std::string ext = file_ext_lower(filename);
        if (ext != ".txt" && ext != ".pdf") {
            res.status = 400;
            res.set_content(dump_json_safe(make_error(400, "only .txt and .pdf are supported")), "application/json");
            log_event("rag.upload.error", "unsupported_ext filename=" + filename + " ext=" + ext);
            return;
        }

        log_event("rag.upload", "filename=" + filename + " size=" + std::to_string(file.content.size()));

        std::string stored = std::to_string(now_ms_epoch()) + "_" + sanitize_filename(filename);
        std::filesystem::path outpath = upload_dir / path_from_utf8(stored);

        std::string err;
        if (!write_file(outpath, file.content, &err)) {
            res.status = 500;
            res.set_content(dump_json_safe(make_error(500, err)), "application/json");
            log_event("rag.upload.error", "write_failed err=" + err);
            return;
        }

        std::vector<std::string> trace;
        trace.push_back("saved to " + outpath.string());
        size_t doc_id = 0;
        size_t chunks = 0;
        size_t doc_count = 0;
        size_t chunk_count = 0;
        {
            std::lock_guard<std::mutex> lock(rag_mutex);
            if (!ingest_document(filename,
                                 ext == ".pdf" ? "application/pdf" : "text/plain",
                                 outpath,
                                 rag,
                                 opt,
                                 &trace,
                                 &doc_id,
                                 &chunks,
                                 &err)) {
                res.status = 500;
                res.set_content(dump_json_safe(make_error(500, err)), "application/json");
                log_event("rag.upload.error", "ingest_failed err=" + err);
                return;
            }
            doc_count = rag.doc_count();
            chunk_count = rag.chunk_count();
        }

        log_event("rag.upload.done", "filename=" + filename +
                                     " doc_id=" + std::to_string(doc_id) +
                                     " chunks=" + std::to_string(chunks) +
                                     " doc_count=" + std::to_string(doc_count) +
                                     " chunk_count=" + std::to_string(chunk_count));

        json resp = {
            {"ok", true},
            {"doc", {
                {"id", doc_id},
                {"filename", filename},
                {"mime", ext == ".pdf" ? "application/pdf" : "text/plain"},
                {"chunks", chunks}
            }},
            {"trace", trace},
            {"rag", {
                {"doc_count", doc_count},
                {"chunk_count", chunk_count}
            }}
        };
        res.set_content(dump_json_safe(resp), "application/json");
    });

    server.Get("/rag/info", [&](const httplib::Request&, httplib::Response& res) {
        size_t doc_count = 0;
        size_t chunk_count = 0;
        int embed_dim = 0;
        {
            std::lock_guard<std::mutex> lock(rag_mutex);
            doc_count = rag.doc_count();
            chunk_count = rag.chunk_count();
            embed_dim = rag.embed_dim();
        }
        json info = {
            {"enabled", opt.rag_enabled && rag_ready},
            {"ready", rag_ready},
            {"doc_count", doc_count},
            {"chunk_count", chunk_count},
            {"embed_dim", embed_dim}
        };
        if (!rag_ready && !rag_open_err.empty()) info["error"] = rag_open_err;
        res.set_content(dump_json_safe(info), "application/json");
    });

    server.Get("/rag/docs", [&](const httplib::Request& req, httplib::Response& res) {
        if (!rag_ready) {
            res.status = 500;
            std::string msg = "RAG database not ready";
            if (!rag_open_err.empty()) msg += ": " + rag_open_err;
            res.set_content(dump_json_safe(make_error(500, msg)), "application/json");
            return;
        }
        size_t limit = 200;
        if (auto it = req.params.find("limit"); it != req.params.end()) {
            if (auto v = parse_int(it->second)) {
                if (*v > 0) limit = static_cast<size_t>(*v);
            }
        }
        std::vector<RagDocInfo> docs;
        {
            std::lock_guard<std::mutex> lock(rag_mutex);
            docs = rag.list_docs(limit, 0);
        }
        json out = {{"docs", json::array()}};
        for (const auto& d : docs) {
            out["docs"].push_back({
                {"id", d.id},
                {"filename", sanitize_utf8_strict(d.filename)},
                {"mime", sanitize_utf8_strict(d.mime)},
                {"added_at", d.added_at},
                {"chunk_count", d.chunk_count},
                {"url", "/rag/doc/" + std::to_string(d.id)}
            });
        }
        res.set_content(dump_json_safe(out), "application/json");
    });

    server.Delete(R"(/rag/doc/(\d+))", [&](const httplib::Request& req, httplib::Response& res) {
        if (!rag_ready) {
            res.status = 500;
            std::string msg = "RAG database not ready";
            if (!rag_open_err.empty()) msg += ": " + rag_open_err;
            res.set_content(dump_json_safe(make_error(500, msg)), "application/json");
            return;
        }
        size_t doc_id = 0;
        try {
            doc_id = static_cast<size_t>(std::stoull(req.matches[1]));
        } catch (...) {
            res.status = 400;
            res.set_content(dump_json_safe(make_error(400, "invalid doc id")), "application/json");
            return;
        }

        std::string err;
        size_t doc_count = 0;
        size_t chunk_count = 0;
        {
            std::lock_guard<std::mutex> lock(rag_mutex);
            if (!rag.delete_doc(doc_id, &err)) {
                res.status = 404;
                res.set_content(dump_json_safe(make_error(404, err)), "application/json");
                log_event("rag.doc.delete.error", "doc_id=" + std::to_string(doc_id) + " err=" + err);
                return;
            }
            doc_count = rag.doc_count();
            chunk_count = rag.chunk_count();
        }

        json out = {
            {"ok", true},
            {"doc_id", doc_id},
            {"doc_count", doc_count},
            {"chunk_count", chunk_count}
        };
        res.set_content(dump_json_safe(out), "application/json");
        log_event("rag.doc.delete", "doc_id=" + std::to_string(doc_id));
    });

    server.Get(R"(/rag/doc/(\d+))", [&](const httplib::Request& req, httplib::Response& res) {
        if (!rag_ready) {
            res.status = 500;
            res.set_content("RAG database not ready", "text/plain; charset=utf-8");
            return;
        }

        size_t doc_id = 0;
        try {
            doc_id = static_cast<size_t>(std::stoull(req.matches[1]));
        } catch (...) {
            res.status = 400;
            res.set_content("invalid doc id", "text/plain; charset=utf-8");
            return;
        }

        std::string filename;
        std::vector<RagSearchHit> chunks;
        std::string err;
        {
            std::lock_guard<std::mutex> lock(rag_mutex);
            if (!rag.get_document_chunks(doc_id, &filename, &chunks, &err)) {
                res.status = 404;
                res.set_content("document not found", "text/plain; charset=utf-8");
                log_event("rag.doc.error", "doc_id=" + std::to_string(doc_id) + " err=" + err);
                return;
            }
        }

        std::ostringstream html;
        html << "<!doctype html><html><head><meta charset=\"utf-8\"/>"
             << "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\"/>"
             << "<title>RAG Doc " << doc_id << "</title>"
             << "<style>"
             << "body{font-family:system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif;margin:20px;}"
             << "h1{margin:0 0 6px 0;font-size:18px;}"
             << ".meta{color:#555;margin:0 0 18px 0;font-size:13px;}"
             << "h2{margin:18px 0 6px 0;font-size:15px;}"
             << "pre{white-space:pre-wrap;word-break:break-word;background:#f6f6f6;padding:12px;border-radius:8px;}"
             << "a.anchor{color:#888;text-decoration:none;margin-right:8px;}"
             << "a.back{display:inline-block;margin:0 0 14px 0;color:#06c;text-decoration:none;}"
             << "</style></head><body>";
        html << "<a class=\"back\" href=\"/\">← Back</a>";
        html << "<h1>Document " << doc_id << "</h1>";
        html << "<p class=\"meta\">filename: " << escape_html(sanitize_utf8_strict(filename)) << " · chunks: " << chunks.size() << "</p>";
        for (const auto& c : chunks) {
            html << "<div id=\"chunk-" << c.chunk_index << "\"></div>";
            html << "<h2><a class=\"anchor\" href=\"#chunk-" << c.chunk_index << "\">#</a>"
                 << "Chunk " << c.chunk_index << "</h2>";
            html << "<pre>" << escape_html(sanitize_utf8_strict(c.text)) << "</pre>";
        }
        html << "</body></html>";

        res.set_content(html.str(), "text/html; charset=utf-8");
        log_event("rag.doc", "doc_id=" + std::to_string(doc_id) + " chunks=" + std::to_string(chunks.size()));
    });

    server.Post("/v1/chat/completions", [&](const httplib::Request& req, httplib::Response& res) {
        json body;
        try {
            body = json::parse(req.body);
        } catch (const std::exception& e) {
            res.status = 400;
            res.set_content(dump_json_safe(make_error(400, std::string("Invalid JSON: ") + e.what())), "application/json");
            log_event("chat.error", std::string("invalid_json=") + e.what());
            return;
        }

        if (!body.contains("messages") || !body["messages"].is_array()) {
            res.status = 400;
            res.set_content(dump_json_safe(make_error(400, "`messages` must be an array")), "application/json");
            log_event("chat.error", "invalid_messages");
            return;
        }

        std::vector<Message> messages = parse_messages(body["messages"]);
        if (messages.empty()) {
            res.status = 400;
            res.set_content(dump_json_safe(make_error(400, "`messages` cannot be empty")), "application/json");
            log_event("chat.error", "empty_messages");
            return;
        }

        std::string user_query;
        for (auto it = messages.rbegin(); it != messages.rend(); ++it) {
            if (it->role == "user") {
                user_query = it->content;
                break;
            }
        }

        const std::string rag_mode = body.value("rag_mode", std::string("server"));
        const bool client_rag = (rag_mode == "client");
        const bool rag_enabled = client_rag ? false : body.value("rag_enable", opt.rag_enabled);
        size_t rag_top_k = opt.rag_top_k;
        if (body.contains("rag_top_k") && body["rag_top_k"].is_number_integer()) {
            int v = body["rag_top_k"].get<int>();
            if (v > 0) rag_top_k = static_cast<size_t>(v);
        }
        const bool stream = body.value("stream", false);
        const bool enable_thinking = body.value("enable_thinking", false);
        const std::string model_name = body.value("model", std::string("qwen3-0.6b"));
        const std::string resp_id = make_response_id();

        log_event("chat.request", "id=" + resp_id +
                                   " " + summarize_messages(messages, user_query) +
                                   " rag_mode=" + rag_mode +
                                   " rag_enabled=" + std::string(rag_enabled ? "1" : "0") +
                                   " rag_ready=" + std::string(rag_ready ? "1" : "0") +
                                   " rag_top_k=" + std::to_string(rag_top_k) +
                                   " stream=" + std::string(stream ? "1" : "0") +
                                   " thinking=" + std::string(enable_thinking ? "1" : "0") +
                                   " model=" + model_name);

        std::vector<std::string> rag_trace;
        std::string rag_error;
        std::vector<RagSearchHit> hits;
        if (!client_rag && rag_enabled && rag_ready && !user_query.empty()) {
            rag_trace.push_back("tokenize+embed");
            rag_trace.push_back("vector search");
            auto t0 = std::chrono::steady_clock::now();
            std::vector<float> qvec = embedder.embed(user_query);
            {
                std::lock_guard<std::mutex> lock(rag_mutex);
                hits = rag.search(qvec, rag_top_k);
                rag_trace.push_back("expand neighbors");
                expand_hits_with_neighbors(rag, hits, opt.rag_neighbor_chunks, opt.rag_chunk_max_chars);
            }
            auto t1 = std::chrono::steady_clock::now();
            int64_t elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
            log_event("rag.search", "id=" + resp_id +
                                      " query_len=" + std::to_string(user_query.size()) +
                                      " top_k=" + std::to_string(rag_top_k) +
                                      " " + summarize_hits(hits, 3) +
                                      " elapsed_ms=" + std::to_string(elapsed_ms));
        } else {
            std::string reason;
            if (client_rag) {
                reason = "client_mode";
            } else if (!rag_enabled) {
                reason = "disabled";
            } else if (!rag_ready) {
                reason = "db_not_ready";
                rag_error = rag_open_err;
            } else if (user_query.empty()) {
                reason = "empty_query";
            }
            if (!reason.empty()) {
                rag_trace.push_back("skip: " + reason);
                log_event("rag.search.skip", "id=" + resp_id + " reason=" + reason);
            }
        }

        std::string rag_context;
        if (client_rag) {
            if (messages.empty() || messages.front().role != "system") {
                messages.insert(messages.begin(), Message{"system", "You are a helpful assistant."});
            }
        } else {
            rag_context = build_rag_context(hits);
            std::string system_prompt = build_system_prompt(rag_context, rag_enabled && rag_ready);

            if (!messages.empty() && messages.front().role == "system") {
                if (!messages.front().content.empty()) {
                    system_prompt += "\n\nOriginal system message:\n" + messages.front().content;
                }
                messages.front().content = system_prompt;
            } else {
                messages.insert(messages.begin(), Message{"system", system_prompt});
            }
        }

        GenerateConfig cfg;
        cfg.max_new_tokens = body.value("max_tokens", cfg.max_new_tokens);
        cfg.temperature = body.value("temperature", cfg.temperature);
        cfg.top_p = body.value("top_p", cfg.top_p);
        cfg.top_k = body.value("top_k", cfg.top_k);
        cfg.repetition_penalty = body.value("repetition_penalty", cfg.repetition_penalty);
        cfg.beam_size = body.value("beam_size", cfg.beam_size);
        if (body.contains("do_sample") && body["do_sample"].is_boolean()) {
            cfg.do_sample = body["do_sample"].get<bool>() ? 1 : 0;
        } else if (cfg.temperature <= 0.0f) {
            cfg.do_sample = 0;
        }

        log_event("gen.config", "id=" + resp_id +
                                  " max_new_tokens=" + std::to_string(cfg.max_new_tokens) +
                                  " temperature=" + std::to_string(cfg.temperature) +
                                  " top_p=" + std::to_string(cfg.top_p) +
                                  " top_k=" + std::to_string(cfg.top_k) +
                                  " repetition_penalty=" + std::to_string(cfg.repetition_penalty) +
                                  " beam_size=" + std::to_string(cfg.beam_size) +
                                  " do_sample=" + std::to_string(cfg.do_sample));

        const std::string prompt = apply_chat_template(messages, {}, true, enable_thinking);
        size_t system_prompt_len = 0;
        if (!messages.empty() && messages.front().role == "system") {
            system_prompt_len = messages.front().content.size();
        }
        log_event("prompt.build", "id=" + resp_id +
                                  " prompt_len=" + std::to_string(prompt.size()) +
                                  " system_prompt_len=" + std::to_string(system_prompt_len) +
                                  " rag_context_len=" + std::to_string(rag_context.size()) +
                                  " messages=" + std::to_string(messages.size()));
        json rag_payload;
        if (client_rag && body.contains("rag_payload") && body["rag_payload"].is_object()) {
            rag_payload = body["rag_payload"];
        } else {
            size_t doc_count = 0;
            size_t chunk_count = 0;
            {
                std::lock_guard<std::mutex> lock(rag_mutex);
                doc_count = rag.doc_count();
                chunk_count = rag.chunk_count();
            }
            rag_payload = build_rag_payload(hits,
                                            rag_enabled && rag_ready,
                                            rag_top_k,
                                            doc_count,
                                            chunk_count,
                                            rag_trace.empty() ? nullptr : &rag_trace,
                                            rag_error.empty() ? nullptr : &rag_error);
        }

	        if (stream) {
	            res.set_header("Content-Type", "text/event-stream");
	            res.set_header("Cache-Control", "no-cache");
	            res.set_header("Connection", "keep-alive");

	            res.set_chunked_content_provider(
	                "text/event-stream",
	                [&, prompt, cfg, resp_id, model_name, rag_payload](size_t, httplib::DataSink& sink) mutable {
	                    std::lock_guard<std::mutex> lock(model_mutex);

	                    log_event("chat.prefill.start", "id=" + resp_id + " prompt_len=" + std::to_string(prompt.size()));
	                    auto prefill_start = std::chrono::steady_clock::now();
	                    auto ctx = prefill_chunked(model, prompt, opt.llm_prefill_chunk_bytes, resp_id);
	                    const size_t prompt_tokens = (!ctx || ctx->kv_cache.empty()) ? 0u : static_cast<size_t>(ctx->kv_cache[0].first.h);
	                    auto prefill_end = std::chrono::steady_clock::now();
	                    int64_t prefill_ms = std::chrono::duration_cast<std::chrono::milliseconds>(prefill_end - prefill_start).count();
	                    log_event("chat.prefill.done", "id=" + resp_id + " elapsed_ms=" + std::to_string(prefill_ms));

	                    size_t token_count = 0;
	                    size_t output_bytes = 0;
	                    auto gen_start = std::chrono::steady_clock::now();
	                    model.generate(ctx, cfg, [&](const std::string& token) {
	                        std::string safe_token = sanitize_utf8(token);
	                        ++token_count;
	                        output_bytes += safe_token.size();
	                        json chunk = {
	                            {"id", resp_id},
	                            {"object", "chat.completion.chunk"},
	                            {"model", model_name},
	                            {"choices", json::array({
	                                json{
	                                    {"index", 0},
	                                    {"delta", {{"role", "assistant"}, {"content", safe_token}}},
	                                    {"finish_reason", nullptr}
	                                }
	                            })},
	                            {"usage", {{"prompt_tokens", prompt_tokens},
	                                       {"completion_tokens", token_count},
	                                       {"total_tokens", prompt_tokens + token_count}}}
	                        };
	                        std::string data = "data: " + dump_json_safe(chunk) + "\n\n";
	                        sink.write(data.data(), data.size());
	                    });
	                    auto gen_end = std::chrono::steady_clock::now();
	                    int64_t gen_ms = std::chrono::duration_cast<std::chrono::milliseconds>(gen_end - gen_start).count();
	                    log_event("chat.generate.done", "id=" + resp_id +
	                                                   " tokens=" + std::to_string(token_count) +
	                                                   " output_bytes=" + std::to_string(output_bytes) +
	                                                   " elapsed_ms=" + std::to_string(gen_ms));

	                    const size_t kv_bytes = kv_cache_bytes(ctx);
	                    ctx.reset();
	                    maybe_malloc_trim(opt.malloc_trim);
	                    MemSnapshot mem = read_self_mem_snapshot();
	                    json done_chunk = {
	                        {"id", resp_id},
	                        {"object", "chat.completion.chunk"},
	                        {"model", model_name},
	                        {"choices", json::array({
	                            json{
	                                {"index", 0},
	                                {"delta", json::object()},
	                                {"finish_reason", "stop"}
	                            }
	                        })},
	                        {"usage", {{"prompt_tokens", prompt_tokens},
	                                   {"completion_tokens", token_count},
	                                   {"total_tokens", prompt_tokens + token_count}}},
	                        {"mem", {{"rss_bytes", mem.rss_bytes},
	                                 {"hwm_bytes", mem.hwm_bytes},
	                                 {"kv_cache_bytes", kv_bytes}}},
	                        {"rag", rag_payload}
	                    };

	                    std::string end_data = "data: " + dump_json_safe(done_chunk) + "\n\n";
	                    sink.write(end_data.data(), end_data.size());

                    const char done[] = "data: [DONE]\n\n";
                    sink.write(done, sizeof(done) - 1);
                    return false;
                },
                [](bool) {});
            return;
        }

	        std::string generated;
	        size_t prompt_tokens = 0;
	        size_t completion_tokens = 0;
	        size_t kv_bytes = 0;
	        MemSnapshot mem;
	        {
	            std::lock_guard<std::mutex> lock(model_mutex);
	            log_event("chat.prefill.start", "id=" + resp_id + " prompt_len=" + std::to_string(prompt.size()));
	            auto prefill_start = std::chrono::steady_clock::now();
	            auto ctx = prefill_chunked(model, prompt, opt.llm_prefill_chunk_bytes, resp_id);
	            prompt_tokens = (!ctx || ctx->kv_cache.empty()) ? 0u : static_cast<size_t>(ctx->kv_cache[0].first.h);
	            auto prefill_end = std::chrono::steady_clock::now();
	            int64_t prefill_ms = std::chrono::duration_cast<std::chrono::milliseconds>(prefill_end - prefill_start).count();
	            log_event("chat.prefill.done", "id=" + resp_id + " elapsed_ms=" + std::to_string(prefill_ms));

            size_t token_count = 0;
            auto gen_start = std::chrono::steady_clock::now();
            model.generate(ctx, cfg, [&](const std::string& token) {
                generated += sanitize_utf8(token);
                ++token_count;
            });
	            auto gen_end = std::chrono::steady_clock::now();
	            completion_tokens = token_count;
	            kv_bytes = kv_cache_bytes(ctx);
	            ctx.reset();
	            int64_t gen_ms = std::chrono::duration_cast<std::chrono::milliseconds>(gen_end - gen_start).count();
	            log_event("chat.generate.done", "id=" + resp_id +
	                                               " tokens=" + std::to_string(token_count) +
	                                               " output_bytes=" + std::to_string(generated.size()) +
	                                               " elapsed_ms=" + std::to_string(gen_ms));
	        }
	        maybe_malloc_trim(opt.malloc_trim);
	        mem = read_self_mem_snapshot();

        json resp = {
            {"id", resp_id},
            {"object", "chat.completion"},
            {"model", model_name},
            {"choices", json::array({
                json{
                    {"index", 0},
                    {"message", {{"role", "assistant"}, {"content", generated}}},
                    {"finish_reason", "stop"}
                }
            })},
	            {"usage", {{"prompt_tokens", prompt_tokens},
	                       {"completion_tokens", completion_tokens},
	                       {"total_tokens", prompt_tokens + completion_tokens}}},
	            {"mem", {{"rss_bytes", mem.rss_bytes},
	                     {"hwm_bytes", mem.hwm_bytes},
	                     {"kv_cache_bytes", kv_bytes}}},
	            {"rag", rag_payload}
	        };
	        res.set_content(dump_json_safe(resp), "application/json");
	    });

    std::cout << "RAG web app listening on http://0.0.0.0:" << opt.port << "\n";
    std::cout << "POST /v1/chat/completions and open / for the demo UI.\n";
    server.listen("0.0.0.0", opt.port);

#if defined(NCNN_RAG_HAS_VULKAN_API) && NCNN_RAG_HAS_VULKAN_API
    if (opt.use_vulkan) {
        ncnn::destroy_gpu_instance();
    }
#endif
    return 0;
}
