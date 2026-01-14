#include "rag_ingest.h"
#include "rag_text.h"
#include "rag_vector_db.h"

#include "json_utils.h"
#include "ncnn_llm_gpt.h"
#include "util.h"
#include "utils/prompt.h"

#include <httplib.h>
#include <nlohmann/json.hpp>

#include <cstdlib>
#include <algorithm>
#include <chrono>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>

using nlohmann::json;

namespace {

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

void log_event(const std::string& tag, const std::string& msg) {
    std::cerr << "[" << now_ms_epoch() << "] " << tag << " " << msg << "\n";
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

struct AppOptions {
    std::string model_path = "assets/qwen3_0.6b";
    std::string web_root = "src/web";
    std::string docs_path = "assets/rag";
    std::string data_dir = "data";
    std::string db_path = "data/rag.sqlite";
    std::string pdf_txt_dir = "data/pdf_txt";
    size_t chunk_size = 900;
    int embed_dim = 256;
    int port = 8080;
    bool use_vulkan = false;
    bool rag_enabled = true;
    size_t rag_top_k = 10;
    int rag_neighbor_chunks = 1;
    size_t rag_chunk_max_chars = 1800;
    bool save_pdf_txt = true;
};

void print_usage(const char* argv0) {
    std::cout << "Usage: " << argv0 << " [options]\n"
              << "  --model PATH      Model directory (default: assets/qwen3_0.6b)\n"
              << "  --docs PATH       Seed docs directory (default: assets/rag)\n"
              << "  --web PATH        Web root to serve (default: src/web)\n"
              << "  --data PATH       Data directory (default: data)\n"
              << "  --db PATH         SQLite database path (default: data/rag.sqlite)\n"
              << "  --pdf-txt PATH    Exported PDF text directory (default: data/pdf_txt)\n"
              << "  --chunk-size N    Chunk size for indexing (default: 900)\n"
              << "  --embed-dim N     Embedding dimension (default: 256)\n"
              << "  --port N          HTTP port (default: 8080)\n"
              << "  --rag-top-k N     Retrieved chunks (default: 10)\n"
              << "  --rag-neighbors N Include neighbor chunks around each hit (default: 1)\n"
              << "  --rag-chunk-max N Max chars per returned chunk after expansion (default: 1800)\n"
              << "  --no-rag          Disable retrieval\n"
              << "  --no-pdf-txt      Disable exporting extracted PDF text\n"
              << "  --vulkan          Enable Vulkan compute\n"
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
        } else if (arg == "--no-rag") {
            opt.rag_enabled = false;
        } else if (arg == "--no-pdf-txt") {
            opt.save_pdf_txt = false;
        } else if (arg == "--vulkan") {
            opt.use_vulkan = true;
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
        ctx += "[" + std::to_string(i + 1) + "] Source: " + hits[i].source + "\n";
        ctx += hits[i].text + "\n\n";
    }
    return ctx;
}

void expand_hits_with_neighbors(const RagVectorDb& rag,
                                std::vector<RagSearchHit>& hits,
                                int neighbor_chunks,
                                size_t max_chunk_chars) {
    if (neighbor_chunks <= 0 || hits.empty()) return;
    for (auto& hit : hits) {
        std::string expanded = rag.expand_neighbors(hit.doc_id, hit.chunk_index, neighbor_chunks);
        if (!expanded.empty()) {
            hit.text = shorten_text(expanded, max_chunk_chars);
        }
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
                       size_t chunk_count) {
    json rag = {
        {"enabled", rag_enabled},
        {"top_k", top_k},
        {"doc_count", doc_count},
        {"chunk_count", chunk_count},
        {"chunks", json::array()}
    };
    for (const auto& hit : hits) {
        rag["chunks"].push_back({
            {"source", hit.source},
            {"score", hit.score},
            {"text", hit.text}
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
            for (auto& hit : hits) {
                std::string expanded = rag.expand_neighbors(hit.doc_id, hit.chunk_index, neighbor_chunks);
                if (!expanded.empty()) {
                    hit.text = shorten_text(expanded, max_chunk_chars);
                }
            }
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
            {"source", hit.source},
            {"score", hit.score},
            {"text", hit.text}
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
        if (c == '/' || c == '\\' || c == ':' || c == '\0') c = '_';
    }
    return s;
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
    std::string ext = file_ext_lower(filename);

    if (trace) trace->push_back("read content");
    if (ext == ".txt") {
        if (!read_text_file(path.string(), &text, &local_err)) {
            if (err) *err = local_err;
            return false;
        }
    } else if (ext == ".pdf") {
        if (!extract_pdf_text(path.string(), &text, &local_err)) {
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

    if (trace) trace->push_back("chunk+embed+store");
    size_t doc_id = 0;
    size_t chunk_count = 0;
    if (!rag.add_document(filename, mime, text, opt.chunk_size, &local_err, &doc_id, &chunk_count)) {
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
    opt.model_path = normalize_path(opt.model_path, "./assets");
    opt.docs_path = normalize_path(opt.docs_path, ".");
    opt.data_dir = normalize_path(opt.data_dir, ".");
    opt.db_path = normalize_path(opt.db_path, opt.data_dir);
    opt.pdf_txt_dir = normalize_path(opt.pdf_txt_dir, opt.data_dir);

    log_event("startup", "model_path=" + opt.model_path +
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
                         " rag_enabled=" + std::string(opt.rag_enabled ? "1" : "0") +
                         " save_pdf_txt=" + std::string(opt.save_pdf_txt ? "1" : "0") +
                         " vulkan=" + std::string(opt.use_vulkan ? "1" : "0"));

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
    std::string rag_err;
    bool rag_ready = rag.open(opt.db_path, opt.embed_dim, &rag_err);
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

    ncnn_llm_gpt model(opt.model_path, opt.use_vulkan);
    std::mutex model_mutex;

    httplib::Server server;
    if (!server.set_mount_point("/", opt.web_root.c_str())) {
        std::cerr << "Warning: failed to mount web root at " << opt.web_root << "\n";
    }

    server.Get("/mcp/tools/list", [&](const httplib::Request&, httplib::Response& res) {
        json tools = json::array();
        tools.push_back(rag_tool_schema());
        res.set_content(tools.dump(), "application/json");
    });

    server.Post("/mcp/tools/call", [&](const httplib::Request& req, httplib::Response& res) {
        json body;
        try {
            body = json::parse(req.body);
        } catch (const std::exception& e) {
            res.status = 400;
            res.set_content(make_error(400, std::string("Invalid JSON: ") + e.what()).dump(), "application/json");
            log_event("mcp.call.error", std::string("invalid_json=") + e.what());
            return;
        }

        const std::string name = body.value("name", "");
        json args = body.value("arguments", json::object());
        if (name != "rag_search") {
            res.status = 400;
            res.set_content(make_error(400, "Unknown tool: " + name).dump(), "application/json");
            log_event("mcp.call.error", "unknown_tool name=" + name);
            return;
        }
        if (!rag_ready) {
            res.status = 500;
            res.set_content(make_error(500, "RAG database not ready").dump(), "application/json");
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

        json result = rag_tool_call(args, rag, embedder, opt.rag_top_k, opt.rag_neighbor_chunks, opt.rag_chunk_max_chars);
        size_t hit_count = 0;
        if (result.contains("chunks") && result["chunks"].is_array()) {
            hit_count = result["chunks"].size();
        }
        log_event("mcp.call.done", "name=" + name +
                                  " hits=" + std::to_string(hit_count) +
                                  " elapsed_ms=" + std::to_string(result.value("elapsed_ms", 0)));
        json resp = {{"name", name}, {"result", result}};
        res.set_content(resp.dump(), "application/json");
    });

    server.Post("/rag/upload", [&](const httplib::Request& req, httplib::Response& res) {
        if (!req.is_multipart_form_data() || !req.has_file("file")) {
            res.status = 400;
            res.set_content(make_error(400, "multipart file field 'file' required").dump(), "application/json");
            log_event("rag.upload.error", "invalid_form");
            return;
        }
        if (!rag_ready) {
            res.status = 500;
            res.set_content(make_error(500, "RAG database not ready").dump(), "application/json");
            log_event("rag.upload.error", "rag_not_ready");
            return;
        }

        const auto file = req.get_file_value("file");
        std::string filename = file.filename.empty() ? "upload.txt" : file.filename;
        std::string ext = file_ext_lower(filename);
        if (ext != ".txt" && ext != ".pdf") {
            res.status = 400;
            res.set_content(make_error(400, "only .txt and .pdf are supported").dump(), "application/json");
            log_event("rag.upload.error", "unsupported_ext filename=" + filename + " ext=" + ext);
            return;
        }

        log_event("rag.upload", "filename=" + filename + " size=" + std::to_string(file.content.size()));

        std::string stored = std::to_string(now_ms_epoch()) + "_" + filename;
        for (char& c : stored) {
            if (c == '/' || c == '\\') c = '_';
        }
        std::filesystem::path outpath = upload_dir / stored;

        std::string err;
        if (!write_file(outpath, file.content, &err)) {
            res.status = 500;
            res.set_content(make_error(500, err).dump(), "application/json");
            log_event("rag.upload.error", "write_failed err=" + err);
            return;
        }

        std::vector<std::string> trace;
        trace.push_back("saved to " + outpath.string());
        size_t doc_id = 0;
        size_t chunks = 0;
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
            res.set_content(make_error(500, err).dump(), "application/json");
            log_event("rag.upload.error", "ingest_failed err=" + err);
            return;
        }

        log_event("rag.upload.done", "filename=" + filename +
                                     " doc_id=" + std::to_string(doc_id) +
                                     " chunks=" + std::to_string(chunks) +
                                     " doc_count=" + std::to_string(rag.doc_count()) +
                                     " chunk_count=" + std::to_string(rag.chunk_count()));

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
                {"doc_count", rag.doc_count()},
                {"chunk_count", rag.chunk_count()}
            }}
        };
        res.set_content(resp.dump(), "application/json");
    });

    server.Get("/rag/info", [&](const httplib::Request&, httplib::Response& res) {
        json info = {
            {"enabled", opt.rag_enabled && rag_ready},
            {"doc_count", rag.doc_count()},
            {"chunk_count", rag.chunk_count()},
            {"embed_dim", rag.embed_dim()}
        };
        res.set_content(info.dump(), "application/json");
    });

    server.Post("/v1/chat/completions", [&](const httplib::Request& req, httplib::Response& res) {
        json body;
        try {
            body = json::parse(req.body);
        } catch (const std::exception& e) {
            res.status = 400;
            res.set_content(make_error(400, std::string("Invalid JSON: ") + e.what()).dump(), "application/json");
            log_event("chat.error", std::string("invalid_json=") + e.what());
            return;
        }

        if (!body.contains("messages") || !body["messages"].is_array()) {
            res.status = 400;
            res.set_content(make_error(400, "`messages` must be an array").dump(), "application/json");
            log_event("chat.error", "invalid_messages");
            return;
        }

        std::vector<Message> messages = parse_messages(body["messages"]);
        if (messages.empty()) {
            res.status = 400;
            res.set_content(make_error(400, "`messages` cannot be empty").dump(), "application/json");
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

        std::vector<RagSearchHit> hits;
        if (!client_rag && rag_enabled && rag_ready && !user_query.empty()) {
            auto t0 = std::chrono::steady_clock::now();
            std::vector<float> qvec = embedder.embed(user_query);
            hits = rag.search(qvec, rag_top_k);
            expand_hits_with_neighbors(rag, hits, opt.rag_neighbor_chunks, opt.rag_chunk_max_chars);
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
            } else if (user_query.empty()) {
                reason = "empty_query";
            }
            if (!reason.empty()) {
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
            rag_payload = build_rag_payload(hits, rag_enabled && rag_ready, rag_top_k, rag.doc_count(), rag.chunk_count());
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
                    auto ctx = model.prefill(prompt);
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
                            })}
                        };
                        std::string data = "data: " + chunk.dump() + "\n\n";
                        sink.write(data.data(), data.size());
                    });
                    auto gen_end = std::chrono::steady_clock::now();
                    int64_t gen_ms = std::chrono::duration_cast<std::chrono::milliseconds>(gen_end - gen_start).count();
                    log_event("chat.generate.done", "id=" + resp_id +
                                                   " tokens=" + std::to_string(token_count) +
                                                   " output_bytes=" + std::to_string(output_bytes) +
                                                   " elapsed_ms=" + std::to_string(gen_ms));

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
                        {"rag", rag_payload}
                    };

                    std::string end_data = "data: " + done_chunk.dump() + "\n\n";
                    sink.write(end_data.data(), end_data.size());

                    const char done[] = "data: [DONE]\n\n";
                    sink.write(done, sizeof(done) - 1);
                    return false;
                },
                [](bool) {});
            return;
        }

        std::string generated;
        {
            std::lock_guard<std::mutex> lock(model_mutex);
            log_event("chat.prefill.start", "id=" + resp_id + " prompt_len=" + std::to_string(prompt.size()));
            auto prefill_start = std::chrono::steady_clock::now();
            auto ctx = model.prefill(prompt);
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
            int64_t gen_ms = std::chrono::duration_cast<std::chrono::milliseconds>(gen_end - gen_start).count();
            log_event("chat.generate.done", "id=" + resp_id +
                                               " tokens=" + std::to_string(token_count) +
                                               " output_bytes=" + std::to_string(generated.size()) +
                                               " elapsed_ms=" + std::to_string(gen_ms));
        }

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
            {"usage", {{"prompt_tokens", 0}, {"completion_tokens", 0}}},
            {"rag", rag_payload}
        };
        res.set_content(resp.dump(), "application/json");
    });

    std::cout << "RAG web app listening on http://0.0.0.0:" << opt.port << "\n";
    std::cout << "POST /v1/chat/completions and open / for the demo UI.\n";
    server.listen("0.0.0.0", opt.port);

    return 0;
}
