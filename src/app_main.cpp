#include "rag_index.h"

#include "json_utils.h"
#include "ncnn_llm_gpt.h"
#include "util.h"
#include "utils/prompt.h"

#include <httplib.h>
#include <nlohmann/json.hpp>

#include <cstdlib>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <mutex>
#include <string>

using nlohmann::json;

namespace {

struct AppOptions {
    std::string model_path = "assets/qwen3_0.6b";
    std::string docs_path = "assets/rag";
    std::string web_root = "src/web";
    int port = 8080;
    bool use_vulkan = false;
    bool rag_enabled = true;
    size_t rag_top_k = 4;
};

void print_usage(const char* argv0) {
    std::cout << "Usage: " << argv0 << " [options]\n"
              << "  --model PATH      Model directory (default: assets/qwen3_0.6b)\n"
              << "  --docs PATH       Docs directory for RAG (default: assets/rag)\n"
              << "  --web PATH        Web root to serve (default: src/web)\n"
              << "  --port N          HTTP port (default: 8080)\n"
              << "  --rag-top-k N     Retrieved chunks (default: 4)\n"
              << "  --no-rag          Disable retrieval\n"
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
        } else if (arg == "--port" && i + 1 < argc) {
            if (auto v = parse_int(argv[++i])) opt.port = *v;
        } else if (arg == "--rag-top-k" && i + 1 < argc) {
            if (auto v = parse_int(argv[++i])) opt.rag_top_k = static_cast<size_t>(*v);
        } else if (arg == "--no-rag") {
            opt.rag_enabled = false;
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

std::string build_rag_context(const std::vector<RagHit>& hits) {
    if (hits.empty()) return {};
    std::string ctx;
    for (size_t i = 0; i < hits.size(); ++i) {
        ctx += "[" + std::to_string(i + 1) + "] Source: " + hits[i].source + "\n";
        ctx += hits[i].text + "\n\n";
    }
    return ctx;
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

json build_rag_payload(const std::vector<RagHit>& hits, bool rag_enabled, size_t top_k, const RagIndex& index) {
    json rag = {
        {"enabled", rag_enabled},
        {"top_k", top_k},
        {"doc_count", index.doc_count()},
        {"chunk_count", index.chunk_count()},
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

json rag_tool_call(const json& args, const RagIndex& rag, size_t default_top_k) {
    const std::string query = args.value("query", std::string());
    size_t top_k = default_top_k;
    if (args.contains("top_k") && args["top_k"].is_number_integer()) {
        int v = args["top_k"].get<int>();
        if (v > 0) top_k = static_cast<size_t>(v);
    }

    auto t0 = std::chrono::steady_clock::now();
    std::vector<RagHit> hits;
    if (!query.empty()) {
        hits = rag.search(query, top_k);
    }
    auto t1 = std::chrono::steady_clock::now();
    int64_t elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

    json result = {
        {"query", query},
        {"top_k", top_k},
        {"elapsed_ms", elapsed_ms},
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

} // namespace

int main(int argc, char** argv) {
    AppOptions opt = parse_options(argc, argv);
    opt.model_path = normalize_path(opt.model_path, "./assets");

    RagIndex rag;
    std::string rag_err;
    bool rag_ready = rag.load_directory(opt.docs_path, &rag_err);
    if (!rag_ready) {
        std::cerr << "RAG index warning: " << rag_err << "\n";
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
            return;
        }

        const std::string name = body.value("name", "");
        json args = body.value("arguments", json::object());
        if (name != "rag_search") {
            res.status = 400;
            res.set_content(make_error(400, "Unknown tool: " + name).dump(), "application/json");
            return;
        }
        if (!rag_ready) {
            res.status = 500;
            res.set_content(make_error(500, "RAG index not ready").dump(), "application/json");
            return;
        }

        json result = rag_tool_call(args, rag, opt.rag_top_k);
        json resp = {{"name", name}, {"result", result}};
        res.set_content(resp.dump(), "application/json");
    });

    server.Get("/rag/info", [&](const httplib::Request&, httplib::Response& res) {
        json info = {
            {"enabled", opt.rag_enabled && rag_ready},
            {"doc_count", rag.doc_count()},
            {"chunk_count", rag.chunk_count()}
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
            return;
        }

        if (!body.contains("messages") || !body["messages"].is_array()) {
            res.status = 400;
            res.set_content(make_error(400, "`messages` must be an array").dump(), "application/json");
            return;
        }

        std::vector<Message> messages = parse_messages(body["messages"]);
        if (messages.empty()) {
            res.status = 400;
            res.set_content(make_error(400, "`messages` cannot be empty").dump(), "application/json");
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

        std::vector<RagHit> hits;
        if (!client_rag && rag_enabled && rag_ready && !user_query.empty()) {
            hits = rag.search(user_query, rag_top_k);
        }

        if (client_rag) {
            if (messages.empty() || messages.front().role != "system") {
                messages.insert(messages.begin(), Message{"system", "You are a helpful assistant."});
            }
        } else {
            std::string rag_context = build_rag_context(hits);
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

        const bool stream = body.value("stream", false);
        const bool enable_thinking = body.value("enable_thinking", false);
        const std::string model_name = body.value("model", std::string("qwen3-0.6b"));
        const std::string prompt = apply_chat_template(messages, {}, true, enable_thinking);
        const std::string resp_id = make_response_id();
        json rag_payload;
        if (client_rag && body.contains("rag_payload") && body["rag_payload"].is_object()) {
            rag_payload = body["rag_payload"];
        } else {
            rag_payload = build_rag_payload(hits, rag_enabled && rag_ready, rag_top_k, rag);
        }

        if (stream) {
            res.set_header("Content-Type", "text/event-stream");
            res.set_header("Cache-Control", "no-cache");
            res.set_header("Connection", "keep-alive");

            res.set_chunked_content_provider(
                "text/event-stream",
                [&, prompt, cfg, resp_id, model_name, rag_payload](size_t, httplib::DataSink& sink) mutable {
                    std::lock_guard<std::mutex> lock(model_mutex);

                    auto ctx = model.prefill(prompt);
                    model.generate(ctx, cfg, [&](const std::string& token) {
                        std::string safe_token = sanitize_utf8(token);
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
            auto ctx = model.prefill(prompt);
            model.generate(ctx, cfg, [&](const std::string& token) {
                generated += sanitize_utf8(token);
            });
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
