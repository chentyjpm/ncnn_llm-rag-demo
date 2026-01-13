#include "rag_index.h"

#include <algorithm>
#include <cmath>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <unordered_set>

namespace fs = std::filesystem;

namespace {

bool has_text_extension(const fs::path& path) {
    const std::string ext = path.extension().string();
    if (ext == ".txt" || ext == ".md" || ext == ".mdx" || ext == ".markdown" ||
        ext == ".rst" || ext == ".log") {
        return true;
    }
    return ext.empty();
}

bool read_file(const fs::path& path, std::string& out) {
    std::ifstream ifs(path, std::ios::in | std::ios::binary);
    if (!ifs) return false;
    std::ostringstream oss;
    oss << ifs.rdbuf();
    out = oss.str();
    return true;
}

size_t utf8_char_len(unsigned char c) {
    if (c < 0x80) return 1;
    if ((c >> 5) == 0x6) return 2;
    if ((c >> 4) == 0xE) return 3;
    if ((c >> 3) == 0x1E) return 4;
    return 1;
}

bool is_ascii_word_char(unsigned char c) {
    return (c >= '0' && c <= '9') || (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z');
}

} // namespace

bool RagIndex::load_directory(const std::string& dir, std::string* err) {
    chunks_.clear();
    doc_freq_.clear();
    idf_.clear();
    doc_count_ = 0;
    avg_len_ = 0.0;

    fs::path root(dir);
    std::error_code ec;
    if (!fs::exists(root, ec)) {
        if (err) *err = "docs directory not found: " + root.string();
        return false;
    }

    for (auto it = fs::recursive_directory_iterator(root, ec); it != fs::recursive_directory_iterator(); it.increment(ec)) {
        if (ec) break;
        if (!it->is_regular_file()) continue;
        const fs::path path = it->path();
        if (!has_text_extension(path)) continue;

        std::string content;
        if (!read_file(path, content)) continue;
        content = trim(content);
        if (content.empty()) continue;

        std::vector<std::string> parts = split_chunks(content, 900);
        if (parts.empty()) continue;

        std::string rel = fs::relative(path, root, ec).generic_string();
        if (ec || rel.empty()) rel = path.filename().string();
        ++doc_count_;

        size_t chunk_index = 0;
        for (const auto& raw_chunk : parts) {
            std::string chunk_text = trim(raw_chunk);
            if (chunk_text.empty()) continue;

            Chunk chunk;
            chunk.source = rel + "#" + std::to_string(chunk_index++);
            chunk.text = chunk_text;

            std::vector<std::string> tokens = tokenize(chunk_text);
            if (tokens.empty()) continue;

            std::unordered_set<std::string> unique;
            for (const auto& tok : tokens) {
                ++chunk.term_freq[tok];
                unique.insert(tok);
            }
            chunk.length = tokens.size();
            for (const auto& tok : unique) {
                ++doc_freq_[tok];
            }
            chunks_.push_back(std::move(chunk));
        }
    }

    build_stats();
    if (chunks_.empty()) {
        if (err) *err = "no readable text chunks found in " + root.string();
        return false;
    }
    return true;
}

std::vector<RagHit> RagIndex::search(const std::string& query, size_t top_k) const {
    std::vector<RagHit> out;
    if (chunks_.empty() || query.empty() || top_k == 0) return out;

    std::vector<std::string> q_tokens = tokenize(query);
    if (q_tokens.empty()) return out;

    std::unordered_map<std::string, int> q_tf;
    for (const auto& tok : q_tokens) {
        ++q_tf[tok];
    }

    const double k1 = 1.5;
    const double b = 0.75;
    const double avg_len = avg_len_ > 0.0 ? avg_len_ : 1.0;

    struct Scored {
        size_t idx;
        double score;
    };
    std::vector<Scored> scored;
    scored.reserve(chunks_.size());

    for (size_t i = 0; i < chunks_.size(); ++i) {
        const Chunk& chunk = chunks_[i];
        double score = 0.0;
        for (const auto& kv : q_tf) {
            auto tf_it = chunk.term_freq.find(kv.first);
            if (tf_it == chunk.term_freq.end()) continue;
            const int tf = tf_it->second;
            auto idf_it = idf_.find(kv.first);
            if (idf_it == idf_.end()) continue;
            const double idf = idf_it->second;
            const double denom = tf + k1 * (1.0 - b + b * (static_cast<double>(chunk.length) / avg_len));
            score += idf * (tf * (k1 + 1.0)) / denom;
        }
        if (score > 0.0) scored.push_back({i, score});
    }

    if (scored.empty()) return out;

    const size_t limit = std::min(top_k, scored.size());
    std::partial_sort(scored.begin(), scored.begin() + limit, scored.end(),
                      [](const Scored& a, const Scored& b) { return a.score > b.score; });

    out.reserve(limit);
    for (size_t i = 0; i < limit; ++i) {
        const Chunk& chunk = chunks_[scored[i].idx];
        RagHit hit;
        hit.source = chunk.source;
        hit.text = shorten(chunk.text, 520);
        hit.score = scored[i].score;
        out.push_back(std::move(hit));
    }
    return out;
}

std::vector<std::string> RagIndex::tokenize(const std::string& text) {
    std::vector<std::string> tokens;
    std::string cur;
    cur.reserve(32);

    for (size_t i = 0; i < text.size(); ++i) {
        unsigned char c = static_cast<unsigned char>(text[i]);
        if (c < 0x80) {
            if (is_ascii_word_char(c)) {
                cur.push_back(static_cast<char>(std::tolower(c)));
            } else {
                if (!cur.empty()) {
                    if (cur.size() > 1) tokens.push_back(cur);
                    cur.clear();
                }
            }
            continue;
        }

        if (!cur.empty()) {
            if (cur.size() > 1) tokens.push_back(cur);
            cur.clear();
        }
        size_t len = utf8_char_len(c);
        if (len == 0 || i + len > text.size()) continue;
        tokens.push_back(text.substr(i, len));
        i += len - 1;
    }

    if (!cur.empty() && cur.size() > 1) tokens.push_back(cur);
    return tokens;
}

std::vector<std::string> RagIndex::split_chunks(const std::string& text, size_t max_chars) {
    std::vector<std::string> chunks;
    std::istringstream iss(text);
    std::string line;
    std::string current;

    auto flush_current = [&]() {
        std::string trimmed = trim(current);
        if (!trimmed.empty()) chunks.push_back(std::move(trimmed));
        current.clear();
    };

    while (std::getline(iss, line)) {
        if (trim(line).empty()) {
            if (!current.empty()) flush_current();
            continue;
        }
        if (current.size() + line.size() + 1 > max_chars && !current.empty()) {
            flush_current();
        }
        if (!current.empty()) current.push_back('\n');
        current += line;
    }
    if (!current.empty()) flush_current();

    std::vector<std::string> expanded;
    for (const auto& chunk : chunks) {
        if (chunk.size() <= max_chars) {
            expanded.push_back(chunk);
            continue;
        }
        size_t pos = 0;
        while (pos < chunk.size()) {
            size_t len = std::min(max_chars, chunk.size() - pos);
            expanded.push_back(chunk.substr(pos, len));
            pos += len;
        }
    }
    return expanded.empty() ? chunks : expanded;
}

std::string RagIndex::trim(const std::string& s) {
    size_t start = 0;
    while (start < s.size() && std::isspace(static_cast<unsigned char>(s[start]))) ++start;
    size_t end = s.size();
    while (end > start && std::isspace(static_cast<unsigned char>(s[end - 1]))) --end;
    return s.substr(start, end - start);
}

std::string RagIndex::shorten(const std::string& s, size_t max_chars) {
    if (s.size() <= max_chars) return s;
    size_t cut = max_chars;
    if (cut > 3) cut -= 3;
    return s.substr(0, cut) + "...";
}

void RagIndex::build_stats() {
    if (chunks_.empty()) return;
    double total_len = 0.0;
    for (const auto& chunk : chunks_) {
        total_len += static_cast<double>(chunk.length);
    }
    avg_len_ = total_len / static_cast<double>(chunks_.size());

    const double n_docs = static_cast<double>(chunks_.size());
    for (const auto& kv : doc_freq_) {
        const double df = static_cast<double>(kv.second);
        double idf = std::log((n_docs - df + 0.5) / (df + 0.5) + 1.0);
        idf_[kv.first] = idf;
    }
}
