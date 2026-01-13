#pragma once

#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>

struct RagHit {
    std::string source;
    std::string text;
    double score = 0.0;
};

class RagIndex {
public:
    bool load_directory(const std::string& dir, std::string* err);
    std::vector<RagHit> search(const std::string& query, size_t top_k) const;
    size_t doc_count() const { return doc_count_; }
    size_t chunk_count() const { return chunks_.size(); }

private:
    struct Chunk {
        std::string source;
        std::string text;
        std::unordered_map<std::string, int> term_freq;
        size_t length = 0;
    };

    std::vector<Chunk> chunks_;
    std::unordered_map<std::string, int> doc_freq_;
    std::unordered_map<std::string, double> idf_;
    size_t doc_count_ = 0;
    double avg_len_ = 0.0;

    static std::vector<std::string> tokenize(const std::string& text);
    static std::vector<std::string> split_chunks(const std::string& text, size_t max_chars);
    static std::string trim(const std::string& s);
    static std::string shorten(const std::string& s, size_t max_chars);
    void build_stats();
};
