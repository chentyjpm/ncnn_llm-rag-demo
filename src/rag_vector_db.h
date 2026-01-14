#pragma once

#include <cstddef>
#include <string>
#include <vector>

struct RagSearchHit {
    std::string source;
    std::string text;
    double score = 0.0;
    size_t doc_id = 0;
    int chunk_index = 0;
};

class RagEmbedder {
public:
    explicit RagEmbedder(int dim);
    int dim() const { return dim_; }
    std::vector<float> embed(const std::string& text) const;

private:
    int dim_;
};

class RagVectorDb {
public:
    RagVectorDb();
    ~RagVectorDb();
    RagVectorDb(const RagVectorDb&) = delete;
    RagVectorDb& operator=(const RagVectorDb&) = delete;

    bool open(const std::string& path, int embed_dim, std::string* err);
    bool add_document(const std::string& filename,
                      const std::string& mime,
                      const std::string& text,
                      size_t chunk_chars,
                      std::string* err,
                      size_t* out_doc_id,
                      size_t* out_chunk_count);

    std::vector<RagSearchHit> search(const std::vector<float>& query_vec, size_t top_k) const;
    std::string expand_neighbors(size_t doc_id, int center_chunk_index, int neighbor_chunks) const;

    size_t doc_count() const { return doc_count_; }
    size_t chunk_count() const { return chunk_count_; }
    int embed_dim() const { return embed_dim_; }

private:
    struct sqlite3* db_ = nullptr;
    int embed_dim_ = 0;
    size_t doc_count_ = 0;
    size_t chunk_count_ = 0;

    bool exec(const std::string& sql, std::string* err) const;
    bool ensure_schema(std::string* err);
    bool load_counts(std::string* err);
};
