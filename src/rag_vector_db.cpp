#include "rag_vector_db.h"

#include "rag_text.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <unordered_map>

#include <sqlite3.h>

namespace {

struct Stmt {
    sqlite3_stmt* stmt = nullptr;
    ~Stmt() {
        if (stmt) sqlite3_finalize(stmt);
    }
};

std::vector<float> l2_normalize(std::vector<float> v) {
    double sum = 0.0;
    for (float x : v) sum += x * x;
    if (sum <= 0.0) return v;
    float inv = 1.0f / std::sqrt(sum);
    for (float& x : v) x *= inv;
    return v;
}

uint32_t hash_token(const std::string& s) {
    uint32_t h = 2166136261u;
    for (unsigned char c : s) {
        h ^= c;
        h *= 16777619u;
    }
    return h;
}

} // namespace

RagEmbedder::RagEmbedder(int dim) : dim_(dim > 0 ? dim : 256) {}

std::vector<float> RagEmbedder::embed(const std::string& text) const {
    std::vector<float> vec(dim_, 0.0f);
    std::vector<std::string> tokens = tokenize_text(text);
    if (tokens.empty()) return vec;

    std::unordered_map<int, int> counts;
    counts.reserve(tokens.size());
    for (const auto& tok : tokens) {
        uint32_t h = hash_token(tok);
        int idx = static_cast<int>(h % static_cast<uint32_t>(dim_));
        ++counts[idx];
    }

    for (const auto& kv : counts) {
        vec[kv.first] = std::log1p(static_cast<float>(kv.second));
    }
    return l2_normalize(std::move(vec));
}

RagVectorDb::RagVectorDb() = default;

RagVectorDb::~RagVectorDb() {
    if (db_) sqlite3_close(db_);
}

bool RagVectorDb::open(const std::string& path, int embed_dim, std::string* err) {
    if (db_) {
        sqlite3_close(db_);
        db_ = nullptr;
    }
    if (sqlite3_open(path.c_str(), &db_) != SQLITE_OK) {
        if (err) *err = sqlite3_errmsg(db_ ? db_ : nullptr);
        if (db_) sqlite3_close(db_);
        db_ = nullptr;
        return false;
    }
    embed_dim_ = embed_dim > 0 ? embed_dim : 256;
    if (!ensure_schema(err)) return false;
    if (!load_counts(err)) return false;
    return true;
}

bool RagVectorDb::exec(const std::string& sql, std::string* err) const {
    char* errmsg = nullptr;
    if (sqlite3_exec(db_, sql.c_str(), nullptr, nullptr, &errmsg) != SQLITE_OK) {
        if (err) *err = errmsg ? errmsg : "sqlite exec failed";
        sqlite3_free(errmsg);
        return false;
    }
    return true;
}

bool RagVectorDb::ensure_schema(std::string* err) {
    if (!exec("PRAGMA journal_mode=WAL;", err)) return false;
    if (!exec("CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT);", err)) return false;
    if (!exec("CREATE TABLE IF NOT EXISTS docs("
              "id INTEGER PRIMARY KEY AUTOINCREMENT,"
              "filename TEXT,"
              "mime TEXT,"
              "added_at INTEGER,"
              "chunk_count INTEGER);", err)) return false;
    if (!exec("CREATE TABLE IF NOT EXISTS chunks("
              "id INTEGER PRIMARY KEY AUTOINCREMENT,"
              "doc_id INTEGER,"
              "chunk_index INTEGER,"
              "source TEXT,"
              "text TEXT);", err)) return false;
    if (!exec("CREATE TABLE IF NOT EXISTS vectors("
              "chunk_id INTEGER PRIMARY KEY,"
              "dim INTEGER,"
              "vec BLOB);", err)) return false;
    if (!exec("CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);", err)) return false;

    const char* sql = "SELECT value FROM meta WHERE key='embed_dim';";
    Stmt stmt;
    if (sqlite3_prepare_v2(db_, sql, -1, &stmt.stmt, nullptr) != SQLITE_OK) {
        if (err) *err = sqlite3_errmsg(db_);
        return false;
    }
    int rc = sqlite3_step(stmt.stmt);
    if (rc == SQLITE_ROW) {
        const char* v = reinterpret_cast<const char*>(sqlite3_column_text(stmt.stmt, 0));
        if (v) {
            int stored = std::atoi(v);
            if (stored > 0 && stored != embed_dim_) {
                if (err) *err = "embedding dim mismatch in existing database";
                return false;
            }
        }
        return true;
    }

    const char* insert_sql = "INSERT OR REPLACE INTO meta(key, value) VALUES('embed_dim', ?);";
    Stmt ins;
    if (sqlite3_prepare_v2(db_, insert_sql, -1, &ins.stmt, nullptr) != SQLITE_OK) {
        if (err) *err = sqlite3_errmsg(db_);
        return false;
    }
    sqlite3_bind_int(ins.stmt, 1, embed_dim_);
    if (sqlite3_step(ins.stmt) != SQLITE_DONE) {
        if (err) *err = sqlite3_errmsg(db_);
        return false;
    }
    return true;
}

bool RagVectorDb::load_counts(std::string* err) {
    const char* doc_sql = "SELECT COUNT(*) FROM docs;";
    Stmt doc_stmt;
    if (sqlite3_prepare_v2(db_, doc_sql, -1, &doc_stmt.stmt, nullptr) != SQLITE_OK) {
        if (err) *err = sqlite3_errmsg(db_);
        return false;
    }
    if (sqlite3_step(doc_stmt.stmt) == SQLITE_ROW) {
        doc_count_ = static_cast<size_t>(sqlite3_column_int64(doc_stmt.stmt, 0));
    }

    const char* chunk_sql = "SELECT COUNT(*) FROM chunks;";
    Stmt chunk_stmt;
    if (sqlite3_prepare_v2(db_, chunk_sql, -1, &chunk_stmt.stmt, nullptr) != SQLITE_OK) {
        if (err) *err = sqlite3_errmsg(db_);
        return false;
    }
    if (sqlite3_step(chunk_stmt.stmt) == SQLITE_ROW) {
        chunk_count_ = static_cast<size_t>(sqlite3_column_int64(chunk_stmt.stmt, 0));
    }
    return true;
}

bool RagVectorDb::add_document(const std::string& filename,
                               const std::string& mime,
                               const std::string& text,
                               size_t chunk_chars,
                               std::string* err,
                               size_t* out_doc_id,
                               size_t* out_chunk_count) {
    if (!db_) {
        if (err) *err = "database not initialized";
        return false;
    }

    std::vector<std::string> chunks = split_text_chunks(text, chunk_chars);
    if (chunks.empty()) {
        if (err) *err = "no text chunks generated";
        return false;
    }

    if (!exec("BEGIN TRANSACTION;", err)) return false;

    const char* insert_doc_sql = "INSERT INTO docs(filename, mime, added_at, chunk_count) VALUES(?, ?, strftime('%s','now'), ?);";
    Stmt doc_stmt;
    if (sqlite3_prepare_v2(db_, insert_doc_sql, -1, &doc_stmt.stmt, nullptr) != SQLITE_OK) {
        if (err) *err = sqlite3_errmsg(db_);
        exec("ROLLBACK;", nullptr);
        return false;
    }
    sqlite3_bind_text(doc_stmt.stmt, 1, filename.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(doc_stmt.stmt, 2, mime.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_int(doc_stmt.stmt, 3, static_cast<int>(chunks.size()));
    if (sqlite3_step(doc_stmt.stmt) != SQLITE_DONE) {
        if (err) *err = sqlite3_errmsg(db_);
        exec("ROLLBACK;", nullptr);
        return false;
    }
    sqlite3_int64 doc_id = sqlite3_last_insert_rowid(db_);

    const char* insert_chunk_sql = "INSERT INTO chunks(doc_id, chunk_index, source, text) VALUES(?, ?, ?, ?);";
    const char* insert_vec_sql = "INSERT INTO vectors(chunk_id, dim, vec) VALUES(?, ?, ?);";
    Stmt chunk_stmt;
    Stmt vec_stmt;
    if (sqlite3_prepare_v2(db_, insert_chunk_sql, -1, &chunk_stmt.stmt, nullptr) != SQLITE_OK ||
        sqlite3_prepare_v2(db_, insert_vec_sql, -1, &vec_stmt.stmt, nullptr) != SQLITE_OK) {
        if (err) *err = sqlite3_errmsg(db_);
        exec("ROLLBACK;", nullptr);
        return false;
    }

    RagEmbedder embedder(embed_dim_);
    size_t idx = 0;
    for (const auto& chunk : chunks) {
        std::string trimmed = trim_text(chunk);
        if (trimmed.empty()) continue;
        std::string source = filename + "#" + std::to_string(idx);

        sqlite3_reset(chunk_stmt.stmt);
        sqlite3_bind_int64(chunk_stmt.stmt, 1, doc_id);
        sqlite3_bind_int(chunk_stmt.stmt, 2, static_cast<int>(idx));
        sqlite3_bind_text(chunk_stmt.stmt, 3, source.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_text(chunk_stmt.stmt, 4, trimmed.c_str(), -1, SQLITE_TRANSIENT);
        if (sqlite3_step(chunk_stmt.stmt) != SQLITE_DONE) {
            if (err) *err = sqlite3_errmsg(db_);
            exec("ROLLBACK;", nullptr);
            return false;
        }

        sqlite3_int64 chunk_id = sqlite3_last_insert_rowid(db_);
        std::vector<float> vec = embedder.embed(trimmed);

        sqlite3_reset(vec_stmt.stmt);
        sqlite3_bind_int64(vec_stmt.stmt, 1, chunk_id);
        sqlite3_bind_int(vec_stmt.stmt, 2, embed_dim_);
        sqlite3_bind_blob(vec_stmt.stmt, 3, vec.data(), static_cast<int>(vec.size() * sizeof(float)), SQLITE_TRANSIENT);
        if (sqlite3_step(vec_stmt.stmt) != SQLITE_DONE) {
            if (err) *err = sqlite3_errmsg(db_);
            exec("ROLLBACK;", nullptr);
            return false;
        }
        ++idx;
    }

    if (!exec("COMMIT;", err)) {
        exec("ROLLBACK;", nullptr);
        return false;
    }

    doc_count_ += 1;
    chunk_count_ += idx;
    if (out_doc_id) *out_doc_id = static_cast<size_t>(doc_id);
    if (out_chunk_count) *out_chunk_count = idx;
    return true;
}

bool RagVectorDb::delete_doc(size_t doc_id, std::string* err) {
    if (!db_) {
        if (err) *err = "database not initialized";
        return false;
    }

    if (!exec("BEGIN TRANSACTION;", err)) return false;

    {
        const char* sql = "SELECT id FROM docs WHERE id = ?;";
        Stmt stmt;
        if (sqlite3_prepare_v2(db_, sql, -1, &stmt.stmt, nullptr) != SQLITE_OK) {
            if (err) *err = sqlite3_errmsg(db_);
            exec("ROLLBACK;", nullptr);
            return false;
        }
        sqlite3_bind_int64(stmt.stmt, 1, static_cast<sqlite3_int64>(doc_id));
        if (sqlite3_step(stmt.stmt) != SQLITE_ROW) {
            if (err) *err = "document not found";
            exec("ROLLBACK;", nullptr);
            return false;
        }
    }

    {
        const char* sql = "DELETE FROM vectors WHERE chunk_id IN (SELECT id FROM chunks WHERE doc_id = ?);";
        Stmt stmt;
        if (sqlite3_prepare_v2(db_, sql, -1, &stmt.stmt, nullptr) != SQLITE_OK) {
            if (err) *err = sqlite3_errmsg(db_);
            exec("ROLLBACK;", nullptr);
            return false;
        }
        sqlite3_bind_int64(stmt.stmt, 1, static_cast<sqlite3_int64>(doc_id));
        if (sqlite3_step(stmt.stmt) != SQLITE_DONE) {
            if (err) *err = sqlite3_errmsg(db_);
            exec("ROLLBACK;", nullptr);
            return false;
        }
    }

    {
        const char* sql = "DELETE FROM chunks WHERE doc_id = ?;";
        Stmt stmt;
        if (sqlite3_prepare_v2(db_, sql, -1, &stmt.stmt, nullptr) != SQLITE_OK) {
            if (err) *err = sqlite3_errmsg(db_);
            exec("ROLLBACK;", nullptr);
            return false;
        }
        sqlite3_bind_int64(stmt.stmt, 1, static_cast<sqlite3_int64>(doc_id));
        if (sqlite3_step(stmt.stmt) != SQLITE_DONE) {
            if (err) *err = sqlite3_errmsg(db_);
            exec("ROLLBACK;", nullptr);
            return false;
        }
    }

    {
        const char* sql = "DELETE FROM docs WHERE id = ?;";
        Stmt stmt;
        if (sqlite3_prepare_v2(db_, sql, -1, &stmt.stmt, nullptr) != SQLITE_OK) {
            if (err) *err = sqlite3_errmsg(db_);
            exec("ROLLBACK;", nullptr);
            return false;
        }
        sqlite3_bind_int64(stmt.stmt, 1, static_cast<sqlite3_int64>(doc_id));
        if (sqlite3_step(stmt.stmt) != SQLITE_DONE) {
            if (err) *err = sqlite3_errmsg(db_);
            exec("ROLLBACK;", nullptr);
            return false;
        }
    }

    std::string local_err;
    if (!load_counts(&local_err)) {
        if (err) *err = local_err;
        exec("ROLLBACK;", nullptr);
        return false;
    }

    if (!exec("COMMIT;", err)) {
        exec("ROLLBACK;", nullptr);
        return false;
    }
    return true;
}

std::vector<RagSearchHit> RagVectorDb::search(const std::vector<float>& query_vec, size_t top_k) const {
    std::vector<RagSearchHit> out;
    if (!db_ || query_vec.empty() || top_k == 0) return out;

    const char* sql =
        "SELECT chunks.source, chunks.text, vectors.vec, vectors.dim, chunks.doc_id, chunks.chunk_index "
        "FROM vectors JOIN chunks ON vectors.chunk_id = chunks.id;";
    Stmt stmt;
    if (sqlite3_prepare_v2(db_, sql, -1, &stmt.stmt, nullptr) != SQLITE_OK) {
        return out;
    }

    struct Scored {
        RagSearchHit hit;
    };
    std::vector<Scored> scored;

    while (sqlite3_step(stmt.stmt) == SQLITE_ROW) {
        const unsigned char* source = sqlite3_column_text(stmt.stmt, 0);
        const unsigned char* text = sqlite3_column_text(stmt.stmt, 1);
        const void* blob = sqlite3_column_blob(stmt.stmt, 2);
        int dim = sqlite3_column_int(stmt.stmt, 3);
        sqlite3_int64 doc_id = sqlite3_column_int64(stmt.stmt, 4);
        int chunk_index = sqlite3_column_int(stmt.stmt, 5);
        int bytes = sqlite3_column_bytes(stmt.stmt, 2);
        if (!blob || dim <= 0 || bytes != dim * static_cast<int>(sizeof(float))) continue;
        if (static_cast<int>(query_vec.size()) != dim) continue;

        const float* vec = reinterpret_cast<const float*>(blob);
        double score = 0.0;
        for (int i = 0; i < dim; ++i) {
            score += static_cast<double>(query_vec[i]) * static_cast<double>(vec[i]);
        }
        if (score <= 0.0) continue;

        RagSearchHit hit;
        hit.source = source ? reinterpret_cast<const char*>(source) : "";
        hit.text = text ? reinterpret_cast<const char*>(text) : "";
        hit.score = score;
        hit.doc_id = static_cast<size_t>(doc_id);
        hit.chunk_index = chunk_index;
        scored.push_back({std::move(hit)});
    }

    if (scored.empty()) return out;
    std::sort(scored.begin(), scored.end(), [](const Scored& a, const Scored& b) {
        return a.hit.score > b.hit.score;
    });
    size_t limit = std::min(top_k, scored.size());
    out.reserve(limit);
    for (size_t i = 0; i < limit; ++i) {
        RagSearchHit hit = scored[i].hit;
        hit.text = shorten_text(hit.text, 520);
        out.push_back(std::move(hit));
    }
    return out;
}

std::string RagVectorDb::expand_neighbors(size_t doc_id, int center_chunk_index, int neighbor_chunks) const {
    if (!db_ || neighbor_chunks <= 0) return {};
    if (center_chunk_index < 0) return {};

    int start = center_chunk_index - neighbor_chunks;
    if (start < 0) start = 0;
    int end = center_chunk_index + neighbor_chunks;
    return expand_range(doc_id, start, end, center_chunk_index);
}

std::string RagVectorDb::expand_range(size_t doc_id,
                                     int start_chunk_index,
                                     int end_chunk_index,
                                     int center_chunk_index) const {
    if (!db_) return {};
    if (center_chunk_index < 0) return {};
    if (start_chunk_index < 0) start_chunk_index = 0;
    if (end_chunk_index < start_chunk_index) return {};

    const char* sql =
        "SELECT chunk_index, text "
        "FROM chunks "
        "WHERE doc_id = ? AND chunk_index BETWEEN ? AND ? "
        "ORDER BY chunk_index ASC;";
    Stmt stmt;
    if (sqlite3_prepare_v2(db_, sql, -1, &stmt.stmt, nullptr) != SQLITE_OK) {
        return {};
    }
    sqlite3_bind_int64(stmt.stmt, 1, static_cast<sqlite3_int64>(doc_id));
    sqlite3_bind_int(stmt.stmt, 2, start_chunk_index);
    sqlite3_bind_int(stmt.stmt, 3, end_chunk_index);

    std::string out;
    bool first = true;
    while (sqlite3_step(stmt.stmt) == SQLITE_ROW) {
        int idx = sqlite3_column_int(stmt.stmt, 0);
        const unsigned char* text = sqlite3_column_text(stmt.stmt, 1);
        if (!text) continue;
        if (!first) out += "\n\n";
        first = false;
        if (idx != center_chunk_index) {
            out += "(neighbor chunk " + std::to_string(idx) + ")\n";
        } else {
            out += "(matched chunk " + std::to_string(idx) + ")\n";
        }
        out += reinterpret_cast<const char*>(text);
    }
    return out;
}

bool RagVectorDb::get_document_chunks(size_t doc_id,
                                      std::string* out_filename,
                                      std::vector<RagSearchHit>* out_chunks,
                                      std::string* err) const {
    if (!db_) {
        if (err) *err = "database not initialized";
        return false;
    }
    if (!out_filename || !out_chunks) {
        if (err) *err = "invalid output pointers";
        return false;
    }

    out_filename->clear();
    out_chunks->clear();

    {
        const char* sql = "SELECT filename FROM docs WHERE id = ?;";
        Stmt stmt;
        if (sqlite3_prepare_v2(db_, sql, -1, &stmt.stmt, nullptr) != SQLITE_OK) {
            if (err) *err = sqlite3_errmsg(db_);
            return false;
        }
        sqlite3_bind_int64(stmt.stmt, 1, static_cast<sqlite3_int64>(doc_id));
        int rc = sqlite3_step(stmt.stmt);
        if (rc != SQLITE_ROW) {
            if (err) *err = "document not found";
            return false;
        }
        const unsigned char* filename = sqlite3_column_text(stmt.stmt, 0);
        if (filename) *out_filename = reinterpret_cast<const char*>(filename);
    }

    const char* chunk_sql =
        "SELECT chunk_index, source, text "
        "FROM chunks "
        "WHERE doc_id = ? "
        "ORDER BY chunk_index ASC;";
    Stmt chunk_stmt;
    if (sqlite3_prepare_v2(db_, chunk_sql, -1, &chunk_stmt.stmt, nullptr) != SQLITE_OK) {
        if (err) *err = sqlite3_errmsg(db_);
        return false;
    }
    sqlite3_bind_int64(chunk_stmt.stmt, 1, static_cast<sqlite3_int64>(doc_id));
    while (sqlite3_step(chunk_stmt.stmt) == SQLITE_ROW) {
        int chunk_index = sqlite3_column_int(chunk_stmt.stmt, 0);
        const unsigned char* source = sqlite3_column_text(chunk_stmt.stmt, 1);
        const unsigned char* text = sqlite3_column_text(chunk_stmt.stmt, 2);

        RagSearchHit hit;
        hit.doc_id = doc_id;
        hit.chunk_index = chunk_index;
        hit.source = source ? reinterpret_cast<const char*>(source) : "";
        hit.text = text ? reinterpret_cast<const char*>(text) : "";
        hit.score = 0.0;
        out_chunks->push_back(std::move(hit));
    }

    return true;
}

std::vector<RagDocInfo> RagVectorDb::list_docs(size_t limit, size_t offset) const {
    std::vector<RagDocInfo> out;
    if (!db_ || limit == 0) return out;

    const char* sql =
        "SELECT id, filename, mime, added_at, chunk_count "
        "FROM docs "
        "ORDER BY id DESC "
        "LIMIT ? OFFSET ?;";
    Stmt stmt;
    if (sqlite3_prepare_v2(db_, sql, -1, &stmt.stmt, nullptr) != SQLITE_OK) {
        return out;
    }
    sqlite3_bind_int64(stmt.stmt, 1, static_cast<sqlite3_int64>(limit));
    sqlite3_bind_int64(stmt.stmt, 2, static_cast<sqlite3_int64>(offset));

    while (sqlite3_step(stmt.stmt) == SQLITE_ROW) {
        sqlite3_int64 id = sqlite3_column_int64(stmt.stmt, 0);
        const unsigned char* filename = sqlite3_column_text(stmt.stmt, 1);
        const unsigned char* mime = sqlite3_column_text(stmt.stmt, 2);
        sqlite3_int64 added_at = sqlite3_column_int64(stmt.stmt, 3);
        sqlite3_int64 chunk_count = sqlite3_column_int64(stmt.stmt, 4);

        RagDocInfo info;
        info.id = static_cast<size_t>(id);
        info.filename = filename ? reinterpret_cast<const char*>(filename) : "";
        info.mime = mime ? reinterpret_cast<const char*>(mime) : "";
        info.added_at = static_cast<int64_t>(added_at);
        info.chunk_count = chunk_count > 0 ? static_cast<size_t>(chunk_count) : 0;
        out.push_back(std::move(info));
    }
    return out;
}
