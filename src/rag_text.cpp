#include "rag_text.h"

#include <algorithm>
#include <cctype>
#include <sstream>
#include <string_view>

std::string trim_text(const std::string& s);

namespace {

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

bool is_utf8_continuation(unsigned char c) {
    return (c & 0xC0) == 0x80;
}

size_t utf8_safe_cut_pos(const std::string& s, size_t pos) {
    if (pos >= s.size()) return s.size();
    while (pos > 0 && is_utf8_continuation(static_cast<unsigned char>(s[pos]))) {
        --pos;
    }
    return pos;
}

std::string normalize_newlines(std::string s) {
    // Convert CRLF/CR to LF.
    std::string out;
    out.reserve(s.size());
    for (size_t i = 0; i < s.size(); ++i) {
        char c = s[i];
        if (c == '\r') {
            if (i + 1 < s.size() && s[i + 1] == '\n') ++i;
            out.push_back('\n');
        } else {
            out.push_back(c);
        }
    }
    return out;
}

bool starts_with(std::string_view s, std::string_view prefix) {
    return s.size() >= prefix.size() && s.substr(0, prefix.size()) == prefix;
}

bool match_at(const std::string& s, size_t pos, std::string_view needle) {
    return pos + needle.size() <= s.size() && std::string_view(s).substr(pos, needle.size()) == needle;
}

bool looks_like_heading(const std::string& line) {
    // Heuristics for CN/EN headings: short line + numbering patterns.
    std::string t = trim_text(line);
    if (t.empty()) return false;
    if (t.size() > 120) return false;

    // Common CN headings.
    if (starts_with(t, "第") && (t.find("章") != std::string::npos || t.find("节") != std::string::npos ||
                                 t.find("条") != std::string::npos || t.find("部分") != std::string::npos)) {
        return true;
    }
    if (starts_with(t, "附录") || starts_with(t, "目录")) return true;

    // "一、" / "二、" / "三、" ...
    if (t.size() >= 3) {
        unsigned char c0 = static_cast<unsigned char>(t[0]);
        if (c0 >= 0x80) {
            if (t.find("、") != std::string::npos && t.find("、") <= 6) return true;
        }
    }

    // 1 / 1. / 1.2 / 1.2.3
    size_t i = 0;
    int dot_count = 0;
    while (i < t.size() && std::isdigit(static_cast<unsigned char>(t[i]))) ++i;
    if (i > 0) {
        auto consume_dot = [&]() -> bool {
            if (i < t.size() && t[i] == '.') {
                ++i;
                return true;
            }
            if (match_at(t, i, u8"．")) {
                i += std::string_view(u8"．").size();
                return true;
            }
            return false;
        };
        while (consume_dot()) {
            ++dot_count;
            while (i < t.size() && std::isdigit(static_cast<unsigned char>(t[i]))) ++i;
        }
        if (dot_count >= 1 && (i < t.size()) && (t[i] == ' ' || t[i] == '\t' || static_cast<unsigned char>(t[i]) >= 0x80)) {
            return true;
        }
        // "1)" / "1、"
        if (i < t.size() && (t[i] == ')' || match_at(t, i, u8"）") || match_at(t, i, u8"、"))) return true;
    }
    return false;
}

bool looks_like_list_item(const std::string& line) {
    std::string t = trim_text(line);
    if (t.empty()) return false;
    if (starts_with(t, "- ") || starts_with(t, "* ") || starts_with(t, "•")) return true;

    // "(1)" / "（一）" / "1)" / "1、"
    if (starts_with(t, "(") || starts_with(t, "（")) return true;
    size_t i = 0;
    while (i < t.size() && std::isdigit(static_cast<unsigned char>(t[i]))) ++i;
    if (i > 0 && i < t.size()) {
        if (t[i] == ')' || t[i] == '.' || match_at(t, i, u8"）") || match_at(t, i, u8"．") || match_at(t, i, u8"、")) return true;
    }
    return false;
}

bool looks_like_table_line(const std::string& line) {
    // Very light heuristic: many pipes or many runs of multiple spaces.
    int pipe = 0;
    for (char c : line) {
        if (c == '|') ++pipe;
    }
    if (pipe >= 2) return true;

    int multi_space_runs = 0;
    int run = 0;
    for (char c : line) {
        if (c == ' ' || c == '\t') {
            ++run;
        } else {
            if (run >= 3) ++multi_space_runs;
            run = 0;
        }
    }
    if (run >= 3) ++multi_space_runs;
    return multi_space_runs >= 2;
}

size_t find_last_sentence_boundary(const std::string& s, size_t start, size_t end) {
    // Search backwards for a "good" cut point (sentence end / paragraph).
    if (end <= start) return start;
    size_t i = end;
    auto is_delim = [&](std::string_view v) -> bool {
        // Common sentence delimiters in CN/EN.
        return v == "\n" || v == "." || v == "!" || v == "?" || v == ";" ||
               v == "。" || v == "！" || v == "？" || v == "；";
    };

    // Look back up to 256 bytes for a delimiter.
    size_t window_start = (i > start + 256) ? (i - 256) : start;
    while (i > window_start) {
        unsigned char c = static_cast<unsigned char>(s[i - 1]);
        if (c < 0x80) {
            std::string_view v(&s[i - 1], 1);
            if (is_delim(v)) return i;
            --i;
            continue;
        }

        // For UTF-8, step to char start.
        size_t j = i - 1;
        while (j > start && is_utf8_continuation(static_cast<unsigned char>(s[j]))) --j;
        size_t len = i - j;
        if (len == 3) {
            std::string_view v(&s[j], 3);
            if (is_delim(v)) return i;
        }
        i = j;
    }
    return start;
}

std::vector<std::string> split_long_block(const std::string& block, size_t max_chars) {
    std::vector<std::string> out;
    if (block.size() <= max_chars) {
        out.push_back(block);
        return out;
    }

    size_t pos = 0;
    while (pos < block.size()) {
        size_t remaining = block.size() - pos;
        size_t want = std::min(max_chars, remaining);
        size_t end = pos + want;
        if (end < block.size()) {
            size_t cut = find_last_sentence_boundary(block, pos, end);
            if (cut > pos) end = cut;
            end = utf8_safe_cut_pos(block, end);
            if (end <= pos) {
                end = utf8_safe_cut_pos(block, pos + want);
                if (end <= pos) end = std::min(pos + want, block.size());
            }
        }
        std::string piece = trim_text(block.substr(pos, end - pos));
        if (!piece.empty()) out.push_back(std::move(piece));
        pos = end;
    }
    return out;
}

} // namespace

std::string trim_text(const std::string& s) {
    size_t start = 0;
    while (start < s.size() && std::isspace(static_cast<unsigned char>(s[start]))) ++start;
    size_t end = s.size();
    while (end > start && std::isspace(static_cast<unsigned char>(s[end - 1]))) --end;
    return s.substr(start, end - start);
}

std::string shorten_text(const std::string& s, size_t max_chars) {
    if (s.size() <= max_chars) return s;
    size_t cut = max_chars;
    if (cut > 3) cut -= 3;
    return s.substr(0, cut) + "...";
}

std::vector<std::string> split_text_chunks(const std::string& text, size_t max_chars) {
    if (max_chars == 0) max_chars = 512;
    std::vector<std::string> blocks;
    std::istringstream iss(normalize_newlines(text));
    std::string line;
    std::string cur_block;
    enum class BlockType { Paragraph, List, Table } cur_type = BlockType::Paragraph;

    auto flush_block = [&]() {
        std::string trimmed = trim_text(cur_block);
        if (!trimmed.empty()) blocks.push_back(std::move(trimmed));
        cur_block.clear();
        cur_type = BlockType::Paragraph;
    };

    while (std::getline(iss, line)) {
        std::string trimmed = trim_text(line);
        if (trimmed.empty()) {
            if (!cur_block.empty()) flush_block();
            continue;
        }

        // Headings become hard boundaries.
        if (looks_like_heading(trimmed)) {
            if (!cur_block.empty()) flush_block();
            cur_block = trimmed;
            flush_block();
            continue;
        }

        BlockType t = BlockType::Paragraph;
        if (looks_like_table_line(line)) t = BlockType::Table;
        else if (looks_like_list_item(line)) t = BlockType::List;

        if (!cur_block.empty() && t != cur_type) {
            flush_block();
        }
        cur_type = t;

        if (!cur_block.empty()) cur_block.push_back('\n');
        cur_block += line;
    }
    if (!cur_block.empty()) flush_block();

    // Assemble blocks into final chunks near max_chars.
    std::vector<std::string> chunks;
    std::string current;
    current.reserve(max_chars + 64);

    auto flush_chunk = [&]() {
        std::string trimmed = trim_text(current);
        if (!trimmed.empty()) chunks.push_back(std::move(trimmed));
        current.clear();
    };

    for (const auto& b : blocks) {
        if (b.size() > max_chars) {
            if (!current.empty()) flush_chunk();
            auto pieces = split_long_block(b, max_chars);
            for (auto& p : pieces) chunks.push_back(std::move(p));
            continue;
        }

        size_t extra = b.size() + (current.empty() ? 0 : 2);
        if (!current.empty() && current.size() + extra > max_chars) {
            flush_chunk();
        }
        if (!current.empty()) current += "\n\n";
        current += b;
    }
    if (!current.empty()) flush_chunk();

    return chunks;
}

std::vector<std::string> tokenize_text(const std::string& text) {
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
