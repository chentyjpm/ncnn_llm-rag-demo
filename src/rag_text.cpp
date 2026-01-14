#include "rag_text.h"

#include <algorithm>
#include <cctype>
#include <sstream>

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
    std::vector<std::string> chunks;
    std::istringstream iss(text);
    std::string line;
    std::string current;

    auto flush_current = [&]() {
        std::string trimmed = trim_text(current);
        if (!trimmed.empty()) chunks.push_back(std::move(trimmed));
        current.clear();
    };

    while (std::getline(iss, line)) {
        if (trim_text(line).empty()) {
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
