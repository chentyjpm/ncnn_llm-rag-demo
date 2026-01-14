#pragma once

#include <string>
#include <vector>

std::string trim_text(const std::string& s);
std::string shorten_text(const std::string& s, size_t max_chars);
std::vector<std::string> split_text_chunks(const std::string& text, size_t max_chars);
std::vector<std::string> tokenize_text(const std::string& text);
