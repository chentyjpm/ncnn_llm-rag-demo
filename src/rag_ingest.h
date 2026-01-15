#pragma once

#include <filesystem>
#include <string>

// Normalize a string to valid UTF-8 if possible (handles BOM/UTF-16, and best-effort legacy encodings).
// Returns false if input is not UTF-8 and conversion failed.
bool normalize_utf8(std::string* s, std::string* err);

bool read_text_file(const std::filesystem::path& path, std::string* out, std::string* err);
bool extract_pdf_text(const std::filesystem::path& path, std::string* out, std::string* err);
