#pragma once

#include <string>

bool read_text_file(const std::string& path, std::string* out, std::string* err);
bool extract_pdf_text(const std::string& path, std::string* out, std::string* err);
