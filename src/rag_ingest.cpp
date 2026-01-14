#include "rag_ingest.h"

#include "rag_text.h"

#include <cstdlib>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

namespace fs = std::filesystem;

namespace {

std::string shell_escape(const std::string& s) {
    std::string out = "'";
    for (char c : s) {
        if (c == '\'') {
            out += "'\\''";
        } else {
            out.push_back(c);
        }
    }
    out.push_back('\'');
    return out;
}

bool command_exists(const std::string& name) {
    std::string cmd = "command -v " + name + " >/dev/null 2>&1";
    return std::system(cmd.c_str()) == 0;
}

} // namespace

bool read_text_file(const std::string& path, std::string* out, std::string* err) {
    std::ifstream ifs(path, std::ios::in | std::ios::binary);
    if (!ifs) {
        if (err) *err = "failed to open file";
        return false;
    }
    std::ostringstream oss;
    oss << ifs.rdbuf();
    *out = trim_text(oss.str());
    if (out->empty()) {
        if (err) *err = "empty text file";
        return false;
    }
    return true;
}

bool extract_pdf_text(const std::string& path, std::string* out, std::string* err) {
    if (!command_exists("pdftotext")) {
        if (err) *err = "pdftotext not found; please install poppler-utils";
        return false;
    }
    std::string cmd = "pdftotext -layout -q -enc UTF-8 " + shell_escape(path) + " -";
    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) {
        if (err) *err = "failed to execute pdftotext";
        return false;
    }

    std::string result;
    char buf[4096];
    while (true) {
        size_t n = fread(buf, 1, sizeof(buf), pipe);
        if (n > 0) result.append(buf, n);
        if (n < sizeof(buf)) break;
    }
    int rc = pclose(pipe);
    if (rc != 0) {
        if (err) *err = "pdftotext failed";
        return false;
    }

    *out = trim_text(result);
    if (out->empty()) {
        if (err) *err = "pdf contains no extractable text";
        return false;
    }
    return true;
}
