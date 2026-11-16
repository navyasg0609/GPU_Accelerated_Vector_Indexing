#include "vs/common.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <sstream>

namespace vs {

Args::Args(int argc, char** argv, std::string description)
    : prog_(argv[0]), description_(std::move(description)) {
    for (int i = 1; i < argc; ++i) {
        std::string tok = argv[i];
        if (tok == "--help" || tok == "-h") {
            want_help_ = true;
            continue;
        }
        if (tok.rfind("--", 0) != 0) {
            std::fprintf(stderr, "Unexpected positional argument '%s'\n", tok.c_str());
            std::exit(EXIT_FAILURE);
        }
        tok = tok.substr(2);
        const size_t eq = tok.find('=');
        if (eq != std::string::npos) {
            values_[tok.substr(0, eq)] = tok.substr(eq + 1);
        } else if (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0) {
            values_[tok] = argv[++i];
        } else {
            values_[tok] = "true";
        }
    }
}

const std::string* Args::find(const std::string& key, const std::string& def_repr,
                              const std::string& help) {
    declared_[key] = true;
    help_lines_.push_back("  --" + key + " (default: " + def_repr + ")\n      " + help);
    auto it = values_.find(key);
    return it == values_.end() ? nullptr : &it->second;
}

std::string Args::str(const std::string& key, const std::string& def, const std::string& help) {
    const std::string* v = find(key, def.empty() ? "\"\"" : def, help);
    return v ? *v : def;
}

int64_t Args::i64(const std::string& key, int64_t def, const std::string& help) {
    const std::string* v = find(key, std::to_string(def), help);
    if (!v) return def;
    try {
        size_t pos = 0;
        const int64_t x = std::stoll(*v, &pos);
        if (pos != v->size()) throw std::invalid_argument("trailing characters");
        return x;
    } catch (const std::exception&) {
        std::fprintf(stderr, "--%s expects an integer, got '%s'\n", key.c_str(), v->c_str());
        std::exit(EXIT_FAILURE);
    }
}

int Args::i32(const std::string& key, int def, const std::string& help) {
    return static_cast<int>(i64(key, def, help));
}

double Args::f64(const std::string& key, double def, const std::string& help) {
    const std::string* v = find(key, fmt(def), help);
    if (!v) return def;
    try {
        return std::stod(*v);
    } catch (const std::exception&) {
        std::fprintf(stderr, "--%s expects a number, got '%s'\n", key.c_str(), v->c_str());
        std::exit(EXIT_FAILURE);
    }
}

bool Args::flag(const std::string& key, bool def, const std::string& help) {
    const std::string* v = find(key, def ? "true" : "false", help);
    if (!v) return def;
    if (*v == "true" || *v == "1") return true;
    if (*v == "false" || *v == "0") return false;
    std::fprintf(stderr, "--%s expects true/false, got '%s'\n", key.c_str(), v->c_str());
    std::exit(EXIT_FAILURE);
}

std::vector<int> Args::int_list(const std::string& key, const std::string& def,
                                const std::string& help) {
    const std::string* v = find(key, def, help);
    std::vector<int> out;
    for (const auto& p : split(v ? *v : def, ',')) {
        if (p.empty()) continue;
        try {
            out.push_back(std::stoi(p));
        } catch (const std::exception&) {
            std::fprintf(stderr, "--%s expects a comma-separated integer list\n", key.c_str());
            std::exit(EXIT_FAILURE);
        }
    }
    return out;
}

std::vector<std::string> Args::str_list(const std::string& key, const std::string& def,
                                        const std::string& help) {
    const std::string* v = find(key, def, help);
    std::vector<std::string> out;
    for (const auto& p : split(v ? *v : def, ','))
        if (!p.empty()) out.push_back(p);
    return out;
}

void Args::finish() {
    if (want_help_) {
        std::cout << description_ << "\n\nUsage: " << prog_ << " [options]\n";
        for (const auto& l : help_lines_) std::cout << l << "\n";
        std::exit(EXIT_SUCCESS);
    }
    bool bad = false;
    for (const auto& kv : values_) {
        if (!declared_.count(kv.first)) {
            std::fprintf(stderr, "Unknown option --%s (see --help)\n", kv.first.c_str());
            bad = true;
        }
    }
    if (bad) std::exit(EXIT_FAILURE);
}

std::vector<std::string> split(const std::string& s, char sep) {
    std::vector<std::string> out;
    std::string cur;
    std::istringstream in(s);
    while (std::getline(in, cur, sep)) out.push_back(cur);
    return out;
}

Stats compute_stats(const std::vector<double>& xs) {
    Stats s;
    s.n = xs.size();
    if (xs.empty()) return s;
    double sum = 0;
    for (double x : xs) sum += x;
    s.mean = sum / xs.size();
    double var = 0;
    for (double x : xs) var += (x - s.mean) * (x - s.mean);
    s.std = xs.size() > 1 ? std::sqrt(var / (xs.size() - 1)) : 0.0;
    s.min = *std::min_element(xs.begin(), xs.end());
    s.max = *std::max_element(xs.begin(), xs.end());
    return s;
}

CsvWriter::CsvWriter(const std::string& path, const std::vector<std::string>& header)
    : path_(path), ncols_(header.size()) {
    const std::filesystem::path p(path);
    if (p.has_parent_path()) std::filesystem::create_directories(p.parent_path());
    const bool fresh = !std::filesystem::exists(p) || std::filesystem::file_size(p) == 0;
    out_.open(path, std::ios::app);
    VS_REQUIRE(out_.good(), "could not open CSV file for writing");
    if (fresh) row(header);
}

void CsvWriter::row(const std::vector<std::string>& values) {
    VS_REQUIRE(values.size() == ncols_, "CSV row has the wrong number of columns");
    for (size_t i = 0; i < values.size(); ++i) {
        if (i) out_ << ',';
        const bool quote = values[i].find(',') != std::string::npos;
        if (quote) out_ << '"' << values[i] << '"';
        else out_ << values[i];
    }
    out_ << '\n';
    out_.flush();
}

std::string fmt(double v, int precision) {
    if (std::isnan(v)) return "";
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.*g", precision, v);
    return buf;
}

std::vector<std::string> concat(std::vector<std::string> a, const std::vector<std::string>& b) {
    a.insert(a.end(), b.begin(), b.end());
    return a;
}

}  // namespace vs
