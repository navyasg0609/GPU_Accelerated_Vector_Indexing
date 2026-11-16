// Host-only utilities: argument parsing, statistics, timers, CSV output.
#pragma once

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#define VS_REQUIRE(cond, msg)                                                              \
    do {                                                                                   \
        if (!(cond)) {                                                                     \
            std::fprintf(stderr, "Requirement failed: %s [%s] at %s:%d\n", msg, #cond,     \
                         __FILE__, __LINE__);                                              \
            std::exit(EXIT_FAILURE);                                                       \
        }                                                                                  \
    } while (0)

namespace vs {

inline int64_t ceil_div(int64_t a, int64_t b) { return (a + b - 1) / b; }

// Parses --key=value, --key value and bare --flag options. Every option is declared by a
// getter call (which also records its help text); finish() then rejects unknown options.
class Args {
public:
    Args(int argc, char** argv, std::string description);

    std::string str(const std::string& key, const std::string& def, const std::string& help);
    int64_t i64(const std::string& key, int64_t def, const std::string& help);
    int i32(const std::string& key, int def, const std::string& help);
    double f64(const std::string& key, double def, const std::string& help);
    bool flag(const std::string& key, bool def, const std::string& help);
    std::vector<int> int_list(const std::string& key, const std::string& def, const std::string& help);
    std::vector<std::string> str_list(const std::string& key, const std::string& def,
                                      const std::string& help);

    // Prints help and exits if --help was given; exits with an error on unknown options.
    void finish();

private:
    const std::string* find(const std::string& key, const std::string& def_repr,
                            const std::string& help);

    std::string prog_;
    std::string description_;
    std::map<std::string, std::string> values_;
    std::map<std::string, bool> declared_;
    std::vector<std::string> help_lines_;
    bool want_help_ = false;
};

std::vector<std::string> split(const std::string& s, char sep);

struct Stats {
    double mean = 0, std = 0, min = 0, max = 0;
    size_t n = 0;
};
// Mean and sample standard deviation.
Stats compute_stats(const std::vector<double>& xs);

class HostTimer {
public:
    HostTimer() : t0_(std::chrono::steady_clock::now()) {}
    void reset() { t0_ = std::chrono::steady_clock::now(); }
    double ms() const {
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0_)
            .count();
    }

private:
    std::chrono::steady_clock::time_point t0_;
};

// Appends rows to a CSV file, writing the header only when the file is new or empty.
class CsvWriter {
public:
    CsvWriter(const std::string& path, const std::vector<std::string>& header);
    void row(const std::vector<std::string>& values);
    const std::string& path() const { return path_; }

private:
    std::string path_;
    size_t ncols_;
    std::ofstream out_;
};

std::string fmt(double v, int precision = 6);
std::vector<std::string> concat(std::vector<std::string> a, const std::vector<std::string>& b);

}  // namespace vs
