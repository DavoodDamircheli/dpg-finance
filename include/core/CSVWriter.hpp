#pragma once
#ifndef DPG_FINANCE_CSVWRITER_HPP
#define DPG_FINANCE_CSVWRITER_HPP

#include <fstream>
#include <initializer_list>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace dpg_finance {

/**
 * @brief Lightweight CSV writer for convergence tables, solution snapshots,
 *        and Greek surfaces.
 *
 * Usage:
 *   CSVWriter w("results/convergence/v1_european_1d_primal.csv");
 *   w.header({"level", "ndof", "h", "L2_error", "EOC"});
 *   w.row({0, 65, 0.015625, 1.23e-3, 0.0});
 *   w.row({1, 129, 0.0078125, 3.1e-4, 1.99});
 */
class CSVWriter {
public:
    /**
     * @brief Open a CSV file for writing, overwriting if it exists.
     * @param path  Output file path (parent directory must exist)
     * @throws std::runtime_error if file cannot be opened
     */
    explicit CSVWriter(const std::string& path) : path_(path) {
        ofs_.open(path);
        if (!ofs_.is_open())
            throw std::runtime_error("CSVWriter: cannot open " + path);
    }

    ~CSVWriter() { if (ofs_.is_open()) ofs_.close(); }

    CSVWriter(const CSVWriter&) = delete;
    CSVWriter& operator=(const CSVWriter&) = delete;

    /**
     * @brief Write header row.
     */
    void header(std::initializer_list<std::string> cols) {
        write_row(cols);
    }

    /**
     * @brief Write a data row from doubles and ints (via implicit conversion).
     */
    template<typename... Args>
    void row(Args&&... args) {
        std::ostringstream oss;
        bool first = true;
        (void)std::initializer_list<int>{
            (write_val(oss, first, std::forward<Args>(args)), 0)...
        };
        ofs_ << oss.str() << "\n";
    }

    /**
     * @brief Flush the output buffer.
     */
    void flush() { ofs_.flush(); }

    const std::string& path() const { return path_; }

private:
    std::string   path_;
    std::ofstream ofs_;

    void write_row(std::initializer_list<std::string> cols) {
        bool first = true;
        for (const auto& c : cols) {
            if (!first) ofs_ << ",";
            ofs_ << c;
            first = false;
        }
        ofs_ << "\n";
    }

    template<typename T>
    void write_val(std::ostringstream& oss, bool& first, T&& val) {
        if (!first) oss << ",";
        oss << val;
        first = false;
    }
};

} // namespace dpg_finance

#endif // DPG_FINANCE_CSVWRITER_HPP
