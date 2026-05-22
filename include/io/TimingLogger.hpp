#pragma once
#ifndef DPG_FINANCE_TIMINGLOGGER_HPP
#define DPG_FINANCE_TIMINGLOGGER_HPP

#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace dpg_finance {

/**
 * @brief Wall-clock timing logger for HPC profiling.
 *
 * Records named timers (e.g. "assembly", "solve", "error_calc") and writes
 * a CSV summary to results/logs/.
 *
 * In MPI contexts, call flush() only on rank 0 to avoid duplicate output.
 *
 * Usage:
 *   TimingLogger tl("results/logs/v1_timing.csv");
 *   tl.start("assembly");
 *   // ... assemble ...
 *   tl.stop("assembly");
 *   tl.start("solve");
 *   // ... solve ...
 *   tl.stop("solve");
 *   tl.flush();          // writes CSV
 *   tl.print_summary();  // prints to stdout
 */
class TimingLogger {
public:
    using Clock    = std::chrono::steady_clock;
    using Duration = std::chrono::duration<double>;

    /**
     * @brief Construct a logger; output will be written to path on flush().
     * @param path  CSV output path (e.g. "results/logs/v1_timing.csv")
     */
    explicit TimingLogger(const std::string& path = "") : path_(path) {}

    /**
     * @brief Start (or restart) a named timer.
     * @param name  Timer label
     */
    void start(const std::string& name) {
        starts_[name] = Clock::now();
    }

    /**
     * @brief Stop a named timer and accumulate elapsed time.
     * @param name  Must match a prior start() call
     * @throws std::runtime_error if name was not started
     */
    void stop(const std::string& name) {
        auto it = starts_.find(name);
        if (it == starts_.end())
            throw std::runtime_error("TimingLogger: timer '" + name + "' was not started");
        double elapsed = Duration(Clock::now() - it->second).count();
        accum_[name] += elapsed;
        ++counts_[name];
        starts_.erase(it);
    }

    /**
     * @brief Return accumulated time for a named timer in seconds.
     */
    double elapsed(const std::string& name) const {
        auto it = accum_.find(name);
        return (it != accum_.end()) ? it->second : 0.0;
    }

    /**
     * @brief Write CSV summary (name, total_seconds, call_count, avg_seconds).
     */
    void flush() const {
        if (path_.empty()) return;
        std::ofstream ofs(path_);
        if (!ofs.is_open())
            throw std::runtime_error("TimingLogger: cannot open " + path_);
        ofs << "timer,total_s,calls,avg_s\n";
        for (const auto& [name, total] : accum_) {
            int cnt = counts_.at(name);
            ofs << name << "," << total << "," << cnt
                << "," << total / cnt << "\n";
        }
    }

    /**
     * @brief Print a formatted summary to stdout.
     */
    void print_summary() const {
        std::cout << "\n--- Timing Summary ---\n";
        std::cout << std::left << std::setw(28) << "Timer"
                  << std::right << std::setw(12) << "Total(s)"
                  << std::setw(8)  << "Calls"
                  << std::setw(12) << "Avg(s)\n";
        std::cout << std::string(60, '-') << "\n";
        for (const auto& [name, total] : accum_) {
            int cnt = counts_.at(name);
            std::cout << std::left  << std::setw(28) << name
                      << std::right << std::setw(12) << std::fixed
                      << std::setprecision(4) << total
                      << std::setw(8) << cnt
                      << std::setw(12) << total / cnt << "\n";
        }
        std::cout << std::string(60, '-') << "\n";
    }

private:
    std::string path_;
    std::unordered_map<std::string, Clock::time_point> starts_;
    std::unordered_map<std::string, double>            accum_;
    std::unordered_map<std::string, int>               counts_;
};

} // namespace dpg_finance

#endif // DPG_FINANCE_TIMINGLOGGER_HPP
