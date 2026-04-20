#pragma once
#include <cmath>
#include <cstddef>
#include <iostream>
#include <iomanip>
#include <numeric>
#include <string>
#include <vector>

// Element-wise floating point comparison with absolute and relative tolerance.
// Returns true if all elements pass: |ref[i] - test[i]| <= atol + rtol * |ref[i]|
inline bool compareFloat(const float* ref, const float* test, size_t n,
                         float atol = 1e-3f, float rtol = 1e-3f) {
    for (size_t i = 0; i < n; i++) {
        float diff = std::fabs(ref[i] - test[i]);
        if (diff > atol + rtol * std::fabs(ref[i])) {
            return false;
        }
    }
    return true;
}

inline double computeStddev(const std::vector<double>& times) {
    if (times.empty()) return 0.0;
    double mean = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
    double sq_sum = 0.0;
    for (double v : times) sq_sum += (v - mean) * (v - mean);
    return std::sqrt(sq_sum / times.size());
}

// Print structured output for metal_eval.py to parse.
inline void printResult(bool correct, double medianMs, double stddevMs,
                        const std::string& error = "") {
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "CORRECT: " << (correct ? "true" : "false") << "\n";
    std::cout << "MEDIAN_MS: " << medianMs << "\n";
    std::cout << "STDDEV_MS: " << stddevMs << "\n";
    if (!error.empty()) {
        std::cout << "ERROR: " << error << "\n";
    }
}
