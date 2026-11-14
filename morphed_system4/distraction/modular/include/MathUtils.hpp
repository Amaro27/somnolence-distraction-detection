#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <numeric>
#include <vector>

namespace MathUtils {

constexpr size_t MAX_MEDIAN_STACK_CAPACITY = 32;

/**
 * @brief Computes the median of an iterable container (std::deque, std::vector, std::array)
 * without heap allocation for window sizes up to MAX_MEDIAN_STACK_CAPACITY (32).
 */
template <typename Container>
auto compute_median(const Container& values) -> typename Container::value_type {
    using T = typename Container::value_type;
    if (values.empty()) return T{0};
    const size_t count = values.size();
    if (count == 1) return *values.begin();

    const size_t mid = count / 2;

    if (count <= MAX_MEDIAN_STACK_CAPACITY) {
        std::array<T, MAX_MEDIAN_STACK_CAPACITY> stack_buf{};
        std::copy(values.begin(), values.end(), stack_buf.begin());
        std::nth_element(stack_buf.begin(), stack_buf.begin() + mid, stack_buf.begin() + count);
        if (count % 2 != 0) {
            return stack_buf[mid];
        }
        auto max_lower = *std::max_element(stack_buf.begin(), stack_buf.begin() + mid);
        return (max_lower + stack_buf[mid]) * static_cast<T>(0.5);
    }

    std::vector<T> heap_copy(values.begin(), values.end());
    std::nth_element(heap_copy.begin(), heap_copy.begin() + mid, heap_copy.end());
    if (count % 2 != 0) {
        return heap_copy[mid];
    }
    auto max_lower = *std::max_element(heap_copy.begin(), heap_copy.begin() + mid);
    return (max_lower + heap_copy[mid]) * static_cast<T>(0.5);
}

/**
 * @brief Computes arithmetic mean across an arbitrary iterable container.
 */
template <typename Container>
double compute_mean(const Container& c) {
    if (c.empty()) return 0.0;
    double sum = std::accumulate(c.begin(), c.end(), 0.0);
    return sum / static_cast<double>(c.size());
}

} // namespace MathUtils
