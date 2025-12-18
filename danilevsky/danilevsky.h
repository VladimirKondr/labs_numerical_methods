#pragma once

#include "../linalg/matrix.cpp"
#include "../linalg/operations.cpp"

#include <cmath>
#include <stdexcept>
#include <vector>

template <typename T = double>
struct DanilevskyResult {
    matrix<T> frobenius_matrix;
    std::vector<T> characteristic_poly_coeffs;
    std::vector<matrix<T>> transform_matrices;
};

template <typename T>
DanilevskyResult<T> perform_danilevsky(const matrix<T>& a_in) {
    const uint64_t n = a_in.rows();

    if (n != a_in.cols()) {
        throw std::invalid_argument("Matrix must be square");
    }

    matrix<T> a = a_in;

    std::vector<matrix<T>> m_matrices;
    m_matrices.clear();

    for (uint64_t k = n; k >= 2; --k) {
        const uint64_t row_idx = k - 1;   
        const uint64_t pivot_idx = k - 2; 
        const T pivot = a(row_idx, pivot_idx);

        if (std::abs(pivot) < 1e-8) {
            throw std::runtime_error(
                "Pivot element is too small (near zero). Cannot continue with regular case.");
        }

        matrix<T> m(n, n, T{0});
        for (uint64_t i = 0; i < n; ++i) {
            m(i, i) = T{1};
        }

        for (uint64_t j = 0; j < n; ++j) {
            if (j == pivot_idx) {
                m(pivot_idx, j) = T{1} / pivot;
            } else {
                m(pivot_idx, j) = -a(row_idx, j) / pivot;
            }
        }

        matrix<T> m_inv = m.inverse();
        a = m_inv * a * m;
        m_matrices.push_back(m);
    }

    std::vector<T> coeffs;
    for (uint64_t j = 0; j < n; ++j) {
        coeffs.push_back(-a(0, j)); 
    }

    DanilevskyResult<T> result;
    result.frobenius_matrix = a;
    result.characteristic_poly_coeffs = coeffs;
    result.transform_matrices = m_matrices;

    return result;
}
