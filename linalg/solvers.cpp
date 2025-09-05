#ifndef SOLVERS_CPP
#define SOLVERS_CPP

#include "matrix.cpp"
#include "vector.cpp"
#include <stdexcept>
#include <algorithm>
#include <iostream>

/**
 * @brief Solves a system of linear equations Ax = b using Gaussian elimination without pivoting.
 * @param a The matrix A.
 * @param b The vector b.
 * @return The solution vector x.
 */
template <typename T>
vector<T> gauss_solve_no_pivot(matrix<T> a, vector<T> b) {
    const uint64_t n = a.size1();
    if (n != a.size2() || n != b.size()) {
        throw std::invalid_argument("Matrix must be square and sizes must match.");
    }

    // Forward elimination
    for (uint64_t i = 0; i < n; ++i) {
        if (std::abs(a[i][i]) < 1e-12) {
            // This would be handled by pivoting. Without it, we might fail.
            // For this assignment, we proceed, but in general, this is an issue.
        }
        for (uint64_t k = i + 1; k < n; ++k) {
            T factor = a[k][i] / a[i][i];
            for (uint64_t j = i; j < n; ++j) {
                a[k][j] -= factor * a[i][j];
            }
            b[k] -= factor * b[i];
        }
    }

    // Backward substitution
    vector<T> x(n);
    for (int i = n - 1; i >= 0; --i) {
        T sum = 0;
        for (uint64_t j = i + 1; j < n; ++j) {
            sum += a[i][j] * x[j];
        }
        x[i] = (b[i] - sum) / a[i][i];
    }

    return x;
}

/**
 * @brief Solves a system of linear equations Ax = b using Gaussian elimination with partial pivoting.
 * @param a The matrix A.
 * @param b The vector b.
 * @return The solution vector x.
 */
template <typename T>
vector<T> gauss_solve_pivot(matrix<T> a, vector<T> b) {
    const uint64_t n = a.size1();
    if (n != a.size2() || n != b.size()) {
        throw std::invalid_argument("Matrix must be square and sizes must match.");
    }

    // Forward elimination with partial pivoting
    for (uint64_t i = 0; i < n; ++i) {
        // Find pivot
        uint64_t max_row = i;
        for (uint64_t k = i + 1; k < n; ++k) {
            if (std::abs(a[k][i]) > std::abs(a[max_row][i])) {
                max_row = k;
            }
        }

        // Swap rows
        if (max_row != i) {
            std::swap(a[i], a[max_row]);
            std::swap(b[i], b[max_row]);
        }

        // If pivot is close to zero, matrix is singular
        if (std::abs(a[i][i]) < 1e-12) {
            throw std::runtime_error("Matrix is singular or nearly singular.");
        }

        // Elimination
        for (uint64_t k = i + 1; k < n; ++k) {
            T factor = a[k][i] / a[i][i];
            for (uint64_t j = i; j < n; ++j) {
                a[k][j] -= factor * a[i][j];
            }
            b[k] -= factor * b[i];
        }
    }

    // Backward substitution
    vector<T> x(n);
    for (int i = n - 1; i >= 0; --i) {
        T sum = 0;
        for (uint64_t j = i + 1; j < n; ++j) {
            sum += a[i][j] * x[j];
        }
        x[i] = (b[i] - sum) / a[i][i];
    }

    return x;
}

#endif // SOLVERS_CPP
