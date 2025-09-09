#ifndef SOLVERS_CPP
#define SOLVERS_CPP

#include "matrix.cpp"
#include "vector.cpp"
#include <stdexcept>
#include <algorithm>
#include <cmath>

/**
 * @brief Defines the pivoting strategy for Gaussian elimination.
 */
enum class PivotStrategy {
    NO_PIVOT,
    PARTIAL_PIVOT
};

/**
 * @brief Solves a system of linear equations Ax = b using Gaussian elimination.
 * @tparam T The data type of the matrix and vector elements.
 * @param a The matrix A.
 * @param b The vector b.
 * @param strategy The pivoting strategy to use (NO_PIVOT or PARTIAL_PIVOT).
 * @return The solution vector x.
 */
template <typename T>
vector<T> gauss_solve(const matrix<T>& a_in, const vector<T>& b_in, PivotStrategy strategy = PivotStrategy::PARTIAL_PIVOT) {
    matrix<T> a = a_in;
    vector<T> b = b_in;
    const uint64_t n = a.rows();
    if (n != a.cols() || n != b.size()) {
        throw std::invalid_argument("Matrix must be square and sizes must match.");
    }

    // Forward elimination
    for (uint64_t i = 0; i < n; ++i) {
        if (strategy == PivotStrategy::PARTIAL_PIVOT) {
            uint64_t max_row = i;
            for (uint64_t k = i + 1; k < n; ++k) {
                if (std::abs(a[k][i]) > std::abs(a[max_row][i])) {
                    max_row = k;
                }
            }

            if (max_row != i) {
                std::swap(a[i], a[max_row]);
                std::swap(b[i], b[max_row]);
            }
        }

        // If pivot is close to zero, matrix is singular
        if (std::abs(a[i][i]) < 1e-12) {
            throw std::runtime_error("Matrix is singular or nearly singular.");
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
 * @brief Performs LDLT decomposition of a symmetric matrix.
 * The lower triangular matrix L (without the unit diagonal) is stored in the lower part of a,
 * and the diagonal matrix D is stored on the main diagonal of a.
 * @tparam T The data type of the matrix elements.
 * @param a The symmetric matrix to decompose.
 * https://math.stackexchange.com/questions/2512046/give-an-algorithm-to-compute-the-ldlt-decomposition
 */
template <typename T>
std::pair<matrix<T>, matrix<T>> ldlt_decomposition(const matrix<T>& a) {
    const uint64_t n = a.rows();
    if (n != a.cols()) {
        throw std::invalid_argument("Matrix must be square for LDLT decomposition.");
    }                   

    matrix<T> l(n, n);
    matrix<T> d(n, n);

    for (uint64_t j = 0; j < n; ++j) {
        l[j][j] = 1;
        T temp_sum = 0.0;
        for (uint64_t k = 0; k < j; ++k) {
            temp_sum += l[j][k] * l[j][k] * d[k][k];
        }
        d[j][j] = a[j][j] - temp_sum;

        for (uint64_t i = j + 1; i < n; ++i) {
            T temp_sum = 0.0;
            for (uint64_t k = 0; k < j; ++k) {
                temp_sum += l[i][k] * l[j][k] * d[k][k];
            }
            if (d[j][i] < 1e-12) {
                throw std::runtime_error("LDLT algorithm has encountered zero-element of D");
            }
            l[i][j] = (a[j][i] - temp_sum) / d[j][j];
        }
    }
    return {l, d};
}

/**
 * @brief Solves a system of linear equations Ax = b using LDLT decomposition.
 * This method is suitable for symmetric positive-definite matrices.
 * @tparam T The data type of the matrix and vector elements.
 * @param a_in The symmetric matrix A.
 * @param b The vector b.
 * @return The solution vector x.
 */
template <typename T>
vector<T> ldlt_solve(const matrix<T>& a_in, const vector<T>& b) {
    matrix<T> a = a_in;
    const uint64_t n = a.rows();
    if (n != a.cols() || n != b.size()) {
        throw std::invalid_argument("Matrix must be square and sizes must match.");
    }

    auto [l, d] = ldlt_decomposition(a);
    // LDL^Tx = b
    // 1) z = DL^Tx
    // Lz = b
    // 2) y = L^Tx
    // Dy = z
    // 3) L^Tx = y

    // Forward substitution (Lz = b)
    vector<T> z(n);
    for (uint64_t i = 0; i < n; ++i) {
        T sum = 0;
        for (uint64_t j = 0; j < i; ++j) {
            sum += l[i][j] * z[j];
        }
        z[i] = b[i] - sum;
    }

    // Diagonal scaling (Dy = z)
    vector<T> y(n);
    for (uint64_t i = 0; i < n; ++i) {
        y[i] = z[i] / d[i][i];
    }

    // Backward substitution (L^T x = y)
    vector<T> x(n);
    for (int i = n - 1; i >= 0; --i) {
        T sum = 0;
        for (uint64_t j = i + 1; j < n; ++j) {
            sum += l[j][i] * x[j];
        }
        x[i] = y[i] - sum;
    }

    return x;
}

#endif // SOLVERS_CPP