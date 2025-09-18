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
                if (std::abs(a(k, i)) > std::abs(a(max_row, i))) {
                    max_row = k;
                }
            }

            if (max_row != i) {
                for (uint64_t k = i; k < n; ++k) {
                    std::swap(a(i, k), a(max_row, k));
                }
                std::swap(b[i], b[max_row]);
            }
        }

        // If pivot is close to zero, matrix is singular
        if (std::abs(a(i, i)) < 1e-12) {
            throw std::runtime_error("Matrix is singular or nearly singular.");
        }

        for (uint64_t k = i + 1; k < n; ++k) {
            T factor = a(k, i) / a(i, i);
            for (uint64_t j = i; j < n; ++j) {
                a(k, j) -= factor * a(i, j);
            }
            b[k] -= factor * b[i];
        }
    }

    // Backward substitution
    vector<T> x(n);
    for (int i = n - 1; i >= 0; --i) {
        T sum = 0;
        for (uint64_t j = i + 1; j < n; ++j) {
            sum += a(i, j) * x[j];
        }
        x[i] = (b[i] - sum) / a(i, i);
    }

    return x;
}

/**
 * @brief Performs in-place LDLT decomposition of a symmetric matrix.
 * The lower triangular matrix L (without the unit diagonal) is stored in the lower part of a,
 * and the diagonal matrix D is stored on the main diagonal of a.
 * @tparam T The data type of the matrix elements.
 * @param a The symmetric matrix to decompose.
 * https://math.stackexchange.com/questions/2512046/give-an-algorithm-to-compute-the-ldlt-decomposition
 */
template <typename T>
void ldlt_decomposition(matrix<T>& a) {
    const uint64_t n = a.rows();
    if (n != a.cols()) {
        throw std::invalid_argument("Matrix must be square for LDLT decomposition.");
    }                   

    std::vector<T> v(n);
    for (uint64_t j = 0; j < n; ++j) {
        T temp_sum = 0.0;
        for (uint64_t k = 0; k < j; ++k) {
            v[k] = a(j, k) * a(k, k);
            temp_sum += a(j, k) * v[k];
        }
        a(j, j) -= temp_sum;
        if (std::abs(a(j, j)) < 1e-12) {
            throw std::runtime_error("LDLT algorithm has encountered zero-element of D, meaning non-positive definition");
        }

        const T inv = T(1.0) / a(j, j);
        for (uint64_t i = j + 1; i < n; ++i) {
            T sum_l = 0.0;
            for (uint64_t k = 0; k < j; ++k) {
                sum_l += a(i, k) * v[k];
            }
            a(i, j) = (a(i, j) - sum_l) * inv;
        }
    }
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

    ldlt_decomposition(a);
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
            sum += a(i, j) * z[j];
        }
        z[i] = b[i] - sum;
    }

    // Diagonal scaling (Dy = z)
    vector<T> y(n);
    for (uint64_t i = 0; i < n; ++i) {
        y[i] = z[i] / a(i, i);
    }

    // Backward substitution (L^T x = y)
    vector<T> x(n);
    for (int i = n - 1; i >= 0; --i) {
        T sum = 0;
        for (uint64_t j = i + 1; j < n; ++j) {
            sum += a(j, i) * x[j];
        }
        x[i] = y[i] - sum;
    }

    return x;
}

/**
 * @brief Solves a system of linear equations Ax = d for a tridiagonal matrix A using the Thomas algorithm.
 * @tparam T The data type of the matrix and vector elements.
 * @param a The tridiagonal matrix A.
 * @param d The vector d (the right-hand side of the equation).
 * @return The solution vector x.
 */
template <typename T>
vector<T> tridiagonal_solve(const matrix<T>& a, const vector<T>& d) {
    const uint64_t n = a.rows();
    if (n != a.cols() || n != d.size()) {
        throw std::invalid_argument("Matrix must be square and sizes must match for tridiagonal solver.");
    }

    if (n == 0) {
        return vector<T>();
    }

    vector<T> cc(n);
    vector<T> dc(n);

    // Forward substitution
    T b0 = a(0, 0);
    if (std::abs(b0) < 1e-12) {
        throw std::runtime_error("Tridiagonal matrix is singular (first element is zero).");
    }
    cc[0] = a(0, 1) / b0;
    dc[0] = d[0] / b0;

    for (uint64_t i = 1; i < n; ++i) {
        T a_i = a(i, i - 1);
        T b_i = a(i, i);
        T denom = b_i - a_i * cc[i - 1];
        if (std::abs(denom) < 1e-12) {
            throw std::runtime_error("Tridiagonal matrix is singular (zero pivot encountered).");
        }
        
        if (i < n - 1) {
            T c_i = a(i, i + 1);
            cc[i] = c_i / denom;
        }
        
        dc[i] = (d[i] - a_i * dc[i - 1]) / denom;
    }

    // Backward substitution
    vector<T> x(n);
    x[n - 1] = dc[n - 1];
    for (int i = n - 2; i >= 0; --i) {
        x[i] = dc[i] - cc[i] * x[i + 1];
    }

    return x;
}

#endif // SOLVERS_CPP