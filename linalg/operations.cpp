#ifndef OPERATIONS_CPP
#define OPERATIONS_CPP

#include "matrix.cpp"
#include "vector.cpp"
#include <stdexcept>
#include <cstdint>

/**
 * \brief Computes product of matrix and vector.
 * \param a Matrix.
 * \param b Vector.
 * \return The product (vector).
 */
template <typename T>
vector<T> operator*(const matrix<T>& a, const vector<T>& b) {
    if (a.cols() != b.size()) {
        throw std::invalid_argument("For matrix*vector they must have corresponding sizes");
    }

    vector<T> res(a.rows());
    for (uint64_t i = 0; i < a.rows(); ++i) {
        T temp{};
        for (uint64_t j = 0; j < a.cols(); ++j) {
            temp += b[j] * a[i][j];
        }
        res[i] = temp;
    }
    return res;
}

/**
 * \brief Computes product of matrix and matrix.
 * \param a First matrix.
 * \param b Second matrix.
 * \return The product (matrix).
 */
template <typename T>
matrix<T> operator*(const matrix<T>& a, const matrix<T>& b) {
    const uint64_t a_rows = a.rows();
    const uint64_t a_cols = a.cols();
    const uint64_t b_rows = b.rows();
    const uint64_t b_cols = b.cols();

    if (a_cols != b_rows) {
        throw std::invalid_argument("For matrix*matrix the must have corresponding sizes.");
    }

    matrix<T> res(a_rows, b_cols);

    for (uint64_t i = 0; i < a_rows; ++i) {
        for (uint64_t j = 0; j < b_cols; ++j) {
            for (uint64_t k = 0; k < a_cols; ++k) {
                res[i][j] += a[i][k] * b[k][j];
            }
        }
    }

    return res;
}

#endif // OPERATIONS_CPP