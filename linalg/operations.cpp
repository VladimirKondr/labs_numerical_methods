#ifndef OPERATIONS_CPP
#define OPERATIONS_CPP

#include "matrix.cpp"
#include "vector.cpp"
#include <stdexcept>
#include <cstdint>

/**
 * \brief Computes the dot product of two vectors.
 * \param a First vector.
 * \param b Second vector.
 * \return The dot product (scalar of the same type).
 */
template <typename T>
T operator*(const vector<T>& a, const vector<T>& b) {
    if (a.size() != b.size()) {
        throw std::runtime_error("Dot product takes two same-size vectors");
    }
    T init{};
    for (uint64_t i = 0; i < a.size(); ++i) {
        init += a[i] * b[i];
    }
    return init;
}

/**
 * \brief Computes the cross product of two vectors.
 * \param a First vector.
 * \param b Second vector.
 * \return The cross product (vector of the same type).
 */
template <typename T>
vector<T> operator^(const vector<T>& a, const vector<T>& b) {
    if (a.size() != b.size() || a.size() != 3) {
        throw std::invalid_argument("Cross product takes two 3D vectors");
    }
    const T i = a[1] * b[2] - a[2] * b[1];
    const T j = a[2] * b[0] - a[0] * b[2];
    const T k = a[0] * b[1] - a[1] * b[0];
    return vector<T>{i, j, k};
}

/**
 * \brief Computes product of matrix and vector.
 * \param a Matrix.
 * \param b Vector.
 * \return The product (vector).
 */
template <typename T>
vector<T> operator*(const matrix<T>& a, const vector<T>& b) {
    if (a.size2() != b.size()) {
        throw std::invalid_argument("For matrix*vector they must have corresponding sizes");
    }

    vector<T> res(a.size1());
    for (uint64_t i = 0; i < a.size1(); ++i) {
        T temp{};
        for (uint64_t j = 0; j < a.size2(); ++j) {
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
    const uint64_t a_rows = a.size1();
    const uint64_t a_cols = a.size2();
    const uint64_t b_rows = b.size1();
    const uint64_t b_cols = b.size2();

    if (a_cols != b_rows) {
        throw std::invalid_argument("For matrix*matrix the must have corresponding sizes.");
    }

    matrix<T> res;
    res.resize(a_rows);
    for (uint64_t i = 0; i < a_rows; ++i) {
        res[i].resize(b_cols, T{});
    }

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