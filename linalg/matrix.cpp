#ifndef MATRIX_CPP
#define MATRIX_CPP

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iomanip>
#include <random>
#include <stdexcept>
#include "vector.cpp"

template <typename T = double>
class matrix {
   private:
    uint64_t m_rows_;
    uint64_t m_cols_;
    vector<T> m_data_;

   public:
    matrix() : m_rows_(0), m_cols_(0) {
    }

    matrix(uint64_t rows, uint64_t cols, const T& initial_value = T{})
        : m_rows_(rows), m_cols_(cols), m_data_(rows * cols, initial_value) {
    }

    matrix(const vector<T>& data, uint64_t rows, uint64_t cols) : m_rows_(rows), m_cols_(cols), m_data_(data) {
    }

    T& operator()(uint64_t r, uint64_t c) {
        assert(r < m_rows_ && c < m_cols_ && "Matrix index out of bounds");
        return m_data_[r * m_cols_ + c];
    }

    const T& operator()(uint64_t r, uint64_t c) const {
        assert(r < m_rows_ && c < m_cols_ && "Matrix index out of bounds");
        return m_data_[r * m_cols_ + c];
    }

    uint64_t rows() const {
        return m_rows_;
    }

    uint64_t cols() const {
        return m_cols_;
    }

    T* data() {
        return m_data_.data();
    }

    const T* data() const {
        return m_data_.data();
    }

    /**
     * @brief Создает матрицу со случайными значениями.
     */
    static matrix<T> random(
        uint64_t rows, uint64_t cols, uint64_t seed, T min_val = -100.0, T max_val = 100.0) {
            auto x = vector<T>::random(rows * cols, seed, min_val, max_val);
        return {x, rows, cols};
    }

    /**
     * @brief Вычисляет определитель матрицы.
     */
    T det() const {
        if (m_rows_ != m_cols_) {
            throw std::invalid_argument("Determinant can be calculated only for square matrices.");
        }

        const uint64_t n = m_rows_;
        if (n == 0) {
            return 1;
        }

        matrix<T> temp = *this;
        T determinant = 1;

        for (uint64_t i = 0; i < n; ++i) {
            uint64_t max_row = i;
            for (uint64_t k = i + 1; k < n; k++) {
                if (std::abs(temp(k, i)) > std::abs(temp(max_row, i))) {
                    max_row = k;
                }
            }

            if (max_row != i) {
                for (uint64_t k = i; k < n; ++k) {
                    std::swap(temp(i, k), temp(max_row, k));
                }
                determinant *= -1;
            }

            const T pivot = temp(i, i);

            if (std::abs(pivot) < 1e-12) {
                return 0;
            }

            for (uint64_t k = i + 1; k < n; k++) {
                const T factor = temp(k, i) / pivot;
                for (uint64_t j = i; j < n; j++) {
                    temp(k, j) -= factor * temp(i, j);
                }
            }
            determinant *= pivot;
        }

        return determinant;
    }
};

/**
 * @brief Перегрузка оператора вывода для класса matrix.
 * 
 * @tparam T Тип элементов матрицы.
 * @param os Поток вывода (например, std::cout).
 * @param mat Матрица для вывода.
 * @return std::ostream& Ссылка на поток вывода для возможности цепочного вызова.
 */
template <typename T>
std::ostream& operator<<(std::ostream& os, const matrix<T>& mat) {
    if (mat.rows() == 0 || mat.cols() == 0) {
        os << "[]\n";
        return os;
    }

    for (uint64_t i = 0; i < mat.rows(); ++i) {
        for (uint64_t j = 0; j < mat.cols(); ++j) {
            os << std::setw(8);
            os << mat(i, j);
            if (j < mat.cols() - 1) {
                os << "\t";
            }
        }
        os << "\n";
    }
    return os;
}

#endif  // MATRIX_CPP