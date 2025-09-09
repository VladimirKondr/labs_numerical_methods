#ifndef MATRIX_CPP
#define MATRIX_CPP

#include "vector.cpp"
#include <vector>

template <typename T = double>
class matrix : public std::vector<vector<T>> {
public:
    matrix(int rows, int cols) : std::vector<vector<T>>(rows, vector<T>(cols)) {}

    /**
     * @brief Creates a matrix with random values.
     * @param rows Number of rows.
     * @param cols Number of columns.
     * @param min_val Minimum value for random numbers.
     * @param max_val Maximum value for random numbers.
     * @param seed Seed for the random number generator. If 0, a random seed is used.
     * @return A new matrix with random values.
     */
    static matrix<T> random(int rows, int cols, uint64_t seed, T min_val = -100.0, T max_val = 100.0) {
        matrix<T> result(rows, cols);
        for (int i = 0; i < rows; ++i) {
            result[i] = vector<T>::random(cols, seed + i, min_val, max_val);
        }
        return result;
    }

    /**
     * @brief Calculates the determinant of a submatrix using the Gaussian method.
     * @param i1 The starting index of the submatrix row.
     * @param j1 The starting index of the submatrix column.
     * @param i2 End row index (not including) of the submatrix.
     * @param j2 End column index (not including) of the submatrix.
     * @return The determinant of the specified submatrix.
     */
    T det(uint64_t i1 = 0, uint64_t j1 = 0, uint64_t i2 = -1, uint64_t j2 = -1) const {
        if (i2 == static_cast<uint64_t>(-1)) {
            i2 = this->size();
        }
        if (j2 == static_cast<uint64_t>(-1)) {
            j2 = this->empty() ? 0 : (*this)[0].size();
        }

        const uint64_t height = i2 - i1;
        const uint64_t width = j2 - j1;

        if (height != width) {
            throw std::invalid_argument("Determinant can be calculated only for square matrices.");
        }
        
        const uint64_t n = height;

        if (n == 0) {
            return 1;
        }

        matrix<T> temp;
        temp.resize(n);
        for (uint64_t i = 0; i < n; ++i) {
            temp[i].resize(n);
            for (uint64_t j = 0; j < n; ++j) {
                temp[i][j] = (*this)[i1 + i][j1 + j];
            }
        }

        T determinant = 1;

        for (uint64_t i = 0; i < n; ++i) {
            // Partial pivoting: looking for a row with maximal element
            uint64_t max_row = i;
            for (uint64_t k = i + 1; k < n; k++) {
                if (std::abs(temp[k][i]) > std::abs(temp[max_row][i])) {
                    max_row = k;
                }
            }

            if (max_row != i) {
                std::swap(temp[i], temp[max_row]);
                determinant *= -1;
            }

            const T pivot = temp[i][i];
            
            // If pivot is close to zero, matrix is singular
            if (std::abs(pivot) < 1e-12) {
                return 0; 
            }

            for (uint64_t k = i + 1; k < n; k++) {
                const T factor = temp[k][i] / pivot;
                for (uint64_t j = i; j < n; j++) {
                    temp[k][j] -= factor * temp[i][j];
                }
            }
        }

        for(uint64_t i = 0; i < n; ++i) {
            determinant *= temp[i][i];
        }

        return determinant;
    }

    /**
    * @brief Returns the number of rows and columns in the matrix.
     */
    uint64_t rows() const {
        return this->size();
    }

    /**
    * @brief Returns the number of columns in the matrix.
     */
    uint64_t cols() const {
        return this->empty() ? 0 : (*this)[0].size();
    }


};

#endif // MATRIX_CPP