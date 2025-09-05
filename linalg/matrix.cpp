#ifndef MATRIX_CPP
#define MATRIX_CPP

#include <vector>
#include <cmath> 
#include <stdexcept>

template <typename T = double>
class matrix : public std::vector<std::vector<T>> {
public:
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

    uint64_t size1() const {
        return this->size();
    }

    uint64_t size2() const {
        return this->empty() ? 0 : (*this)[0].size();
    }


};

#endif // MATRIX_CPP