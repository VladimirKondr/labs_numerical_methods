#ifndef vector_CPP
#define vector_CPP

#include <vector>
#include <cmath>
#include <iostream>
#include <stdexcept>

template <typename T = double>
class vector : public std::vector<T> {
public:
    // Default constructor
    vector() = default;

    // Constructor with size
    explicit vector(size_t size) : std::vector<T>(size) {}

    // Constructor with size and value
    vector(size_t size, T value) : std::vector<T>(size, value) {}

    // Copy constructor from std::vector
    vector(const std::vector<T>& other) : std::vector<T>(other) {}


    /**
     * @brief Calculates the Euclidean norm (L2 norm) of the vector.
     * @return The Euclidean norm of the vector.
     */
    T norm() const {
        T sum_sq = 0;
        for (const auto& val : *this) {
            sum_sq += val * val;
        }
        return std::sqrt(sum_sq);
    }

    /**
     * @brief Calculates the infinity norm (max norm) of the vector.
     * @return The infinity norm of the vector.
     */
    T norm_inf() const {
        T max_val = 0;
        if (this->empty()) {
            return 0;
        }
        for (const auto& val : *this) {
            if (std::abs(val) > max_val) {
                max_val = std::abs(val);
            }
        }
        return max_val;
    }

    /**
     * @brief Calculates the p-norm of the vector.
     * @param p The order of the norm.
     * @return The p-norm of the vector.
     */
    T norm_p(int p) const {
        if (p <= 0) {
            throw std::invalid_argument("p must be a positive integer for p-norm.");
        }
        T sum = 0;
        for (const auto& val : *this) {
            sum += std::pow(std::abs(val), p);
        }
        return std::pow(sum, 1.0 / p);
    }
};

/**
 * @brief Overloads the << operator to print the vector.
 * @param os The output stream.
 * @param v The vector to print.
 * @return The output stream.
 */
template <typename T>
std::ostream& operator<<(std::ostream& os, const vector<T>& v) {
    os << "[";
    for (size_t i = 0; i < v.size(); ++i) {
        os << v[i] << (i == v.size() - 1 ? "" : ", ");
    }
    os << "]";
    return os;
}

/**
 * @brief Overloads the - operator for vector subtraction.
 * @param a The first vector.
 * @param b The second vector.
 * @return The resulting vector from the subtraction.
 */
template <typename T>
vector<T> operator-(const vector<T>& a, const vector<T>& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Vectors must have the same size for subtraction.");
    }
    vector<T> result(a.size());
    for (size_t i = 0; i < a.size(); ++i) {
        result[i] = a[i] - b[i];
    }
    return result;
}

#endif // vector_CPP