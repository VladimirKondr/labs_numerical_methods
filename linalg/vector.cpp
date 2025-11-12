#ifndef VECTOR_CPP
#define VECTOR_CPP

#include <vector>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <random>

template<typename T = double>
class vector : public std::vector<T> {
public:
    using std::vector<T>::vector;

    /**
     * @brief Creates a vector with random values.
     * @param size The size of the vector.
     * @param seed Seed for the random number generator.
     * @return A new vector with random values.
     */
    static vector<T> random(uint64_t size, uint64_t seed, T min_val = -100.0, T max_val = 100.0) {
        if (min_val > max_val) {
            throw std::invalid_argument("Min bnorder of random must be less of equal to the max border");
        }
        vector<T> result(size);
        std::mt19937 gen(seed);
        std::uniform_real_distribution<double> dis(min_val, max_val);
        for (uint64_t i = 0; i < size; ++i) {
            result[i] = dis(gen);
        }
        return result;
    }

    /**
     * @brief Calculates the Euclidean norm2 (L2 norm2) of the vector.
     * @return The Euclidean norm2 of the vector.
     */
    T norm2() const {
        T sum_sq = 0;
        for (const auto& val : *this) {
            sum_sq += val * val;
        }
        return std::sqrt(sum_sq);
    }

    /**
     * @brief Calculates the infinity norm2 (max norm2) of the vector.
     * @return The infinity norm2 of the vector.
     */
    T norm_inf() const {
        if (this->empty()) {
            return 0.0;
        }
        T max_val = 0.0;
        for (const auto& val : *this) {
            max_val = std::max(max_val, std::abs(val));
        }
        return max_val;
    }

    /**
     * @brief Calculates the p-norm2 of the vector.
     * @param p The order of the norm2.
     * @return The p-norm2 of the vector.
     */
    T p_norm(double p) const {
        if (this->empty()) {
            return 0.0;
        }
        T sum = 0.0;
        for (const auto& val : *this) {
            sum += std::pow(std::abs(val), p);
        }
        return std::pow(sum, 1.0 / p);
    }

    template<typename U>
    explicit operator vector<U>() const {
        vector<U> result(this->size());
        for (size_t i = 0; i < this->size(); ++i) {
            result[i] = static_cast<U>((*this)[i]);
        }
        return result;
    }
};

/**
 * \brief Adds two vectors element-wise.
 * \param a First vector.
 * \param b Second vector.
 * \return The resulting vector after addition.
 */
template<typename T>
vector<T> operator+(const vector<T>& a, const vector<T>& b) {
    vector<T> result(a.size());
    for (size_t i = 0; i < a.size(); ++i) {
        result[i] = a[i] + b[i];
    }
    return result;
}

/**
 * \brief Subtracts two vectors element-wise.
 * \param a First vector.
 * \param b Second vector.
 * \return The resulting vector after subtraction.
 */
template<typename T>
vector<T> operator-(const vector<T>& a, const vector<T>& b) {
    vector<T> result(a.size());
    for (size_t i = 0; i < a.size(); ++i) {
        result[i] = a[i] - b[i];
    }
    return result;
}

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
 * \brief Outputs the vector to the given output stream.
 * \param os The output stream.
 * \param v The vector to output.
 * \return The output stream.
 */
template<typename T>
std::ostream& operator<<(std::ostream& os, const vector<T>& v) {
    for (const auto& elem : v) {
        os << elem << " ";
    }
    return os;
}

#endif // VECTOR_CPP