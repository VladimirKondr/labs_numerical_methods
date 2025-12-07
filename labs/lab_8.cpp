#include "../linalg/matrix.cpp"
#include "../linalg/operations.cpp"
#include "../linalg/vector.cpp"

#include <cmath>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <vector>

enum class NormType { MAX_NORM, EUCLIDEAN_NORM };

template <typename T>
struct PowerMethodResult {
    T eigenvalue;
    vector<T> eigenvector;
    vector<T> residual;
    T residual_norm_inf;
    T residual_norm_2;
    uint64_t iterations;
};

template <typename T>
PowerMethodResult<T> power_method(
    const matrix<T>& a, uint64_t max_iterations, const vector<T>& u0,
    NormType norm_type = NormType::EUCLIDEAN_NORM) {
    const uint64_t n = a.rows();

    if (n != a.cols()) {
        throw std::invalid_argument("Matrix must be square");
    }

    if (u0.size() != n) {
        throw std::invalid_argument("Initial vector size must match matrix size");
    }

    vector<T> u = u0;

    if (norm_type == NormType::MAX_NORM) {
        T norm_inf = u.norm_inf();
        for (uint64_t i = 0; i < n; ++i) {
            u[i] /= norm_inf;
        }
    } else {
        T norm_2 = u.norm2();
        for (uint64_t i = 0; i < n; ++i) {
            u[i] /= norm_2;
        }
    }

    vector<T> v(n);
    T lambda = T{0};

    for (uint64_t k = 0; k < max_iterations; ++k) {
        v = a * u;

        if (norm_type == NormType::MAX_NORM) {
            uint64_t max_idx = 0;
            T max_val = std::abs(v[0]);
            for (uint64_t i = 1; i < n; ++i) {
                if (std::abs(v[i]) > max_val) {
                    max_val = std::abs(v[i]);
                    max_idx = i;
                }
            }

            T sign_u = (u[max_idx] >= 0) ? T{1} : T{-1};
            lambda = v[max_idx] * sign_u;

            T norm_inf = v.norm_inf();
            for (uint64_t i = 0; i < n; ++i) {
                u[i] = v[i] / norm_inf;
            }
        } else {
            lambda = v * u;

            T norm_2 = v.norm2();
            for (uint64_t i = 0; i < n; ++i) {
                u[i] = v[i] / norm_2;
            }
        }
    }

    v = a * u;

    if (norm_type == NormType::MAX_NORM) {
        uint64_t max_idx = 0;
        T max_val = std::abs(v[0]);
        for (uint64_t i = 1; i < n; ++i) {
            if (std::abs(v[i]) > max_val) {
                max_val = std::abs(v[i]);
                max_idx = i;
            }
        }
        T sign_u = (u[max_idx] >= 0) ? T{1} : T{-1};
        lambda = v[max_idx] * sign_u;
    } else {
        lambda = v * u;
    }

    vector<T> residual(n);
    for (uint64_t i = 0; i < n; ++i) {
        residual[i] = v[i] - lambda * u[i];
    }

    PowerMethodResult<T> result;
    result.eigenvalue = lambda;
    result.eigenvector = u;
    result.residual = residual;
    result.residual_norm_inf = residual.norm_inf();
    result.residual_norm_2 = residual.norm2();
    result.iterations = max_iterations;

    return result;
}

matrix<double> create_symmetric_matrix(
    uint64_t n, uint64_t m [[maybe_unused]], uint64_t k, uint64_t seed) {
    matrix<double> a = matrix<double>::random(n, n, seed, -100.0, 0.0);

    for (uint64_t i = 0; i < n; ++i) {
        for (uint64_t j = i + 1; j < n; ++j) {
            a(i, j) = a(j, i);
        }
    }

    // Заполняем диагональ для строк 1..n-1
    for (uint64_t i = 1; i < n; ++i) {
        double temp_sum = 0.0;
        for (uint64_t j = 0; j < n; ++j) {
            if (j != i) {
                temp_sum += a(i, j);
            }
        }
        a(i, i) = -temp_sum;
    }

    double temp_sum = 0.0;
    for (uint64_t j = 1; j < n; ++j) {
        temp_sum += a(0, j);
    }
    a(0, 0) = -temp_sum + 1.0 / std::pow(10, k - 2);

    a = static_cast<matrix<double>>(static_cast<matrix<int64_t>>(a));

    return a;
}

int main() {
    std::cout << std::fixed << std::setprecision(15);

    const uint64_t n = 10000;
    const uint64_t m = 4;
    const uint64_t k_param = 3;
    const uint64_t seed = static_cast<uint64_t>(std::time(nullptr));

    matrix<double> a = create_symmetric_matrix(n, m, k_param, seed);

    vector<double> u0(n);
    for (uint64_t i = 0; i < n; ++i) {
        u0[i] = 1.0;
    }

    std::cout << "МАКСИМУМ НОРМА" << std::endl;

    std::cout << ">>> При k = 999 итераций:" << std::endl;
    auto result_max_999 = power_method(a, 999, u0, NormType::MAX_NORM);
    std::cout << "lambda1 = " << result_max_999.eigenvalue << std::endl;
    std::cout << "Первые 5 координат собственного вектора u^k:" << std::endl;
    std::cout << "  ";
    for (uint64_t i = 0; i < 5; ++i) {
        std::cout << result_max_999.eigenvector[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "INF RESIDUAL NORM " << result_max_999.residual_norm_inf << std::endl;
    std::cout << "EUC RESIDUAL NORM " << result_max_999.residual_norm_2 << std::endl;
    std::cout << std::endl;

    // k = 1000
    std::cout << ">>> При k = 1000 итераций:" << std::endl;
    auto result_max_1000 = power_method(a, 1000, u0, NormType::MAX_NORM);
    std::cout << "lambda1 = " << result_max_1000.eigenvalue << std::endl;
    std::cout << "Первые 5 координат собственного вектора u^k:" << std::endl;
    std::cout << "  ";
    for (uint64_t i = 0; i < 5; ++i) {
        std::cout << result_max_1000.eigenvector[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "INF RESIDUAL NORM " << result_max_1000.residual_norm_inf << std::endl;
    std::cout << "EUC RESIDUAL NORM " << result_max_1000.residual_norm_2 << std::endl;
    std::cout << std::endl;

    std::cout << "ЕВКЛИДОВА НОРМА" << std::endl;

    std::cout << ">>> При k = 999 итераций:" << std::endl;
    auto result_euc_999 = power_method(a, 999, u0, NormType::EUCLIDEAN_NORM);
    std::cout << "lambda1 = " << result_euc_999.eigenvalue << std::endl;
    std::cout << "Первые 5 координат собственного вектора u^k:" << std::endl;
    std::cout << "  ";
    for (uint64_t i = 0; i < 5; ++i) {
        std::cout << result_euc_999.eigenvector[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "INF RESIDUAL NORM " << result_euc_999.residual_norm_inf << std::endl;
    std::cout << "EUC RESIDUAL NORM " << result_euc_999.residual_norm_2 << std::endl;
    std::cout << std::endl;

    std::cout << ">>> При k = 1000 итераций:" << std::endl;
    auto result_euc_1000 = power_method(a, 1000, u0, NormType::EUCLIDEAN_NORM);
    std::cout << "lambda1 = " << result_euc_1000.eigenvalue << std::endl;
    std::cout << "Первые 5 координат собственного вектора u^k:" << std::endl;
    std::cout << "  ";
    for (uint64_t i = 0; i < 5; ++i) {
        std::cout << result_euc_1000.eigenvector[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "INF RESIDUAL NORM " << result_euc_1000.residual_norm_inf << std::endl;
    std::cout << "EUC RESIDUAL NORM " << result_euc_1000.residual_norm_2 << std::endl;
    std::cout << std::endl;

    std::cout << "Метод 1: изменение lambda1 = "
              << std::abs(result_max_1000.eigenvalue - result_max_999.eigenvalue) << std::endl;
    std::cout << "Метод 2: изменение lambda1 = "
              << std::abs(result_euc_1000.eigenvalue - result_euc_999.eigenvalue) << std::endl;
    std::cout << std::endl;

    return 0;
}
