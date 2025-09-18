#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <random>
#include <numeric>

#include "../linalg/matrix.cpp"
#include "../linalg/vector.cpp"
#include "../linalg/operations.cpp"
#include "../linalg/solvers.cpp"

int main() {
    std::cout << std::fixed << std::setprecision(15);
    const uint64_t n = 2000;
    const uint64_t m = 4;
    const uint64_t k = 3;
    const uint64_t seed = time(0);

    std::cout << "Running ldlt solve for n=" << n << ", m=" << m << ", seed=" << seed << std::endl;
    matrix<> a = matrix<>::random(n, n, seed, -100.0, 0);

    for (uint64_t i = 0; i < n; ++i) {
        for (uint64_t j = i + 1; j < n; ++j) {
            a(i, j) = a(j, i);
        }
    }
    for (uint64_t i = 1; i < n; ++i) {
        double temp_sum = 0.0;
        for (uint64_t j = 0; j < n; ++j) {
            if (j == i) {
                continue;
            }
            temp_sum += a(i, j);
        }
        a(i, i) = -temp_sum;
    }
    double temp_sum = 0.0;
    for (uint64_t j = 1; j < n; ++j) {
        temp_sum += a(0, j);
    }
    a(0, 0) = -temp_sum + 1 / std::pow(10, k - 2);

    vector<> x_exact(n);
    for (uint64_t i = 0; i < n; ++i) {
        x_exact[i] = m + i;
    }
    vector<> b = a * x_exact;
    auto a_copy = a;

    // Solve using LDLT
    auto start_ldlt = std::chrono::high_resolution_clock::now();
    vector<> x_star = ldlt_solve(a, b);
    a = std::move(a_copy);
    auto end_ldlt = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_ldlt = end_ldlt - start_ldlt;

    std::cout << "First 5 coordinates of x*: ";
    for (uint64_t i = 0; i < 5; ++i) {
        std::cout << x_star[i] << " ";
    }
    std::cout << std::endl;

    vector<> residual = (a * x_star) - b;
    double residual_norm = residual.norm2();
    double relative_error = (x_exact - x_star).norm2() / x_exact.norm2();
    std::cout << std::endl;
    std::cout << "L2 norm2 of residual vector: " << residual_norm << std::endl;
    std::cout << "L2 Relative error: " << relative_error << std::endl;
    std::cout << "Execution time: " << duration_ldlt.count() << "s" << std::endl;
    std::cout << std::endl;

    
    std::cout << "\n--- For comparison: Gauss Method ---" << std::endl;
    auto start_gauss = std::chrono::high_resolution_clock::now();
    vector<double> x_gauss = gauss_solve(a, b);
    auto end_gauss = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_gauss = end_gauss - start_gauss;
    std::cout << "Gauss execution time: " << duration_gauss.count() << "s" << std::endl;
    

    return 0;
}
