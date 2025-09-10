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
    const uint64_t n = 2000 - 1;
    const uint64_t m = 4;
    const uint64_t k = 3;
    const uint64_t seed = time(0);

    std::cout << "Running sweep solve for n=" << n << ", m=" << m << ", seed=" << seed << std::endl;
    matrix<> a(n + 1, n + 1);
    
    for (uint64_t i = 0; i < n + 1; ++i) {
        for (uint64_t j = std::max(static_cast<uint64_t>(0), i - 1); j < std::min(n + 1, i + 2); ++j) {

        }
    }
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
