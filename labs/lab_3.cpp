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
    const uint64_t n = 2000 - 1;
    const uint64_t m = 4;
    const uint64_t k = 3;
    const uint64_t seed = time(0);

    std::cout << "Running sweep solve for n=" << n << ", m=" << m << ", seed=" << seed << std::endl;
    matrix<float> a(n + 1, n + 1);

    a(0, 0) = static_cast<float>(m);
    a(0, 1) = static_cast<float>(m - 1);
    a(n, n - 1) = -static_cast<float>(k);
    a(n, n) = -static_cast<float>(m + k + n - 1);
    
    for (uint64_t i = 1; i < n; ++i) {
        a(i, i - 1) = -static_cast<float>(k);
        a(i, i) = static_cast<float>(m + k + i - 1);
        a(i, i + 1) = static_cast<float>(m + i - 1);
    }
    vector<float> x_exact(n + 1);
    for (uint64_t i = 0; i < n + 1; ++i) {
        x_exact[i] = i + 1;
    }
    vector<float> b = a * x_exact;
    auto a_copy = a;

    // Solve using Thomas algorithm
    auto start_thomas = std::chrono::high_resolution_clock::now();
    vector<float> x_star = tridiagonal_solve(a, b);
    a = std::move(a_copy);
    auto end_thomas = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> duration_ldlt = end_thomas - start_thomas;

    std::cout << "First 5 coordinates of x*: ";
    for (uint64_t i = 0; i < 5; ++i) {
        std::cout << x_star[i] << " ";
    }
    std::cout << std::endl;

    float relative_error = (x_exact - x_star).norm2() / x_exact.norm2();
    std::cout << std::endl;
    std::cout << "L2 Relative error: " << relative_error << std::endl;
    std::cout << "Execution time: " << duration_ldlt.count() << "s" << std::endl;
    std::cout << std::endl;

    
    std::cout << "\n--- For comparison: Gauss Method ---" << std::endl;
    auto start_gauss = std::chrono::high_resolution_clock::now();
    vector<float> x_gauss = gauss_solve(a, b);
    auto end_gauss = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> duration_gauss = end_gauss - start_gauss;
    std::cout << "Gauss execution time: " << duration_gauss.count() << "s" << std::endl;
    

    return 0;
}
