#include <iostream>
#include <vector>
#include <chrono>
#include "../linalg/matrix.cpp"
#include "../linalg/vector.cpp"
#include "../linalg/operations.cpp"
#include "../linalg/solvers.cpp"

int main() {
    const uint64_t n = 2000;
    const uint64_t m = 4;
    const uint64_t seed = time(0);

    std::cout << "Running gaussian solve for n=" << n << ", m=" << m << ", seed=" << seed << std::endl;
    matrix<> a = matrix<>::random(n, n, seed);

    vector<> x_exact(n);
    for (uint64_t i = 0; i < n; ++i) {
        x_exact[i] = m + i;
    }

    vector<> b = a * x_exact;

    // --- Solve without pivoting ---
    auto start_no_pivot = std::chrono::high_resolution_clock::now();
    vector<> x_no_pivot = gauss_solve(a, b, PivotStrategy::NO_PIVOT);
    auto end_no_pivot = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_no_pivot = end_no_pivot - start_no_pivot;

    vector<> residual_no_pivot = (a * x_no_pivot) - b;
    double residual_norm_no_pivot = residual_no_pivot.norm2();
    double residual_norm_inf_no_pivot = residual_no_pivot.norm_inf();
    double relative_error_no_pivot = (x_no_pivot - x_exact).norm2() / x_exact.norm2();
    double relative_error_inf_no_pivot = (x_no_pivot - x_exact).norm_inf() / x_exact.norm_inf();

    std::cout << "--- Without Pivoting ---" << std::endl;
    std::cout << "First 5 coordinates of x*: ";
    for (uint64_t i = 0; i < 5; ++i) {
        std::cout << x_no_pivot[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "L2 norm2 of residual vector: " << residual_norm_no_pivot << std::endl;
    std::cout << "Inf norm2 of residual vector: " << residual_norm_inf_no_pivot << std::endl;
    std::cout << "L2 Relative error: " << relative_error_no_pivot << std::endl;
    std::cout << "Inf Relative error: " << relative_error_inf_no_pivot << std::endl;
    std::cout << "Execution time: " << duration_no_pivot.count() << "s" << std::endl;
    std::cout << std::endl;

    // --- Solve with pivoting ---
    auto start_pivot = std::chrono::high_resolution_clock::now();
    vector<> x_pivot = gauss_solve(a, b, PivotStrategy::PARTIAL_PIVOT);
    auto end_pivot = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_pivot = end_pivot - start_pivot;

    vector<> residual_pivot = (a * x_pivot) - b;
    double residual_norm_pivot = residual_pivot.norm2();
    double residual_norm_inf_pivot = residual_pivot.norm_inf();
    double relative_error_pivot = (x_pivot - x_exact).norm2() / x_exact.norm2();
    double relative_error_inf_pivot = (x_pivot - x_exact).norm_inf() / x_exact.norm_inf();

    std::cout << "--- With Pivoting ---" << std::endl;
    std::cout << "First 5 coordinates of x*: ";
    for (uint64_t i = 0; i < 5; ++i) {
        std::cout << x_pivot[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "L2 norm2 of residual vector: " << residual_norm_pivot << std::endl;
    std::cout << "Inf norm2 of residual vector: " << residual_norm_inf_pivot << std::endl;
    std::cout << "L2 Relative error: " << relative_error_pivot << std::endl;
    std::cout << "Inf Relative error: " << relative_error_inf_pivot << std::endl;
    std::cout << "Execution time: " << duration_pivot.count() << "s" << std::endl;
}
