#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include "../linalg/matrix.cpp"
#include "../linalg/vector.cpp"
#include "../linalg/operations.cpp"
#include "../linalg/solvers.cpp"

void run_experiment(int n, int m) {
    // Generate matrix A
    matrix<> A;
    A.resize(n);
    for (int i = 0; i < n; ++i) {
        A[i].resize(n);
        for (int j = 0; j < n; ++j) {
            A[i][j] = -100.0 + static_cast<double>(rand()) / (static_cast<double>(RAND_MAX / 200.0));
        }
    }

    // Generate exact solution vector x_exact
    vector<> x_exact(n);
    for (int i = 0; i < n; ++i) {
        x_exact[i] = m + i;
    }

    // Calculate vector b
    vector<> b = A * x_exact;

    // --- Solve without pivoting ---
    auto start_no_pivot = std::chrono::high_resolution_clock::now();
    vector<> x_no_pivot = gauss_solve_no_pivot(A, b);
    auto end_no_pivot = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_no_pivot = end_no_pivot - start_no_pivot;

    vector<> residual_no_pivot = (A * x_no_pivot) - b;
    double residual_norm_no_pivot = residual_no_pivot.norm();
    double residual_norm_inf_no_pivot = residual_no_pivot.norm_inf();
    double relative_error_no_pivot = (x_no_pivot - x_exact).norm() / x_exact.norm();
    double relative_error_inf_no_pivot = (x_no_pivot - x_exact).norm_inf() / x_exact.norm_inf();

    std::cout << "--- Without Pivoting ---" << std::endl;
    std::cout << "First 5 coordinates of x*: ";
    for (int i = 0; i < 5; ++i) {
        std::cout << x_no_pivot[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "L2 Norm of residual vector: " << residual_norm_no_pivot << std::endl;
    std::cout << "Inf Norm of residual vector: " << residual_norm_inf_no_pivot << std::endl;
    std::cout << "L2 Relative error: " << relative_error_no_pivot << std::endl;
    std::cout << "Inf Relative error: " << relative_error_inf_no_pivot << std::endl;
    std::cout << "Execution time: " << duration_no_pivot.count() << "s" << std::endl;
    std::cout << std::endl;

    // --- Solve with pivoting ---
    auto start_pivot = std::chrono::high_resolution_clock::now();
    vector<> x_pivot = gauss_solve_pivot(A, b);
    auto end_pivot = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_pivot = end_pivot - start_pivot;

    vector<> residual_pivot = (A * x_pivot) - b;
    double residual_norm_pivot = residual_pivot.norm();
    double residual_norm_inf_pivot = residual_pivot.norm_inf();
    double relative_error_pivot = (x_pivot - x_exact).norm() / x_exact.norm();
    double relative_error_inf_pivot = (x_pivot - x_exact).norm_inf() / x_exact.norm_inf();

    std::cout << "--- With Pivoting ---" << std::endl;
    std::cout << "First 5 coordinates of x*: ";
    for (int i = 0; i < 5; ++i) {
        std::cout << x_pivot[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "L2 Norm of residual vector: " << residual_norm_pivot << std::endl;
    std::cout << "Inf Norm of residual vector: " << residual_norm_inf_pivot << std::endl;
    std::cout << "L2 Relative error: " << relative_error_pivot << std::endl;
    std::cout << "Inf Relative error: " << relative_error_inf_pivot << std::endl;
    std::cout << "Execution time: " << duration_pivot.count() << "s" << std::endl;
}

int main() {
    srand(time(0));

    const int n = 2000;
    const int m = 4;

    std::cout << "Running experiment for n=" << n << ", m=" << m << std::endl;
    run_experiment(n, m);

    return 0;
}
