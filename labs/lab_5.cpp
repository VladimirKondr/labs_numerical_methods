#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <iomanip>

#include "../linalg/matrix.cpp"
#include "../linalg/vector.cpp"
#include "../linalg/operations.cpp"
#include "../linalg/solvers.cpp"

int main() {
    std::cout << std::fixed << std::setprecision(6);
    
    const uint64_t n = 2000;
    const uint64_t m = 4;
    const uint64_t k = 3;
    const double epsilon = 1e-6;
    const uint64_t k_max = 2000;
    const uint64_t seed = time(0);

    std::cout << "Running Conjugate Gradient solve for n=" << n << ", m=" << m << ", k=" << k << std::endl;
    std::cout << "Epsilon=" << epsilon << ", k_max=" << k_max << ", seed=" << seed << std::endl;
    std::cout << std::endl;

    matrix<double> a = matrix<double>::random(n, n, seed, -100.0, 0);

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
            temp_sum += std::abs(a(i, j));
        }
        a(i, i) = temp_sum + 100.0; // гарантируем положительную определнность
    }

    double temp_sum = 0.0;
    for (uint64_t j = 1; j < n; ++j) {
        temp_sum += std::abs(a(0, j));
    }
    a(0, 0) = temp_sum + 100.0 + 1.0 / std::pow(10, k - 2);

    vector<double> x_exact(n);
    for (uint64_t i = 0; i < n; ++i) {
        x_exact[i] = m + i;
    }

    vector<double> b = a * x_exact;

    std::cout << "First 5 coordinates of exact solution x:" << std::endl;
    for (uint64_t i = 0; i < 5; ++i) {
        std::cout << x_exact[i] << " ";
    }
    std::cout << std::endl << std::endl;

    std::cout << "========== Метод сопряженных градиентов (CG) ==========" << std::endl;
    auto start_cg = std::chrono::high_resolution_clock::now();
    auto result_cg = cg_solve(a, b, epsilon, k_max);
    auto end_cg = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_cg = end_cg - start_cg;

    if (result_cg.converged) {
        std::cout << "Сходимость достигнута на итерации: " << result_cg.iterations << std::endl;
    } else {
        std::cout << "ВНИМАНИЕ: Превышено максимальное количество итераций (" << k_max << ")" << std::endl;
        std::cout << "Результат получен на итерации: " << result_cg.iterations << std::endl;
    }
    
    std::cout << "First 5 coordinates of x*: ";
    for (uint64_t i = 0; i < 5; ++i) {
        std::cout << result_cg.solution[i] << " ";
    }
    std::cout << std::endl;

    vector<double> residual_cg = (a * result_cg.solution) - b;
    double residual_norm_cg = residual_cg.norm2();
    double relative_error_cg = (result_cg.solution - x_exact).norm2() / x_exact.norm2();
    
    std::cout << "L2 norm of residual vector: " << residual_norm_cg << std::endl;
    std::cout << "L2 Relative error: " << relative_error_cg << std::endl;
    std::cout << "Время выполнения: " << duration_cg.count() << "s" << std::endl;
    std::cout << std::endl;

    std::cout << "========== Метод Гаусса-Зейделя ==========" << std::endl;
    auto start_gs = std::chrono::high_resolution_clock::now();
    auto result_gs = sor_solve(a, b, 1.0, epsilon, k_max);
    auto end_gs = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_gs = end_gs - start_gs;

    if (result_gs.converged) {
        std::cout << "Сходимость достигнута на итерации: " << result_gs.iterations << std::endl;
    } else {
        std::cout << "ВНИМАНИЕ: Превышено максимальное количество итераций (" << k_max << ")" << std::endl;
        std::cout << "Результат получен на итерации: " << result_gs.iterations << std::endl;
    }
    
    std::cout << "First 5 coordinates of x*: ";
    for (uint64_t i = 0; i < 5; ++i) {
        std::cout << result_gs.solution[i] << " ";
    }
    std::cout << std::endl;

    vector<double> residual_gs = (a * result_gs.solution) - b;
    double residual_norm_gs = residual_gs.norm2();
    double relative_error_gs = (result_gs.solution - x_exact).norm2() / x_exact.norm2();
    
    std::cout << "L2 norm of residual vector: " << residual_norm_gs << std::endl;
    std::cout << "L2 Relative error: " << relative_error_gs << std::endl;
    std::cout << "Время выполнения: " << duration_gs.count() << "s" << std::endl;
    std::cout << std::endl;

    std::cout << "========== Для сравнения: Метод LDLT ==========" << std::endl;
    auto start_ldlt = std::chrono::high_resolution_clock::now();
    vector<double> x_ldlt = ldlt_solve(a, b);
    auto end_ldlt = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_ldlt = end_ldlt - start_ldlt;

    std::cout << "First 5 coordinates of x*: ";
    for (uint64_t i = 0; i < 5; ++i) {
        std::cout << x_ldlt[i] << " ";
    }
    std::cout << std::endl;

    vector<double> residual_ldlt = (a * x_ldlt) - b;
    double residual_norm_ldlt = residual_ldlt.norm2();
    double relative_error_ldlt = (x_ldlt - x_exact).norm2() / x_exact.norm2();
    
    std::cout << "L2 norm of residual vector: " << residual_norm_ldlt << std::endl;
    std::cout << "L2 Relative error: " << relative_error_ldlt << std::endl;
    std::cout << "Время выполнения: " << duration_ldlt.count() << "s" << std::endl;
    std::cout << std::endl;

    std::cout << "========== Сводка результатов ==========" << std::endl;
    std::cout << std::setw(25) << "Метод" 
              << std::setw(15) << "Итераций" 
              << std::setw(20) << "Время (с)" 
              << std::setw(20) << "Погрешность" << std::endl;
    std::cout << std::string(80, '-') << std::endl;
    std::cout << std::setw(25) << "CG" 
              << std::setw(15) << result_cg.iterations 
              << std::setw(20) << duration_cg.count() 
              << std::setw(20) << relative_error_cg << std::endl;
    std::cout << std::setw(25) << "Gauss-Seidel" 
              << std::setw(15) << result_gs.iterations 
              << std::setw(20) << duration_gs.count() 
              << std::setw(20) << relative_error_gs << std::endl;
    std::cout << std::setw(25) << "LDLT (прямой)" 
              << std::setw(15) << "-" 
              << std::setw(20) << duration_ldlt.count() 
              << std::setw(20) << relative_error_ldlt << std::endl;

    return 0;
}
