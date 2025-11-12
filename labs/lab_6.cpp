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

    std::cout << "Running Preconditioned Conjugate Gradient for n=" << n << ", m=" << m << ", k=" << k << std::endl;
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
        a(i, i) = temp_sum + 100.0;
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

    std::cout << "========== CG без предобусловливания ==========" << std::endl;
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

    std::cout << "========== PCG с предобусловливанием Якоби ==========" << std::endl;
    auto start_pcg_jacobi = std::chrono::high_resolution_clock::now();
    auto result_pcg_jacobi = pcg_solve(a, b, PreconditionerType::JACOBI, 1.0, epsilon, k_max);
    auto end_pcg_jacobi = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_pcg_jacobi = end_pcg_jacobi - start_pcg_jacobi;

    if (result_pcg_jacobi.converged) {
        std::cout << "Сходимость достигнута на итерации: " << result_pcg_jacobi.iterations << std::endl;
    } else {
        std::cout << "ВНИМАНИЕ: Превышено максимальное количество итераций (" << k_max << ")" << std::endl;
        std::cout << "Результат получен на итерации: " << result_pcg_jacobi.iterations << std::endl;
    }
    
    std::cout << "First 5 coordinates of x*: ";
    for (uint64_t i = 0; i < 5; ++i) {
        std::cout << result_pcg_jacobi.solution[i] << " ";
    }
    std::cout << std::endl;

    vector<double> residual_pcg_jacobi = (a * result_pcg_jacobi.solution) - b;
    double residual_norm_pcg_jacobi = residual_pcg_jacobi.norm2();
    double relative_error_pcg_jacobi = (result_pcg_jacobi.solution - x_exact).norm2() / x_exact.norm2();
    
    std::cout << "L2 norm of residual vector: " << residual_norm_pcg_jacobi << std::endl;
    std::cout << "L2 Relative error: " << relative_error_pcg_jacobi << std::endl;
    std::cout << "Время выполнения: " << duration_pcg_jacobi.count() << "s" << std::endl;
    std::cout << std::endl;

    std::cout << "========== PCG с предобусловливанием SSOR (ω = 1.0) ==========" << std::endl;
    auto start_pcg_ssor1 = std::chrono::high_resolution_clock::now();
    auto result_pcg_ssor1 = pcg_solve(a, b, PreconditionerType::SSOR, 1.0, epsilon, k_max);
    auto end_pcg_ssor1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_pcg_ssor1 = end_pcg_ssor1 - start_pcg_ssor1;

    if (result_pcg_ssor1.converged) {
        std::cout << "Сходимость достигнута на итерации: " << result_pcg_ssor1.iterations << std::endl;
    } else {
        std::cout << "ВНИМАНИЕ: Превышено максимальное количество итераций (" << k_max << ")" << std::endl;
        std::cout << "Результат получен на итерации: " << result_pcg_ssor1.iterations << std::endl;
    }
    
    std::cout << "First 5 coordinates of x*: ";
    for (uint64_t i = 0; i < 5; ++i) {
        std::cout << result_pcg_ssor1.solution[i] << " ";
    }
    std::cout << std::endl;

    vector<double> residual_pcg_ssor1 = (a * result_pcg_ssor1.solution) - b;
    double residual_norm_pcg_ssor1 = residual_pcg_ssor1.norm2();
    double relative_error_pcg_ssor1 = (result_pcg_ssor1.solution - x_exact).norm2() / x_exact.norm2();
    
    std::cout << "L2 norm of residual vector: " << residual_norm_pcg_ssor1 << std::endl;
    std::cout << "L2 Relative error: " << relative_error_pcg_ssor1 << std::endl;
    std::cout << "Время выполнения: " << duration_pcg_ssor1.count() << "s" << std::endl;
    std::cout << std::endl;

    std::cout << "========== PCG с предобусловливанием SSOR (ω = 1.5) ==========" << std::endl;
    auto start_pcg_ssor15 = std::chrono::high_resolution_clock::now();
    auto result_pcg_ssor15 = pcg_solve(a, b, PreconditionerType::SSOR, 1.5, epsilon, k_max);
    auto end_pcg_ssor15 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_pcg_ssor15 = end_pcg_ssor15 - start_pcg_ssor15;

    if (result_pcg_ssor15.converged) {
        std::cout << "Сходимость достигнута на итерации: " << result_pcg_ssor15.iterations << std::endl;
    } else {
        std::cout << "ВНИМАНИЕ: Превышено максимальное количество итераций (" << k_max << ")" << std::endl;
        std::cout << "Результат получен на итерации: " << result_pcg_ssor15.iterations << std::endl;
    }
    
    std::cout << "First 5 coordinates of x*: ";
    for (uint64_t i = 0; i < 5; ++i) {
        std::cout << result_pcg_ssor15.solution[i] << " ";
    }
    std::cout << std::endl;

    vector<double> residual_pcg_ssor15 = (a * result_pcg_ssor15.solution) - b;
    double residual_norm_pcg_ssor15 = residual_pcg_ssor15.norm2();
    double relative_error_pcg_ssor15 = (result_pcg_ssor15.solution - x_exact).norm2() / x_exact.norm2();
    
    std::cout << "L2 norm of residual vector: " << residual_norm_pcg_ssor15 << std::endl;
    std::cout << "L2 Relative error: " << relative_error_pcg_ssor15 << std::endl;
    std::cout << "Время выполнения: " << duration_pcg_ssor15.count() << "s" << std::endl;
    std::cout << std::endl;

    std::cout << "========== Сводка результатов ==========" << std::endl;
    std::cout << std::setw(30) << "Метод" 
              << std::setw(15) << "Итераций" 
              << std::setw(20) << "Время (с)" 
              << std::setw(20) << "Погрешность" << std::endl;
    std::cout << std::string(85, '-') << std::endl;
    std::cout << std::setw(30) << "CG (без предобусл.)" 
              << std::setw(15) << result_cg.iterations 
              << std::setw(20) << duration_cg.count() 
              << std::setw(20) << relative_error_cg << std::endl;
    std::cout << std::setw(30) << "PCG (Якоби)" 
              << std::setw(15) << result_pcg_jacobi.iterations 
              << std::setw(20) << duration_pcg_jacobi.count() 
              << std::setw(20) << relative_error_pcg_jacobi << std::endl;
    std::cout << std::setw(30) << "PCG (SSOR, ω=1.0)" 
              << std::setw(15) << result_pcg_ssor1.iterations 
              << std::setw(20) << duration_pcg_ssor1.count() 
              << std::setw(20) << relative_error_pcg_ssor1 << std::endl;
    std::cout << std::setw(30) << "PCG (SSOR, ω=1.5)" 
              << std::setw(15) << result_pcg_ssor15.iterations 
              << std::setw(20) << duration_pcg_ssor15.count() 
              << std::setw(20) << relative_error_pcg_ssor15 << std::endl;

    std::cout << std::endl;
    std::cout << "========== Анализ эффективности предобусловливания ==========" << std::endl;
    
    double speedup_jacobi = static_cast<double>(result_cg.iterations) / result_pcg_jacobi.iterations;
    double speedup_ssor1 = static_cast<double>(result_cg.iterations) / result_pcg_ssor1.iterations;
    double speedup_ssor15 = static_cast<double>(result_cg.iterations) / result_pcg_ssor15.iterations;
    
    std::cout << "Ускорение сходимости (по количеству итераций):" << std::endl;
    std::cout << "  Якоби: " << speedup_jacobi << "x" << std::endl;
    std::cout << "  SSOR (ω=1.0): " << speedup_ssor1 << "x" << std::endl;
    std::cout << "  SSOR (ω=1.5): " << speedup_ssor15 << "x" << std::endl;

    return 0;
}
