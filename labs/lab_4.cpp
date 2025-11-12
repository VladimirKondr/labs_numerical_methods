#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <random>
#include <iomanip>

#include "../linalg/matrix.cpp"
#include "../linalg/vector.cpp"
#include "../linalg/operations.cpp"
#include "../linalg/solvers.cpp"

int main() {
    std::cout << std::fixed << std::setprecision(6);
    
    const uint64_t n = 10;
    const uint64_t m = 4;
    const float epsilon = 0.0001f;
    const uint64_t k_max = 2000;
    const uint64_t seed = 42;

    std::cout << "Параметры: n=" << n << ", m=" << m << ", epsilon=" << epsilon << ", k_max=" << k_max << std::endl;
    std::cout << "Seed=" << seed << std::endl;
    std::cout << std::endl;

    matrix<float> a(n, n, 0.0f);
    std::mt19937 gen(seed);
    std::uniform_int_distribution<int> dis(0, 4);

    for (uint64_t i = 0; i < n; ++i) {
        for (uint64_t j = 0; j < n; ++j) {
            if (i != j) {
                int val = dis(gen);
                a(i, j) = -static_cast<float>(val);
            }
        }
    }

    for (uint64_t i = 1; i < n; ++i) {
        float sum = 0.0f;
        for (uint64_t j = 0; j < n; ++j) {
            if (j != i) {
                sum += a(i, j);
            }
        }
        a(i, i) = -sum;
    }

    float sum = 0.0f;
    for (uint64_t j = 1; j < n; ++j) {
        sum += a(0, j);
    }
    a(0, 0) = -sum + 1.0f;

    // std::cout << "Матрица A:" << std::endl;
    // std::cout << a << std::endl;

    vector<float> x_exact(n);
    for (uint64_t i = 0; i < n; ++i) {
        x_exact[i] = static_cast<float>(m + i);
    }

    std::cout << "Точное решение x:" << std::endl;
    std::cout << x_exact << std::endl;
    std::cout << std::endl;

    vector<float> f = a * x_exact;

    std::cout << "Правая часть f = Ax:" << std::endl;
    std::cout << f << std::endl;
    std::cout << std::endl;

    std::cout << "========== Метод Якоби ==========" << std::endl;
    auto start_jacobi = std::chrono::high_resolution_clock::now();
    auto result_jacobi = jacobi_solve(a, f, epsilon, k_max);
    auto end_jacobi = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_jacobi = end_jacobi - start_jacobi;

    if (result_jacobi.converged) {
        std::cout << "Сходимость достигнута на итерации: " << result_jacobi.iterations << std::endl;
    } else {
        std::cout << "ВНИМАНИЕ: Превышено максимальное количество итераций (" << k_max << ")" << std::endl;
        std::cout << "Результат получен на итерации: " << result_jacobi.iterations << std::endl;
    }
    
    std::cout << "Приближенное решение x*:" << std::endl;
    std::cout << result_jacobi.solution << std::endl;
    
    float error_jacobi = (result_jacobi.solution - x_exact).norm_inf();
    std::cout << "Максимальная погрешность: " << error_jacobi << std::endl;
    std::cout << "Время выполнения: " << duration_jacobi.count() << "s" << std::endl;
    std::cout << std::endl;

    std::cout << "========== Метод релаксации (ω = 0.5) ==========" << std::endl;
    auto start_sor05 = std::chrono::high_resolution_clock::now();
    auto result_sor05 = sor_solve(a, f, 0.5f, epsilon, k_max);
    auto end_sor05 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_sor05 = end_sor05 - start_sor05;

    if (result_sor05.converged) {
        std::cout << "Сходимость достигнута на итерации: " << result_sor05.iterations << std::endl;
    } else {
        std::cout << "ВНИМАНИЕ: Превышено максимальное количество итераций (" << k_max << ")" << std::endl;
        std::cout << "Результат получен на итерации: " << result_sor05.iterations << std::endl;
    }
    
    std::cout << "Приближенное решение x*:" << std::endl;
    std::cout << result_sor05.solution << std::endl;
    
    float error_sor05 = (result_sor05.solution - x_exact).norm_inf();
    std::cout << "Максимальная погрешность (infinity norm): " << error_sor05 << std::endl;
    std::cout << "Время выполнения: " << duration_sor05.count() << "s" << std::endl;
    std::cout << std::endl;

    std::cout << "========== Метод Гаусса-Зейделя (ω = 1.0) ==========" << std::endl;
    auto start_gs = std::chrono::high_resolution_clock::now();
    auto result_gs = sor_solve(a, f, 1.0f, epsilon, k_max);
    auto end_gs = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_gs = end_gs - start_gs;

    if (result_gs.converged) {
        std::cout << "Сходимость достигнута на итерации: " << result_gs.iterations << std::endl;
    } else {
        std::cout << "ВНИМАНИЕ: Превышено максимальное количество итераций (" << k_max << ")" << std::endl;
        std::cout << "Результат получен на итерации: " << result_gs.iterations << std::endl;
    }
    
    std::cout << "Приближенное решение x*:" << std::endl;
    std::cout << result_gs.solution << std::endl;
    
    float error_gs = (result_gs.solution - x_exact).norm_inf();
    std::cout << "Максимальная погрешность: " << error_gs << std::endl;
    std::cout << "Время выполнения: " << duration_gs.count() << "s" << std::endl;
    std::cout << std::endl;

    std::cout << "========== Метод релаксации (ω = 1.5) ==========" << std::endl;
    auto start_sor15 = std::chrono::high_resolution_clock::now();
    auto result_sor15 = sor_solve(a, f, 1.5f, epsilon, k_max);
    auto end_sor15 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_sor15 = end_sor15 - start_sor15;

    if (result_sor15.converged) {
        std::cout << "Сходимость достигнута на итерации: " << result_sor15.iterations << std::endl;
    } else {
        std::cout << "ВНИМАНИЕ: Превышено максимальное количество итераций (" << k_max << ")" << std::endl;
        std::cout << "Результат получен на итерации: " << result_sor15.iterations << std::endl;
    }
    
    std::cout << "Приближенное решение x*:" << std::endl;
    std::cout << result_sor15.solution << std::endl;
    
    float error_sor15 = (result_sor15.solution - x_exact).norm_inf();
    std::cout << "Максимальная погрешность: " << error_sor15 << std::endl;
    std::cout << "Время выполнения: " << duration_sor15.count() << "s" << std::endl;
    std::cout << std::endl;

    std::cout << "========== Сравнение методов ==========" << std::endl;
    std::cout << std::fixed << std::setprecision(6);
    
    const int method_width = 28;
    const int iter_width = 15;
    const int error_width = 20;
    const int time_width = 15;

    std::cout << std::left << std::setw(method_width) << "Метод"
              << std::right << std::setw(iter_width) << "Итерации"
              << std::setw(error_width) << "Погрешность"
              << std::setw(time_width) << "Время (ms)" << std::endl;
    
    std::cout << std::string(method_width + iter_width + error_width + time_width, '-') << std::endl;

    std::cout << std::left << std::setw(method_width) << "Якоби"
              << std::right << std::setw(iter_width) << result_jacobi.iterations
              << std::setw(error_width) << error_jacobi
              << std::setw(time_width) << duration_jacobi.count() << std::endl;
    
    std::cout << std::left << std::setw(method_width) << "Релаксация (ω=0.5)"
              << std::right << std::setw(iter_width) << result_sor05.iterations
              << std::setw(error_width) << error_sor05
              << std::setw(time_width) << duration_sor05.count() << std::endl;

    std::cout << std::left << std::setw(method_width) << "Гаусс-Зейдель (ω=1.0)"
              << std::right << std::setw(iter_width) << result_gs.iterations
              << std::setw(error_width) << error_gs
              << std::setw(time_width) << duration_gs.count() << std::endl;

    std::cout << std::left << std::setw(method_width) << "Релаксация (ω=1.5)"
              << std::right << std::setw(iter_width) << result_sor15.iterations
              << std::setw(error_width) << error_sor15
              << std::setw(time_width) << duration_sor15.count() << std::endl;

    return 0;
}
