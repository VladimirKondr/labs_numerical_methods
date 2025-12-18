#include "../danilevsky/danilevsky.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

double bisection_method(
    const std::function<double(double)>& f, double a, double b, double epsilon = 1e-9) {
    if (f(a) * f(b) > 0) {
        throw std::runtime_error(
            "Bisection method error: Function must have different signs at the interval "
            "endpoints.");
    }

    double c;
    while (std::abs(b - a) > epsilon) {
        c = a + (b - a) / 2.0;
        double fc = f(c);

        if (std::abs(fc) < epsilon) {
            return c;
        }

        if (f(a) * fc < 0) {
            b = c;
        } else {
            a = c;
        }
    }

    return (a + b) / 2.0;
}

double newton_method(
    const std::function<double(double)>& f, const std::function<double(double)>& f_prime, double x0,
    double epsilon = 1e-9, int max_iter = 1000) {
    double x = x0;

    for (int i = 0; i < max_iter; ++i) {
        double fx = f(x);
        double f_prime_x = f_prime(x);

        if (std::abs(fx) < epsilon) {
            return x;
        }

        if (std::abs(f_prime_x) < 1e-15) {
            throw std::runtime_error("Newton's method error: Derivative is too small.");
        }

        double x_new = x - fx / f_prime_x;

        if (std::abs(x_new - x) < epsilon) {
            return x_new;
        }

        x = x_new;
    }

    throw std::runtime_error("Newton's method failed to converge within maximum iterations.");
}

std::vector<double> solve_quadratic(double a, double b, double c) {
    std::vector<double> roots;

    if (std::abs(a) < 1e-15) {
        if (std::abs(b) > 1e-15) {
            roots.push_back(-c / b);
        }
        return roots;
    }

    double discriminant = b * b - 4 * a * c;

    if (discriminant > 0) {
        double sqrt_d = std::sqrt(discriminant);
        roots.push_back((-b + sqrt_d) / (2 * a));
        roots.push_back((-b - sqrt_d) / (2 * a));
    } else if (std::abs(discriminant) < 1e-15) {
        roots.push_back(-b / (2 * a));
    }

    std::sort(roots.begin(), roots.end());
    return roots;
}

matrix<double> generate_valid_matrix(uint64_t n, uint64_t seed, int max_attempts = 1000) {
    for (int attempt = 0; attempt < max_attempts; ++attempt) {
        matrix<double> a = matrix<double>::random(n, n, seed + attempt, -50.0f, 50.0f);
        
        try {
            perform_danilevsky(a);
            std::cout << "valid attempt number " << attempt << std::endl;
            return a;
        } catch (const std::runtime_error&) {
            continue;
        }
    }
    
    throw std::runtime_error("Failed to generate valid matrix after maximum attempts");
}

int main() {
    std::cout << std::fixed << std::setprecision(6);

    const uint64_t n = 4;
    const uint64_t seed = 42;
    matrix<double> A;
    try {
        A = generate_valid_matrix(n, seed);
    } catch (const std::runtime_error& e) {
        std::cerr << "ОШИБКА генерации матрицы: " << e.what() << std::endl;
        return 1;
    }

    std::cout << "Исходная матрица A:\n" << A << "\n";

    DanilevskyResult<double> result;
    try {
        result = perform_danilevsky(A);
    } catch (const std::exception& e) {
        std::cerr << "ОШИБКА метода Данилевского: " << e.what() << std::endl;
        return 1;
    }

    const auto& coeffs = result.characteristic_poly_coeffs;
    double p1 = coeffs[0], p2 = coeffs[1], p3 = coeffs[2], p4 = coeffs[3];

    std::cout << "Характеристический многочлен P(lambda) = lambda⁴ + p₁lambda³ + p₂lambda² + p₃lambda + p₄:\n";
    std::cout << "p₁=" << p1 << ", p₂=" << p2 << ", p₃=" << p3 << ", p₄=" << p4 << "\n\n";

    auto P = [p1, p2, p3, p4](double lambda) {
        return std::pow(lambda, 4) + p1 * std::pow(lambda, 3) + p2 * std::pow(lambda, 2) + p3 * lambda + p4;
    };
    auto P_prime = [p1, p2, p3](double lambda) {
        return 4 * std::pow(lambda, 3) + 3 * p1 * std::pow(lambda, 2) + 2 * p2 * lambda + p3;
    };

    std::vector<double> roots_P_prime;
    
    std::vector<double> roots_P_double_prime = solve_quadratic(12.0, 6.0 * p1, 2.0 * p2);

    std::vector<double> points = roots_P_double_prime;
    points.insert(points.begin(), -200.0);
    points.push_back(200.0);
    
    for (size_t i = 0; i < points.size() - 1; ++i) {
        if (P_prime(points[i]) * P_prime(points[i+1]) < 0) {
            try {
                roots_P_prime.push_back(bisection_method(P_prime, points[i], points[i+1]));
            } catch (...) {}
        }
    }

    if (roots_P_prime.empty()) {
        std::cout << "Корней P'(lambda) не найдено. P(lambda) монотонна.\n\n";
    } else {
        std::cout << "Корни P'(lambda) (точки экстремума P(lambda)): ";
        for (const auto& root : roots_P_prime) std::cout << root << " ";
        std::cout << "\n\n";
    }
    
    std::vector<double> eigenvalues;
    std::vector<double> search_points = roots_P_prime;
    search_points.insert(search_points.begin(), -200.0);
    search_points.push_back(200.0);

    for (size_t i = 0; i < search_points.size() - 1; ++i) {
        double a = search_points[i], b = search_points[i+1];
        if (P(a) * P(b) <= 0) {
            try {
                eigenvalues.push_back(newton_method(P, P_prime, (a + b) / 2.0));
            } catch (...) {
                try {
                    eigenvalues.push_back(bisection_method(P, a, b));
                } catch(...) {}
            }
        }
    }
    
    std::sort(eigenvalues.begin(), eigenvalues.end());
    eigenvalues.erase(std::unique(eigenvalues.begin(), eigenvalues.end(), [](double a, double b){ return std::abs(a-b) < 1e-6; }), eigenvalues.end());

    if (eigenvalues.empty()) {
        std::cout << "Вещественных собственных значений не найдено.\n";
        return 0;
    }
    
    std::cout << "Найденные собственные значения:\n";
    for (size_t i = 0; i < eigenvalues.size(); ++i) {
        std::cout << "  lambda_" << i + 1 << " = " << eigenvalues[i] << "\n";
    }
    std::cout << "\n";
    
    double lambda_to_check = eigenvalues[0];
    std::cout << "Проверка для lambda = " << lambda_to_check << "\n";

    vector<double> y(n);
    for(uint64_t i = 0; i < n; ++i) {
        y[i] = std::pow(lambda_to_check, n - 1 - i);
    }
    
    vector<double> u = y;
    for (int i = static_cast<int>(result.transform_matrices.size()) - 1; i >= 0; --i) {
        u = result.transform_matrices[i] * u;
    }
    std::cout << "Найденный собственный вектор u:\n" << u << "\n";

    vector<double> residual = (A * u) - (u * lambda_to_check);
    double residual_norm = residual.norm2();
    
    std::cout << "Норма вектора невязки = " << residual_norm << "\n";
    return 0;
}