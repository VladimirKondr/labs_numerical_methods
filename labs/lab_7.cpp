#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <ctime>

#include "../linalg/matrix.cpp"
#include "../linalg/operations.cpp"


matrix<float> danilevsky_method(const matrix<float>& a_in, std::vector<matrix<float>>& m_matrices) {
    const uint64_t n = a_in.rows();
    
    if (n != a_in.cols()) {
        throw std::invalid_argument("Matrix must be square");
    }
    
    matrix<float> a = a_in;
    
    m_matrices.clear();
    
    for (uint64_t k = n; k >= 2; --k) {
        const uint64_t row_idx = k - 1;
        const uint64_t pivot_idx = k - 2;
        
        const float pivot = a(row_idx, pivot_idx);
        
        if (std::abs(pivot) < 1e-8f) {
            throw std::runtime_error("Pivot element is too small (near zero). Cannot continue with regular case.");
        }
        
        matrix<float> m(n, n, 0.0f);
        for (uint64_t i = 0; i < n; ++i) {
            m(i, i) = 1.0f;
        }
        
        for (uint64_t j = 0; j < n; ++j) {
            if (j == pivot_idx) {
                m(pivot_idx, j) = 1.0f / pivot;
            } else {
                m(pivot_idx, j) = -a(row_idx, j) / pivot;
            }
        }
        
        matrix<float> m_inv = m.inverse();
        
        a = m_inv * a * m; 
        
        m_matrices.push_back(m);
    }
    
    return a;
}

matrix<float> generate_valid_matrix(uint64_t n, uint64_t seed, int max_attempts = 1000) {
    std::vector<matrix<float>> dummy_m_matrices;
    
    for (int attempt = 0; attempt < max_attempts; ++attempt) {
        matrix<float> a = matrix<float>::random(n, n, seed + attempt, -50.0f, 50.0f);
        
        try {
            danilevsky_method(a, dummy_m_matrices);
            std::cout << "valid attempt number " << attempt << std::endl;
            return a;
        } catch (const std::runtime_error&) {
            continue;
        }
    }
    
    throw std::runtime_error("Failed to generate valid matrix after maximum attempts");
}

int main() {
    std::cout << std::fixed << std::setprecision(8);

    
    const uint64_t n = 4;
    const uint64_t seed = static_cast<uint64_t>(std::time(nullptr));
    
    matrix<float> a;
    try {
        a = generate_valid_matrix(n, seed);
    } catch (const std::runtime_error& e) {
        std::cerr << "ОШИБКА: " << e.what() << std::endl;
        return 1;
    }

    std::cout << "Матрица A" << std::endl;
    std::cout << a << std::endl;
    
    std::vector<matrix<float>> m_matrices;
    matrix<float> frobenius;
    
    try {
        frobenius = danilevsky_method(a, m_matrices);
        std::cout << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "ОШИБКА при выполнении метода: " << e.what() << std::endl;
        return 1;
    }
    
    std::cout << "Каноническая форма Фробениуса:" << std::endl;
    std::cout << frobenius << std::endl;
    
    std::cout << "Преобразующие матрицы M_i:" << std::endl;
    std::cout << "(Всего матриц: " << m_matrices.size() << ")" << std::endl;
    std::cout << std::endl;
    
    for (int i = static_cast<int>(m_matrices.size()) - 1; i >= 0; --i) {
        std::cout << "Матрица M_" << (i + 1) << ":" << std::endl;
        std::cout << m_matrices[i] << std::endl;
    }
    
    const float p1 = frobenius(0, 0);
    const float trace_a = a.trace();
    
    std::cout << "Коэффициент p_1 (из формы Фробениуса): " << p1 << std::endl;
    std::cout << "След матрицы A (Sp A):                  " << trace_a << std::endl;
    std::cout << "Разность |p_1 - Sp A|:                  " << std::abs(p1 - trace_a) << std::endl;
    std::cout << std::endl;
    
    const float tolerance = 1e-3f;
    if (std::abs(p1 - trace_a) < tolerance) {
        std::cout << "ПРОВЕРКА ПРОЙДЕНА: p_1 ≈ Sp A (с точностью " << tolerance << ")" << std::endl;
    } else {
        std::cout << "ВНИМАНИЕ: Разность p_1 и Sp A превышает допустимую погрешность!" << std::endl;
    }
    std::cout << std::endl;
    return 0;
}
