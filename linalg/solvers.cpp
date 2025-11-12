#ifndef SOLVERS_CPP
#define SOLVERS_CPP

#include "matrix.cpp"
#include "vector.cpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

/**
 * @brief Defines the pivoting strategy for Gaussian elimination.
 */
enum class PivotStrategy { NO_PIVOT, PARTIAL_PIVOT };

/**
 * @brief Solves a system of linear equations Ax = b using Gaussian elimination.
 * @tparam T The data type of the matrix and vector elements.
 * @param a The matrix A.
 * @param b The vector b.
 * @param strategy The pivoting strategy to use (NO_PIVOT or PARTIAL_PIVOT).
 * @return The solution vector x.
 */
template <typename T>
vector<T> gauss_solve(
    const matrix<T>& a_in, const vector<T>& b_in,
    PivotStrategy strategy = PivotStrategy::PARTIAL_PIVOT) {
    matrix<T> a = a_in;
    vector<T> b = b_in;
    const uint64_t n = a.rows();
    if (n != a.cols() || n != b.size()) {
        throw std::invalid_argument("Matrix must be square and sizes must match.");
    }

    // Forward elimination
    for (uint64_t i = 0; i < n; ++i) {
        if (strategy == PivotStrategy::PARTIAL_PIVOT) {
            uint64_t max_row = i;
            for (uint64_t k = i + 1; k < n; ++k) {
                if (std::abs(a(k, i)) > std::abs(a(max_row, i))) {
                    max_row = k;
                }
            }

            if (max_row != i) {
                for (uint64_t k = i; k < n; ++k) {
                    std::swap(a(i, k), a(max_row, k));
                }
                std::swap(b[i], b[max_row]);
            }
        }

        // If pivot is close to zero, matrix is singular
        if (std::abs(a(i, i)) < 1e-12) {
            throw std::runtime_error("Matrix is singular or nearly singular.");
        }

        for (uint64_t k = i + 1; k < n; ++k) {
            T factor = a(k, i) / a(i, i);
            for (uint64_t j = i; j < n; ++j) {
                a(k, j) -= factor * a(i, j);
            }
            b[k] -= factor * b[i];
        }
    }

    // Backward substitution
    vector<T> x(n);
    for (int i = n - 1; i >= 0; --i) {
        T sum = 0;
        for (uint64_t j = i + 1; j < n; ++j) {
            sum += a(i, j) * x[j];
        }
        x[i] = (b[i] - sum) / a(i, i);
    }

    return x;
}

/**
 * @brief Performs in-place LDLT decomposition of a symmetric matrix.
 * The lower triangular matrix L (without the unit diagonal) is stored in the lower part of a,
 * and the diagonal matrix D is stored on the main diagonal of a.
 * @tparam T The data type of the matrix elements.
 * @param a The symmetric matrix to decompose.
 * https://math.stackexchange.com/questions/2512046/give-an-algorithm-to-compute-the-ldlt-decomposition
 */
template <typename T>
void ldlt_decomposition(matrix<T>& a) {
    const uint64_t n = a.rows();
    if (n != a.cols()) {
        throw std::invalid_argument("Matrix must be square for LDLT decomposition.");
    }

    std::vector<T> v(n);
    for (uint64_t j = 0; j < n; ++j) {
        T temp_sum = 0.0;
        for (uint64_t k = 0; k < j; ++k) {
            v[k] = a(j, k) * a(k, k);
            temp_sum += a(j, k) * v[k];
        }
        a(j, j) -= temp_sum;
        if (std::abs(a(j, j)) < 1e-12) {
            throw std::runtime_error(
                "LDLT algorithm has encountered zero-element of D, meaning non-positive "
                "definition");
        }

        const T inv = T(1.0) / a(j, j);
        for (uint64_t i = j + 1; i < n; ++i) {
            T sum_l = 0.0;
            for (uint64_t k = 0; k < j; ++k) {
                sum_l += a(i, k) * v[k];
            }
            a(i, j) = (a(i, j) - sum_l) * inv;
        }
    }
}

/**
 * @brief Solves a system of linear equations Ax = b using LDLT decomposition.
 * This method is suitable for symmetric positive-definite matrices.
 * @tparam T The data type of the matrix and vector elements.
 * @param a_in The symmetric matrix A.
 * @param b The vector b.
 * @return The solution vector x.
 */
template <typename T>
vector<T> ldlt_solve(const matrix<T>& a_in, const vector<T>& b) {
    matrix<T> a = a_in;
    const uint64_t n = a.rows();
    if (n != a.cols() || n != b.size()) {
        throw std::invalid_argument("Matrix must be square and sizes must match.");
    }

    ldlt_decomposition(a);
    // LDL^Tx = b
    // 1) z = DL^Tx
    // Lz = b
    // 2) y = L^Tx
    // Dy = z
    // 3) L^Tx = y

    // Forward substitution (Lz = b)
    vector<T> z(n);
    for (uint64_t i = 0; i < n; ++i) {
        T sum = 0;
        for (uint64_t j = 0; j < i; ++j) {
            sum += a(i, j) * z[j];
        }
        z[i] = b[i] - sum;
    }

    // Diagonal scaling (Dy = z)
    vector<T> y(n);
    for (uint64_t i = 0; i < n; ++i) {
        y[i] = z[i] / a(i, i);
    }

    // Backward substitution (L^T x = y)
    vector<T> x(n);
    for (int i = n - 1; i >= 0; --i) {
        T sum = 0;
        for (uint64_t j = i + 1; j < n; ++j) {
            sum += a(j, i) * x[j];
        }
        x[i] = y[i] - sum;
    }

    return x;
}

/**
 * @brief Solves a system of linear equations Ax = d for a tridiagonal matrix A using the Thomas
 * algorithm.
 * @tparam T The data type of the matrix and vector elements.
 * @param a The tridiagonal matrix A.
 * @param d The vector d (the right-hand side of the equation).
 * @return The solution vector x.
 */
template <typename T>
vector<T> tridiagonal_solve(const matrix<T>& a, const vector<T>& d) {
    const uint64_t n = a.rows();
    if (n != a.cols() || n != d.size()) {
        throw std::invalid_argument(
            "Matrix must be square and sizes must match for tridiagonal solver.");
    }

    if (n == 0) {
        return vector<T>();
    }

    vector<T> cc(n);
    vector<T> dc(n);

    // Forward substitution
    T b0 = a(0, 0);
    if (std::abs(b0) < 1e-12) {
        throw std::runtime_error("Tridiagonal matrix is singular (first element is zero).");
    }
    cc[0] = a(0, 1) / b0;
    dc[0] = d[0] / b0;

    for (uint64_t i = 1; i < n; ++i) {
        T a_i = a(i, i - 1);
        T b_i = a(i, i);
        T denom = b_i - a_i * cc[i - 1];
        if (std::abs(denom) < 1e-12) {
            throw std::runtime_error("Tridiagonal matrix is singular (zero pivot encountered).");
        }

        if (i < n - 1) {
            T c_i = a(i, i + 1);
            cc[i] = c_i / denom;
        }

        dc[i] = (d[i] - a_i * dc[i - 1]) / denom;
    }

    // Backward substitution
    vector<T> x(n);
    x[n - 1] = dc[n - 1];
    for (int i = n - 2; i >= 0; --i) {
        x[i] = dc[i] - cc[i] * x[i + 1];
    }

    return x;
}

/**
 * @brief Struct to hold the result of iterative methods.
 * @tparam T The data type of the solution vector.
 */
template <typename T>
struct IterativeResult {
    vector<T> solution;
    uint64_t iterations;
    bool converged;
};

/**
 * @brief Solves a system of linear equations Ax = f using the Jacobi method.
 * @tparam T The data type of the matrix and vector elements.
 * @param a The matrix A.
 * @param f The vector f.
 * @param epsilon The tolerance for convergence.
 * @param max_iterations The maximum number of iterations.
 * @return An IterativeResult containing the solution, number of iterations, and convergence status.
 */
template <typename T>
IterativeResult<T> jacobi_solve(
    const matrix<T>& a, const vector<T>& f, T epsilon = 1e-4, uint64_t max_iterations = 2000) {
    const uint64_t n = a.rows();
    if (n != a.cols() || n != f.size()) {
        throw std::invalid_argument(
            "Matrix must be square and sizes must match for Jacobi method.");
    }

    vector<T> x(n, T{0});
    vector<T> x_new(n);

    for (uint64_t k = 0; k < max_iterations; ++k) {
        for (uint64_t i = 0; i < n; ++i) {
            T sum = 0;
            for (uint64_t j = 0; j < n; ++j) {
                if (j != i) {
                    sum += a(i, j) * x[j];
                }
            }
            x_new[i] = (f[i] - sum) / a(i, i);
        }

        // Check convergence
        T max_diff = 0;
        for (uint64_t i = 0; i < n; ++i) {
            T diff = std::abs(x_new[i] - x[i]);
            if (diff > max_diff) {
                max_diff = diff;
            }
        }

        x = x_new;

        if (max_diff < epsilon) {
            return {x, k + 1, true};
        }
    }

    return {x, max_iterations, false};
}

/**
 * @brief Solves a system of linear equations Ax = f using the Successive Over-Relaxation (SOR)
 * method. When omega = 1, this becomes the Gauss-Seidel method. 0 < omega < 1: Lower relaxation. 1
 * < ω < 2: Upper relaxation.
 * @tparam T The data type of the matrix and vector elements.
 * @param a The matrix A.
 * @param f The vector f.
 * @param omega The relaxation parameter (0 < omega < 2).
 * @param epsilon The tolerance for convergence.
 * @param max_iterations The maximum number of iterations.
 * @return An IterativeResult containing the solution, number of iterations, and convergence status.
 */
template <typename T>
IterativeResult<T> sor_solve(
    const matrix<T>& a, const vector<T>& f, T omega = 1.0, T epsilon = 1e-4,
    uint64_t max_iterations = 2000) {
    const uint64_t n = a.rows();
    if (n != a.cols() || n != f.size()) {
        throw std::invalid_argument("Matrix must be square and sizes must match for SOR method.");
    }

    vector<T> x(n, T{0});
    vector<T> x_old(n);

    for (uint64_t k = 0; k < max_iterations; ++k) {
        x_old = x;

        for (uint64_t i = 0; i < n; ++i) {
            T sum1 = 0;
            for (uint64_t j = 0; j < i; ++j) {
                sum1 += a(i, j) * x[j];
            }

            T sum2 = 0;
            for (uint64_t j = i + 1; j < n; ++j) {
                sum2 += a(i, j) * x_old[j];
            }

            x[i] = (1 - omega) * x[i] + (omega / a(i, i)) * (f[i] - sum1 - sum2);
        }

        // Check convergence
        T max_diff = 0;
        for (uint64_t i = 0; i < n; ++i) {
            T diff = std::abs(x[i] - x_old[i]);
            if (diff > max_diff) {
                max_diff = diff;
            }
        }

        if (max_diff < epsilon) {
            return {x, k + 1, true};
        }
    }

    return {x, max_iterations, false};
}

/**
 * @brief Solves a system of linear equations Ax = b using the Conjugate Gradient (CG) method.
 * This method is suitable for symmetric positive-definite matrices.
 * @tparam T The data type of the matrix and vector elements.
 * @param a The symmetric positive-definite matrix A.
 * @param b The vector b.
 * @param epsilon The tolerance for convergence.
 * @param max_iterations The maximum number of iterations.
 * @return An IterativeResult containing the solution, number of iterations, and convergence status.
 */
template <typename T>
IterativeResult<T> cg_solve(
    const matrix<T>& a, const vector<T>& b, T epsilon = 1e-4, uint64_t max_iterations = 2000) {
    const T eps_sqr = epsilon * epsilon;
    const uint64_t n = a.rows();
    if (n != a.cols() || n != b.size()) {
        throw std::invalid_argument("Matrix must be square and sizes must match for CG method.");
    }

    vector<T> x(n, T{0});
    vector<T> r = b - (a * x);  // r_0 = b - Ax_0
    vector<T> p = r;            // p_0 = r_0
    T r_norm_sq = r * r;        // ||r_0||^2

    for (uint64_t k = 0; k < max_iterations; ++k) {
        vector<T> Ap = a * p;
        T alpha = r_norm_sq / (p * Ap);
        
        for (uint64_t i = 0; i < n; ++i) {
            x[i] += alpha * p[i];
            r[i] -= alpha * Ap[i];
        }

        T r_norm_sq_new = r * r;
        
        if (r_norm_sq_new < eps_sqr) {
            return {x, k + 1, true};
        }

        T beta = r_norm_sq_new / r_norm_sq;
        
        for (uint64_t i = 0; i < n; ++i) {
            p[i] = r[i] + beta * p[i];
        }

        r_norm_sq = r_norm_sq_new;
    }

    return {x, max_iterations, false};
}

/**
 * @brief Enum for preconditioning strategies.
 */
enum class PreconditionerType { NONE, JACOBI, SSOR };

/**
 * @brief Base class for preconditioners used in iterative solvers.
 * @tparam T The data type of matrix and vector elements.
 */
template <typename T>
class Preconditioner {
   public:
    virtual ~Preconditioner() = default;
    
    /**
     * @brief Applies the preconditioner: solves M*z = r for z.
     * @param r The residual vector.
     * @return The preconditioned vector z.
     */
    virtual vector<T> apply(const vector<T>& r) const = 0;
};

/**
 * @brief Identity (no) preconditioner: M = I.
 */
template <typename T>
class NoPreconditioner : public Preconditioner<T> {
   public:
    vector<T> apply(const vector<T>& r) const override {
        return r;
    }
};

/**
 * @brief Jacobi preconditioner: M = diag(A).
 */
template <typename T>
class JacobiPreconditioner : public Preconditioner<T> {
   private:
    vector<T> inv_diag_;
    
   public:
    explicit JacobiPreconditioner(const matrix<T>& a) : inv_diag_(a.rows()) {
        for (uint64_t i = 0; i < a.rows(); ++i) {
            inv_diag_[i] = T{1} / a(i, i);
        }
    }
    
    vector<T> apply(const vector<T>& r) const override {
        vector<T> z(r.size());
        for (uint64_t i = 0; i < r.size(); ++i) {
            z[i] = inv_diag_[i] * r[i];
        }
        return z;
    }
};

/**
 * @brief SSOR preconditioner: M = (D + ωL)D^(-1)(D + ωL^T) / ω(2-ω).
 */
template <typename T>
class SSORPreconditioner : public Preconditioner<T> {
   private:
    matrix<T> lower_;  // D + ωL
    T scale_;          // 1 / (ω(2-ω))
    uint64_t n_;
    
   public:
    SSORPreconditioner(const matrix<T>& a, T omega) 
        : lower_(a.rows(), a.rows(), T{0}), 
          scale_(T{1} / (omega * (T{2} - omega))),
          n_(a.rows()) {
        for (uint64_t i = 0; i < n_; ++i) {
            lower_(i, i) = a(i, i);
            for (uint64_t j = 0; j < i; ++j) {
                lower_(i, j) = omega * a(i, j);
            }
        }
    }
    
    vector<T> apply(const vector<T>& r) const override {
        // Forward substitution (D + ωL)y = r
        vector<T> y(n_);
        for (uint64_t i = 0; i < n_; ++i) {
            T sum = T{0};
            for (uint64_t j = 0; j < i; ++j) {
                sum += lower_(i, j) * y[j];
            }
            y[i] = (r[i] - sum) / lower_(i, i);
        }
        
        // w = D^(-1) y
        vector<T> w(n_);
        for (uint64_t i = 0; i < n_; ++i) {
            w[i] = y[i] / lower_(i, i);
        }
        
        // Backward substitution (D + ωL^T)z = Dw
        vector<T> z(n_);
        for (int i = n_ - 1; i >= 0; --i) {
            T sum = T{0};
            for (uint64_t j = i + 1; j < n_; ++j) {
                sum += lower_(j, i) * z[j];
            }
            z[i] = (lower_(i, i) * w[i] - sum) / lower_(i, i);
        }
        
        // Scale by 1/(ω(2-ω))
        for (uint64_t i = 0; i < n_; ++i) {
            z[i] *= scale_;
        }
        
        return z;
    }
};

/**
 * @brief Factory function to create a preconditioner.
 */
template <typename T>
Preconditioner<T>* create_preconditioner(
    PreconditionerType type, const matrix<T>& a, T omega = T{1}) {
    switch (type) {
        case PreconditionerType::NONE:
            return new NoPreconditioner<T>();
        case PreconditionerType::JACOBI:
            return new JacobiPreconditioner<T>(a);
        case PreconditionerType::SSOR:
            return new SSORPreconditioner<T>(a, omega);
        default:
            return new NoPreconditioner<T>();
    }
}

/**
 * @brief Solves a system of linear equations Ax = b using the Preconditioned Conjugate Gradient (PCG) method.
 * @tparam T The data type of the matrix and vector elements.
 * @param a The symmetric positive-definite matrix A.
 * @param b The vector b.
 * @param preconditioner_type The type of preconditioner to use.
 * @param omega The relaxation parameter for SSOR preconditioner (0 < omega < 2).
 * @param epsilon The tolerance for convergence.
 * @param max_iterations The maximum number of iterations.
 * @return An IterativeResult containing the solution, number of iterations, and convergence status.
 */
template <typename T>
IterativeResult<T> pcg_solve(
    const matrix<T>& a, const vector<T>& b, 
    PreconditionerType preconditioner_type = PreconditionerType::JACOBI,
    T omega = 1.0, T epsilon = 1e-4, uint64_t max_iterations = 2000) {
    const uint64_t n = a.rows();
    if (n != a.cols() || n != b.size()) {
        throw std::invalid_argument("Matrix must be square and sizes must match for PCG method.");
    }

    Preconditioner<T>* precond = create_preconditioner(preconditioner_type, a, omega);
    
    vector<T> x(n, T{0});
    vector<T> r = b - (a * x);
    vector<T> z = precond->apply(r);
    vector<T> p = z;
    T rz = r * z; // beta

    for (uint64_t k = 0; k < max_iterations; ++k) {
        vector<T> Ap = a * p;
        T alpha = rz / (p * Ap);
        
        for (uint64_t i = 0; i < n; ++i) {
            x[i] += alpha * p[i];
            r[i] -= alpha * Ap[i];
        }

        if (r.norm2() < epsilon) {
            delete precond;
            return {x, k + 1, true};
        }

        z = precond->apply(r);
        T rz_new = r * z;
        T beta = rz_new / rz;
        
        for (uint64_t i = 0; i < n; ++i) {
            p[i] = z[i] + beta * p[i];
        }

        rz = rz_new;
    }

    delete precond;
    return {x, max_iterations, false};
}

#endif  // SOLVERS_CPP