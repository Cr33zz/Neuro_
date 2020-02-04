#include <sstream>
#include <iomanip>

#include "ComputationalGraph/Session.h"
#include "Optimizers/LBFGS.h"
#include "Tensors/TensorOpCpu.h"
#include "ComputationalGraph/Variable.h"
#include "ComputationalGraph/Graph.h"

namespace Neuro
{
    //////////////////////////////////////////////////////////////////////////
    static void PackParams(const vector<Variable*>& vars, Tensor& x)
    {
        size_t xOffset = 0;
        for (auto v : vars)
        {
            v->Output().CopyTo(0, x, xOffset, v->Output().Length());
            xOffset += v->Output().Length();
        }
    }

    //////////////////////////////////////////////////////////////////////////
    static void UnPackParams(const Tensor& x, vector<Variable*>& vars)
    {
        size_t xOffset = 0;
        for (auto v : vars)
        {
            x.CopyTo(xOffset, v->Output(), 0, v->Output().Length());
            xOffset += v->Output().Length();
        }
    }

    //////////////////////////////////////////////////////////////////////////
    static void PackGrads(const vector<Variable*>& vars, Tensor& grad)
    {
        size_t gradOffset = 0;
        for (auto v : vars)
        {
            v->OutputGrad().CopyTo(0, grad, gradOffset, v->Output().Length());
            gradOffset += v->Output().Length();
        }
    }

    //////////////////////////////////////////////////////////////////////////
    static void UnPackGrads(const Tensor& grad, vector<Variable*>& vars)
    {
        size_t gradOffset = 0;
        for (auto v : vars)
        {
            grad.CopyTo(gradOffset, v->OutputGrad(), gradOffset, v->Output().Length());
            gradOffset += v->Output().Length();
        }
    }

    //////////////////////////////////////////////////////////////////////////
    class Foo
    {
    public:
        Foo(const vector<TensorLike*>& losses, const vector<Variable*>& vars)
            : m_Losses(losses), m_Vars(vars)
        {
            m_BwdOrder = Graph::Default()->BuildBackwardOrder(losses, m_NodesAffectingLosses, vars);
        }

        // evaluate computational graph for the purpose of l-bfgs optimization step
        // both x and grad contain flatten and merged parameters and their gradients
        float operator()(const Tensor& x, Tensor& grad)
        {
            UnPackParams(x, m_Vars);
            Session::Default()->Run(m_Losses);
            auto vars = Graph::Default()->ComputeGradientsInOrder(m_BwdOrder, m_Losses, m_NodesAffectingLosses, m_Vars);
            PackGrads(vars, grad);

            float fx = 0.f; // loss is considered sum of all input tensors values...
            for (auto inputNode : m_Losses)
                fx += inputNode->Output().Sum(NoneAxis)(0);

            return fx;
        }

    private:
        vector<TensorLike*> m_Losses;
        vector<Variable*> m_Vars;
        vector<TensorLike*> m_BwdOrder;
        unordered_set<TensorLike*> m_NodesAffectingLosses;
    };

    //////////////////////////////////////////////////////////////////////////
    class LineSearchBacktracking
    {
    private:
        //typedef Eigen::Matrix<float, Eigen::Dynamic, 1> Tensor;

    public:
        ///
        /// Line search by backtracking.
        ///
        /// \param f      A function object such that `f(x, grad)` returns the
        ///               objective function value at `x`, and overwrites `grad` with
        ///               the gradient.
        /// \param fx     In: The objective function value at the current point.
        ///               Out: The function value at the new point.
        /// \param x      Out: The new point moved to.
        /// \param grad   In: The current gradient vector. Out: The gradient at the
        ///               new point.
        /// \param step   In: The initial step length. Out: The calculated step length.
        /// \param drt    The current moving direction.
        /// \param xp     The current point.
        /// \param param  Parameters for the LBFGS algorithm
        ///
        static void LineSearch(Foo& f, float& fx, Tensor& x, Tensor& grad,
            float& step,
            const Tensor& drt, const Tensor& xp,
            const LBFGS::Param& param)
        {
            // Decreasing and increasing factors
            const float dec = 0.5f;
            const float inc = 2.1f;

            // Check the value of step
            if (step <= 0)
                std::invalid_argument("'step' must be positive");

            // Save the function value at the current x
            const float fx_init = fx;
            // Projection of gradient on the search direction
            const float dg_init = grad.Dot(drt);
            // Make sure d points to a descent direction
            if (dg_init > 0)
                std::logic_error("the moving direction increases the objective function value");

            const float dg_test = param.ftol * dg_init;
            float width;

            int iter;
            for (iter = 0; iter < param.max_linesearch; iter++)
            {
                // x_{k+1} = x_k + step * d_k
                x = xp.Add(drt.Mul(step));
                // Evaluate this candidate
                fx = f(x, grad);

                if (fx > fx_init + step * dg_test)
                {
                    width = dec;
                }
                else
                {
                    // Armijo condition is met
                    if (param.linesearch == LBFGS::LINESEARCH_BACKTRACKING_ARMIJO)
                        break;

                    const float dg = grad.Dot(drt);
                    if (dg < param.wolfe * dg_init)
                    {
                        width = inc;
                    }
                    else {
                        // Regular Wolfe condition is met
                        if (param.linesearch == LBFGS::LINESEARCH_BACKTRACKING_WOLFE)
                            break;

                        if (dg > -param.wolfe * dg_init)
                        {
                            width = dec;
                        }
                        else {
                            // Strong Wolfe condition is met
                            break;
                        }
                    }
                }

                if (iter >= param.max_linesearch)
                    throw std::runtime_error("the line search routine reached the maximum number of iterations");

                if (step < param.min_step)
                    throw std::runtime_error("the line search step became smaller than the minimum value allowed");

                if (step > param.max_step)
                    throw std::runtime_error("the line search step became larger than the maximum value allowed");

                step *= width;
            }
        }
    };

    //////////////////////////////////////////////////////////////////////////
    LBFGS::LBFGS(size_t maxIterations, float epsilon)
        : m_MaxIterations(maxIterations), m_Epsilon(epsilon)
    {
    }

    //////////////////////////////////////////////////////////////////////////
    OptimizerBase* LBFGS::Clone() const
    {
        return new LBFGS(*this);
    }

    //////////////////////////////////////////////////////////////////////////
    string LBFGS::ToString()
    {
        stringstream ss;
        ss << setprecision(5) << "L-BFGS()";
        return ss.str();
    }

    //////////////////////////////////////////////////////////////////////////
    const char* LBFGS::ClassName() const
    {
        return "LBFGS";
    }

    //////////////////////////////////////////////////////////////////////////
    LBFGS::MinimizationOperation::MinimizationOperation(const vector<TensorLike*>& losses, const vector<Variable*>& vars, size_t maxIterations, float epsilon)
        : Operation({}, "L_BFGS_minimize"), m_Vars(vars), m_Losses(losses)
    {
        m_param.epsilon = epsilon;
        m_param.max_iterations = maxIterations;

        if (vars.empty())
        {
            // we need all trainable tensors to figure out final number of parameters
            unordered_set<TensorLike*> nodesAffectingLosses;
            auto order = Graph::Default()->BuildBackwardOrder(losses, nodesAffectingLosses, vars);

            for (auto node : nodesAffectingLosses)
            {
                if (node->CareAboutGradient() && node->IsVar())
                {
                    Variable* var = static_cast<Variable*>(node);
                    if (var->Trainable())
                        m_Vars.push_back(var);
                }
            }
        }

        for (auto v : m_Vars)
            m_ParamsNum += v->Output().Length();
    }

    //////////////////////////////////////////////////////////////////////////
    void LBFGS::MinimizationOperation::ComputeInternal()
    {
        m_InputsManuallyConsumed = true; // loss outputs will be completely obliterated after gradients computation

        Foo f(m_Losses, m_Vars);
        Tensor x = zeros(Shape(m_ParamsNum));
        PackParams(m_Vars, x);

        const uint32_t n = m_ParamsNum;
        const size_t fpast = m_param.past;
        Reset(n);

        // Evaluate function and compute gradient
        float fx = f(x, m_Grad);
        
        // Evaluate function and compute gradient
        float xnorm = x.L2Norm();
        float gnorm = m_Grad.L2Norm();
        if (fpast > 0)
            m_fx[0] = fx;

        // Early exit if the initial x is already a minimizer
        if (gnorm <= m_param.epsilon * std::max(xnorm, 1.f))
        {
            return;// 1;
        }

        // Initial direction
        m_Drt = m_Grad.Negated();
        // Initial step
        float step = 1.f / m_Drt.L2Norm();

        size_t k = 1;
        size_t end = 0;
        for (; ; )
        {
            // Save the current x and gradient
            m_xp = x;
            m_GradP = m_Grad;

            // Line search to update x, fx and gradient
            LineSearchBacktracking::LineSearch(f, fx, x, m_Grad, step, m_Drt, m_xp, m_param);

            // New x norm and gradient norm
            xnorm = x.L2Norm();
            gnorm = m_Grad.L2Norm();

            // Convergence test -- gradient
            if (gnorm <= m_param.epsilon * std::max(xnorm, 1.f))
            {
                return;// k;
            }

            // Convergence test -- objective function value
            if (fpast > 0)
            {
                if (k >= fpast && std::abs((m_fx[k % fpast] - fx) / fx) < m_param.delta)
                    return;// k;

                m_fx[k % fpast] = fx;
            }
            // Maximum number of iterations
            if (m_param.max_iterations != 0 && k >= m_param.max_iterations)
            {
                return;// k;
            }

            // Update s and y
            // s_{k+1} = x_{k+1} - x_k
            // y_{k+1} = g_{k+1} - g_k
            Tensor& svec = m_s[end];
            Tensor& yvec = m_y[end];
            x.Sub(m_xp, svec);
            m_Grad.Sub(m_GradP, yvec);

            // ys = y's = 1/rho
            // yy = y'y
            float ys = yvec.Dot(svec);
            float yy = yvec.SquaredL2Norm();
            m_ys[end] = ys;

            // Recursive formula to compute d = -H * g
            m_Drt = -m_Grad;
            size_t bound = std::min(m_param.m, k);
            end = (end + 1) % m_param.m;
            size_t j = end;
            for (size_t i = 0; i < bound; i++)
            {
                j = (j + m_param.m - 1) % m_param.m;
                const Tensor& sj = m_s[j];
                const Tensor& yj = m_y[j];
                m_alpha[j] = sj.Dot(m_Drt) / m_ys[j];
                m_Drt.Sub(yj.Mul(m_alpha[j]), m_Drt);
            }

            m_Drt.Mul(ys / yy, m_Drt);

            for (int i = 0; i < bound; i++)
            {
                const Tensor& sj = m_s[j];
                const Tensor& yj = m_y[j];
                float beta = yj.Dot(m_Drt) / m_ys[j];
                m_Drt.Add(sj.Mul(m_alpha[j] - beta), m_Drt);
                j = (j + 1) % m_param.m;
            }

            // step = 1.0 as initial guess
            step = 1.f;
            k++;
        }
    }

    //////////////////////////////////////////////////////////////////////////
    //void LBFGS::MinimizationOperation::store_curvature_pair(self, s, y)
    //{
    //    """Updates the L-BFGS memory with a new curvature pair."""
    //        sy = sdot(s, y)
    //        if sy > 1e-10:
    //    self.sk.append(s)
    //        self.yk.append(y)
    //        self.syk.append(sy)
    //        if len(self.sk) > self.n_corr:
    //    self.sk, self.yk, self.syk = self.sk[1:], self.yk[1:], self.syk[1:]
    //}

    ////////////////////////////////////////////////////////////////////////////
    //Tensor LBFGS::MinimizationOperation::inv_hv(Variable* var)
    //{
    //    /// Computes the product of a vector with an approximation of the inverse Hessian.
    //    p = p.copy();
    //    alphas = [];

    //    for (int i = sk.size() - 1; i >= 0; ++i)
    //    {
    //        auto& s = sk[i];
    //        auto& y = yk[i];
    //        auto& sy = syk[i];

    //        alphas.append(sdot(s, p) / sy)
    //            saxpy(-alphas[-1], y, p)
    //    }
    //            
    //            

    //    if self.sk :
    //        sy, y = self.syk[-1], self.yk[-1]
    //        p *= sy / sdot(y, y)

    //        for (int i = sk.size() - 1; i >= 0; ++i)
    //        {
    //            auto& s = sk[i];
    //            auto& y = yk[i];
    //            auto& sy = syk[i];
    //            auto& alpha = alphas[i];
    //    for s, y, sy, alpha in zip(self.sk, self.yk, self.syk, reversed(alphas)) :
    //        beta = sdot(y, p) / sy
    //        saxpy(alpha - beta, s, p)

    //        return p
    //}

    ////////////////////////////////////////////////////////////////////////
    void LBFGS::MinimizationOperation::Reset(uint32_t n)
    {
        // n is number of parameters
        // m is number of correction steps
        const size_t m = m_param.m;
        m_s.resize(m); fill(m_s.begin(), m_s.end(), zeros(Shape(n)));
        m_y.resize(m); fill(m_y.begin(), m_y.end(), zeros(Shape(n)));
        m_ys.resize(m);
        m_alpha.resize(m);
        m_xp.Resize(n);
        m_Grad.Resize(n);
        m_GradP.Resize(n);
        m_Drt.Resize(n);
        if (m_param.past > 0)
            m_fx.resize(m_param.past);
    }

    ////////////////////////////////////////////////////////////////////////////
    //int LBFGS::MinimizationOperation::minimize(Foo& f, Tensor& x, float& fx)
    //{
    //    const int n = x.size();
    //    const int fpast = m_param.past;
    //    reset(n);

    //    // Evaluate function and compute gradient
    //    fx = f(x, m_grad);
    //    float xnorm = x.Norm();
    //    float gnorm = m_grad.Norm();
    //    if (fpast > 0)
    //        m_fx(0) = fx;

    //    // Early exit if the initial x is already a minimizer
    //    if (gnorm <= m_param.epsilon * std::max(xnorm, float(1.0)))
    //    {
    //        return 1;
    //    }

    //    // Initial direction
    //    m_drt = m_grad.Negated();
    //    // Initial step
    //    float step = 1.f / m_drt.Norm();

    //    int k = 1;
    //    int end = 0;
    //    for (; ; )
    //    {
    //        // Save the current x and gradient
    //        m_xp = x;
    //        m_gradp = m_grad;

    //        // Line search to update x, fx and gradient
    //        LineSearch<float>::LineSearch(f, fx, x, m_grad, step, m_drt, m_xp, m_param);

    //        // New x norm and gradient norm
    //        xnorm = x.Norm();
    //        gnorm = m_grad.Norm();

    //        // Convergence test -- gradient
    //        if (gnorm <= m_param.epsilon * std::max(xnorm, float(1.0)))
    //        {
    //            return k;
    //        }
    //        // Convergence test -- objective function value
    //        if (fpast > 0)
    //        {
    //            if (k >= fpast && std::abs((m_fx[k % fpast] - fx) / fx) < m_param.delta)
    //                return k;

    //            m_fx[k % fpast] = fx;
    //        }
    //        // Maximum number of iterations
    //        if (m_param.max_iterations != 0 && k >= m_param.max_iterations)
    //        {
    //            return k;
    //        }

    //        // Update s and y
    //        // s_{k+1} = x_{k+1} - x_k
    //        // y_{k+1} = g_{k+1} - g_k
    //        MapVec svec(&m_s(0, end), n);
    //        MapVec yvec(&m_y(0, end), n);
    //        svec.noalias() = x - m_xp;
    //        yvec.noalias() = m_grad - m_gradp;

    //        // ys = y's = 1/rho
    //        // yy = y'y
    //        float ys = yvec.dot(svec);
    //        float yy = yvec.squaredNorm();
    //        m_ys[end] = ys;

    //        // Recursive formula to compute d = -H * g
    //        m_drt.noalias() = -m_grad;
    //        int bound = std::min(m_param.m, k);
    //        end = (end + 1) % m_param.m;
    //        int j = end;
    //        for (int i = 0; i < bound; i++)
    //        {
    //            j = (j + m_param.m - 1) % m_param.m;
    //            MapVec sj(&m_s(0, j), n);
    //            MapVec yj(&m_y(0, j), n);
    //            m_alpha[j] = sj.dot(m_drt) / m_ys[j];
    //            m_drt.noalias() -= m_alpha[j] * yj;
    //        }

    //        m_drt *= (ys / yy);

    //        for (int i = 0; i < bound; i++)
    //        {
    //            MapVec sj(&m_s(0, j), n);
    //            MapVec yj(&m_y(0, j), n);
    //            float beta = yj.dot(m_drt) / m_ys[j];
    //            m_drt += (m_alpha[j] - beta) * sj;
    //            j = (j + 1) % m_param.m;
    //        }

    //        // step = 1.0 as initial guess
    //        step = float(1.0);
    //        k++;
    //    }

    //    return k;
    //}

    ////////////////////////////////////////////////////////////////////////////
    //void LBFGS::MinimizationOperation::LBFGSParam::check_param() const
    //{
    //    if (m <= 0)
    //        throw std::invalid_argument("'m' must be positive");
    //    if (epsilon <= 0)
    //        throw std::invalid_argument("'epsilon' must be positive");
    //    if (past < 0)
    //        throw std::invalid_argument("'past' must be non-negative");
    //    if (delta < 0)
    //        throw std::invalid_argument("'delta' must be non-negative");
    //    if (max_iterations < 0)
    //        throw std::invalid_argument("'max_iterations' must be non-negative");
    //    if (linesearch < LBFGS_LINESEARCH_BACKTRACKING_ARMIJO ||
    //        linesearch > LBFGS_LINESEARCH_BACKTRACKING_STRONG_WOLFE)
    //        throw std::invalid_argument("unsupported line search algorithm");
    //    if (max_linesearch <= 0)
    //        throw std::invalid_argument("'max_linesearch' must be positive");
    //    if (min_step < 0)
    //        throw std::invalid_argument("'min_step' must be positive");
    //    if (max_step < min_step)
    //        throw std::invalid_argument("'max_step' must be greater than 'min_step'");
    //    if (ftol <= 0 || ftol >= 0.5)
    //        throw std::invalid_argument("'ftol' must satisfy 0 < ftol < 0.5");
    //    if (wolfe <= ftol || wolfe >= 1)
    //        throw std::invalid_argument("'wolfe' must satisfy ftol < wolfe < 1");
    //}

}
