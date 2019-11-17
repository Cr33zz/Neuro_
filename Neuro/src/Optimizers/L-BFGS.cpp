#include <sstream>
#include <iomanip>

#include "Optimizers/L-BFGS.h"
#include "Tensors/TensorOpCpu.h"
#include "ComputationalGraph/Variable.h"
#include "ComputationalGraph/Graph.h"

namespace Neuro
{
    //////////////////////////////////////////////////////////////////////////
    L_BFGS::L_BFGS(float lr)
    {
        m_LearningRate = lr;
    }

    //////////////////////////////////////////////////////////////////////////
    OptimizerBase* L_BFGS::Clone() const
    {
        return new L_BFGS(*this);
    }

    //////////////////////////////////////////////////////////////////////////
    string L_BFGS::ToString()
    {
        stringstream ss;
        ss << setprecision(5) << "L-BFGS(lr=" << m_LearningRate << ")";
        return ss.str();
    }

    //////////////////////////////////////////////////////////////////////////
    const char* L_BFGS::ClassName() const
    {
        return "L_BFGS";
    }

    //////////////////////////////////////////////////////////////////////////
    L_BFGS::MinimizationOperation::MinimizationOperation(const vector<TensorLike*>& losses, const vector<Variable*>& vars, float maxIterations, float epsilon)
        : Operation(losses, "L_BFGS_minimize"), m_Vars(vars)
    {
        m_param.epsilon = epsilon;
        m_param.max_iterations = maxIterations;
        m_Order = Graph::Default()->BuildBackwardOrder(losses, m_NodesAffectingLosses, vars);
    }

    //////////////////////////////////////////////////////////////////////////
    void L_BFGS::MinimizationOperation::ComputeInternal()
    {
        m_InputsManuallyConsumed = true; // loss outputs will be completely obliterated after gradients computation
        auto vars = Graph::Default()->ComputeGradientsInOrder(m_Order, m_InputNodes, m_NodesAffectingLosses, m_Vars);

        //for (auto v : vars)
        //    Tensor::ActiveOp()->L_BFGSStep(v->Output(), v->OutputGrad(), /*batchSize, */m_LearningRate);
    }

    //////////////////////////////////////////////////////////////////////////
    void L_BFGS::MinimizationOperation::reset(int n)
    {
        const int m = m_param.m;
        m_s.Resize(Shape(n, m));
        m_y.Resize(Shape(n, m));
        m_ys.Resize(m);
        m_alpha.Resize(m);
        m_xp.Resize(n);
        m_grad.Resize(n);
        m_gradp.Resize(n);
        m_drt.Resize(n);
        if (m_param.past > 0)
            m_fx.Resize(m_param.past);
    }

    //////////////////////////////////////////////////////////////////////////
    int L_BFGS::MinimizationOperation::minimize(Foo& f, Tensor& x, float& fx)
    {
        const int n = x.size();
        const int fpast = m_param.past;
        reset(n);

        // Evaluate function and compute gradient
        fx = f(x, m_grad);
        float xnorm = x.Norm();
        float gnorm = m_grad.Norm();
        if (fpast > 0)
            m_fx(0) = fx;

        // Early exit if the initial x is already a minimizer
        if (gnorm <= m_param.epsilon * std::max(xnorm, float(1.0)))
        {
            return 1;
        }

        // Initial direction
        m_drt = m_grad.Negated();
        // Initial step
        float step = 1.f / m_drt.Norm();

        int k = 1;
        int end = 0;
        for (; ; )
        {
            // Save the current x and gradient
            m_xp = x;
            m_gradp = m_grad;

            // Line search to update x, fx and gradient
            LineSearch<float>::LineSearch(f, fx, x, m_grad, step, m_drt, m_xp, m_param);

            // New x norm and gradient norm
            xnorm = x.Norm();
            gnorm = m_grad.Norm();

            // Convergence test -- gradient
            if (gnorm <= m_param.epsilon * std::max(xnorm, float(1.0)))
            {
                return k;
            }
            // Convergence test -- objective function value
            if (fpast > 0)
            {
                if (k >= fpast && std::abs((m_fx[k % fpast] - fx) / fx) < m_param.delta)
                    return k;

                m_fx[k % fpast] = fx;
            }
            // Maximum number of iterations
            if (m_param.max_iterations != 0 && k >= m_param.max_iterations)
            {
                return k;
            }

            // Update s and y
            // s_{k+1} = x_{k+1} - x_k
            // y_{k+1} = g_{k+1} - g_k
            MapVec svec(&m_s(0, end), n);
            MapVec yvec(&m_y(0, end), n);
            svec.noalias() = x - m_xp;
            yvec.noalias() = m_grad - m_gradp;

            // ys = y's = 1/rho
            // yy = y'y
            float ys = yvec.dot(svec);
            float yy = yvec.squaredNorm();
            m_ys[end] = ys;

            // Recursive formula to compute d = -H * g
            m_drt.noalias() = -m_grad;
            int bound = std::min(m_param.m, k);
            end = (end + 1) % m_param.m;
            int j = end;
            for (int i = 0; i < bound; i++)
            {
                j = (j + m_param.m - 1) % m_param.m;
                MapVec sj(&m_s(0, j), n);
                MapVec yj(&m_y(0, j), n);
                m_alpha[j] = sj.dot(m_drt) / m_ys[j];
                m_drt.noalias() -= m_alpha[j] * yj;
            }

            m_drt *= (ys / yy);

            for (int i = 0; i < bound; i++)
            {
                MapVec sj(&m_s(0, j), n);
                MapVec yj(&m_y(0, j), n);
                float beta = yj.dot(m_drt) / m_ys[j];
                m_drt += (m_alpha[j] - beta) * sj;
                j = (j + 1) % m_param.m;
            }

            // step = 1.0 as initial guess
            step = float(1.0);
            k++;
        }

        return k;
    }

    //////////////////////////////////////////////////////////////////////////
    L_BFGS::MinimizationOperation::LBFGSParam::LBFGSParam()
    {
        m = 6;
        epsilon = float(1e-5);
        past = 0;
        delta = float(0);
        max_iterations = 0;
        linesearch = LBFGS_LINESEARCH_BACKTRACKING_ARMIJO;
        max_linesearch = 20;
        min_step = float(1e-20);
        max_step = float(1e+20);
        ftol = float(1e-4);
        wolfe = float(0.9);
    }

    //////////////////////////////////////////////////////////////////////////
    void L_BFGS::MinimizationOperation::LBFGSParam::check_param() const
    {
        if (m <= 0)
            throw std::invalid_argument("'m' must be positive");
        if (epsilon <= 0)
            throw std::invalid_argument("'epsilon' must be positive");
        if (past < 0)
            throw std::invalid_argument("'past' must be non-negative");
        if (delta < 0)
            throw std::invalid_argument("'delta' must be non-negative");
        if (max_iterations < 0)
            throw std::invalid_argument("'max_iterations' must be non-negative");
        if (linesearch < LBFGS_LINESEARCH_BACKTRACKING_ARMIJO ||
            linesearch > LBFGS_LINESEARCH_BACKTRACKING_STRONG_WOLFE)
            throw std::invalid_argument("unsupported line search algorithm");
        if (max_linesearch <= 0)
            throw std::invalid_argument("'max_linesearch' must be positive");
        if (min_step < 0)
            throw std::invalid_argument("'min_step' must be positive");
        if (max_step < min_step)
            throw std::invalid_argument("'max_step' must be greater than 'min_step'");
        if (ftol <= 0 || ftol >= 0.5)
            throw std::invalid_argument("'ftol' must satisfy 0 < ftol < 0.5");
        if (wolfe <= ftol || wolfe >= 1)
            throw std::invalid_argument("'wolfe' must satisfy ftol < wolfe < 1");
    }

}
