#include <sstream>
#include <iomanip>

#include "ComputationalGraph/Session.h"
#include "Optimizers/LBFGS.h"
#include "Tensors/TensorOpCpu.h"
#include "ComputationalGraph/Variable.h"
#include "ComputationalGraph/Graph.h"
#include "ComputationalGraph/Operations/GradientsOp.h"
#include "Tools.h"

namespace Neuro
{
    ////////////////////////////////////////////////////////////////////////
    Foo::Foo(TensorLike* loss, const vector<Variable*>& vars) : m_Vars(vars)
    {
        m_Fetches = MergeVectors({ vector<TensorLike*>{ loss }, gradients(loss, vars) });
    }

    ////////////////////////////////////////////////////////////////////////
    float Foo::operator()(const Tensor& x, Tensor& grad)
    {
        UnPackParams(x, m_Vars);
        auto res = Session::Default()->Run(m_Fetches);

        size_t gradOffset = 0;
        for (size_t i = 1; i < res.size(); ++i)
        {
            res[i]->CopyTo(0, grad, gradOffset, res[i]->Length());
            gradOffset += res[i]->Length();
        }
        //PackGrads(m_Vars, grad);
        return (*res[0])(0);
    }

    //////////////////////////////////////////////////////////////////////////
    class LineSearchBacktracking
    {
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
        static void LineSearch(
            Foo& f, 
            float& fx, 
            Tensor& x, 
            Tensor& grad,
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
            const float fxInit = fx;
            // Projection of gradient on the search direction
            const float dgInit = grad.Dot(drt);
            // Make sure d points to a descent direction
            if (dgInit > 0)
                std::logic_error("the moving direction increases the objective function value");

            const float dgTest = param.ftol * dgInit;
            float width = 0;

            for (int iter = 0; iter < param.max_linesearch; ++iter)
            {
                // x_{k+1} = x_k + step * d_k
                x = xp.Add(drt.Mul(step));
                // Evaluate this candidate
                fx = f(x, grad);

                if (fx > fxInit + step * dgTest)
                {
                    width = dec;
                }
                else
                {
                    // Armijo condition is met
                    if (param.linesearch == LBFGS::LINESEARCH_BACKTRACKING_ARMIJO)
                        break;

                    const float dg = grad.Dot(drt);
                    if (dg < param.wolfe * dgInit)
                    {
                        width = inc;
                    }
                    else
                    {
                        // Regular Wolfe condition is met
                        if (param.linesearch == LBFGS::LINESEARCH_BACKTRACKING_WOLFE)
                            break;

                        if (dg > -param.wolfe * dgInit)
                            width = dec;
                        else // Strong Wolfe condition is met
                            break;
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
    LBFGS::LBFGS(/*size_t maxIterations, */float epsilon)
        : /*m_MaxIterations(maxIterations), */m_Epsilon(epsilon)
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
        : Operation({}, "L_BFGS_minimize"), m_Vars(vars), m_Loss(losses[0])
    {
        NEURO_ASSERT(losses.size() == 1 && losses[0]->Output().Length() == 1, "L-BFGS expects single loss returning scalar value.");
        m_Param.epsilon = epsilon;
        m_Param.max_iterations = maxIterations;

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

         m_F = Foo(m_Loss, m_Vars);
         m_X = zeros(Shape(m_ParamsNum));

         Reset(m_ParamsNum);
    }

    //////////////////////////////////////////////////////////////////////////
    void LBFGS::MinimizationOperation::ComputeInternal()
    {
        m_InputsManuallyConsumed = true; // loss outputs will be completely obliterated after gradients computation

        if (m_Done)
            return;

        const size_t past = m_Param.past;
        
        if (m_Iter == 1)
        {
            PackParams(m_Vars, m_X);

            // Evaluate function and compute gradient
            m_LocalFx = m_F(m_X, m_Grad);

            // Evaluate function and compute gradient
            auto xnorm = m_X.L2Norm();
            auto gnorm = m_Grad.L2Norm();

            if (past > 0)
                m_Fx[0] = m_LocalFx;

            // Early exit if the initial x is already a minimizer
            if (gnorm <= m_Param.epsilon * std::max(xnorm, 1.f))
            {
                m_Done = true;
                return;
            }

            // Initial direction
            m_Grad.Negated(m_Drt);
            // Initial step
            m_Step = 1.f / m_Drt.L2Norm();
        }

        // Save the current x and gradient
        m_PrevX = m_X;
        m_PrevGrad = m_Grad;

        // Line search to update x, fx and gradient
        LineSearchBacktracking::LineSearch(m_F, m_LocalFx, m_X, m_Grad, m_Step, m_Drt, m_PrevX, m_Param);

        // New x norm and gradient norm
        auto xnorm = m_X.L2Norm();
        auto gnorm = m_Grad.L2Norm();

        // Convergence test -- gradient
        if (gnorm <= m_Param.epsilon * std::max(xnorm, 1.f))
        {
            m_Done = true;
            return;
        }

        // Convergence test -- objective function value
        if (past > 0)
        {
            if (m_Iter >= past && std::abs((m_Fx[m_Iter % past] - m_LocalFx) / m_LocalFx) < m_Param.delta)
            {
                m_Done = true;
                return;
            }

            m_Fx[m_Iter % past] = m_LocalFx;
        }
        // Maximum number of iterations
        if (m_Param.max_iterations != 0 && m_Iter >= m_Param.max_iterations)
        {
            m_Done = true;
            return;
        }

        // Update s and y
        // s_{k+1} = x_{k+1} - x_k
        // y_{k+1} = g_{k+1} - g_k
        Tensor& svec = m_S[m_End];
        Tensor& yvec = m_Y[m_End];
        m_X.Sub(m_PrevX, svec);
        m_Grad.Sub(m_PrevGrad, yvec);

        // ys = y's = 1/rho
        // yy = y'y
        float ys = yvec.Dot(svec);
        float yy = yvec.SquaredL2Norm();
        m_YS[m_End] = ys;

        // Recursive formula to compute d = -H * g
        m_Grad.Negated(m_Drt);
        size_t bound = std::min(m_Param.m, m_Iter);
        m_End = (m_End + 1) % m_Param.m;
        size_t j = m_End;
        for (size_t i = 0; i < bound; i++)
        {
            j = (j + m_Param.m - 1) % m_Param.m;
            const Tensor& sj = m_S[j];
            const Tensor& yj = m_Y[j];
            m_Alpha[j] = sj.Dot(m_Drt) / m_YS[j];
            m_Drt.Sub(yj.Mul(m_Alpha[j]), m_Drt);
        }

        m_Drt.Mul(ys / yy, m_Drt);

        for (int i = 0; i < bound; i++)
        {
            const Tensor& sj = m_S[j];
            const Tensor& yj = m_Y[j];
            float beta = yj.Dot(m_Drt) / m_YS[j];
            m_Drt.Add(sj.Mul(m_Alpha[j] - beta), m_Drt);
            j = (j + 1) % m_Param.m;
        }

        // step = 1.0 as initial guess
        m_Step = 1.f;
        ++m_Iter;
    }

    ////////////////////////////////////////////////////////////////////////
    void LBFGS::MinimizationOperation::Reset(uint32_t n)
    {
        // n is number of parameters
        // m is number of correction steps
        const size_t m = m_Param.m;
        m_S.resize(m); fill(m_S.begin(), m_S.end(), zeros(Shape(n)));
        m_Y.resize(m); fill(m_Y.begin(), m_Y.end(), zeros(Shape(n)));
        m_YS.resize(m);
        m_Alpha.resize(m);
        m_PrevX.Resize(n);
        m_Grad.Resize(n);
        m_PrevGrad.Resize(n);
        m_Drt.Resize(n);
        if (m_Param.past > 0)
            m_Fx.resize(m_Param.past);
    }
}
