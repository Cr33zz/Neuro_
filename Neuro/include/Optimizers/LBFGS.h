#pragma once

#include <unordered_set>

#include "Optimizers/OptimizerBase.h"

namespace Neuro
{
    // Computational graph evaluation wrapper, used to compute total loss and gradients at any
    // point of parameters space
    class Foo
    {
    public:
        Foo() {}
        Foo(TensorLike* loss, const vector<Variable*>& vars);

        // Evaluate computational graph for the purpose of l-bfgs optimization step
        // both x and grad contain flatten and merged parameters and their gradients
        float operator()(const Tensor& x, Tensor& grad);

    private:
        vector<Variable*> m_Vars;
        vector<TensorLike*> m_Fetches;
    };

    // Limited-memory Broyden-Fletcher-Goldfarb-Shanno optimizer
    // Implementation based on https://github.com/yixuan/LBFGSpp
    class LBFGS : public OptimizerBase
    {
    public:
        LBFGS(/*size_t maxIterations = 100, */float epsilon = 1e-6f);
        virtual OptimizerBase* Clone() const override;
        virtual string ToString() override;
        const char* ClassName() const;

        enum ELineSearchAlgo
        {
            LINESEARCH_BACKTRACKING_ARMIJO = 1,
            LINESEARCH_BACKTRACKING = 2,
            LINESEARCH_BACKTRACKING_WOLFE = 2,
            LINESEARCH_BACKTRACKING_STRONG_WOLFE = 3
        };

        /// https://rdrr.io/cran/lbfgs/man/lbfgs.html
        struct Param
        {
            size_t m = 6;
            float epsilon = 1e-5f;
            size_t past = 0;
            float delta = 0.f;
            size_t max_iterations = 0;
            ELineSearchAlgo linesearch = LINESEARCH_BACKTRACKING_ARMIJO;
            size_t max_linesearch = 20;
            float min_step = 1e-20f;
            float max_step = 1e+20f;
            float ftol = 1e-4f;
            float wolfe = 0.9f;
            //void check_param() const;
        };

        virtual Operation* Minimize(const vector<TensorLike*>& losses, const vector<Variable*>& vars = {}, Variable* globalStep = nullptr) override { return new MinimizationOperation(losses, vars, m_MaxIterations, m_Epsilon); }

    private:
        class MinimizationOperation : public Operation
        {
        public:
            MinimizationOperation(const vector<TensorLike*>& losses, const vector<Variable*>& vars, size_t maxIterations, float epsilon);

        protected:
            virtual void ComputeInternal();
            virtual void ComputeGradientInternal(const Tensor& grad) {}

        private:
            void Reset(uint32_t n);

            LBFGS::Param m_Param;
            vector<Tensor> m_S;
            vector<Tensor> m_Y;
            vector<float> m_YS;
            vector<float> m_Alpha; // history of the step lengths
            vector<float> m_Fx; // history of the objective function values
            Tensor m_PrevX; // old x
            Tensor m_Grad; // new gradient
            Tensor m_PrevGrad; // old gradient
            Tensor m_Drt; // moving direction
            uint32_t m_ParamsNum = 0;
            Foo m_F;
            size_t m_Iter = 1;
            size_t m_End = 0;
            float m_Step;
            Tensor m_X;
            float m_LocalFx;
            bool m_Done = false;

            vector<Variable*> m_Vars;
            TensorLike* m_Loss;
        };

        size_t m_MaxIterations;
        float m_Epsilon;

        friend class MinimizationOperation;
    };
}
