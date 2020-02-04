#pragma once

#include <unordered_set>

#include "Optimizers/OptimizerBase.h"

namespace Neuro
{
    // Limited-memory Broyden-Fletcher-Goldfarb-Shanno optimizer
    class LBFGS : public OptimizerBase
    {
    public:
        LBFGS(size_t maxIterations = 100, float epsilon = 1e-6f);
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

            LBFGS::Param m_param;  // Parameters to control the LBFGS algorithm
            vector<Tensor> m_s;      // History of the s vectors
            vector<Tensor> m_y;      // History of the y vectors
            vector<float> m_ys;     // History of the s'y values
            vector<float> m_alpha;  // History of the step lengths
            vector<float> m_fx;     // History of the objective function values
            Tensor m_xp;     // Old x
            Tensor m_Grad;   // New gradient
            Tensor m_GradP;  // Old gradient
            Tensor m_Drt;    // Moving direction
            uint32_t m_ParamsNum;

            vector<Variable*> m_Vars;
            vector<TensorLike*> m_Losses;
        };

        size_t m_MaxIterations;
        float m_Epsilon;

        friend class MinimizationOperation;
    };
}
