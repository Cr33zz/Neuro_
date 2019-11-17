#pragma once

#include <unordered_set>

#include "Optimizers/OptimizerBase.h"

namespace Neuro
{
    class L_BFGS : public OptimizerBase
    {
    public:
        L_BFGS(float lr = 0.01f);
        virtual OptimizerBase* Clone() const override;
        virtual string ToString() override;
        const char* ClassName() const;

        virtual Operation* Minimize(const Tensor<TensorLike*>& losses, const Tensor<Variable*>& vars = {}, Variable* globalStep = nullptr) override { return new MinimizationOperation(losses, vars, m_MaxIterations, m_Epsilon); }

    private:
        class MinimizationOperation : public Operation
        {
        public:
            MinimizationOperation(const Tensor<TensorLike*>& losses, const Tensor<Variable*>& vars, float maxIterations, float epsilon);
        protected:
            virtual void ComputeInternal();
            virtual void ComputeGradientInternal(const Tensor& grad) {}

        private:
            /*typedef Eigen::Matrix<float, Eigen::Dynamic, 1> Tensor;
            typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> Matrix;
            typedef Eigen::Map<Tensor> MapVec;*/

            enum LINE_SEARCH_ALGORITHM
            {
                LBFGS_LINESEARCH_BACKTRACKING_ARMIJO = 1,
                LBFGS_LINESEARCH_BACKTRACKING = 2,
                LBFGS_LINESEARCH_BACKTRACKING_WOLFE = 2,
                LBFGS_LINESEARCH_BACKTRACKING_STRONG_WOLFE = 3
            };

            /// Parameters to control the LBFGS algorithm.
            class LBFGSParam
            {
            public:
                int m;
                float epsilon;
                int past;
                float delta;
                int max_iterations;
                int linesearch;
                int max_linesearch;
                float min_step;
                float max_step;
                float ftol;
                float wolfe;

                LBFGSParam();

                void check_param() const;
            };

            LBFGSParam m_param;  // Parameters to control the LBFGS algorithm
            Tensor m_s;      // History of the s Tensors
            Tensor m_y;      // History of the y Tensors
            Tensor m_ys;     // History of the s'y values
            Tensor m_alpha;  // History of the step lengths
            Tensor m_fx;     // History of the objective function values
            Tensor m_xp;     // Old x
            Tensor m_grad;   // New gradient
            Tensor m_gradp;  // Old gradient
            Tensor m_drt;    // Moving direction

            void reset(int n);

        public:
            int minimize(Foo& f, Tensor& x, float& fx);

        private:
            float m_LearningRate;
            Tensor<Variable*> m_Vars;
            Tensor<TensorLike*> m_Order;
            unordered_set<TensorLike*> m_NodesAffectingLosses;
        };

        int m_MaxIterations;
        float m_Epsilon;

        friend class MinimizationOperation;
    };
}
