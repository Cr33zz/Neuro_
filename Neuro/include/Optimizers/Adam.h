#pragma once

#include <unordered_set>

#include "Optimizers/OptimizerBase.h"

namespace Neuro
{
    // Implementation based on https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/training/adam.py
    class Adam : public OptimizerBase
    {
	public:
        Adam(float lr = 0.001f, float beta1 = 0.9f, float beta2 = 0.999f, float epsilon = 1e-8f);

        virtual OptimizerBase* Clone() const override;
        virtual string ToString() override;
		const char* ClassName() const;

        virtual Operation* Minimize(const vector<TensorLike*>& losses, const vector<Variable*>& vars = {}) override { return new MinimizationOperation(losses, vars, m_LearningRate, m_Beta1, m_Beta2, m_Epsilon); }

    private:
        class MinimizationOperation : public Operation
        {
        public:
            MinimizationOperation(const vector<TensorLike*>& losses, const vector<Variable*>& vars, float lr, float beta1, float beta2, float epsilon);
        protected:
            virtual void ComputeInternal();
            virtual void ComputeGradientInternal(const Tensor& grad) {}

            float m_LearningRate;
            float m_Beta1;
            float m_Beta2;
            float m_Epsilon;
            vector<Variable*> m_Vars;
            vector<Tensor> m_MGradients;
            vector<Tensor> m_VGradients;
            vector<TensorLike*> m_Order;
            unordered_set<TensorLike*> m_NodesAffectingLosses;
            float m_Iteration = 0;
        };

    private:
        float m_LearningRate;
        float m_Beta1;
        float m_Beta2;
        float m_Epsilon;

        friend class MinimizationOperation;
	};
}
