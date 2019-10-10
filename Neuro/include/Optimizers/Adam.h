﻿#pragma once

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

        virtual Operation* Minimize(const vector<TensorLike*>& losses, const vector<Variable*>& vars = {}) override { return new MinimizationOperation(losses, vars, this); }

    private:
        class MinimizationOperation : public Operation
        {
        public:
            MinimizationOperation(const vector<TensorLike*>& losses, const vector<Variable*>& vars, Adam* owner);
        protected:
            virtual void ComputeInternal();
            virtual void ComputeGradientInternal(const Tensor& grad) {}

            Adam* m_Owner;
            vector<Variable*> m_Vars;
            vector<Tensor> m_MGradients;
            vector<Tensor> m_VGradients;
            vector<TensorLike*> m_Order;
        };

    private:
        float m_LearningRate;
        float m_Beta1;
        float m_Beta2;
        float m_Epsilon;

        friend class MinimizationOperation;
	};
}
