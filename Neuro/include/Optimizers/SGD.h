#pragma once

#include "Optimizers/OptimizerBase.h"

namespace Neuro
{
    class SGD : public OptimizerBase
    {
	public:
        SGD(float lr = 0.01f);
        virtual OptimizerBase* Clone() const override;
		virtual string ToString() override;
		const char* ClassName() const;

        virtual Operation* Minimize(const vector<TensorLike*>& losses, const vector<Variable*>& vars = {}) override { return new MinimizationOperation(losses, vars, this); }

    private:
        class MinimizationOperation : public Operation
        {
        public:
            MinimizationOperation(const vector<TensorLike*>& losses, const vector<Variable*>& vars, SGD* owner);
        protected:
            virtual void ComputeInternal();
            virtual void ComputeGradientInternal(const Tensor& grad) {}

        private:
            SGD* m_Owner;
            vector<Variable*> m_Vars;
            vector<TensorLike*> m_Order;
        };

        float m_LearningRate;

        friend class MinimizationOperation;
	};
}
