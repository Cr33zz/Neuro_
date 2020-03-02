#pragma once

#include <unordered_set>

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

        virtual Operation* Minimize(const vector<TensorLike*>& losses, const vector<Variable*>& vars = {}, Variable* globalStep = nullptr) override { return new MinimizationOperation(losses, vars, m_LearningRate); }

        class MinimizationOperation : public Operation
        {
        public:
            MinimizationOperation(const vector<TensorLike*>& losses, const vector<Variable*>& vars, float lr);
            virtual bool IsTrainingOp() const override { return true; }
        protected:
            virtual void UpdateOutputShape() override {}
            virtual void ComputeInternal() override;
            virtual void ComputeGradientInternal(const Tensor& grad) override {}

        private:
            float m_LearningRate;
            vector<Variable*> m_Vars;
            vector<TensorLike*> m_Order;
            unordered_set<TensorLike*> m_NodesAffectingLosses;
        };

    private:
        float m_LearningRate;

        friend class MinimizationOperation;
	};
}
