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

        virtual Operation* Minimize(TensorLike* loss) override { return new MinimizationOperation(loss, this); }

    private:
        class MinimizationOperation : public Operation
        {
        public:
            MinimizationOperation(TensorLike* loss, SGD* owner) :Operation({ loss }, "sgd_minimize"), m_Owner(owner) {}
        protected:
            virtual void ComputeInternal();
            virtual void ComputeGradientInternal(const Tensor& grad) {}

            SGD* m_Owner;
        };

        float m_LearningRate;

        friend class MinimizationOperation;
	};
}
