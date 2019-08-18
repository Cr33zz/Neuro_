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

	protected:
        virtual void OnStep(vector<ParametersAndGradients>& paramsAndGrads, int batchSize) override;

	private:
        float m_LearningRate;
	};
}
