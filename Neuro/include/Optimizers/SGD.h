#pragma once

#include "Optimizers/OptimizerBase.h"

namespace Neuro
{
    class SGD : public OptimizerBase
    {
	public:
        SGD(float lr = 0.01f);
		virtual string ToString() override;
		virtual const char* ClassName() const override;

	protected:
        virtual void OnStep(vector<ParametersAndGradients>& paramsAndGrads, int batchSize) override;

	private:
        float LearningRate;
	};
}
