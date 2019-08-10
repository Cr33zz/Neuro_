#pragma once

#include "Optimizers/OptimizerBase.h"

namespace Neuro
{
    // Implementation based on https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/training/adam.py
    class Adam : public OptimizerBase
    {
	public:
        Adam(float lr = 0.001f);

        virtual void OnStep(vector<ParametersAndGradients>& paramsAndGrads, int batchSize) override;
        virtual OptimizerBase* Clone() const override;
        virtual string ToString() override;
		virtual const char* ClassName() const override;

	private:
        float LearningRate;
        float Beta1 = 0.9f;
        float Beta2 = 0.999f;
        float Epsilon = 1e-8f;

        vector<Tensor> MGradients;
        vector<Tensor> VGradients;
	};
}
