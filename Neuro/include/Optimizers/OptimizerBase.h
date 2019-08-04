#pragma once

#include <vector>
#include <string>

#include "ParametersAndGradients.h"

namespace Neuro
{
	using namespace std;

    class OptimizerBase
    {
	public:
        void Step(vector<ParametersAndGradients>& paramsAndGrads, int batchSize);
		virtual string ToString() = 0;

	protected:
        virtual void OnStep(vector<ParametersAndGradients>& paramsAndGrads, int batchSize) = 0;

		float Iteration;
	};
}
