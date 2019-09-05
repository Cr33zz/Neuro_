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
        virtual OptimizerBase* Clone() const = 0;
		virtual string ToString() = 0;
		virtual const char* ClassName() const = 0;
        int Iteration() const { return (int)m_Iteration; }

	protected:
        virtual void OnStep(vector<ParametersAndGradients>& paramsAndGrads, int batchSize) = 0;

		float m_Iteration = 0;
	};
}
