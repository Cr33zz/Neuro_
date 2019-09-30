#pragma once

#include <vector>
#include <string>

#include "ParameterAndGradient.h"

namespace Neuro
{
	using namespace std;

    class OptimizerBase
    {
	public:
        void Step(vector<ParameterAndGradient>& paramsAndGrads, int batchSize);
        virtual OptimizerBase* Clone() const = 0;
		virtual string ToString() = 0;
		virtual const char* ClassName() const = 0;
        int Iteration() const { return (int)m_Iteration; }

	protected:
        virtual void OnStep(vector<ParameterAndGradient>& paramsAndGrads, int batchSize) = 0;

		float m_Iteration = 0;
	};
}
