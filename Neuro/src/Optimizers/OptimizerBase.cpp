#include "Optimizers/OptimizerBase.h"

namespace Neuro
{
	//////////////////////////////////////////////////////////////////////////
	void OptimizerBase::Step(vector<ParameterAndGradient>& paramsAndGrads, int batchSize)
	{
		++m_Iteration;
		OnStep(paramsAndGrads, batchSize);
	}
}
