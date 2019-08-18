#include "Optimizers/OptimizerBase.h"

namespace Neuro
{
	//////////////////////////////////////////////////////////////////////////
	void OptimizerBase::Step(vector<ParametersAndGradients>& paramsAndGrads, int batchSize)
	{
		++m_Iteration;
		OnStep(paramsAndGrads, batchSize);
	}
}
