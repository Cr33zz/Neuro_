#include "Optimizers/OptimizerBase.h"

namespace Neuro
{
	//////////////////////////////////////////////////////////////////////////
	void OptimizerBase::Step(vector<ParametersAndGradients>& paramsAndGrads, int batchSize)
	{
		++Iteration;
		OnStep(paramsAndGrads, batchSize);
	}
}
