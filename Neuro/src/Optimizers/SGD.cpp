#include <sstream>

#include "Optimizers/SGD.h"
#include "Tensors/Tensor.h"

namespace Neuro
{
	//////////////////////////////////////////////////////////////////////////
	SGD::SGD(float lr /*= 0.01f*/)
	{
		LearningRate = lr;
	}

	//////////////////////////////////////////////////////////////////////////
	std::string SGD::ToString()
	{
		stringstream ss;
		ss << "SGD(lr=" << LearningRate << ")";
		return ss.str();
	}

	//////////////////////////////////////////////////////////////////////////
	const char* SGD::ClassName() const
	{
		return "SGD";
	}

	//////////////////////////////////////////////////////////////////////////
	void SGD::OnStep(vector<ParametersAndGradients>& paramsAndGrads, int batchSize)
	{
		for (int i = 0; i < paramsAndGrads.size(); ++i)
		{
			auto& parametersAndGradient = paramsAndGrads[i];
			auto parameters = parametersAndGradient.Parameters;
			auto gradients = parametersAndGradient.Gradients;

			float tempLearningRate = LearningRate / batchSize;

			gradients->Mul(tempLearningRate, *gradients);
			parameters->Sub(*gradients, *parameters);

			gradients->Zero();
		}
	}

}
