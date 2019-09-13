#include <sstream>
#include <iomanip>

#include "Optimizers/SGD.h"
#include "Tensors/Tensor.h"
#include "Tensors/TensorOpCpu.h"

namespace Neuro
{
	//////////////////////////////////////////////////////////////////////////
	SGD::SGD(float lr /*= 0.01f*/)
	{
		m_LearningRate = lr;
	}

    //////////////////////////////////////////////////////////////////////////
    Neuro::OptimizerBase* SGD::Clone() const
    {
        return new SGD(*this);
    }

    //////////////////////////////////////////////////////////////////////////
	std::string SGD::ToString()
	{
		stringstream ss;
		ss << setprecision(5) << "SGD(lr=" << m_LearningRate << ")";
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
        float learningRate = m_LearningRate;

		for (auto i = 0; i < paramsAndGrads.size(); ++i)
		{
			auto& parametersAndGradient = paramsAndGrads[i];
			auto parameter = parametersAndGradient.Parameters;
			auto gradient = parametersAndGradient.Gradients;

            Tensor::ActiveOp()->SgdStep(*parameter, *gradient, (float)batchSize, learningRate);
		}
	}

}
