#include <sstream>

#include "Optimizers/Adam.h"
#include "Tensors/Tensor.h"
#include "Tensors/TensorOpCpu.h"

namespace Neuro
{    
	//////////////////////////////////////////////////////////////////////////
    Adam::Adam(float lr, float beta1, float beta2)
	{
		m_LearningRate = lr;
        m_Beta1 = beta1;
        m_Beta2 = beta2;
	}

	//////////////////////////////////////////////////////////////////////////
	void Adam::OnStep(vector<ParametersAndGradients>& paramsAndGrads, int batchSize)
	{
		if (m_MGradients.size() != paramsAndGrads.size())
		{
            assert(m_MGradients.empty() && m_VGradients.empty());
			
            for (uint32_t i = 0; i < paramsAndGrads.size(); ++i)
			{
				auto gradients = paramsAndGrads[i].Gradients;

				m_MGradients.push_back(Tensor(gradients->GetShape()));
				m_VGradients.push_back(Tensor(gradients->GetShape()));
			}
		}

        float learningRate = m_LearningRate * (float)sqrt(1.0 - pow(m_Beta2, m_Iteration)) / (1.0f - (float)pow(m_Beta1, m_Iteration));

		for (uint32_t i = 0; i < paramsAndGrads.size(); ++i)
		{
			auto& parameterAndGradient = paramsAndGrads[i];
			auto parameter = parameterAndGradient.Parameters;
			auto gradient = parameterAndGradient.Gradients;
            auto& mGrad = m_MGradients[i];
			auto& vGrad = m_VGradients[i];

            Tensor::ActiveOp()->AdamStep(*parameter, *gradient, mGrad, vGrad, (float)batchSize, learningRate, m_Beta1, m_Beta2, m_Epsilon);
		}
	}

    //////////////////////////////////////////////////////////////////////////
    OptimizerBase* Adam::Clone() const
    {
        return new Adam(*this);
    }

    //////////////////////////////////////////////////////////////////////////
	string Adam::ToString()
	{
		stringstream ss;
		ss << "Adam(lr=" << m_LearningRate << ")";
		return ss.str();
		//return $"Adam(lr={LearningRate}, beta1={Beta1}, beta2={Beta2}, epsilon={Epsilon})";
	}

	//////////////////////////////////////////////////////////////////////////
	const char* Adam::ClassName() const
	{
		return "Adam";
	}

}
