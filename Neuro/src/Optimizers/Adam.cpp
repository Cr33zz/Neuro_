#include <sstream>

#include "Optimizers/Adam.h"
#include "Tensors/Tensor.h"

namespace Neuro
{    
	//////////////////////////////////////////////////////////////////////////
	Adam::Adam(float lr)
	{
		m_LearningRate = lr;
	}

	//////////////////////////////////////////////////////////////////////////
	void Adam::OnStep(vector<ParametersAndGradients>& paramsAndGrads, int batchSize)
	{
		if (m_MGradients.size() != paramsAndGrads.size())
		{
			for (uint i = 0; i < paramsAndGrads.size(); ++i)
			{
				auto gradients = paramsAndGrads[i].Gradients;

				m_MGradients.push_back(Tensor(gradients->GetShape()));
				m_VGradients.push_back(Tensor(gradients->GetShape()));
			}
		}

		for (uint i = 0; i < paramsAndGrads.size(); ++i)
		{
			auto& parametersAndGradient = paramsAndGrads[i];
			auto parameters = parametersAndGradient.Parameters;
			auto gradients = parametersAndGradient.Gradients;
			auto& mGrad = m_MGradients[i];
			auto& vGrad = m_VGradients[i];

			gradients->Div((float)batchSize, *gradients);

			float tempLearningRate = m_LearningRate * (float)sqrt(1.0 - pow(m_Beta2, m_Iteration)) / (1.0f - (float)pow(m_Beta1, m_Iteration));

			//mGrad.Map((m, g) => m * Beta1 + (1 - Beta1) * g, gradients, mGrad);
			mGrad.Add(m_Beta1, 1 - m_Beta1, *gradients, mGrad);
			vGrad.Map([&](float v, float g) { return v * m_Beta2 + (1 - m_Beta2) * g * g; }, *gradients, vGrad);

			parameters->Sub(mGrad.Div(vGrad.Map([&](float x) { return (float)sqrt(x) + m_Epsilon; })).Mul(tempLearningRate), *parameters);

			gradients->Zero();
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
