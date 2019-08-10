#include <sstream>

#include "Optimizers/Adam.h"
#include "Tensors/Tensor.h"

namespace Neuro
{    
	//////////////////////////////////////////////////////////////////////////
	Adam::Adam(float lr)
	{
		LearningRate = lr;
	}

	//////////////////////////////////////////////////////////////////////////
	void Adam::OnStep(vector<ParametersAndGradients>& paramsAndGrads, int batchSize)
	{
		if (MGradients.size() != paramsAndGrads.size())
		{
			for (int i = 0; i < paramsAndGrads.size(); ++i)
			{
				auto gradients = paramsAndGrads[i].Gradients;

				MGradients.push_back(Tensor(gradients->GetShape()));
				VGradients.push_back(Tensor(gradients->GetShape()));
			}
		}

		for (int i = 0; i < paramsAndGrads.size(); ++i)
		{
			auto& parametersAndGradient = paramsAndGrads[i];
			auto parameters = parametersAndGradient.Parameters;
			auto gradients = parametersAndGradient.Gradients;
			auto& mGrad = MGradients[i];
			auto& vGrad = VGradients[i];

			gradients->Div((float)batchSize, *gradients);

			float tempLearningRate = LearningRate * (float)sqrt(1.0 - pow(Beta2, Iteration)) / (1.0f - (float)pow(Beta1, Iteration));

			//mGrad.Map((m, g) => m * Beta1 + (1 - Beta1) * g, gradients, mGrad);
			mGrad.Add(Beta1, 1 - Beta1, *gradients, mGrad);
			vGrad.Map([&](float v, float g) { return v * Beta2 + (1 - Beta2) * g * g; }, *gradients, vGrad);

			parameters->Sub(mGrad.Div(vGrad.Map([&](float x) { return (float)sqrt(x) + Epsilon; })).Mul(tempLearningRate), *parameters);

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
		ss << "Adam(lr=" << LearningRate << ")";
		return ss.str();
		//return $"Adam(lr={LearningRate}, beta1={Beta1}, beta2={Beta2}, epsilon={Epsilon})";
	}

	//////////////////////////////////////////////////////////////////////////
	const char* Adam::ClassName() const
	{
		return "Adam";
	}

}
