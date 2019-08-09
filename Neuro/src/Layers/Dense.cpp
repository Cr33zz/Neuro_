#include "Layers/Dense.h"
#include "Tensors/Tensor.h"
#include "Initializers/GlorotUniform.h"
#include "Initializers/Zeros.h"

namespace Neuro
{
	//////////////////////////////////////////////////////////////////////////
	Dense::Dense(LayerBase* inputLayer, int outputs, ActivationBase* activation, const string& name)
		: LayerBase(inputLayer, Shape(1, outputs), activation, name.empty() ? GenerateName() : name)
	{
	}

	//////////////////////////////////////////////////////////////////////////
	Dense::Dense(int inputs, int outputs, ActivationBase* activation, const string& name)
		: LayerBase(Shape(1, inputs), Shape(1, outputs), activation, name.empty() ? GenerateName() : name)
	{
	}

	//////////////////////////////////////////////////////////////////////////
	Dense::Dense()
	{
	}

	//////////////////////////////////////////////////////////////////////////
	Dense::~Dense()
	{
		delete WeightsInitializer;
		delete BiasInitializer;
	}

	//////////////////////////////////////////////////////////////////////////
	Neuro::LayerBase* Dense::GetCloneInstance() const
	{
		return new Dense();
	}

	//////////////////////////////////////////////////////////////////////////
	void Dense::OnClone(const LayerBase& source)
	{
		__super::OnClone(source);

		auto& sourceDense = static_cast<const Dense&>(source);
		Weights = sourceDense.Weights;
		Bias = sourceDense.Bias;
		UseBias = sourceDense.UseBias;
	}

	//////////////////////////////////////////////////////////////////////////
	void Dense::OnInit()
	{
		__super::OnInit();

		Weights = Tensor(Shape(InputShape().Length, m_OutputShape.Length));
		Bias = Tensor(m_OutputShape);

		WeightsGradient = Tensor(Weights.GetShape());
		BiasGradient = Tensor(Bias.GetShape());

		WeightsInitializer->Init(Weights, InputShape().Length, m_OutputShape.Length);

		if (UseBias)
			BiasInitializer->Init(Bias, InputShape().Length, m_OutputShape.Length);
	}

	//////////////////////////////////////////////////////////////////////////
	void Dense::FeedForwardInternal()
	{
		Weights.Mul(*m_Inputs[0], m_Output);
		if (UseBias)
			m_Output.Add(Bias, m_Output);
	}

	//////////////////////////////////////////////////////////////////////////
	void Dense::BackPropInternal(Tensor& outputGradient)
	{
		// for explanation watch https://www.youtube.com/watch?v=8H2ODPNxEgA&t=898s
		// each input is responsible for the output error proportionally to weights it is multiplied by
		Weights.Transposed().Mul(outputGradient, m_InputsGradient[0]);

		WeightsGradient.Add(outputGradient.Mul(m_Inputs[0]->Transposed()).SumBatches(), WeightsGradient);
		if (UseBias)
			BiasGradient.Add(outputGradient.SumBatches(), BiasGradient);
	}

	//////////////////////////////////////////////////////////////////////////
	void Dense::CopyParametersTo(LayerBase& target, float tau) const
	{
		__super::CopyParametersTo(target, tau);

		auto& targetDense = static_cast<Dense&>(target);
		Weights.CopyTo(targetDense.Weights, tau);
		Bias.CopyTo(targetDense.Bias, tau);
	}

	//////////////////////////////////////////////////////////////////////////
	int Dense::GetParamsNum() const
	{
		return InputShape().Length * m_OutputShape.Length;
	}

	//////////////////////////////////////////////////////////////////////////
	void Dense::GetParametersAndGradients(vector<ParametersAndGradients>& result)
	{
		result.push_back(ParametersAndGradients(&Weights, &WeightsGradient));

		if (UseBias)
			result.push_back(ParametersAndGradients(&Bias, &BiasGradient));
	}

	//////////////////////////////////////////////////////////////////////////
	const char* Dense::ClassName() const
	{
		return "Dense";
	}
}
