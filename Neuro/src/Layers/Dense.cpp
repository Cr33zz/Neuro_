#include "Layers/Dense.h"
#include "Tensors/Tensor.h"
#include "Initializers/GlorotUniform.h"
#include "Initializers/Zeros.h"

namespace Neuro
{
	//////////////////////////////////////////////////////////////////////////
	Dense::Dense(LayerBase* inputLayer, int outputs, ActivationFunc* activation, const string& name)
		: LayerBase(inputLayer, Shape(1, outputs), activation, name)
	{
	}

	//////////////////////////////////////////////////////////////////////////
	Dense::Dense(int inputs, int outputs, ActivationFunc* activation, const string& name)
		: LayerBase(Shape(1, inputs), Shape(1, outputs), activation, name)
	{
	}

	//////////////////////////////////////////////////////////////////////////
	Dense::Dense()
	{
	}

	//////////////////////////////////////////////////////////////////////////
	Neuro::LayerBase* Dense::GetCloneInstance()
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

		Weights = Tensor(Shape(InputShape().Length, OutputShape.Length));
		Bias = Tensor(OutputShape);

		WeightsGradient = Tensor(Weights.GetShape());
		BiasGradient = Tensor(Bias.GetShape());

		if (KernelInitializer == nullptr)
			KernelInitializer = new GlorotUniform();
		KernelInitializer->Init(Weights, InputShape().Length, OutputShape.Length);

		if (UseBias)
		{
			if (BiasInitializer == nullptr)
				BiasInitializer = new Zeros();
			BiasInitializer->Init(Bias, InputShape().Length, OutputShape.Length);
		}
	}

	//////////////////////////////////////////////////////////////////////////
	void Dense::FeedForwardInternal()
	{
		Weights.Mul(*Inputs[0], Output);
		if (UseBias)
			Output.Add(Bias, Output);
	}

	//////////////////////////////////////////////////////////////////////////
	void Dense::BackPropInternal(Tensor& outputGradient)
	{
		// for explanation watch https://www.youtube.com/watch?v=8H2ODPNxEgA&t=898s
		// each input is responsible for the output error proportionally to weights it is multiplied by
		Weights.Transposed().Mul(outputGradient, InputsGradient[0]);

		WeightsGradient.Add(outputGradient.Mul(Inputs[0]->Transposed()).SumBatches(), WeightsGradient);
		if (UseBias)
			BiasGradient.Add(outputGradient.SumBatches(), BiasGradient);
	}

	//////////////////////////////////////////////////////////////////////////
	void Dense::CopyParametersTo(LayerBase& target, float tau)
	{
		__super::CopyParametersTo(target, tau);

		auto& targetDense = static_cast<Dense&>(target);
		Weights.CopyTo(targetDense.Weights, tau);
		Bias.CopyTo(targetDense.Bias, tau);
	}

	//////////////////////////////////////////////////////////////////////////
	int Dense::GetParamsNum()
	{
		return InputShape().Length * OutputShape.Length;
	}

	//////////////////////////////////////////////////////////////////////////
	void Dense::GetParametersAndGradients(vector<ParametersAndGradients>& result)
	{
		result.push_back(ParametersAndGradients(&Weights, &WeightsGradient));

		if (UseBias)
			result.push_back(ParametersAndGradients(&Bias, &BiasGradient));
	}

}
