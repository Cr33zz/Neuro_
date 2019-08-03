#include <algorithm>
#include "Activations.h"
#include "Tensors/Tensor.h"

namespace Neuro
{	
	//////////////////////////////////////////////////////////////////////////
	void Linear::Compute(const Tensor& input, Tensor& result)
	{
		input.CopyTo(result);
	}

	//////////////////////////////////////////////////////////////////////////
	void Linear::Derivative(const Tensor& output, const Tensor& outputGradient, Tensor& result)
	{
		outputGradient.CopyTo(result);
	}

	//////////////////////////////////////////////////////////////////////////
	void Sigmoid::Compute(const Tensor& input, Tensor& result)
	{
		input.Map([&](float x){ return 1 / (1 + (float)exp(-x)); }, result);
	}

	//////////////////////////////////////////////////////////////////////////
	void Sigmoid::Derivative(const Tensor& output, const Tensor& outputGradient, Tensor& result)
	{
		output.Map([&](float x, float x2){ return x * (1 - x) * x2; }, outputGradient, result);
	}

	//////////////////////////////////////////////////////////////////////////
	void Tanh::Compute(const Tensor& input, Tensor& result)
	{
		input.Map([&](float x) { return 2 / (1 + (float)exp(-2 * x)) - 1; }, result);
	}

	//////////////////////////////////////////////////////////////////////////
	void Tanh::Derivative(const Tensor& output, const Tensor& outputGradient, Tensor& result)
	{
		output.Map([&](float x, float x2) { return (1 - x * x) * x2; }, outputGradient, result);
	}

	//////////////////////////////////////////////////////////////////////////
	void ReLU::Compute(const Tensor& input, Tensor& result)
	{
		input.Map([&](float x) { return max(0.f, x); }, result);
	}

	//////////////////////////////////////////////////////////////////////////
	void ReLU::Derivative(const Tensor& output, const Tensor& outputGradient, Tensor& result)
	{
		output.Map([&](float x, float x2) { return x > 0 ? x2 : 0; }, outputGradient, result);
	}

	//////////////////////////////////////////////////////////////////////////
	ELU::ELU(float alpha)
		: ALPHA(alpha)
	{
	}

	//////////////////////////////////////////////////////////////////////////
	void ELU::Compute(const Tensor& input, Tensor& result)
	{
		input.Elu(ALPHA, result);
	}

	//////////////////////////////////////////////////////////////////////////
	void ELU::Derivative(const Tensor& output, const Tensor& outputGradient, Tensor& result)
	{
		output.EluGradient(output, outputGradient, ALPHA, result);
	}

	//////////////////////////////////////////////////////////////////////////
	void Softmax::Compute(const Tensor& input, Tensor& result)
	{
		input.Softmax(result);
	}

	//////////////////////////////////////////////////////////////////////////
	void Softmax::Derivative(const Tensor& output, const Tensor& outputGradient, Tensor& result)
	{
		output.SoftmaxGradient(output, outputGradient, result);
	}
}
