#include <algorithm>
#include "Activations.h"
#include "Tensors/Tensor.h"

namespace Neuro
{	
	//////////////////////////////////////////////////////////////////////////
    void Linear::Compute(const Tensor& input, Tensor& result) const
	{
		input.CopyTo(result);
	}

	//////////////////////////////////////////////////////////////////////////
	void Linear::Derivative(const Tensor& output, const Tensor& outputGradient, Tensor& result) const
	{
		outputGradient.CopyTo(result);
	}

	//////////////////////////////////////////////////////////////////////////
	void Sigmoid::Compute(const Tensor& input, Tensor& result) const
	{
		input.Map([&](float x){ return 1 / (1 + (float)exp(-x)); }, result);
	}

	//////////////////////////////////////////////////////////////////////////
	void Sigmoid::Derivative(const Tensor& output, const Tensor& outputGradient, Tensor& result) const
	{
		output.Map([&](float x, float x2){ return x * (1 - x) * x2; }, outputGradient, result);
	}

	//////////////////////////////////////////////////////////////////////////
	void Tanh::Compute(const Tensor& input, Tensor& result) const
	{
		input.Map([&](float x) { return 2 / (1 + (float)exp(-2 * x)) - 1; }, result);
	}

	//////////////////////////////////////////////////////////////////////////
	void Tanh::Derivative(const Tensor& output, const Tensor& outputGradient, Tensor& result) const
	{
		output.Map([&](float x, float x2) { return (1 - x * x) * x2; }, outputGradient, result);
	}

	//////////////////////////////////////////////////////////////////////////
	void ReLU::Compute(const Tensor& input, Tensor& result) const
	{
		input.Map([&](float x) { return max(0.f, x); }, result);
	}

	//////////////////////////////////////////////////////////////////////////
	void ReLU::Derivative(const Tensor& output, const Tensor& outputGradient, Tensor& result) const
	{
		output.Map([&](float x, float x2) { return x > 0 ? x2 : 0; }, outputGradient, result);
	}

	//////////////////////////////////////////////////////////////////////////
	ELU::ELU(float alpha)
		: ALPHA(alpha)
	{
	}

	//////////////////////////////////////////////////////////////////////////
	void ELU::Compute(const Tensor& input, Tensor& result) const
	{
		input.Elu(ALPHA, result);
	}

	//////////////////////////////////////////////////////////////////////////
	void ELU::Derivative(const Tensor& output, const Tensor& outputGradient, Tensor& result) const
	{
		output.EluGradient(output, outputGradient, ALPHA, result);
	}

	//////////////////////////////////////////////////////////////////////////
	void Softmax::Compute(const Tensor& input, Tensor& result) const
	{
		input.Softmax(result);
	}

	//////////////////////////////////////////////////////////////////////////
	void Softmax::Derivative(const Tensor& output, const Tensor& outputGradient, Tensor& result) const
	{
		output.SoftmaxGradient(output, outputGradient, result);
	}
}
