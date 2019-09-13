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
        input.Sigmoid(result);
	}

	//////////////////////////////////////////////////////////////////////////
	void Sigmoid::Derivative(const Tensor& output, const Tensor& outputGradient, Tensor& result) const
	{
        output.SigmoidGradient(output, outputGradient, result);
	}

	//////////////////////////////////////////////////////////////////////////
	void Tanh::Compute(const Tensor& input, Tensor& result) const
	{
        input.Tanh(result);
	}

	//////////////////////////////////////////////////////////////////////////
	void Tanh::Derivative(const Tensor& output, const Tensor& outputGradient, Tensor& result) const
	{
        output.TanhGradient(output, outputGradient, result);
	}

	//////////////////////////////////////////////////////////////////////////
	void ReLU::Compute(const Tensor& input, Tensor& result) const
	{
        input.ReLU(result);
	}

	//////////////////////////////////////////////////////////////////////////
	void ReLU::Derivative(const Tensor& output, const Tensor& outputGradient, Tensor& result) const
	{
        output.ReLUGradient(output, outputGradient, result);
	}

	//////////////////////////////////////////////////////////////////////////
	ELU::ELU(float alpha)
		: m_Alpha(alpha)
	{
	}

	//////////////////////////////////////////////////////////////////////////
	void ELU::Compute(const Tensor& input, Tensor& result) const
	{
		input.Elu(m_Alpha, result);
	}

	//////////////////////////////////////////////////////////////////////////
	void ELU::Derivative(const Tensor& output, const Tensor& outputGradient, Tensor& result) const
	{
		output.EluGradient(output, outputGradient, m_Alpha, result);
	}

    //////////////////////////////////////////////////////////////////////////
    LeakyReLU::LeakyReLU(float alpha)
        : m_Alpha(alpha)
    {
    }

    //////////////////////////////////////////////////////////////////////////
    void LeakyReLU::Compute(const Tensor& input, Tensor& result) const
    {
        input.LeakyReLU(m_Alpha, result);
    }

    //////////////////////////////////////////////////////////////////////////
    void LeakyReLU::Derivative(const Tensor& output, const Tensor& outputGradient, Tensor& result) const
    {
        output.LeakyReLUGradient(output, outputGradient, m_Alpha, result);
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
