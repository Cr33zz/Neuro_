#include "Layers/Flatten.h"

namespace Neuro
{
	//////////////////////////////////////////////////////////////////////////
	Flatten::Flatten(LayerBase* inputLayer, const string& name)
        : LayerBase(__FUNCTION__, inputLayer, Shape(1, inputLayer->OutputShape().Length), nullptr, name)
	{
	}

	//////////////////////////////////////////////////////////////////////////
	Flatten::Flatten(const Shape& inputShape, const string& name)
		: LayerBase(__FUNCTION__, inputShape, Shape(1, inputShape.Length), nullptr, name)
	{
	}

	//////////////////////////////////////////////////////////////////////////
	Flatten::Flatten()
	{
	}

	//////////////////////////////////////////////////////////////////////////
	LayerBase* Flatten::GetCloneInstance() const
	{
		return new Flatten();
	}

	//////////////////////////////////////////////////////////////////////////
	void Flatten::FeedForwardInternal(bool training)
	{
		// output is already of proper shape thanks to LayerBase.FeedForward
		m_Inputs[0]->CopyTo(m_Output);
	}

	//////////////////////////////////////////////////////////////////////////
	void Flatten::BackPropInternal(Tensor& outputGradient)
	{
		m_InputsGradient[0] = outputGradient.Reshaped(m_Inputs[0]->GetShape());
	}
}
