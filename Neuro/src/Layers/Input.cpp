#include "Layers/Input.h"

namespace Neuro
{
	//////////////////////////////////////////////////////////////////////////
	Input::Input(const Shape& inputShape, const string& name)
		: LayerBase(__FUNCTION__, inputShape, inputShape, nullptr, name)
	{
	}

	//////////////////////////////////////////////////////////////////////////
	Input::Input()
	{
	}

	//////////////////////////////////////////////////////////////////////////
	LayerBase* Input::GetCloneInstance() const
	{
		return new Input();
	}

	//////////////////////////////////////////////////////////////////////////
	void Input::FeedForwardInternal(bool training)
	{
		// output is already of proper shape thanks to LayerBase.FeedForward
		m_Inputs[0]->CopyTo(m_Outputs[0]);
	}

	//////////////////////////////////////////////////////////////////////////
	void Input::BackPropInternal(vector<Tensor>& outputsGradient)
	{
		m_InputsGradient[0] = outputsGradient[0];
	}
}
