#include "Layers/Input.h"
#include "ComputationalGraph/Placeholder.h"
#include "ComputationalGraph/Ops.h"

namespace Neuro
{
	//////////////////////////////////////////////////////////////////////////
	Input::Input(const Shape& inputShape, const string& name)
		: LayerBase(__FUNCTION__, inputShape, name)
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
    void Input::Build(const vector<Shape>& inputShapes)
    {
        m_Placeholder = new Placeholder(inputShapes[0], "input_placeholder");
        m_Built = true;
    }

    //////////////////////////////////////////////////////////////////////////
    vector<TensorLike*> Input::InternalCall(const vector<TensorLike*>& inputNodes, TensorLike* training)
    {
        return { assign(m_Placeholder, inputNodes[0]) };
    }
}
