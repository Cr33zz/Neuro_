#include "Layers/Input.h"
#include "ComputationalGraph/TensorLike.h"
#include "ComputationalGraph/Placeholder.h"
#include "ComputationalGraph/Ops.h"

namespace Neuro
{
	//////////////////////////////////////////////////////////////////////////
	Input::Input(const Shape& inputShape, const string& name)
		: Input(new Placeholder(inputShape, "input_placeholder"), name)
	{
	}

    //////////////////////////////////////////////////////////////////////////
    Input::Input(Placeholder* tensor, const string& name)
        : LayerBase(__FUNCTION__, tensor->GetShape(), name)
    {
        m_Placeholder = tensor;
        m_Built = true;

        tensor->m_Metadata = new TensorLike::metadata{this, 0, 0};
        new node(this, {}, {}, {}, { tensor }, { tensor }, { m_ExpectedInputShape }, { m_ExpectedInputShape });
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
    /*void Input::Build(const vector<Shape>& inputShapes)
    {
        m_Placeholder = new Placeholder(inputShapes[0], "input_placeholder");
        m_Built = true;
    }*/

    //////////////////////////////////////////////////////////////////////////
    vector<TensorLike*> Input::InternalCall(const vector<TensorLike*>& inputs, TensorLike* training)
    {
        return { assign(m_Placeholder, inputs[0]) };
    }
}
