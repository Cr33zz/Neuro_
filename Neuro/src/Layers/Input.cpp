#include "Layers/Input.h"
#include "ComputationalGraph/TensorLike.h"
#include "ComputationalGraph/Placeholder.h"
#include "ComputationalGraph/NameScope.h"
#include "ComputationalGraph/Ops.h"

namespace Neuro
{
	//////////////////////////////////////////////////////////////////////////
	Input::Input(const Shape& inputShape, const string& name)
		: LayerBase(__FUNCTION__, inputShape, name)
	{
        NameScope scope(Name());
        InitPlaceholder(new Placeholder(inputShape, "placeholder"));
	}

    //////////////////////////////////////////////////////////////////////////
    Input::Input(Placeholder* placeholder, const string& name)
        : LayerBase(__FUNCTION__, placeholder->GetShape(), name)
    {
        InitPlaceholder(placeholder);
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

    //////////////////////////////////////////////////////////////////////////
    void Input::InitPlaceholder(Placeholder* placeholder)
    {
        m_Placeholder = placeholder;
        m_Built = true;

        placeholder->m_Metadata = new TensorLike::metadata{ this, 0, 0 };
        new node(this, {}, {}, {}, { placeholder }, { placeholder }, { m_ExpectedInputShape }, { m_ExpectedInputShape });
    }
}
