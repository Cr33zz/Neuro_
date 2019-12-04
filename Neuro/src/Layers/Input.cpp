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
        InitInput(new Placeholder(inputShape, "placeholder"));
	}

    //////////////////////////////////////////////////////////////////////////
    Input::Input(TensorLike* input, const string& name)
        : LayerBase(__FUNCTION__, input->GetShape(), name)
    {
        InitInput(input);
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
    vector<TensorLike*> Input::InternalCall(const vector<TensorLike*>& inputs)
    {
        NEURO_ASSERT(inputs.size() == 1, "Input layer accepts single input.");
        return { identity(inputs[0]) };
    }

    //////////////////////////////////////////////////////////////////////////
    void Input::InitInput(TensorLike* input)
    {
        m_Input = input;
        m_Built = true;

        input->m_Metadata = new TensorLike::metadata{ this, 0, 0 };
        new node(this, {}, {}, {}, { input }, { input }, { m_ExpectedInputShape }, { m_ExpectedInputShape });
    }
}
