#include "Layers/Flatten.h"
#include "ComputationalGraph/Ops.h"
#include "ComputationalGraph/TensorLike.h"

namespace Neuro
{
	//////////////////////////////////////////////////////////////////////////
    Flatten::Flatten(const string& name)
        : Reshape(__FUNCTION__, Shape(), name)
    {
    }

	//////////////////////////////////////////////////////////////////////////
	Flatten::Flatten(const Shape& inputShape, const string& name)
		: Reshape(__FUNCTION__, inputShape, Shape(inputShape.Length), name)
	{
	}

    //////////////////////////////////////////////////////////////////////////
	LayerBase* Flatten::GetCloneInstance() const
	{
		return new Flatten();
	}

    //////////////////////////////////////////////////////////////////////////
    vector<TensorLike*> Flatten::InternalCall(const vector<TensorLike*>& inputs)
    {
        NEURO_ASSERT(inputs[0]->GetShape().Batch() == 1, "");
        return { batch_reshape(inputs[0], Shape(inputs[0]->GetShape().Length)) };
    }
}
