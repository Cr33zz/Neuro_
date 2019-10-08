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
    vector<TensorLike*> Flatten::InternalCall(const vector<TensorLike*>& inputNodes, TensorLike* training)
    {
        return { reshape(inputNodes[0], Shape(inputNodes[0]->GetShape().Length)) };
    }
}
