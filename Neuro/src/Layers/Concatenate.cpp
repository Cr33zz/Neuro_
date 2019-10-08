#include "Layers/Concatenate.h"
#include "ComputationalGraph/Ops.h"

namespace Neuro
{
    //////////////////////////////////////////////////////////////////////////
    Concatenate::Concatenate(EAxis axis, const string& name)
        : SingleLayer(__FUNCTION__, Shape(), nullptr, name), m_Axis(axis)
    {
    }

    //////////////////////////////////////////////////////////////////////////
    LayerBase* Concatenate::GetCloneInstance() const
    {
        return new Concatenate();
    }

    //////////////////////////////////////////////////////////////////////////
    vector<TensorLike*> Concatenate::InternalCall(const vector<TensorLike*>& inputs, TensorLike* training)
    {
        return { concat(inputs, m_Axis) };
    }

}
