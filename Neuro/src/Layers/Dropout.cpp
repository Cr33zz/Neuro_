#include "Layers/Dropout.h"
#include "Tools.h"
#include "ComputationalGraph/Ops.h"

namespace Neuro
{
    //////////////////////////////////////////////////////////////////////////
    Dropout::Dropout(const Shape& inputShape, float p, const string& name)
        : SingleLayer(__FUNCTION__, inputShape, nullptr, name), m_Prob(p)
    {
    }

    //////////////////////////////////////////////////////////////////////////
    Dropout::Dropout(float p, const string& name)
        : SingleLayer(__FUNCTION__, Shape(), nullptr, name), m_Prob(p)
    {
    }

    //////////////////////////////////////////////////////////////////////////
    LayerBase* Dropout::GetCloneInstance() const
    {
        return new Dropout();
    }

    //////////////////////////////////////////////////////////////////////////
    vector<TensorLike*> Dropout::InternalCall(const vector<TensorLike*>& inputs, TensorLike* training)
    {
        return { dropout(inputs[0], m_Prob, training) };
    }

}
