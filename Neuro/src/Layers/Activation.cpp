#include "Layers/Activation.h"
#include "Activations.h"

namespace Neuro
{
    //////////////////////////////////////////////////////////////////////////
    Activation::Activation(ActivationBase* activation, const string& name)
        : SingleLayer(__FUNCTION__, Shape(), activation, name)
    {
    }

    //////////////////////////////////////////////////////////////////////////
    Activation::Activation(const Shape& inputShape, ActivationBase* activation, const string& name)
        : SingleLayer(__FUNCTION__, inputShape, activation, name)
    {
    }

    //////////////////////////////////////////////////////////////////////////
    Activation::Activation()
    {
    }
    
    //////////////////////////////////////////////////////////////////////////
    LayerBase* Activation::GetCloneInstance() const
    {
        return new Activation();
    }

    //////////////////////////////////////////////////////////////////////////
    vector<TensorLike*> Activation::InternalCall(const vector<TensorLike*>& inputs, TensorLike* training)
    {
        NEURO_ASSERT(m_Activation, "Activation is required.");
        return { m_Activation->Build(inputs[0]) };
    }
}
