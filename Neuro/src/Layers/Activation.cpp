#include "Layers/Activation.h"

namespace Neuro
{
    //////////////////////////////////////////////////////////////////////////
    Activation::Activation(LayerBase* inputLayer, ActivationBase* activation, const string& name)
        : LayerBase(__FUNCTION__, inputLayer, inputLayer->OutputShape(), activation, name)
    {
    }

    //////////////////////////////////////////////////////////////////////////
    Activation::Activation(const Shape& inputShape, ActivationBase* activation, const string& name)
        : LayerBase(__FUNCTION__, inputShape, inputShape, activation, name)
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
}
