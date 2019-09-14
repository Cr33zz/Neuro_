#include "Layers/Activation.h"

namespace Neuro
{
    //////////////////////////////////////////////////////////////////////////
    Activation::Activation(LayerBase* inputLayer, ActivationBase* activation, const string& name)
        : SingleLayer(__FUNCTION__, inputLayer, inputLayer->OutputShape(), activation, name)
    {
    }

    //////////////////////////////////////////////////////////////////////////
    Activation::Activation(ActivationBase* activation, const string& name)
        : SingleLayer(__FUNCTION__, Shape(), activation, name)
    {
    }

    //////////////////////////////////////////////////////////////////////////
    Activation::Activation(const Shape& inputShape, ActivationBase* activation, const string& name)
        : SingleLayer(__FUNCTION__, inputShape, inputShape, activation, name)
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
    void Activation::OnLink(LayerBase* layer, bool input)
    {
        __super::OnLink(layer, input);

        if (input)
            m_OutputShapes[0] = m_InputShapes[0];
    }
}
