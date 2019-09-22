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
    void Activation::OnLinkInput(const vector<LayerBase*>& inputLayers)
    {
        assert(inputLayers.size() == 1);
        __super::OnLinkInput(inputLayers);

        m_OutputsShapes[0] = m_InputShape;
    }
}
