#include "Layers/Reshape.h"

namespace Neuro
{
    //////////////////////////////////////////////////////////////////////////
    Reshape::Reshape(LayerBase* inputLayer, const Shape& shape, const string& name)
        : Reshape(__FUNCTION__, inputLayer, shape, name)
    {
    }

    //////////////////////////////////////////////////////////////////////////
    Reshape::Reshape(const Shape& inputShape, const Shape& shape, const string& name)
        : Reshape(__FUNCTION__, inputShape, shape, name)
    {
    }

    //////////////////////////////////////////////////////////////////////////
    Reshape::Reshape(const string& constructorName, LayerBase* inputLayer, const Shape& shape, const string& name)
        : LayerBase(constructorName, inputLayer, shape, nullptr, name)
    {
    }

    //////////////////////////////////////////////////////////////////////////
    Reshape::Reshape(const string& constructorName, const Shape& inputShape, const Shape& shape, const string& name)
        : LayerBase(constructorName, inputShape, shape, nullptr, name)
    {
    }

    //////////////////////////////////////////////////////////////////////////
    Reshape::Reshape(const string& constructorName, const Shape& shape, const string& name)
        : LayerBase(constructorName, shape, nullptr, name)
    {
    }

    //////////////////////////////////////////////////////////////////////////
    LayerBase* Reshape::GetCloneInstance() const
    {
        return new Reshape();
    }

    //////////////////////////////////////////////////////////////////////////
    void Reshape::FeedForwardInternal(bool training)
    {
        // output is already of proper shape thanks to LayerBase.FeedForward
        m_Inputs[0]->CopyTo(m_Outputs[0]);
    }

    //////////////////////////////////////////////////////////////////////////
    void Reshape::BackPropInternal(vector<Tensor>& outputGradients)
    {
        outputGradients[0].Reshaped(m_Inputs[0]->GetShape(), m_InputGradients[0]);
    }
}
