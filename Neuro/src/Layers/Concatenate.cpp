#include "Layers/Concatenate.h"

namespace Neuro
{
    //////////////////////////////////////////////////////////////////////////
    Concatenate::Concatenate(const vector<LayerBase*>& inputLayers, const string& name)
        : LayerBase(__FUNCTION__, inputLayers, Shape(), nullptr, name)
    {
        int totalLen = 0;
        for (auto input : inputLayers)
            totalLen += input->OutputShape().Length;
        m_OutputShape = Shape(1, totalLen);
    }

    //////////////////////////////////////////////////////////////////////////
    Concatenate::Concatenate()
    {
    }

    //////////////////////////////////////////////////////////////////////////
    Neuro::LayerBase* Concatenate::GetCloneInstance() const
    {
        return new Concatenate();
    }

    //////////////////////////////////////////////////////////////////////////
    void Concatenate::FeedForwardInternal()
    {
        // output is already of proper shape thanks to LayerBase.FeedForward
        Tensor::Concat(m_Inputs, m_Output);
    }

    //////////////////////////////////////////////////////////////////////////
    void Concatenate::BackPropInternal(Tensor& outputGradient)
    {
        outputGradient.Split(m_InputsGradient);
    }

    //////////////////////////////////////////////////////////////////////////
    const char* Concatenate::ClassName() const
    {
        return "Concat";
    }

}
