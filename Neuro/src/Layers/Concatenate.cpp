#include "Layers/Concatenate.h"

namespace Neuro
{
    //////////////////////////////////////////////////////////////////////////
    Concatenate::Concatenate(const vector<LayerBase*>& inputLayers, const string& name)
        : LayerBase(__FUNCTION__, inputLayers, Shape(), nullptr, name)
    {
        OnLink(inputLayers, true);
    }

    //////////////////////////////////////////////////////////////////////////
    Concatenate::Concatenate(const string& name)
        : LayerBase(__FUNCTION__, Shape(), nullptr, name)
    {
    }

    //////////////////////////////////////////////////////////////////////////
    Neuro::LayerBase* Concatenate::GetCloneInstance() const
    {
        return new Concatenate(false);
    }

    //////////////////////////////////////////////////////////////////////////
    void Concatenate::OnLink(const vector<LayerBase*>& layers, bool input)
    {
        __super::OnLink(layers, input);

        if (input)
        {
            int totalLen = 0;
            for (auto input : InputLayers())
                totalLen += input->OutputShape().Length;
            m_OutputShapes[0] = Shape(1, totalLen);
        }
    }

    //////////////////////////////////////////////////////////////////////////
    void Concatenate::FeedForwardInternal(bool training)
    {
        // output is already of proper shape thanks to LayerBase.FeedForward
        Tensor::Concat(EAxis::Sample, m_Inputs, m_Outputs[0]);
    }

    //////////////////////////////////////////////////////////////////////////
    void Concatenate::BackPropInternal(vector<Tensor>& outputsGradient)
    {
        outputsGradient[0].Split(EAxis::Sample, m_InputsGradient);
    }
}
