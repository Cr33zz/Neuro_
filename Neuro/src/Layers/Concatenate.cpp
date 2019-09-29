#include "Layers/Concatenate.h"

namespace Neuro
{
    //////////////////////////////////////////////////////////////////////////
    Concatenate::Concatenate(const vector<LayerBase*>& inputLayers, const string& name)
        : SingleLayer(__FUNCTION__, inputLayers, Shape(), nullptr, name)
    {
        int totalLen = 0;
        for (auto input : inputLayers)
            totalLen += input->OutputShape().Length;
        m_OutputsShapes[0] = Shape(totalLen);
    }

    //////////////////////////////////////////////////////////////////////////
    Concatenate::Concatenate(const string& name)
        : SingleLayer(__FUNCTION__, Shape(), nullptr, name)
    {
    }

    //////////////////////////////////////////////////////////////////////////
    LayerBase* Concatenate::GetCloneInstance() const
    {
        return new Concatenate(false);
    }

    //////////////////////////////////////////////////////////////////////////
    void Concatenate::OnLinkInput(const vector<LayerBase*>& inputLayers)
    {
        __super::OnLinkInput(inputLayers);

        int totalLen = 0;
        for (auto input : InputLayers())
            totalLen += input->OutputShape().Length;
        m_OutputsShapes[0] = Shape(totalLen);
    }

    //////////////////////////////////////////////////////////////////////////
    void Concatenate::FeedForwardInternal(bool training)
    {
        // output is already of proper shape thanks to LayerBase.FeedForward
        Tensor::Concat(_012Axes, m_Inputs, *m_Outputs[0]);
    }

    //////////////////////////////////////////////////////////////////////////
    void Concatenate::BackPropInternal(const tensor_ptr_vec_t& outputsGradient)
    {
        outputsGradient[0]->Split(_012Axes, m_InputsGradient);
    }
}
