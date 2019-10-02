#include "Layers/Concatenate.h"
#include "ComputationalGraph/Ops.h"

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
    void Concatenate::InitOps(TensorLike* training, bool initValues)
    {
        m_OutputOps[0] = concatenate(m_InputOps);
    }
}
