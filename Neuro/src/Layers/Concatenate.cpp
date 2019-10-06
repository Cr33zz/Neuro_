#include "Layers/Concatenate.h"
#include "ComputationalGraph/Ops.h"

namespace Neuro
{
    //////////////////////////////////////////////////////////////////////////
    Concatenate::Concatenate(const vector<LayerBase*>& inputLayers, EAxis axis, const string& name)
        : SingleLayer(__FUNCTION__, inputLayers, Shape(), nullptr, name), m_Axis(axis)
    {
        int totalLen = 0;
        for (auto input : inputLayers)
            totalLen += input->OutputShape().Length;
        m_OutputsShapes[0] = Shape(totalLen);
    }

    //////////////////////////////////////////////////////////////////////////
    Concatenate::Concatenate(EAxis axis, const string& name)
        : SingleLayer(__FUNCTION__, Shape(), nullptr, name), m_Axis(axis)
    {
    }

    //////////////////////////////////////////////////////////////////////////
    LayerBase* Concatenate::GetCloneInstance() const
    {
        return new Concatenate();
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
        m_OutputNodes[0] = concat(m_InputNodes, m_Axis);
    }
}
