#include "Layers/Merge.h"
#include "ComputationalGraph/Ops.h"

namespace Neuro
{
    //////////////////////////////////////////////////////////////////////////
    Merge::Merge(const vector<LayerBase*>& inputLayers, EMergeMode mergeMode, ActivationBase* activation, const string& name)
        : SingleLayer(__FUNCTION__, inputLayers, inputLayers[0]->OutputShape(), activation, name)
    {
        m_MergeMode = mergeMode;
    }

    //////////////////////////////////////////////////////////////////////////
    Merge::Merge(EMergeMode mergeMode, ActivationBase* activation, const string& name)
        : SingleLayer(__FUNCTION__, Shape(), activation, name)
    {
        m_MergeMode = mergeMode;
    }

    //////////////////////////////////////////////////////////////////////////
    Merge::Merge(const Shape& inputsShape, EMergeMode mergeMode, ActivationBase* activation, const string& name)
        : SingleLayer(__FUNCTION__, inputsShape, inputsShape, activation, name)
    {
        m_MergeMode = mergeMode;
    }

    //////////////////////////////////////////////////////////////////////////
    LayerBase* Merge::GetCloneInstance() const
    {
        return new Merge();
    }

    //////////////////////////////////////////////////////////////////////////
    void Merge::OnClone(const LayerBase& source)
    {
        __super::OnClone(source);

        auto sourceMerge = static_cast<const Merge&>(source);
        m_MergeMode = sourceMerge.m_MergeMode;
    }

    //////////////////////////////////////////////////////////////////////////
    void Merge::OnLinkInput(const vector<LayerBase*>& inputLayers)
    {
        __super::OnLinkInput(inputLayers);

        m_OutputsShapes[0] = m_InputShape;
    }

    //////////////////////////////////////////////////////////////////////////
    void Merge::InternalCall(TensorLike* training, bool initValues)
    {
        switch (m_MergeMode)
        {
        case MergeAvg:
            m_OutputNodes[0] = merge_avg(m_InputNodes);
            break;
        case MergeMax:
            m_OutputNodes[0] = merge_max(m_InputNodes);
            break;
        case MergeMin:
            m_OutputNodes[0] = merge_min(m_InputNodes);
            break;
        case MergeSum:
            m_OutputNodes[0] = merge_sum(m_InputNodes);
            break;
        }
    }
}
