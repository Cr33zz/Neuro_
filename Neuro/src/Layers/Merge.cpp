#include "Layers/Merge.h"
#include "ComputationalGraph/Ops.h"

namespace Neuro
{
    //////////////////////////////////////////////////////////////////////////
    Merge::Merge(EMergeMode mergeMode, ActivationBase* activation, const string& name)
        : SingleLayer(__FUNCTION__, Shape(), activation, name)
    {
        m_MergeMode = mergeMode;
    }

    //////////////////////////////////////////////////////////////////////////
    Merge::Merge(const Shape& inputsShape, EMergeMode mergeMode, ActivationBase* activation, const string& name)
        : SingleLayer(__FUNCTION__, inputsShape, activation, name)
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
    vector<TensorLike*> Merge::InternalCall(const vector<TensorLike*>& inputs, TensorLike* training)
    {
        TensorLike* output = nullptr;
        switch (m_MergeMode)
        {
        case MergeAvg:
            output = merge_avg(inputs);
            break;
        case MergeMax:
            output = merge_max(inputs);
            break;
        case MergeMin:
            output = merge_min(inputs);
            break;
        case MergeSum:
            output = merge_sum(inputs);
            break;
        }

        return { output };
    }
}
