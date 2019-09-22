﻿#include "Layers/Merge.h"

namespace Neuro
{
    //////////////////////////////////////////////////////////////////////////
    Merge::Merge(const vector<LayerBase*>& inputLayers, EMergeMode mergeMode, const string& name)
        : SingleLayer(__FUNCTION__, inputLayers, inputLayers[0]->OutputShape(), nullptr, name)
    {
        m_MergeMode = mergeMode;
    }

    //////////////////////////////////////////////////////////////////////////
    Merge::Merge(EMergeMode mergeMode, const string& name)
        : SingleLayer(__FUNCTION__, Shape(), nullptr, name)
    {
        m_MergeMode = mergeMode;
    }

    //////////////////////////////////////////////////////////////////////////
    Merge::Merge(const Shape& inputsShape, EMergeMode mergeMode, const string& name)
        : SingleLayer(__FUNCTION__, inputsShape, inputsShape, nullptr, name)
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
    void Merge::OnLink(LayerBase* layer, bool input)
    {
        __super::OnLink(layer, input);

        if (input)
            m_OutputsShapes[0] = m_InputShape;
    }

    //////////////////////////////////////////////////////////////////////////
    void Merge::FeedForwardInternal(bool training)
    {
        switch (m_MergeMode)
        {
        case MergeAvg:
            Tensor::MergeAvg(m_Inputs, *m_Outputs[0]);
            break;
        case MergeMax:
            Tensor::MergeMax(m_Inputs, *m_Outputs[0]);
            break;
        case MergeMin:
            Tensor::MergeMin(m_Inputs, *m_Outputs[0]);
            break;
        case MergeSum:
            Tensor::MergeSum(m_Inputs, *m_Outputs[0]);
            break;
        }
    }

    //////////////////////////////////////////////////////////////////////////
    void Merge::BackPropInternal(const tensor_ptr_vec_t& outputsGradient)
    {
        switch (m_MergeMode)
        {
        case MergeAvg:
            Tensor::MergeAvgGradient(*m_Outputs[0], m_Inputs, *outputsGradient[0], m_InputsGradient);
            break;
        case MergeMax:
        case MergeMin:
            Tensor::MergeMinMaxGradient(*m_Outputs[0], m_Inputs, *outputsGradient[0], m_InputsGradient);
            break;
        case MergeSum:
            Tensor::MergeSumGradient(*m_Outputs[0], m_Inputs, *outputsGradient[0], m_InputsGradient);
            break;
        }
    }
}
