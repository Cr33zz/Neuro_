#include "Layers/Merge.h"

namespace Neuro
{
    //////////////////////////////////////////////////////////////////////////
    Merge::Merge(const vector<LayerBase*>& inputLayers, Mode mergeMode, const string& name)
        : SingleLayer(__FUNCTION__, inputLayers, inputLayers[0]->OutputShape(), nullptr, name)
    {
        m_MergeMode = mergeMode;
    }

    //////////////////////////////////////////////////////////////////////////
    Merge::Merge(Mode mergeMode, const string& name)
        : SingleLayer(__FUNCTION__, Shape(), nullptr, name)
    {
        m_MergeMode = mergeMode;
    }

    //////////////////////////////////////////////////////////////////////////
    Merge::Merge(const Shape& inputsShape, Mode mergeMode, const string& name)
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
    void Merge::OnLink(const vector<LayerBase*>& layers, bool input)
    {
        __super::OnLink(layers, input);

        if (input)
            m_OutputShapes[0] = m_InputShapes[0];
    }

    //////////////////////////////////////////////////////////////////////////
    void Merge::FeedForwardInternal(bool training)
    {
        switch (m_MergeMode)
        {
        case Mode::Avg:
            Tensor::MergeAvg(m_Inputs, m_Outputs[0]);
            break;
        case Mode::Max:
            Tensor::MergeMax(m_Inputs, m_Outputs[0]);
            break;
        case Mode::Min:
            Tensor::MergeMin(m_Inputs, m_Outputs[0]);
            break;
        case Mode::Sum:
            Tensor::MergeSum(m_Inputs, m_Outputs[0]);
            break;
        }
    }

    //////////////////////////////////////////////////////////////////////////
    void Merge::BackPropInternal(vector<Tensor>& outputsGradient)
    {
        switch (m_MergeMode)
        {
        case Mode::Avg:
            Tensor::MergeAvgGradient(m_Outputs[0], m_Inputs, outputsGradient[0], m_InputsGradient);
            break;
        case Mode::Max:
        case Mode::Min:
            Tensor::MergeMinMaxGradient(m_Outputs[0], m_Inputs, outputsGradient[0], m_InputsGradient);
            break;
        case Mode::Sum:
            Tensor::MergeSumGradient(m_Outputs[0], m_Inputs, outputsGradient[0], m_InputsGradient);
            break;
        }
    }
}
