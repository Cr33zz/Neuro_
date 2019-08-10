#include "Layers/Merge.h"

namespace Neuro
{
    //////////////////////////////////////////////////////////////////////////
    Merge::Merge(const vector<LayerBase*>& inputLayers, Mode mergeMode, const string& name)
        : LayerBase(inputLayers, inputLayers[0]->OutputShape(), nullptr, name.empty() ? GenerateName() : name)
    {
        MergeMode = mergeMode;
    }

    //////////////////////////////////////////////////////////////////////////
    /*Merge::Merge(const vector<Shape>& inputShapes, Mode mergeMode, const string& name)
        : LayerBase(inputShapes, inputShapes[0], nullptr, name.empty() ? GenerateName() : name)
    {
        MergeMode = mergeMode;
    }*/

    //////////////////////////////////////////////////////////////////////////
    Merge::Merge()
    {
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
        MergeMode = sourceMerge.MergeMode;
    }

    //////////////////////////////////////////////////////////////////////////
    void Merge::FeedForwardInternal()
    {
        switch (MergeMode)
        {
        case Mode::Avg:
            Tensor::MergeAvg(m_Inputs, m_Output);
            break;
        case Mode::Max:
            Tensor::MergeMax(m_Inputs, m_Output);
            break;
        case Mode::Min:
            Tensor::MergeMin(m_Inputs, m_Output);
            break;
        case Mode::Sum:
            Tensor::MergeSum(m_Inputs, m_Output);
            break;
        }
    }

    //////////////////////////////////////////////////////////////////////////
    void Merge::BackPropInternal(Tensor& outputGradient)
    {
        switch (MergeMode)
        {
        case Mode::Avg:
            Tensor::MergeAvgGradient(m_Output, m_Inputs, outputGradient, m_InputsGradient);
            break;
        case Mode::Max:
        case Mode::Min:
            Tensor::MergeMinMaxGradient(m_Output, m_Inputs, outputGradient, m_InputsGradient);
            break;
        case Mode::Sum:
            Tensor::MergeSumGradient(m_Output, m_Inputs, outputGradient, m_InputsGradient);
            break;
        }
    }

    //////////////////////////////////////////////////////////////////////////
    const char* Merge::ClassName() const
    {
        return "Merge";
    }

}
