#include "ComputationalGraph/Operations/MergeOp.h"

namespace Neuro
{
    //////////////////////////////////////////////////////////////////////////
    MergeOp::MergeOp(const vector<TensorLike*>& xs, EMergeMode mode, const string& name)
        : Operation(xs, name.empty() ? "merge" : name), m_Mode(mode)
    {
        auto& shape0 = xs[0]->GetShape();
        for (size_t i = 0; i < xs.size(); ++i)
            assert(xs[i]->GetShape() == shape0);
        
        m_Output.Resize(shape0);
    }

    //////////////////////////////////////////////////////////////////////////
    void MergeOp::ComputeInternal()
    {
        m_Output.ResizeBatch(m_Inputs[0]->Batch());

        switch (m_Mode)
        {
        case MergeAvg:
            Tensor::MergeAvg(m_Inputs, m_Output);
            break;
        case MergeMax:
            Tensor::MergeMax(m_Inputs, m_Output);
            break;
        case MergeMin:
            Tensor::MergeMin(m_Inputs, m_Output);
            break;
        case MergeSum:
            Tensor::MergeSum(m_Inputs, m_Output);
            break;
        }
    }

    //////////////////////////////////////////////////////////////////////////
    void MergeOp::ComputeGradientInternal(const Tensor& grad)
    {
        bool anyInputCareAboutGrad = false;
        for (auto inputNode : m_InputNodes)
        {
            if (inputNode->CareAboutGradient())
            {
                anyInputCareAboutGrad = true;
                break;
            }
        }

        if (anyInputCareAboutGrad)
        {
            switch (m_Mode)
            {
            case MergeAvg:
                Tensor::MergeAvgGradient(m_Output, m_Inputs, grad, m_InputsGradsPtrs);
                break;
            case MergeMax:
            case MergeMin:
                Tensor::MergeMinMaxGradient(m_Output, m_Inputs, grad, m_InputsGradsPtrs);
                break;
            case MergeSum:
                Tensor::MergeSumGradient(m_Output, m_Inputs, grad, m_InputsGradsPtrs);
                break;
            }
        }
    }
}