#include "ComputationalGraph/Operations/MergeOp.h"

namespace Neuro
{
    //////////////////////////////////////////////////////////////////////////
    MergeOp::MergeOp(const vector<TensorLike*>& xs, EMergeMode mode, const string& name)
        : Operation(xs, name.empty() ? "merge" : name), m_Mode(mode)
    {
        UpdateOutputShape();
    }

    //////////////////////////////////////////////////////////////////////////
    void MergeOp::UpdateOutputShape()
    {
        auto& shape = m_InputNodes[0]->GetShape();
        for (size_t i = 0; i < m_InputNodes.size(); ++i)
            NEURO_ASSERT(m_InputNodes[i]->GetShape() == shape, "All inputs must be of the same shape being " << shape.ToString() << ", input " << i << " is of shape " << m_InputNodes[i]->GetShape().ToString() << ".");

        m_Output.Resize(shape);
    }

    //////////////////////////////////////////////////////////////////////////
    void MergeOp::ComputeInternal()
    {
        m_Output.ResizeBatch(m_Inputs[0]->Batch());

        switch (m_Mode)
        {
        case AvgMerge:
            Tensor::MergeAvg(m_Inputs, m_Output);
            break;
        case MaxMerge:
            Tensor::MergeMax(m_Inputs, m_Output);
            break;
        case MinMerge:
            Tensor::MergeMin(m_Inputs, m_Output);
            break;
        case SumMerge:
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
            case AvgMerge:
                Tensor::MergeAvgGradient(m_Output, m_Inputs, grad, m_InputsGradsPtrs);
                break;
            case MaxMerge:
            case MinMerge:
                Tensor::MergeMinMaxGradient(m_Output, m_Inputs, grad, m_InputsGradsPtrs);
                break;
            case SumMerge:
                Tensor::MergeSumGradient(m_Output, m_Inputs, grad, m_InputsGradsPtrs);
                break;
            }
        }
    }
}