#include "ComputationalGraph/Operations/MergeOp.h"

namespace Neuro
{
    //////////////////////////////////////////////////////////////////////////
    MergeOp::MergeOp(const vector<TensorLike*>& xs, EMergeMode mode)
        : Operation(xs, "merge"), m_Mode(mode)
    {
    }

    //////////////////////////////////////////////////////////////////////////
    void MergeOp::ComputeInternal()
    {
        m_Output.Resize(m_Inputs[0]->GetShape());

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