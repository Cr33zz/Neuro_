#include "ComputationalGraph/Operations/SoftmaxOp.h"

namespace Neuro
{
    //////////////////////////////////////////////////////////////////////////
    SoftmaxOp::SoftmaxOp(TensorLike* x, const string& name)
        : Operation({ x }, name.empty() ? "softmax" : name)
    {
        m_Output.Resize(x->GetShape());
    }

    //////////////////////////////////////////////////////////////////////////
    void SoftmaxOp::ComputeInternal()
    {
        m_Output.ResizeBatch(m_Inputs[0]->Batch());
        m_Inputs[0]->Softmax(m_Output);
    }

    //////////////////////////////////////////////////////////////////////////
    void SoftmaxOp::ComputeGradientInternal(const Tensor& grad)
    {
        m_Output.SoftmaxGradient(m_Output, grad, m_InputsGrads[0]);
    }
}