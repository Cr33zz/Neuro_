#include "ComputationalGraph/Operations/TransposeOp.h"

namespace Neuro
{
    //////////////////////////////////////////////////////////////////////////
    TransposeOp::TransposeOp(TensorLike* x, const string& name)
        : Operation({ x }, name.empty() ? "transpose" : name)
    {
        m_Output.Resize(Shape(x->GetShape().Height(), x->GetShape().Width(), x->GetShape().Depth()));
    }

    //////////////////////////////////////////////////////////////////////////
    void TransposeOp::ComputeInternal()
    {
        m_Output.ResizeBatch(m_Inputs[0]->Batch());
        m_Inputs[0]->Transpose(m_Output);
    }

    //////////////////////////////////////////////////////////////////////////
    void TransposeOp::ComputeGradientInternal(const Tensor& grad)
    {
        grad.Transpose(m_InputsGrads[0]);
    }
}