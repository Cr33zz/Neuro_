#include "ComputationalGraph/Operations/AssignOp.h"

namespace Neuro
{
    //////////////////////////////////////////////////////////////////////////
    AssignOp::AssignOp(TensorLike* x, TensorLike* val)
        : Operation({ x, val })
    {
    }

    //////////////////////////////////////////////////////////////////////////
    void AssignOp::ComputeInternal()
    {
        m_Output.Resize(m_Inputs[0]->GetShape());
        m_Inputs[0]->CopyTo(m_Output);
    }

    //////////////////////////////////////////////////////////////////////////
    void AssignOp::ComputeGradientInternal(const Tensor& grad)
    {
        grad.CopyTo(m_InputsGrads[0]);
    }
}