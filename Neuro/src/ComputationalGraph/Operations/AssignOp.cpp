#include "ComputationalGraph/Operations/AssignOp.h"

namespace Neuro
{
    //////////////////////////////////////////////////////////////////////////
    AssignOp::AssignOp(TensorLike* x, TensorLike* val, const string& name)
        : Operation({ x, val }, name.empty() ? "assign" : name)
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