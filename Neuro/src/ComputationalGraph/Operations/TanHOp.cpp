#include "ComputationalGraph/Operations/TanHOp.h"

namespace Neuro
{
    //////////////////////////////////////////////////////////////////////////
    TanHOp::TanHOp(TensorLike* x, const string& name)
        : Operation({ x }, name.empty() ? "tanh" : name)
    {
    }

    //////////////////////////////////////////////////////////////////////////
    void TanHOp::ComputeInternal()
    {
        m_Output.Resize(m_Inputs[0]->GetShape());
        m_Inputs[0]->Tanh(m_Output);
    }

    //////////////////////////////////////////////////////////////////////////
    void TanHOp::ComputeGradientInternal(const Tensor& grad)
    {
        m_Output.TanhGradient(m_Output, grad, m_InputsGrads[0]);
    }
}