#include "ComputationalGraph/Operations/ReLUOp.h"

namespace Neuro
{
    //////////////////////////////////////////////////////////////////////////
    ReLUOp::ReLUOp(TensorLike* x, const string& name)
        : Operation({ x }, name.empty() ? "relu" : name)
    {
    }

    //////////////////////////////////////////////////////////////////////////
    void ReLUOp::ComputeInternal()
    {
        m_Output.Resize(m_Inputs[0]->GetShape());
        m_Inputs[0]->ReLU(m_Output);
    }

    //////////////////////////////////////////////////////////////////////////
    void ReLUOp::ComputeGradientInternal(const Tensor& grad)
    {
        m_Output.ReLUGradient(m_Output, grad, m_InputsGrads[0]);
    }
}