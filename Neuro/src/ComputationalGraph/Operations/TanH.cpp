#include "ComputationalGraph/Operations/TanH.h"

namespace Neuro
{
    //////////////////////////////////////////////////////////////////////////
    Op::TanH::TanH(NodeBase* x)
        : Operation({ x })
    {
    }

    //////////////////////////////////////////////////////////////////////////
    void Op::TanH::ComputeInternal()
    {
        m_Output.Resize(m_Inputs[0]->GetShape());
        m_Inputs[0]->Tanh(m_Output);
    }

    //////////////////////////////////////////////////////////////////////////
    void Op::TanH::ComputeGradientInternal(const Tensor& grad)
    {
        m_Output.TanhGradient(m_Output, grad, m_InputsGrads[0]);
    }
}