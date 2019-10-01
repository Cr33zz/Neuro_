#include "ComputationalGraph/Operations/PowOp.h"

namespace Neuro
{
    //////////////////////////////////////////////////////////////////////////
    PowOp::PowOp(NodeBase* x, float p)
        : Operation({ x }), m_Power(p)
    {
    }

    //////////////////////////////////////////////////////////////////////////
    void PowOp::ComputeInternal()
    {
        m_Output.Resize(m_Inputs[0]->GetShape());
        m_Inputs[0]->Map([&](float x) {return ::pow(x, m_Power); }, m_Output);
    }

    //////////////////////////////////////////////////////////////////////////
    void PowOp::ComputeGradientInternal(const Tensor& grad)
    {
        //in_grad = p * grad^(p-1)
        if (m_Power == 2)
            grad.Mul(m_Power, m_InputsGrads[0]);
        else
            grad.Map([&](float g) {return m_Power * ::pow(g, m_Power - 1); }, m_InputsGrads[0]);
    }
}