#include "ComputationalGraph/Operations/PowOp.h"

namespace Neuro
{
    //////////////////////////////////////////////////////////////////////////
    PowOp::PowOp(TensorLike* x, float p, const string& name)
        : Operation({ x }, name.empty() ? "pow" : name), m_Power(p)
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
        //in_grad = grad * p * x^(p-1)
        if (m_Power == 2)
            grad.Map([&](float g, float x) {return g * 2 * x; }, *m_Inputs[0], m_InputsGrads[0]);
        else
            grad.Map([&](float g, float x) {return g * m_Power * ::pow(x, m_Power - 1); }, *m_Inputs[0], m_InputsGrads[0]);
    }
}