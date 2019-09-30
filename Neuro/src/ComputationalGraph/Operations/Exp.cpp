#include "ComputationalGraph/Operations/Exp.h"

namespace Neuro
{
    //////////////////////////////////////////////////////////////////////////
    Op::Exp::Exp(NodeBase* x)
        : Operation({x})
    {
    }

    //////////////////////////////////////////////////////////////////////////
    void Op::Exp::ComputeInternal()
    {
        m_Inputs[0]->Map([](float x) {return ::exp(x); }, m_Output);
    }

    //////////////////////////////////////////////////////////////////////////
    void Op::Exp::ComputeGradientInternal(const Tensor& grad)
    {
        // in_grad = grad * e^x
        grad.Map([](float g, float x) {return g * x; }, m_Output, m_InputsGrads[0]);
    }
}