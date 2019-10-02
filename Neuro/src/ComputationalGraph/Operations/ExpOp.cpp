#include "ComputationalGraph/Operations/ExpOp.h"

namespace Neuro
{
    //////////////////////////////////////////////////////////////////////////
    ExpOp::ExpOp(TensorLike* x)
        : Operation({x}, "exp")
    {
    }

    //////////////////////////////////////////////////////////////////////////
    void ExpOp::ComputeInternal()
    {
        m_Output.Resize(m_Inputs[0]->GetShape());
        m_Inputs[0]->Map([](float x) {return ::exp(x); }, m_Output);
    }

    //////////////////////////////////////////////////////////////////////////
    void ExpOp::ComputeGradientInternal(const Tensor& grad)
    {
        // in_grad = grad * e^x
        grad.Map([](float g, float x) {return g * x; }, m_Output, m_InputsGrads[0]);
    }
}