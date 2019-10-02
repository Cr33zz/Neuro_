#include "ComputationalGraph/Operations/SqrtOp.h"

namespace Neuro
{
    //////////////////////////////////////////////////////////////////////////
    SqrtOp::SqrtOp(TensorLike* x)
        : Operation({ x }, "sqrt")
    {
    }

    //////////////////////////////////////////////////////////////////////////
    void SqrtOp::ComputeInternal()
    {
        m_Output.Resize(m_Inputs[0]->GetShape());
        m_Inputs[0]->Map([](float x) {return ::sqrt(x); }, m_Output);
    }

    //////////////////////////////////////////////////////////////////////////
    void SqrtOp::ComputeGradientInternal(const Tensor& grad)
    {
        //in_grad = grad / (2 * sqrt(x))
        grad.Map([](float g, float x) {return g / 2 * x; }, m_Output, m_InputsGrads[0]);
    }
}