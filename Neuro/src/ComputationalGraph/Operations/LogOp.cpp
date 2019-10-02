#include "ComputationalGraph/Operations/LogOp.h"

namespace Neuro
{
    //////////////////////////////////////////////////////////////////////////
    LogOp::LogOp(TensorLike* x)
        : Operation({x}, "log")
    {
    }

    //////////////////////////////////////////////////////////////////////////
    void LogOp::ComputeInternal()
    {
        m_Output.Resize(m_Inputs[0]->GetShape());
        m_Inputs[0]->Map([](float x) {return ::log(x); }, m_Output);
    }

    //////////////////////////////////////////////////////////////////////////
    void LogOp::ComputeGradientInternal(const Tensor& grad)
    {
        //in_grad = grad / x
        grad.Map([](float g, float x) {return g / x; }, *m_Inputs[0], m_InputsGrads[0]);
    }
}