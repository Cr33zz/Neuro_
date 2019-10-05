#include "ComputationalGraph/Operations/ExpOp.h"

namespace Neuro
{
    //////////////////////////////////////////////////////////////////////////
    ExpOp::ExpOp(TensorLike* x, const string& name)
        : Operation({ x }, name.empty() ? "exp" : name)
    {
        m_Output.Resize(x->GetShape());
    }

    //////////////////////////////////////////////////////////////////////////
    void ExpOp::ComputeInternal()
    {
        m_Output.ResizeBatch(m_Inputs[0]->Batch());
        m_Inputs[0]->Map([](float x) {return ::exp(x); }, m_Output);
    }

    //////////////////////////////////////////////////////////////////////////
    void ExpOp::ComputeGradientInternal(const Tensor& grad)
    {
        // in_grad = grad * e^x
        grad.Map([](float g, float x) {return g * x; }, m_Output, m_InputsGrads[0]);
    }
}