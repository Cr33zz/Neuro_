#include "ComputationalGraph/Operations/ExpOp.h"

namespace Neuro
{
    //////////////////////////////////////////////////////////////////////////
    ExpOp::ExpOp(TensorLike* x, const string& name)
        : Operation({ x }, name.empty() ? "exp" : name)
    {
        UpdateOutputShape();
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
        if (m_InputNodes[0]->CareAboutGradient())
            grad.Map([](float g, float x) {return g * x; }, m_Output, m_InputsGrads[0]);
    }
}