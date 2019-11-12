#include "ComputationalGraph/Operations/SqrtOp.h"

namespace Neuro
{
    //////////////////////////////////////////////////////////////////////////
    SqrtOp::SqrtOp(TensorLike* x, const string& name)
        : Operation({ x }, name.empty() ? "sqrt" : name)
    {
        m_Output.Resize(x->GetShape());
    }

    //////////////////////////////////////////////////////////////////////////
    void SqrtOp::ComputeInternal()
    {
        m_Output.ResizeBatch(m_Inputs[0]->Batch());
        m_Inputs[0]->Sqrt(m_Output);
    }

    //////////////////////////////////////////////////////////////////////////
    void SqrtOp::ComputeGradientInternal(const Tensor& grad)
    {
        //in_grad = grad / (2 * sqrt(x))
        if (m_InputNodes[0]->CareAboutGradient())
            grad.Div(1.f, 2.f, m_Output, m_InputsGrads[0]);
    }
}