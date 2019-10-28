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
        m_Inputs[0]->Map([](float x) {return ::sqrt(x); }, m_Output);
    }

    //////////////////////////////////////////////////////////////////////////
    void SqrtOp::ComputeGradientInternal(const Tensor& grad)
    {
        //in_grad = grad / (2 * sqrt(x))
        if (m_InputNodes[0]->CareAboutGradient())
            grad.Map([](float g, float x) {return g / 2 * x; }, m_Output, m_InputsGrads[0]);
    }
}