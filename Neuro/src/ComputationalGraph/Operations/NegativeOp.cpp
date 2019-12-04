#include "ComputationalGraph/Operations/NegativeOp.h"

namespace Neuro
{
    //////////////////////////////////////////////////////////////////////////
    NegativeOp::NegativeOp(TensorLike* x, const string& name)
        : Operation({ x }, name.empty() ? "negative" : name)
    {
        UpdateOutputShape();
    }

    //////////////////////////////////////////////////////////////////////////
    void NegativeOp::ComputeInternal()
    {
        m_Output.ResizeBatch(m_Inputs[0]->Batch());
        m_Inputs[0]->Negated(m_Output);
    }

    //////////////////////////////////////////////////////////////////////////
    void NegativeOp::ComputeGradientInternal(const Tensor& grad)
    {
        //in_grad = -grad
        if (m_InputNodes[0]->CareAboutGradient())
            grad.Negated(m_InputsGrads[0]);
    }
}