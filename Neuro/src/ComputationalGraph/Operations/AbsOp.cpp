#include "ComputationalGraph/Operations/AbsOp.h"

namespace Neuro
{
    //////////////////////////////////////////////////////////////////////////
    AbsOp::AbsOp(TensorLike* x, const string& name)
        : Operation({ x }, name.empty() ? "abs" : name)
    {
        m_Output.Resize(x->GetShape());
    }

    //////////////////////////////////////////////////////////////////////////
    void AbsOp::ComputeInternal()
    {
        m_Output.ResizeBatch(m_Inputs[0]->Batch());
        m_Inputs[0]->Abs(m_Output);
    }

    //////////////////////////////////////////////////////////////////////////
    void AbsOp::ComputeGradientInternal(const Tensor& grad)
    {
        if (m_InputNodes[0]->CareAboutGradient())
            grad.AbsGradient(*m_Inputs[0], grad, m_InputsGrads[0]);
    }
}