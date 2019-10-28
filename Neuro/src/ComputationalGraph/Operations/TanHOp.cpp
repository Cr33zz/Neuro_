#include "ComputationalGraph/Operations/TanHOp.h"

namespace Neuro
{
    //////////////////////////////////////////////////////////////////////////
    TanHOp::TanHOp(TensorLike* x, const string& name)
        : Operation({ x }, name.empty() ? "tanh" : name)
    {
        m_Output.Resize(x->GetShape());
    }

    //////////////////////////////////////////////////////////////////////////
    void TanHOp::ComputeInternal()
    {
        m_Output.ResizeBatch(m_Inputs[0]->Batch());
        m_Inputs[0]->Tanh(m_Output);
    }

    //////////////////////////////////////////////////////////////////////////
    void TanHOp::ComputeGradientInternal(const Tensor& grad)
    {
        if (m_InputNodes[0]->CareAboutGradient())
            m_Output.TanhGradient(m_Output, grad, m_InputsGrads[0]);
    }
}