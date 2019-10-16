#include "ComputationalGraph/Operations/EluOp.h"

namespace Neuro
{
    //////////////////////////////////////////////////////////////////////////
    EluOp::EluOp(TensorLike* x, float alpha, const string& name)
        : Operation({ x }, name.empty() ? "elu" : name), m_Alpha(alpha)
    {
        m_Output.Resize(x->GetShape());
    }

    //////////////////////////////////////////////////////////////////////////
    void EluOp::ComputeInternal()
    {
        m_Output.ResizeBatch(m_Inputs[0]->Batch());
        m_Inputs[0]->Elu(m_Alpha, m_Output);
    }

    //////////////////////////////////////////////////////////////////////////
    void EluOp::ComputeGradientInternal(const Tensor& grad)
    {
        if (m_InputNodes[0]->CareAboutGradient())
            m_Output.EluGradient(m_Output, grad, m_Alpha, m_InputsGrads[0]);
    }
}