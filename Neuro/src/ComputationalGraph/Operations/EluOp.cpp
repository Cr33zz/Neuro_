#include "ComputationalGraph/Operations/EluOp.h"

namespace Neuro
{
    //////////////////////////////////////////////////////////////////////////
    EluOp::EluOp(TensorLike* x, float alpha)
        : Operation({ x }), m_Alpha(alpha)
    {
    }

    //////////////////////////////////////////////////////////////////////////
    void EluOp::ComputeInternal()
    {
        m_Output.Resize(m_Inputs[0]->GetShape());
        m_Inputs[0]->Elu(m_Alpha, m_Output);
    }

    //////////////////////////////////////////////////////////////////////////
    void EluOp::ComputeGradientInternal(const Tensor& grad)
    {
        m_Output.EluGradient(m_Output, grad, m_Alpha, m_InputsGrads[0]);
    }
}