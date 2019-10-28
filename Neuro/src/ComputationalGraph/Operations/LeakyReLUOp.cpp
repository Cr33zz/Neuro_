#include "ComputationalGraph/Operations/LeakyReLUOp.h"

namespace Neuro
{
    //////////////////////////////////////////////////////////////////////////
    LeakyReLUOp::LeakyReLUOp(TensorLike* x, float alpha, const string& name)
        : Operation({ x }, name.empty() ? "leaky_relu" : name), m_Alpha(alpha)
    {
        m_Output.Resize(x->GetShape());
    }

    //////////////////////////////////////////////////////////////////////////
    void LeakyReLUOp::ComputeInternal()
    {
        m_Output.ResizeBatch(m_Inputs[0]->Batch());
        m_Inputs[0]->LeakyReLU(m_Alpha, m_Output);
    }

    //////////////////////////////////////////////////////////////////////////
    void LeakyReLUOp::ComputeGradientInternal(const Tensor& grad)
    {
        if (m_InputNodes[0]->CareAboutGradient())
            m_Output.LeakyReLUGradient(m_Output, grad, m_Alpha, m_InputsGrads[0]);
    }
}