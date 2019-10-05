#include "ComputationalGraph/Operations/LeakyReLUOp.h"

namespace Neuro
{
    //////////////////////////////////////////////////////////////////////////
    LeakyReLUOp::LeakyReLUOp(TensorLike* x, float alpha, const string& name)
        : Operation({ x }, name.empty() ? "leaky_relu" : name), m_Alpha(alpha)
    {
    }

    //////////////////////////////////////////////////////////////////////////
    void LeakyReLUOp::ComputeInternal()
    {
        m_Output.Resize(m_Inputs[0]->GetShape());
        m_Inputs[0]->LeakyReLU(m_Alpha, m_Output);
    }

    //////////////////////////////////////////////////////////////////////////
    void LeakyReLUOp::ComputeGradientInternal(const Tensor& grad)
    {
        m_Output.LeakyReLUGradient(m_Output, grad, m_Alpha, m_InputsGrads[0]);
    }
}