#include "ComputationalGraph/Operations/NormalizeGradientOp.h"

namespace Neuro
{
    //////////////////////////////////////////////////////////////////////////
    NormalizeGradientOp::NormalizeGradientOp(TensorLike* x, size_t order, float scale, const string& name)
        : Operation({ x }, name.empty() ? "normalize_gradient" : name), m_Order((float)order), m_Scale(scale)
    {
        UpdateOutputShape();
    }

    //////////////////////////////////////////////////////////////////////////
    void NormalizeGradientOp::ComputeInternal()
    {
        m_Inputs[0]->CopyTo(m_Output);
    }

    //////////////////////////////////////////////////////////////////////////
    void NormalizeGradientOp::ComputeGradientInternal(const Tensor& grad)
    {
        float norm = 1e-8f;

        if (m_Order == 1)
            norm += grad.AbsSum(NoneAxis)(0);
        else
            norm += ::pow(sum(pow(grad, m_Order), NoneAxis)(0), 1.f / m_Order);

        grad.Mul(m_Scale / norm, m_InputsGrads[0]);
    }
}