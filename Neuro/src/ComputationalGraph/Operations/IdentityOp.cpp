#include "ComputationalGraph/Operations/IdentityOp.h"

namespace Neuro
{
    //////////////////////////////////////////////////////////////////////////
    IdentityOp::IdentityOp(TensorLike* x, const string& name)
        : Operation({ x }, name.empty() ? "identity" : name)
    {
        UpdateOutputShape();
    }

    //////////////////////////////////////////////////////////////////////////
    void IdentityOp::ComputeInternal()
    {
        m_Output.ResizeBatch(m_Inputs[0]->Batch());
        m_Inputs[0]->CopyTo(m_Output);
    }

    //////////////////////////////////////////////////////////////////////////
    void IdentityOp::ComputeGradientInternal(const Tensor& grad)
    {
        if (m_InputNodes[0]->CareAboutGradient())
            grad.CopyTo(m_InputsGrads[0]);
    }
}