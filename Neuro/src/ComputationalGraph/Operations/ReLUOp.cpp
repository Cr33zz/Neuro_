#include "ComputationalGraph/Operations/ReLUOp.h"

namespace Neuro
{
    //////////////////////////////////////////////////////////////////////////
    ReLUOp::ReLUOp(TensorLike* x, const string& name)
        : Operation({ x }, name.empty() ? "relu" : name)
    {
        UpdateOutputShape();
    }

    //////////////////////////////////////////////////////////////////////////
    void ReLUOp::ComputeInternal()
    {
        m_Output.ResizeBatch(m_Inputs[0]->Batch());
        m_Inputs[0]->ReLU(m_Output);
    }

    //////////////////////////////////////////////////////////////////////////
    void ReLUOp::ComputeGradientInternal(const Tensor& grad)
    {
        if (m_InputNodes[0]->CareAboutGradient())
            m_Output.ReLUGradient(m_Output, grad, m_InputsGrads[0]);
    }
}