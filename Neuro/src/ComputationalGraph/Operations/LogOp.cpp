#include "ComputationalGraph/Operations/LogOp.h"

namespace Neuro
{
    //////////////////////////////////////////////////////////////////////////
    LogOp::LogOp(TensorLike* x, const string& name)
        : Operation({ x }, name.empty() ? "log" : name)
    {
        UpdateOutputShape();
    }

    //////////////////////////////////////////////////////////////////////////
    void LogOp::ComputeInternal()
    {
        m_Output.ResizeBatch(m_Inputs[0]->Batch());
        m_Inputs[0]->Log(m_Output);
    }

    //////////////////////////////////////////////////////////////////////////
    void LogOp::ComputeGradientInternal(const Tensor& grad)
    {
        //in_grad = grad / x
        if (m_InputNodes[0]->CareAboutGradient())
            grad.Div(*m_Inputs[0], m_InputsGrads[0]);
    }
}