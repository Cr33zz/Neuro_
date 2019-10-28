#include "ComputationalGraph/Operations/LogOp.h"

namespace Neuro
{
    //////////////////////////////////////////////////////////////////////////
    LogOp::LogOp(TensorLike* x, const string& name)
        : Operation({ x }, name.empty() ? "log" : name)
    {
        m_Output.Resize(x->GetShape());
    }

    //////////////////////////////////////////////////////////////////////////
    void LogOp::ComputeInternal()
    {
        m_Output.ResizeBatch(m_Inputs[0]->Batch());
        m_Inputs[0]->Map([](float x) {return ::log(x); }, m_Output);
    }

    //////////////////////////////////////////////////////////////////////////
    void LogOp::ComputeGradientInternal(const Tensor& grad)
    {
        //in_grad = grad / x
        if (m_InputNodes[0]->CareAboutGradient())
            grad.Map([](float g, float x) {return g / x; }, *m_Inputs[0], m_InputsGrads[0]);
    }
}