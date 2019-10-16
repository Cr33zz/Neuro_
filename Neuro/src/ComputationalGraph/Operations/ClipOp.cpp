#include "ComputationalGraph/Operations/ClipOp.h"

namespace Neuro
{
    //////////////////////////////////////////////////////////////////////////
    ClipOp::ClipOp(TensorLike* x, float min, float max, const string& name)
        : Operation({ x }, name.empty() ? "clip" : name), m_Min(min), m_Max(max)
    {
        m_Output.Resize(x->GetShape());
    }

    //////////////////////////////////////////////////////////////////////////
    void ClipOp::ComputeInternal()
    {
        m_Output.ResizeBatch(m_Inputs[0]->Batch());
        m_Inputs[0]->Clipped(m_Min, m_Max, m_Output);
    }

    //////////////////////////////////////////////////////////////////////////
    void ClipOp::ComputeGradientInternal(const Tensor& grad)
    {
        if (m_InputNodes[0]->CareAboutGradient())
            grad.Map([&](float g, float x) {return (x >= m_Min && x <= m_Max) ? g : 0; }, m_Output, m_InputsGrads[0]);
    }
}