#include "ComputationalGraph/Operations/ClipOp.h"

namespace Neuro
{
    //////////////////////////////////////////////////////////////////////////
    ClipOp::ClipOp(TensorLike* x, float min, float max, const string& name)
        : Operation({ x }, name.empty() ? "clip" : name), m_Min(min), m_Max(max)
    {
    }

    //////////////////////////////////////////////////////////////////////////
    void ClipOp::ComputeInternal()
    {
        m_Output.Resize(m_Inputs[0]->GetShape());
        m_Inputs[0]->Clipped(m_Min, m_Max, m_Output);
    }

    //////////////////////////////////////////////////////////////////////////
    void ClipOp::ComputeGradientInternal(const Tensor& grad)
    {
        grad.Map([&](float g, float x) {return (x >= m_Min && x <= m_Max) ? g : 0; }, m_Output, m_InputsGrads[0]);
    }
}