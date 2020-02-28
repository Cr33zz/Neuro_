#include "ComputationalGraph/Operations/RollOp.h"
#include "Initializers/Uniform.h"

namespace Neuro
{
    //////////////////////////////////////////////////////////////////////////
    RollOp::RollOp(TensorLike* x, int rollX, int rollY, const string& name)
        : Operation({ x }, name.empty() ? "random_roll" : name), m_RollX(rollX), m_RollY(rollY)
    {
        UpdateOutputShape();
    }

    //////////////////////////////////////////////////////////////////////////
    void RollOp::ComputeInternal()
    {
        m_Output.ResizeBatch(m_Inputs[0]->Batch());
        m_Inputs[0]->Roll2D(m_RollX, m_RollY, m_Output);
    }

    //////////////////////////////////////////////////////////////////////////
    void RollOp::ComputeGradientInternal(const Tensor& grad)
    {
        if (m_InputNodes[0]->CareAboutGradient())
            grad.Roll2D(-m_RollX, -m_RollY, m_InputsGrads[0]);
    }

    //////////////////////////////////////////////////////////////////////////
    RandomRollOp::RandomRollOp(TensorLike* x, uint32_t jitterScale, const string& name)
        : Operation({ x }, name.empty() ? "random_roll" : name), m_JitterScale(jitterScale)
    {
        UpdateOutputShape();
    }

    //////////////////////////////////////////////////////////////////////////
    void RandomRollOp::ComputeInternal()
    {
        m_Output.ResizeBatch(m_Inputs[0]->Batch());

        m_LastRollX = (int)::floor(Uniform::NextSingle(0, 1) * m_Output.Width() / (float)m_JitterScale) * m_JitterScale;
        m_LastRollY = (int)::floor(Uniform::NextSingle(0, 1) * m_Output.Height() / (float)m_JitterScale) * m_JitterScale;

        m_Inputs[0]->Roll2D(m_LastRollX, m_LastRollY, m_Output);
    }

    //////////////////////////////////////////////////////////////////////////
    void RandomRollOp::ComputeGradientInternal(const Tensor& grad)
    {
        if (m_InputNodes[0]->CareAboutGradient())
            grad.Roll2D(-m_LastRollX, -m_LastRollY, m_InputsGrads[0]);
    }
}