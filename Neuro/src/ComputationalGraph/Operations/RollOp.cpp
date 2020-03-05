#include "ComputationalGraph/Operations/RollOp.h"
#include "Initializers/Uniform.h"

namespace Neuro
{
    //////////////////////////////////////////////////////////////////////////
    RollOp::RollOp(TensorLike* x, int rollX, int rollY, const string& name)
        : Operation({ x }, name.empty() ? "roll" : name), m_RollX(rollX), m_RollY(rollY)
    {
        UpdateOutputShape();
    }

    RollOp::RollOp(TensorLike* x, TensorLike* rollX, TensorLike* rollY, const string& name)
        : Operation({ x, rollX, rollY }, name.empty() ? "roll" : name)
    {
        NEURO_ASSERT(rollX->GetShape().Length == 1, "Roll x must be a scalar.");
        NEURO_ASSERT(rollY->GetShape().Length == 1, "Roll y must be a scalar.");
        UpdateOutputShape();
    }

    //////////////////////////////////////////////////////////////////////////
    void RollOp::UpdateOutputShape()
    {
        m_Output.Resize(m_InputNodes[0]->GetShape());
    }

    //////////////////////////////////////////////////////////////////////////
    void RollOp::ComputeInternal()
    {
        m_Output.ResizeBatch(m_Inputs[0]->Batch());

        if (m_Inputs.size() == 3)
        {
            m_RollX = (int)(*m_Inputs[1])(0);
            m_RollY = (int)(*m_Inputs[2])(0);
        }

        if (m_RollX == 0 && m_RollY == 0)
            m_Inputs[0]->CopyTo(m_Output);
        else
            m_Inputs[0]->Roll2D(m_RollX, m_RollY, m_Output);
    }

    //////////////////////////////////////////////////////////////////////////
    void RollOp::ComputeGradientInternal(const Tensor& grad)
    {
        if (m_InputNodes[0]->CareAboutGradient())
        {
            if (m_RollX == 0 && m_RollY == 0)
                grad.CopyTo(m_InputsGrads[0]);
            else
                grad.Roll2D(-m_RollX, -m_RollY, m_InputsGrads[0]);
        }
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