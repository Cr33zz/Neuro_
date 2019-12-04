#include "ComputationalGraph/Operations/PowOp.h"

namespace Neuro
{
    //////////////////////////////////////////////////////////////////////////
    PowOp::PowOp(TensorLike* x, TensorLike* p, const string& name)
        : Operation({ x, p }, name.empty() ? "pow" : name)
    {
        assert(p->GetShape().Length == 1);
        UpdateOutputShape();
    }

    //////////////////////////////////////////////////////////////////////////
    PowOp::PowOp(TensorLike* x, float p, const string& name)
        : Operation({ x }, name.empty() ? "pow" : name), m_Power(p)
    {
        m_Output.Resize(x->GetShape());
    }

    //////////////////////////////////////////////////////////////////////////
    void PowOp::ComputeInternal()
    {
        m_Output.ResizeBatch(m_Inputs[0]->Batch());

        float power = m_InputNodes.size() == 1 ? m_Power : (*m_Inputs[1])(0);
        m_Inputs[0]->Pow(power, m_Output);
    }

    //////////////////////////////////////////////////////////////////////////
    void PowOp::ComputeGradientInternal(const Tensor& grad)
    {
        float power = m_InputNodes.size() == 1 ? m_Power : (*m_Inputs[1])(0);

        //in_grad = grad * p * x^(p-1)
        if (m_InputNodes[0]->CareAboutGradient())
        {
            grad.PowGradient(*m_Inputs[0], power, grad, m_InputsGrads[0]);
        }

        if (m_InputNodes.size() > 1 && m_InputNodes[1]->CareAboutGradient())
        {
            //in_grad2 = grad * x^(p) * log(x)
            grad.Map([&](float g, float x) {return g * ::pow(x, power) * ::log(x); }, *m_Inputs[0], m_InputsGrads[1]);
        }
    }
}