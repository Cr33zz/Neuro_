#include <algorithm>
#include "ComputationalGraph/Operations/MultiplyOp.h"

namespace Neuro
{
    //////////////////////////////////////////////////////////////////////////
    MultiplyOp::MultiplyOp(TensorLike* a, TensorLike* b, const string& name)
        : Operation({ a, b }, name.empty() ? "multiply" : name)
    {
        m_Output.Resize(Shape(max(a->GetShape().Width(), b->GetShape().Width()), max(a->GetShape().Height(), b->GetShape().Height()), max(a->GetShape().Depth(), b->GetShape().Depth())));
    }

    //////////////////////////////////////////////////////////////////////////
    MultiplyOp::MultiplyOp(TensorLike* x, float val, const string& name)
        : Operation({ x }, name.empty() ? "multiply" : name), m_Val(val)
    {
        m_Output.Resize(x->GetShape());
    }

    //////////////////////////////////////////////////////////////////////////
    void MultiplyOp::ComputeInternal()
    {
        if (m_Val)
        {
            m_Output.ResizeBatch(m_Inputs[0]->Batch());
            m_Inputs[0]->Mul(m_Val, m_Output);
        }
        else
        {
            m_Output.ResizeBatch(max(m_Inputs[0]->Batch(), m_Inputs[1]->Batch()));
            m_Inputs[0]->MulElem(*m_Inputs[1], m_Output);
        }
    }

    //////////////////////////////////////////////////////////////////////////
    void MultiplyOp::ComputeGradientInternal(const Tensor& grad)
    {
        if (m_Val)
        {
            if (m_InputNodes[0]->CareAboutGradient())
                grad.Mul(m_Val, m_InputsGrads[0]);
        }
        else
        {
            auto progressGrad = [&](size_t idx)
            {
                size_t idx2 = (idx + 1) % 2;
                auto& gShape = grad.GetShape();
                auto& iShape = m_InputsGrads[idx].GetShape();

                if (gShape == iShape)
                    grad.MulElem(*m_Inputs[idx2], m_InputsGrads[idx]);
                else
                {
                    auto gradTemp = grad.MulElem(*m_Inputs[idx2]);
                    for (int i = WidthAxis; i <= BatchAxis; ++i)
                    {
                        if (gradTemp.Len(i) != 1 && m_Inputs[idx]->Len(i) == 1)
                            gradTemp = sum(gradTemp, (EAxis)i);
                    }
                    gradTemp.CopyTo(m_InputsGrads[idx]);
                }
            };

            if (m_InputNodes[0]->CareAboutGradient())
                progressGrad(0);
            if (m_InputNodes[1]->CareAboutGradient())
                progressGrad(1);
        }
    }
}