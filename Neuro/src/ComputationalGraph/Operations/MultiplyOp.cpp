#include <algorithm>
#include "ComputationalGraph/Operations/MultiplyOp.h"

namespace Neuro
{
    //////////////////////////////////////////////////////////////////////////
    MultiplyOp::MultiplyOp(TensorLike* a, TensorLike* b, const string& name)
        : Operation({ a, b }, name.empty() ? "multiply" : name)
    {
        UpdateOutputShape();
    }

    //////////////////////////////////////////////////////////////////////////
    MultiplyOp::MultiplyOp(TensorLike* x, float val, const string& name)
        : Operation({ x }, name.empty() ? "multiply" : name), m_Val(val)
    {
        UpdateOutputShape();
    }

    //////////////////////////////////////////////////////////////////////////
    void MultiplyOp::UpdateOutputShape()
    {
        if (m_InputNodes.size() == 2)
        {
            const Shape& aShape = m_InputNodes[0]->GetShape();
            const Shape& bShape = m_InputNodes[1]->GetShape();
            NEURO_ASSERT(aShape.Width() == bShape.Width() || aShape.Width() == 1 || bShape.Width() == 1, "Mismatched width " << aShape.Width() << " and " << bShape.Width());
            NEURO_ASSERT(aShape.Height() == bShape.Height() || aShape.Height() == 1 || bShape.Height() == 1, "Mismatched height " << aShape.Height() << " and " << bShape.Height());
            NEURO_ASSERT(aShape.Depth() == bShape.Depth() || aShape.Depth() == 1 || bShape.Depth() == 1, "Mismatched depth " << aShape.Depth() << " and " << bShape.Depth());
            NEURO_ASSERT(aShape.Batch() == bShape.Batch() || aShape.Batch() == 1 || bShape.Batch() == 1, "Mismatched batch " << aShape.Batch() << " and " << bShape.Batch());
            m_Output.Resize(Shape(max(aShape.Width(), bShape.Width()), max(aShape.Height(), bShape.Height()), max(aShape.Depth(), bShape.Depth()), max(aShape.Batch(), bShape.Batch())));
        }
        else
            __super::UpdateOutputShape();
    }

    //////////////////////////////////////////////////////////////////////////
    void MultiplyOp::ComputeInternal()
    {
        if (m_InputNodes.size() == 1)
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
        if (m_InputNodes.size() == 1)
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