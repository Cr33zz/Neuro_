#include <algorithm>
#include "ComputationalGraph/Operations/DivideOp.h"

namespace Neuro
{
    //////////////////////////////////////////////////////////////////////////
    DivideOp::DivideOp(TensorLike* a, TensorLike* b, const string& name)
        : Operation({ a, b }, name.empty() ? "divide" : name)
    {
        UpdateOutputShape();
    }

    //////////////////////////////////////////////////////////////////////////
    DivideOp::DivideOp(TensorLike* x, float val, const string& name)
        : Operation({ x }, name.empty() ? "divide" : name), m_Val(val)
    {
        UpdateOutputShape();
    }

    //////////////////////////////////////////////////////////////////////////
    void DivideOp::UpdateOutputShape()
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
    void DivideOp::ComputeInternal()
    {
        if (m_Val)
        {
            m_Output.ResizeBatch(m_Inputs[0]->Batch());
            m_Inputs[0]->Div(m_Val, m_Output);
        }
        else
        {
            m_Output.ResizeBatch(max(m_Inputs[0]->Batch(), m_Inputs[1]->Batch()));
            m_Inputs[0]->Div(*m_Inputs[1], m_Output);
        }
    }

    //////////////////////////////////////////////////////////////////////////
    void DivideOp::ComputeGradientInternal(const Tensor& grad)
    {
        if (m_Val)
        {
            if (m_InputNodes[0]->CareAboutGradient())
                grad.Div(m_Val, m_InputsGrads[0]);
        }
        else
        {
            auto& a = *m_Inputs[0];
            auto& b = *m_Inputs[1];

            auto& gShape = grad.GetShape();

            if (m_InputNodes[0]->CareAboutGradient())
            {
                if (gShape == m_InputsGrads[0].GetShape())
                    grad.Div(b, m_InputsGrads[0]);
                else
                {
                    auto gradTemp = grad.Div(b);
                    for (int i = WidthAxis; i <= BatchAxis; ++i)
                    {
                        if (gradTemp.Len(i) != 1 && a.Len(i) == 1)
                            gradTemp = sum(gradTemp, (EAxis)i);
                    }
                    gradTemp.CopyTo(m_InputsGrads[0]);
                }
            }

            if (m_InputNodes[1]->CareAboutGradient())
            {
                if (gShape == m_InputsGrads[1].GetShape())
                {
                    grad.MulElem(a).Negated().Div(b.Pow(2), m_InputsGrads[1]);
                }
                else
                {
                    auto gradTemp = grad.MulElem(a).Negated().Div(b.Pow(2));
                    for (int i = WidthAxis; i <= BatchAxis; ++i)
                    {
                        if (gradTemp.Len(i) != 1 && b.Len(i) == 1)
                            gradTemp = sum(gradTemp, (EAxis)i);
                    }
                    gradTemp.CopyTo(m_InputsGrads[1]);
                }
            }
        }
    }
}