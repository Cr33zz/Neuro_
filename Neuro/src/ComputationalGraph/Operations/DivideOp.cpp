#include <algorithm>
#include "ComputationalGraph/Operations/DivideOp.h"

namespace Neuro
{
    //////////////////////////////////////////////////////////////////////////
    DivideOp::DivideOp(TensorLike* a, TensorLike* b, const string& name)
        : Operation({ a, b }, name.empty() ? "divide" : name)
    {
        m_Output.Resize(Shape(max(a->GetShape().Width(), b->GetShape().Width()), max(a->GetShape().Height(), b->GetShape().Height()), max(a->GetShape().Depth(), b->GetShape().Depth())));
    }

    //////////////////////////////////////////////////////////////////////////
    void DivideOp::ComputeInternal()
    {
        m_Output.ResizeBatch(max(m_Inputs[0]->Batch(), m_Inputs[1]->Batch()));
        m_Inputs[0]->Div(*m_Inputs[1], m_Output);
    }

    //////////////////////////////////////////////////////////////////////////
    void DivideOp::ComputeGradientInternal(const Tensor& grad)
    {
        auto& a = *m_Inputs[0];
        auto& b = *m_Inputs[1];

        auto& gShape = grad.GetShape();
        
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

        if (gShape == m_InputsGrads[1].GetShape())
            grad.MulElem(a, m_InputsGrads[1]);
        else
        {
            auto gradTemp = grad.MulElem(a);
            for (int i = WidthAxis; i <= BatchAxis; ++i)
            {
                if (gradTemp.Len(i) != 1 && a.Len(i) == 1)
                    gradTemp = sum(gradTemp, (EAxis)i);
            }
            gradTemp.CopyTo(m_InputsGrads[1]);
        }
    }
}