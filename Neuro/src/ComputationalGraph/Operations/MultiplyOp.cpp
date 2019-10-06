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
    void MultiplyOp::ComputeInternal()
    {
        m_Output.ResizeBatch(max(m_Inputs[0]->Batch(), m_Inputs[1]->Batch()));
        return m_Inputs[0]->MulElem(*m_Inputs[1], m_Output);
    }

    //////////////////////////////////////////////////////////////////////////
    void MultiplyOp::ComputeGradientInternal(const Tensor& grad)
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

        progressGrad(0);
        progressGrad(1);
    }
}