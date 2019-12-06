#include <algorithm>
#include "ComputationalGraph/Operations/SubtractOp.h"

namespace Neuro
{
    //////////////////////////////////////////////////////////////////////////
    SubtractOp::SubtractOp(TensorLike* a, TensorLike* b, const string& name)
        : Operation({ a, b }, name.empty() ? "sub" : name)
    {
        UpdateOutputShape();
    }

    //////////////////////////////////////////////////////////////////////////
    void SubtractOp::UpdateOutputShape()
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
    void SubtractOp::ComputeInternal()
    {
        if (m_Name == "disc/loss/cross_entropy/1-yTrue")
        {
            m_Inputs[0]->DebugDumpValues("xxx.log");
            cout << "xxx";
        }
        m_Output.ResizeBatch(max(m_Inputs[0]->Batch(), m_Inputs[1]->Batch()));
        return m_Inputs[0]->Sub(*m_Inputs[1], m_Output);
    }

    //////////////////////////////////////////////////////////////////////////
    void SubtractOp::ComputeGradientInternal(const Tensor& grad)
    {
        if (m_Name == "disc/loss/cross_entropy/1-yTrue")
        {
            cout << "xxx";
        }

        auto progressGrad = [](Tensor& inputGrad, const Tensor& grad)
        {
            auto& gShape = grad.GetShape();
            auto& iShape = inputGrad.GetShape();

            if (gShape == iShape)
                grad.CopyTo(inputGrad);
            // check common cases to utilize optimized sum for combinations of axes
            else if (gShape.Width() == iShape.Width() && gShape.Height() == iShape.Height() && gShape.Depth() == iShape.Depth() && iShape.Batch() == 1)
                grad.Sum(BatchAxis, inputGrad); // used in case of biases in dense layers
            else if (iShape.Width() == 1 && iShape.Height() == 1 && gShape.Depth() == iShape.Depth() && iShape.Batch() == 1)
                grad.Sum(_013Axes, inputGrad); // used in case of biases in convolutional layers
            else if (iShape.Width() == 1 && iShape.Height() == 1 && gShape.Depth() == iShape.Depth() && gShape.Batch() == iShape.Batch())
                grad.Sum(_01Axes, inputGrad);
            else
            {
                auto gradTemp = grad;
                for (int i = WidthAxis; i <= BatchAxis; ++i)
                {
                    if (gradTemp.Len(i) != 1 && inputGrad.Len(i) == 1)
                        gradTemp = sum(gradTemp, (EAxis)i);
                }
                gradTemp.CopyTo(inputGrad);
            }
        };

        if (m_InputNodes[0]->CareAboutGradient())
            progressGrad(m_InputsGrads[0], grad);

        if (m_InputNodes[1]->CareAboutGradient())
            progressGrad(m_InputsGrads[1], grad.Negated());
    }
}