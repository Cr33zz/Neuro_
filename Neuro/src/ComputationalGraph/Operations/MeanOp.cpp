#include "ComputationalGraph/Operations/MeanOp.h"

namespace Neuro
{
    //////////////////////////////////////////////////////////////////////////
    MeanOp::MeanOp(TensorLike* x, EAxis axis, const string& name)
        : Operation({ x }, name.empty() ? "mean" : name), m_Axis(axis)
    {
        UpdateOutputShape();
    }

    //////////////////////////////////////////////////////////////////////////
    void MeanOp::UpdateOutputShape()
    {
        Shape inputShape = m_InputNodes[0]->GetShape();
        if (m_Axis == GlobalAxis)
            m_Output.Resize(Shape(1, 1, 1, 1));
        else if (m_Axis == WidthAxis)
            m_Output.Resize(Shape(1, inputShape.Height(), inputShape.Depth(), inputShape.Batch()));
        else if (m_Axis == HeightAxis)
            m_Output.Resize(Shape(inputShape.Width(), 1, inputShape.Depth(), inputShape.Batch()));
        else if (m_Axis == DepthAxis)
            m_Output.Resize(Shape(inputShape.Width(), inputShape.Height(), 1, inputShape.Batch()));
        else if (m_Axis == BatchAxis)
            m_Output.Resize(Shape(inputShape.Width(), inputShape.Height(), inputShape.Depth(), 1));
        else if (m_Axis == _01Axes)
            m_Output.Resize(Shape(1, 1, inputShape.Depth(), inputShape.Batch()));
        else
            NEURO_ASSERT(false, "Unsupported axis.");
    }

    //////////////////////////////////////////////////////////////////////////
    void MeanOp::ComputeInternal()
    {
        if (m_Axis == WidthAxis || m_Axis == HeightAxis || m_Axis == DepthAxis || m_Axis == _01Axes)
            m_Output.ResizeBatch(m_Inputs[0]->Batch());

        m_Inputs[0]->Mean(m_Axis, m_Output);
    }

    //////////////////////////////////////////////////////////////////////////
    void MeanOp::ComputeGradientInternal(const Tensor& grad)
    {
        if (m_InputNodes[0]->CareAboutGradient())
        {
            float n = (float)m_Inputs[0]->Length();
            if (m_Axis == _01Axes)
                n = (float)(m_Inputs[0]->Width() * m_Inputs[0]->Height());
            else if (m_Axis != GlobalAxis)
                n = (float)m_Inputs[0]->Len(m_Axis);

            m_InputsGrads[0].One();
            m_InputsGrads[0].MulElem(grad.Div(n), m_InputsGrads[0]);
        }
    }
}