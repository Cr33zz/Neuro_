#include "ComputationalGraph/Operations/MeanOp.h"

namespace Neuro
{
    //////////////////////////////////////////////////////////////////////////
    MeanOp::MeanOp(TensorLike* x, EAxis axis, const string& name)
        : Operation({ x }, name.empty() ? "mean" : name), m_Axis(axis)
    {
        Shape inputShape = x->GetShape();
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
        else
            assert(false);
    }

    //////////////////////////////////////////////////////////////////////////
    void MeanOp::ComputeInternal()
    {
        if (m_Axis == WidthAxis || m_Axis == HeightAxis || m_Axis == DepthAxis)
            m_Output.ResizeBatch(m_Inputs[0]->Batch());

        m_Inputs[0]->Mean(m_Axis, m_Output);
    }

    //////////////////////////////////////////////////////////////////////////
    void MeanOp::ComputeGradientInternal(const Tensor& grad)
    {
        float n = (float)m_Inputs[0]->Length();
        if (m_Axis != GlobalAxis)
            n = (float)m_Inputs[0]->Stride(m_Axis);

        m_InputsGrads[0].FillWithValue(1);
        m_InputsGrads[0].MulElem(grad.Div(n), m_InputsGrads[0]);
    }
}