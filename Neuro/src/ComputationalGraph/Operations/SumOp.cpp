#include "ComputationalGraph/Operations/SumOp.h"

namespace Neuro
{
    //////////////////////////////////////////////////////////////////////////
    SumOp::SumOp(TensorLike* x, EAxis axis, const string& name)
        : Operation({ x }, name.empty() ? "sum" : name), m_Axis(axis)
    {
        UpdateOutputShape();
    }

    //////////////////////////////////////////////////////////////////////////
    void SumOp::UpdateOutputShape()
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
        else if (m_Axis == _012Axes)
            m_Output.Resize(Shape(1, 1, 1, inputShape.Batch()));
        else
            NEURO_ASSERT(false, "Unsupported axis.");
    }

    //////////////////////////////////////////////////////////////////////////
    void SumOp::ComputeInternal()
    {
        if (m_Axis == WidthAxis || m_Axis == HeightAxis || m_Axis == DepthAxis || m_Axis == _01Axes || m_Axis == _012Axes)
            m_Output.ResizeBatch(m_Inputs[0]->Batch());

        m_Inputs[0]->Sum(m_Axis, m_Output);
    }

    //////////////////////////////////////////////////////////////////////////
    void SumOp::ComputeGradientInternal(const Tensor& grad)
    {
        if (m_InputNodes[0]->CareAboutGradient())
        {
            m_InputsGrads[0].One();
            m_InputsGrads[0].MulElem(grad, m_InputsGrads[0]);
        }
    }
}