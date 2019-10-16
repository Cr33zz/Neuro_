#include "ComputationalGraph/Operations/SumOp.h"

namespace Neuro
{
    //////////////////////////////////////////////////////////////////////////
    SumOp::SumOp(TensorLike* x, EAxis axis, const string& name)
        : Operation({ x }, name.empty() ? "sum" : name), m_Axis(axis)
    {
        if (m_Axis == GlobalAxis)
            m_Output.Resize(Shape(1, 1, 1, 1));
        else if (m_Axis == WidthAxis)
            m_Output.Resize(Shape(1, m_Inputs[0]->Len(1), m_Inputs[0]->Len(2), m_Inputs[0]->Len(3)));
        else if (m_Axis == HeightAxis)
            m_Output.Resize(Shape(m_Inputs[0]->Len(0), 1, m_Inputs[0]->Len(2), m_Inputs[0]->Len(3)));
        else if (m_Axis == DepthAxis)
            m_Output.Resize(Shape(m_Inputs[0]->Len(0), m_Inputs[0]->Len(1), 1, m_Inputs[0]->Len(3)));
        else if (m_Axis == BatchAxis)
            m_Output.Resize(Shape(m_Inputs[0]->Len(0), m_Inputs[0]->Len(1), m_Inputs[0]->Len(2), 1));
        else
            assert(false);
    }

    //////////////////////////////////////////////////////////////////////////
    void SumOp::ComputeInternal()
    {
        if (m_Axis == WidthAxis || m_Axis == HeightAxis || m_Axis == DepthAxis)
            m_Output.ResizeBatch(m_Inputs[0]->Batch());

        m_Inputs[0]->Sum(m_Axis, m_Output);
    }

    //////////////////////////////////////////////////////////////////////////
    void SumOp::ComputeGradientInternal(const Tensor& grad)
    {
        if (m_InputNodes[0]->CareAboutGradient())
        {
            m_InputsGrads[0].FillWithValue(1);
            m_InputsGrads[0].MulElem(grad, m_InputsGrads[0]);
        }
    }
}