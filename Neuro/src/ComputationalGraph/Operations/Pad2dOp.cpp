#include "ComputationalGraph/Operations/Pad2dOp.h"

namespace Neuro
{
    //////////////////////////////////////////////////////////////////////////
    Pad2dOp::Pad2dOp(TensorLike* x, uint32_t left, uint32_t right, uint32_t top, uint32_t bottom, float value, const string& name)
        : Operation({ x }, name.empty() ? "pad2d" : name), m_Left(left), m_Right(right), m_Top(top), m_Bottom(bottom), m_Value(value)
    {
        m_Output.Resize(Shape(x->GetShape().Width() + m_Left + m_Right, x->GetShape().Height() + m_Top + m_Bottom, x->GetShape().Depth()));
    }

    //////////////////////////////////////////////////////////////////////////
    void Pad2dOp::ComputeInternal()
    {
        auto& x = *m_Inputs[0];
        m_Output.ResizeBatch(x.Batch());
        x.Pad2D(m_Left, m_Right, m_Top, m_Bottom, m_Value, m_Output);
    }

    //////////////////////////////////////////////////////////////////////////
    void Pad2dOp::ComputeGradientInternal(const Tensor& grad)
    {
        if (m_InputNodes[0]->CareAboutGradient())
            grad.Pad2DGradient(grad, m_Left, m_Right, m_Top, m_Bottom, m_InputsGrads[0]);
    }
}