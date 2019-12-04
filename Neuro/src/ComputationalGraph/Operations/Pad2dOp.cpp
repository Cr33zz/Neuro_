#include "ComputationalGraph/Operations/Pad2dOp.h"

namespace Neuro
{
    //////////////////////////////////////////////////////////////////////////
    Pad2dOp::Pad2dOp(TensorLike* x, uint32_t left, uint32_t right, uint32_t top, uint32_t bottom, const string& name)
        : Operation({ x }, name.empty() ? "pad2d" : name), m_Left(left), m_Right(right), m_Top(top), m_Bottom(bottom)
    {
        UpdateOutputShape();
    }

    //////////////////////////////////////////////////////////////////////////
    void Pad2dOp::UpdateOutputShape()
    {
        const auto& shape = m_InputNodes[0]->GetShape();
        m_Output.Resize(Shape(shape.Width() + m_Left + m_Right, shape.Height() + m_Top + m_Bottom, shape.Depth(), shape.Batch()));
    }

    //////////////////////////////////////////////////////////////////////////
    void Pad2dOp::ComputeGradientInternal(const Tensor& grad)
    {
        if (m_InputNodes[0]->CareAboutGradient())
            grad.Pad2DGradient(grad, m_Left, m_Right, m_Top, m_Bottom, m_InputsGrads[0]);
    }

    //////////////////////////////////////////////////////////////////////////
    ConstantPad2dOp::ConstantPad2dOp(TensorLike* x, uint32_t left, uint32_t right, uint32_t top, uint32_t bottom, float value, const string& name)
        : Pad2dOp(x, left, right, top, bottom, name.empty() ? "constant_pad2d" : name), m_Value(value)
    {
    }

    //////////////////////////////////////////////////////////////////////////
    void ConstantPad2dOp::ComputeInternal()
    {
        auto& x = *m_Inputs[0];
        m_Output.ResizeBatch(x.Batch());
        x.ConstantPad2D(m_Left, m_Right, m_Top, m_Bottom, m_Value, m_Output);
    }

    //////////////////////////////////////////////////////////////////////////
    ReflectPad2dOp::ReflectPad2dOp(TensorLike* x, uint32_t left, uint32_t right, uint32_t top, uint32_t bottom, const string& name)
        : Pad2dOp(x, left, right, top, bottom, name.empty() ? "reflect_pad2d" : name)
    {
    }

    //////////////////////////////////////////////////////////////////////////
    void ReflectPad2dOp::ComputeInternal()
    {
        auto& x = *m_Inputs[0];
        m_Output.ResizeBatch(x.Batch());
        x.ReflectPad2D(m_Left, m_Right, m_Top, m_Bottom, m_Output);
    }
}