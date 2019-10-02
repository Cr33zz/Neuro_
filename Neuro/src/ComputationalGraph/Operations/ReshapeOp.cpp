#include "ComputationalGraph/Operations/ReshapeOp.h"

namespace Neuro
{
    //////////////////////////////////////////////////////////////////////////
    ReshapeOp::ReshapeOp(TensorLike* x, const Shape& shape)
        : Operation({x}), m_Shape(shape)
    {
    }

    //////////////////////////////////////////////////////////////////////////
    void ReshapeOp::ComputeInternal()
    {
        m_Output.Resize(Shape::From(m_Shape, m_Inputs[0]->Batch()));
        m_Inputs[0]->CopyTo(m_Output);
    }

    //////////////////////////////////////////////////////////////////////////
    void ReshapeOp::ComputeGradientInternal(const Tensor& grad)
    {
        grad.CopyTo(m_InputsGrads[0]);
    }
}