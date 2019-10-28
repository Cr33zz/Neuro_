#include "ComputationalGraph/Operations/BatchReshapeOp.h"

namespace Neuro
{
    //////////////////////////////////////////////////////////////////////////
    BatchReshapeOp::BatchReshapeOp(TensorLike* x, const Shape& shape, const string& name)
        : Operation({ x }, name.empty() ? "batch_reshape" : name), m_Shape(shape)
    {
        NEURO_ASSERT(shape.Batch() == 1, "");
        m_Output.Resize(shape);
    }

    //////////////////////////////////////////////////////////////////////////
    void BatchReshapeOp::ComputeInternal()
    {
        m_Output.ResizeBatch(m_Inputs[0]->Batch());
        m_Inputs[0]->CopyTo(m_Output);
    }

    //////////////////////////////////////////////////////////////////////////
    void BatchReshapeOp::ComputeGradientInternal(const Tensor& grad)
    {
        if (m_InputNodes[0]->CareAboutGradient())
            grad.CopyTo(m_InputsGrads[0]);
    }
}