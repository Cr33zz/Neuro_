#include "ComputationalGraph/Operations/BatchFlattenOp.h"

namespace Neuro
{
    //////////////////////////////////////////////////////////////////////////
    BatchFlattenOp::BatchFlattenOp(TensorLike* x, const string& name)
        : Operation({ x }, name.empty() ? "batch_flatten" : name)
    {
        Shape shapeNoBatch = Shape::From(x->GetShape(), 1);
        m_Output.Resize(Shape(shapeNoBatch.Length));
    }

    //////////////////////////////////////////////////////////////////////////
    void BatchFlattenOp::ComputeInternal()
    {
        m_Output.ResizeBatch(m_Inputs[0]->Batch());
        m_Inputs[0]->CopyTo(m_Output);
    }

    //////////////////////////////////////////////////////////////////////////
    void BatchFlattenOp::ComputeGradientInternal(const Tensor& grad)
    {
        if (m_InputNodes[0]->CareAboutGradient())
            grad.CopyTo(m_InputsGrads[0]);
    }
}