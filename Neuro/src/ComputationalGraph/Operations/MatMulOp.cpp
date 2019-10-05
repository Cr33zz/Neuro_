#include <algorithm>
#include "ComputationalGraph/Operations/MatMulOp.h"

namespace Neuro
{
    //////////////////////////////////////////////////////////////////////////
    MatMulOp::MatMulOp(TensorLike* a, TensorLike* b, const string& name)
        : Operation({ a, b }, name.empty() ? "matmul" : name)
    {
        /*assert(a->GetShape().Width() == b->GetShape().Height());
        assert(a->GetShape().Depth() == b->GetShape().Depth());*/
    }

    //////////////////////////////////////////////////////////////////////////
    void MatMulOp::ComputeInternal()
    {
        auto& a = *m_Inputs[0];
        auto& b = *m_Inputs[1];

        m_Output.Resize(Shape(b.Width(), a.Height(), a.Depth(), max(a.Batch(), b.Batch())));
        a.Mul(b, m_Output);
    }

    //////////////////////////////////////////////////////////////////////////
    void MatMulOp::ComputeGradientInternal(const Tensor& grad)
    {
        auto& a = *m_Inputs[0];
        auto& b = *m_Inputs[1];

        if (m_InputsGrads[0].Batch() == grad.Batch())
            grad.Mul(b.Transposed(), m_InputsGrads[0]);
        else
            grad.Mul(b.Transposed()).Sum(BatchAxis, m_InputsGrads[0]);

        if (m_InputsGrads[1].Batch() == grad.Batch())
            a.Transposed().Mul(grad, m_InputsGrads[1]);
        else
            a.Transposed().Mul(grad).Sum(BatchAxis, m_InputsGrads[1]);
    }
}