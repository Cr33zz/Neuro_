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

        m_TransTempA.Resize(Shape(a.Height(), a.Width(), a.Depth(), a.Batch()));
        a.Transpose(m_TransTempA);
        m_TransTempB.Resize(Shape(b.Height(), b.Width(), b.Depth(), b.Batch()));
        b.Transpose(m_TransTempB);        

        if (m_InputsGrads[0].Batch() == grad.Batch())
            grad.Mul(m_TransTempB, m_InputsGrads[0]);
        else
        {
            m_MulTempA.Resize(Shape::From(m_InputsGrads[0].GetShape(), grad.Batch()));
            grad.Mul(m_TransTempB, m_MulTempA);
            m_MulTempA.Sum(BatchAxis, m_InputsGrads[0]);
        }

        if (m_InputsGrads[1].Batch() == grad.Batch())
            m_TransTempA.Mul(grad, m_InputsGrads[1]);
        else
        {
            m_MulTempB.Resize(Shape::From(m_InputsGrads[1].GetShape(), grad.Batch()));
            m_TransTempA.Mul(grad, m_MulTempB);
            m_MulTempB.Sum(BatchAxis, m_InputsGrads[1]);
        }
    }
}