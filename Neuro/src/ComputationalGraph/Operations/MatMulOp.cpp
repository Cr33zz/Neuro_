#include <algorithm>
#include "ComputationalGraph/Operations/MatMulOp.h"

namespace Neuro
{
    //////////////////////////////////////////////////////////////////////////
    MatMulOp::MatMulOp(TensorLike* a, TensorLike* b, const string& name)
        : Operation({ a, b }, name.empty() ? "matmul" : name)
    {
        UpdateOutputShape();

        m_MulTempA.Name(m_Name + "/tmp_matmul_0");
        m_MulTempB.Name(m_Name + "/tmp_matmul_1");
        m_TransTempA.Name(m_Name + "/tmp_trans_0");
        m_TransTempB.Name(m_Name + "/tmp_trans_1");
    }

    //////////////////////////////////////////////////////////////////////////
    void MatMulOp::UpdateOutputShape()
    {
        const Shape& aShape = m_InputNodes[0]->GetShape();
        const Shape& bShape = m_InputNodes[1]->GetShape();
        NEURO_ASSERT(aShape.Width() == bShape.Height(), "");
        NEURO_ASSERT(aShape.Depth() == bShape.Depth(), "Depths mismatch.");
        m_Output.Resize(Shape(bShape.Width(), aShape.Height(), aShape.Depth(), max(aShape.Batch(), bShape.Batch())));
    }

    //////////////////////////////////////////////////////////////////////////
    void MatMulOp::ComputeInternal()
    {
        auto& a = *m_Inputs[0];
        auto& b = *m_Inputs[1];

        m_Output.ResizeBatch(max(a.Batch(), b.Batch()));
        a.MatMul(b, m_Output);
    }

    //////////////////////////////////////////////////////////////////////////
    void MatMulOp::ComputeGradientInternal(const Tensor& grad)
    {
        auto& a = *m_Inputs[0];
        auto& b = *m_Inputs[1];

        if (m_InputNodes[0]->CareAboutGradient())
        {
            m_TransTempB.Resize(Shape(b.Height(), b.Width(), b.Depth(), b.Batch()));
            m_TransTempB.TryDeviceAllocate(); // this is actually workspace
            b.Transpose(m_TransTempB);

            if (m_InputsGrads[0].Batch() == grad.Batch())
                grad.MatMul(m_TransTempB, m_InputsGrads[0]);
            else
            {
                m_MulTempA.Resize(Shape::From(m_InputsGrads[0].GetShape(), grad.Batch()));
                m_MulTempA.TryDeviceAllocate(); // this is actually workspace
                grad.MatMul(m_TransTempB, m_MulTempA);
                m_MulTempA.Sum(BatchAxis, m_InputsGrads[0]);
                m_MulTempA.TryDeviceRelease();
            }
            m_TransTempB.TryDeviceRelease();
        }

        if (m_InputNodes[1]->CareAboutGradient())
        {
            m_TransTempA.Resize(Shape(a.Height(), a.Width(), a.Depth(), a.Batch()));
            m_TransTempA.TryDeviceAllocate();
            a.Transpose(m_TransTempA);

            if (m_InputsGrads[1].Batch() == grad.Batch())
                m_TransTempA.MatMul(grad, m_InputsGrads[1]);
            else
            {
                m_MulTempB.Resize(Shape::From(m_InputsGrads[1].GetShape(), grad.Batch()));
                m_MulTempB.TryDeviceAllocate();
                m_TransTempA.MatMul(grad, m_MulTempB);
                m_MulTempB.Sum(BatchAxis, m_InputsGrads[1]);
                m_MulTempB.TryDeviceRelease();
            }
            m_TransTempA.TryDeviceRelease();
        }
    }

    //////////////////////////////////////////////////////////////////////////
    MatMulTransOp::MatMulTransOp(TensorLike* a, bool transposeA, TensorLike* b, bool transposeB, const string& name)
        : Operation({ a, b }, name.empty() ? "matmul" : name), m_TransposeA(transposeA), m_TransposeB(transposeB)
    {
        UpdateOutputShape();
    }

    //////////////////////////////////////////////////////////////////////////
    void MatMulTransOp::UpdateOutputShape()
    {
        const Shape& aShape = m_InputNodes[0]->GetShape();
        const Shape& bShape = m_InputNodes[1]->GetShape();
        NEURO_ASSERT((!m_TransposeA && !m_TransposeB && aShape.Width() == bShape.Height()) ||
                     (m_TransposeA && !m_TransposeB && aShape.Height() == bShape.Height()) ||
                     (!m_TransposeA && m_TransposeB && aShape.Width() == bShape.Width()) ||
                     (m_TransposeA && m_TransposeB && aShape.Height() == bShape.Width()), "Matrices width/height mismatch");
        NEURO_ASSERT(aShape.Depth() == bShape.Depth(), "Depths mismatch.");
        m_Output.Resize(Shape(bShape.Width(), aShape.Height(), aShape.Depth(), max(aShape.Batch(), bShape.Batch())));
    }

    //////////////////////////////////////////////////////////////////////////
    void MatMulTransOp::ComputeInternal()
    {
        auto& a = *m_Inputs[0];
        auto& b = *m_Inputs[1];

        m_Output.ResizeBatch(max(a.Batch(), b.Batch()));
        a.MatMul(m_TransposeA, b, m_TransposeB, m_Output);
    }

    //////////////////////////////////////////////////////////////////////////
    void MatMulTransOp::ComputeGradientInternal(const Tensor& grad)
    {
        auto& a = *m_Inputs[0];
        auto& b = *m_Inputs[1];

        if (m_InputNodes[0]->CareAboutGradient())
        {
            if (m_InputsGrads[0].Batch() == grad.Batch())
                grad.MatMul(false, b, true, m_InputsGrads[0]);
            else
            {
                Tensor tmp(Shape::From(m_InputsGrads[0].GetShape(), grad.Batch()));
                tmp.TryDeviceAllocate(); // this is actually workspace
                grad.MatMul(false, b, true, tmp);
                tmp.Sum(BatchAxis, m_InputsGrads[0]);
                tmp.TryDeviceRelease();
            }
        }

        if (m_InputNodes[1]->CareAboutGradient())
        {
            if (m_InputsGrads[1].Batch() == grad.Batch())
                a.MatMul(true, grad, false, m_InputsGrads[1]);
            else
            {
                Tensor tmp(Shape::From(m_InputsGrads[1].GetShape(), grad.Batch()));
                tmp.TryDeviceAllocate();
                a.MatMul(true, grad, false, tmp);
                tmp.Sum(BatchAxis, m_InputsGrads[1]);
                tmp.TryDeviceRelease();
            }
        }
    }
}