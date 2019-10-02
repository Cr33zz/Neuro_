#include "ComputationalGraph/Operations/Pool2dOp.h"

namespace Neuro
{
    //////////////////////////////////////////////////////////////////////////
    Pool2dOp::Pool2dOp(TensorLike* x, uint32_t filterSize, uint32_t stride, uint32_t padding, EPoolingMode mode, EDataFormat dataFormat)
        : Operation({x}), m_FilterSize(filterSize), m_Stride(stride), m_Padding(padding), m_DataFormat(dataFormat)
    {
    }

    //////////////////////////////////////////////////////////////////////////
    void Pool2dOp::ComputeInternal()
    {
        m_Output.Resize(Tensor::GetPooling2DOutputShape(m_Inputs[0]->GetShape(), m_FilterSize, m_FilterSize, m_Stride, m_Padding, m_Padding, m_DataFormat));
        m_Inputs[0]->Pool2D(m_FilterSize, m_Stride, m_Mode, m_Padding, m_DataFormat, m_Output);
    }

    //////////////////////////////////////////////////////////////////////////
    void Pool2dOp::ComputeGradientInternal(const Tensor& grad)
    {
        m_Inputs[0]->Pool2DGradient(m_Output, *m_Inputs[0], grad, m_FilterSize, m_Stride, m_Mode, m_Padding, m_DataFormat, m_InputsGrads[0]);
    }
}