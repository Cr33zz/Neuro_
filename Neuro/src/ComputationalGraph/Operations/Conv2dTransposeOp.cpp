#include "ComputationalGraph/Operations/Conv2dTransposeOp.h"

namespace Neuro
{
    //////////////////////////////////////////////////////////////////////////
    Conv2dTransposeOp::Conv2dTransposeOp(TensorLike* x, TensorLike* kernels, uint32_t stride, uint32_t padding, EDataFormat dataFormat)
        : Operation({ x, kernels }, "conv2dtranspose"), m_Stride(stride), m_Padding(padding), m_DataFormat(dataFormat)
    {
    }

    //////////////////////////////////////////////////////////////////////////
    void Conv2dTransposeOp::ComputeInternal()
    {
        auto& x = *m_Inputs[0];
        auto& kernels = *m_Inputs[1];

        m_Output.Resize(Tensor::GetConvTransposeOutputShape(x.GetShape(), m_OutputDepth, kernels.Width(), kernels.Height(), m_Stride, m_Padding, m_Padding, m_DataFormat));
        return x.Conv2DTransposed(kernels, m_Stride, m_Padding, m_DataFormat, m_Output);
    }

    //////////////////////////////////////////////////////////////////////////
    void Conv2dTransposeOp::ComputeGradientInternal(const Tensor& grad)
    {
        auto& x = *m_Inputs[0];
        auto& kernels = *m_Inputs[1];

        grad.Conv2DTransposedInputsGradient(grad, kernels, m_Stride, m_Padding, m_DataFormat, m_InputsGrads[0]);
        grad.Conv2DTransposedKernelsGradient(x, grad, m_Stride, m_Padding, m_DataFormat, m_InputsGrads[1]);
    }
}
