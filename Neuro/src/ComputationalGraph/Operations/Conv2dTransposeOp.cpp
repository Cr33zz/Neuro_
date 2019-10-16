#include "ComputationalGraph/Operations/Conv2dTransposeOp.h"

namespace Neuro
{
    //////////////////////////////////////////////////////////////////////////
    Conv2dTransposeOp::Conv2dTransposeOp(TensorLike* x, TensorLike* kernels, uint32_t stride, uint32_t padding, EDataFormat dataFormat, const string& name)
        : Operation({ x, kernels }, name.empty() ? "conv2dtranspose" : name), m_Stride(stride), m_Padding(padding), m_DataFormat(dataFormat)
    {
        m_Output.Resize(Tensor::GetConvTransposeOutputShape(x->GetShape(), m_OutputDepth, kernels->GetShape().Width(), kernels->GetShape().Height(), m_Stride, m_Padding, m_Padding, m_DataFormat));
    }

    //////////////////////////////////////////////////////////////////////////
    void Conv2dTransposeOp::ComputeInternal()
    {
        auto& x = *m_Inputs[0];
        auto& kernels = *m_Inputs[1];

        m_Output.ResizeBatch(x.Batch());
        return x.Conv2DTransposed(kernels, m_Stride, m_Padding, m_DataFormat, m_Output);
    }

    //////////////////////////////////////////////////////////////////////////
    void Conv2dTransposeOp::ComputeGradientInternal(const Tensor& grad)
    {
        auto& x = *m_Inputs[0];
        auto& kernels = *m_Inputs[1];

        if (m_InputNodes[0]->CareAboutGradient())
            grad.Conv2DTransposedInputsGradient(grad, kernels, m_Stride, m_Padding, m_DataFormat, m_InputsGrads[0]);
        if (m_InputNodes[1]->CareAboutGradient())
            grad.Conv2DTransposedKernelsGradient(x, grad, m_Stride, m_Padding, m_DataFormat, m_InputsGrads[1]);
    }
}
