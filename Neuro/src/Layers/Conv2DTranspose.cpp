#include "Layers/Conv2DTranspose.h"
#include "Layers/Conv2D.h"
#include "Tensors/Tensor.h"

namespace Neuro
{
    //////////////////////////////////////////////////////////////////////////
    Conv2DTranspose::Conv2DTranspose(LayerBase* inputLayer, int filterSize, int outputDepth, int stride, int padding, ActivationBase* activation, const string& name)
        : LayerBase(__FUNCTION__, inputLayer, Tensor::GetConvTransposeOutputShape(inputLayer->OutputShape(), outputDepth, filterSize, filterSize, stride, padding, padding), activation, name)
    {
        m_FilterSize = filterSize;
        m_OutputDepth = outputDepth;
        m_Stride = stride;
        m_Padding = padding;
    }

    //////////////////////////////////////////////////////////////////////////
    Conv2DTranspose::Conv2DTranspose(const Shape& inputShape, int filterSize, int outputDepth, int stride, int padding, ActivationBase* activation, const string& name)
        : LayerBase(__FUNCTION__, inputShape, Tensor::GetConvTransposeOutputShape(inputShape, outputDepth, filterSize, filterSize, stride, padding, padding), activation, name)
    {
        m_FilterSize = filterSize;
        m_OutputDepth = outputDepth;
        m_Stride = stride;
        m_Padding = padding;
    }

    //////////////////////////////////////////////////////////////////////////
    Conv2DTranspose::Conv2DTranspose()
    {

    }

    //////////////////////////////////////////////////////////////////////////
    Conv2DTranspose::~Conv2DTranspose()
    {
        delete m_KernelInitializer;
        delete m_BiasInitializer;
    }

    //////////////////////////////////////////////////////////////////////////
    void Conv2DTranspose::OnInit()
    {
        __super::OnInit();

        m_Kernels = Tensor(Shape(m_FilterSize, m_FilterSize, m_OutputDepth, InputShape().Depth()), Name() + "/kernels");
        m_Bias = Tensor(Shape(m_OutputShape.Width(), m_OutputShape.Height(), m_OutputDepth), Name() + "/bias");
        m_KernelsGradient = Tensor(m_Kernels.GetShape(), Name() + "/kernels_grad");
        m_KernelsGradient.Zero();
        m_BiasGradient = Tensor(m_Bias.GetShape(), Name() + "/bias_grad");
        m_BiasGradient.Zero();

        m_KernelInitializer->Init(m_Kernels, m_InputShapes[0].Length, m_OutputShape.Length);
        if (m_UseBias)
            m_BiasInitializer->Init(m_Bias, m_InputShapes[0].Length, m_OutputShape.Length);
    }

    //////////////////////////////////////////////////////////////////////////
    LayerBase* Conv2DTranspose::GetCloneInstance() const
    {
        return new Conv2DTranspose();
    }

    //////////////////////////////////////////////////////////////////////////
    void Conv2DTranspose::OnClone(const LayerBase& source)
    {
        __super::OnClone(source);

        auto& sourceDeconv = static_cast<const Conv2DTranspose&>(source);
        m_Kernels = Tensor(sourceDeconv.m_Kernels);
        m_Bias = Tensor(sourceDeconv.m_Bias);
        m_UseBias = sourceDeconv.m_UseBias;
        m_FilterSize = sourceDeconv.m_FilterSize;
        m_OutputDepth = sourceDeconv.m_OutputDepth;
        m_Stride = sourceDeconv.m_Stride;
        m_Padding = sourceDeconv.m_Padding;
    }

    //////////////////////////////////////////////////////////////////////////
    void Conv2DTranspose::FeedForwardInternal(bool training)
    {
        m_Inputs[0]->Conv2DTransposed(m_Kernels, m_Stride, m_Padding, m_Output);
        if (m_UseBias)
            m_Output.Add(m_Bias, m_Output);
    }

    //////////////////////////////////////////////////////////////////////////
    void Conv2DTranspose::BackPropInternal(Tensor& outputGradient)
    {
        outputGradient.Conv2DTransposedInputsGradient(outputGradient, m_Kernels, m_Stride, m_Padding, m_InputsGradient[0]);
        outputGradient.Conv2DTransposedKernelsGradient(outputGradient, *m_Inputs[0], m_Stride, m_Padding, m_KernelsGradient);

        if (m_UseBias)
            m_BiasGradient.Add(outputGradient.Sum(EAxis::Feature));
    }

    //////////////////////////////////////////////////////////////////////////
    void Conv2DTranspose::GetParametersAndGradients(vector<ParametersAndGradients>& result)
    {
        result.push_back(ParametersAndGradients(&m_Kernels, &m_KernelsGradient));

        if (m_UseBias)
            result.push_back(ParametersAndGradients(&m_Bias, &m_BiasGradient));
    }

    //////////////////////////////////////////////////////////////////////////
    void Conv2DTranspose::CopyParametersTo(LayerBase& target, float tau) const
    {
        __super::CopyParametersTo(target, tau);

        auto& targetConv = static_cast<Conv2DTranspose&>(target);
        m_Kernels.CopyTo(targetConv.m_Kernels, tau);
        m_Bias.CopyTo(targetConv.m_Bias, tau);
    }

    //////////////////////////////////////////////////////////////////////////
    int Conv2DTranspose::GetParamsNum() const
    {
        return m_FilterSize * m_FilterSize * m_OutputDepth;
    }

    //////////////////////////////////////////////////////////////////////////
    Conv2DTranspose* Conv2DTranspose::SetKernelInitializer(InitializerBase* initializer)
    {
        delete m_KernelInitializer;
        m_KernelInitializer = initializer;
        return this;
    }

    //////////////////////////////////////////////////////////////////////////
    Conv2DTranspose* Conv2DTranspose::SetBiasInitializer(InitializerBase* initializer)
    {
        delete m_BiasInitializer;
        m_BiasInitializer = initializer;
        return this;
    }

    //////////////////////////////////////////////////////////////////////////
    Conv2DTranspose* Conv2DTranspose::SetUseBias(bool useBias)
    {
        m_UseBias = useBias;
        return this;
    }
}
