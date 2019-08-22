#include "Layers/Deconvolution.h"
#include "Layers/Convolution.h"
#include "Tensors/Tensor.h"

namespace Neuro
{
    //////////////////////////////////////////////////////////////////////////
    Deconvolution::Deconvolution(LayerBase* inputLayer, int filterSize, int outputDepth, int stride, EPaddingMode paddingMode, ActivationBase* activation, const string& name)
        : LayerBase(__FUNCTION__, inputLayer, GetOutShape(inputLayer->OutputShape(), filterSize, filterSize, stride, outputDepth), activation, name)
    {
        m_FilterSize = filterSize;
        m_OutputDepth = outputDepth;
        m_Stride = stride;
        m_PaddingMode = paddingMode;
    }

    //////////////////////////////////////////////////////////////////////////
    Deconvolution::Deconvolution(const Shape& inputShape, int filterSize, int outputDepth, int stride, EPaddingMode paddingMode, ActivationBase* activation, const string& name)
        : LayerBase(__FUNCTION__, inputShape, GetOutShape(inputShape, filterSize, filterSize, stride, outputDepth), activation, name)
    {
        m_FilterSize = filterSize;
        m_OutputDepth = outputDepth;
        m_Stride = stride;
        m_PaddingMode = paddingMode;
    }

    //////////////////////////////////////////////////////////////////////////
    Deconvolution::Deconvolution()
    {

    }

    //////////////////////////////////////////////////////////////////////////
    Deconvolution::~Deconvolution()
    {
        delete m_KernelInitializer;
        delete m_BiasInitializer;
    }

    //////////////////////////////////////////////////////////////////////////
    void Deconvolution::OnInit()
    {
        __super::OnInit();

        m_Kernels = Tensor(Shape(m_FilterSize, m_FilterSize, m_OutputDepth, InputShape().Depth()));
        m_Bias = Tensor(Shape(m_OutputShape.Width(), m_OutputShape.Height(), m_OutputDepth));
        m_KernelsGradient = Tensor(m_Kernels.GetShape());
        m_BiasGradient = Tensor(m_Bias.GetShape());

        m_KernelInitializer->Init(m_Kernels, m_InputShapes[0].Length, m_OutputShape.Length);
        if (m_UseBias)
            m_BiasInitializer->Init(m_Bias, m_InputShapes[0].Length, m_OutputShape.Length);
    }

    //////////////////////////////////////////////////////////////////////////
    LayerBase* Deconvolution::GetCloneInstance() const
    {
        return new Deconvolution();
    }

    //////////////////////////////////////////////////////////////////////////
    void Deconvolution::OnClone(const LayerBase& source)
    {
        __super::OnClone(source);

        auto& sourceDeconv = static_cast<const Deconvolution&>(source);
        m_Kernels = Tensor(sourceDeconv.m_Kernels);
        m_Bias = Tensor(sourceDeconv.m_Bias);
        m_UseBias = sourceDeconv.m_UseBias;
        m_FilterSize = sourceDeconv.m_FilterSize;
        m_OutputDepth = sourceDeconv.m_OutputDepth;
        m_Stride = sourceDeconv.m_Stride;
        m_PaddingMode = sourceDeconv.m_PaddingMode;
    }

    //////////////////////////////////////////////////////////////////////////
    void Deconvolution::FeedForwardInternal(bool training)
    {
        m_Inputs[0]->Conv2DTransposed(m_Kernels, m_Stride, m_PaddingMode, m_Output);
        if (m_UseBias)
            m_Output.Add(m_Bias, m_Output);
    }

    //////////////////////////////////////////////////////////////////////////
    void Deconvolution::BackPropInternal(Tensor& outputGradient)
    {
        outputGradient.Conv2DTransposedInputsGradient(outputGradient, m_Kernels, m_Stride, Convolution::GetGradientPaddingMode(m_PaddingMode), m_InputsGradient[0]);
        outputGradient.Conv2DTransposedKernelsGradient(outputGradient, *m_Inputs[0], m_Stride, Convolution::GetGradientPaddingMode(m_PaddingMode), m_KernelsGradient);

        if (m_UseBias)
            m_BiasGradient.Add(outputGradient.SumBatches());
    }

    //////////////////////////////////////////////////////////////////////////
    Shape Deconvolution::GetOutShape(const Shape& inputShape, int filterWidth, int filterHeight, int stride, int outputDepth)
    {
        return Shape(stride * (inputShape.Width() - 1) + filterWidth - 2, stride * (inputShape.Height() - 1) + filterWidth - 2, outputDepth);
    }

    //////////////////////////////////////////////////////////////////////////
    void Deconvolution::GetParametersAndGradients(vector<ParametersAndGradients>& result)
    {
        result.push_back(ParametersAndGradients(&m_Kernels, &m_KernelsGradient));

        if (m_UseBias)
            result.push_back(ParametersAndGradients(&m_Bias, &m_BiasGradient));
    }

    //////////////////////////////////////////////////////////////////////////
    void Deconvolution::CopyParametersTo(LayerBase& target, float tau) const
    {
        __super::CopyParametersTo(target, tau);

        auto& targetConv = static_cast<Deconvolution&>(target);
        m_Kernels.CopyTo(targetConv.m_Kernels, tau);
        m_Bias.CopyTo(targetConv.m_Bias, tau);
    }

    //////////////////////////////////////////////////////////////////////////
    int Deconvolution::GetParamsNum() const
    {
        return m_FilterSize * m_FilterSize * m_OutputDepth;
    }

    //////////////////////////////////////////////////////////////////////////
    Deconvolution* Deconvolution::SetKernelInitializer(InitializerBase* initializer)
    {
        delete m_KernelInitializer;
        m_KernelInitializer = initializer;
        return this;
    }

    //////////////////////////////////////////////////////////////////////////
    Deconvolution* Deconvolution::SetBiasInitializer(InitializerBase* initializer)
    {
        delete m_BiasInitializer;
        m_BiasInitializer = initializer;
        return this;
    }

    //////////////////////////////////////////////////////////////////////////
    Deconvolution* Deconvolution::SetUseBias(bool useBias)
    {
        m_UseBias = useBias;
        return this;
    }
}
