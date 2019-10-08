#include "Layers/Conv2DTranspose.h"
#include "Layers/Conv2D.h"
#include "Tensors/Tensor.h"
#include "ComputationalGraph/Variable.h"
#include "ComputationalGraph/Ops.h"
#include "Activations.h"

namespace Neuro
{
    //////////////////////////////////////////////////////////////////////////
    Conv2DTranspose::Conv2DTranspose(uint32_t outputDepth, uint32_t filterSize, uint32_t stride, uint32_t padding, ActivationBase* activation, EDataFormat dataFormat, const string& name)
        : SingleLayer(__FUNCTION__, Shape(), activation, name)
    {
        m_FilterSize = filterSize;
        m_OutputDepth = outputDepth;
        m_Stride = stride;
        m_Padding = padding;
        m_DataFormat = dataFormat;
    }

    //////////////////////////////////////////////////////////////////////////
    Conv2DTranspose::Conv2DTranspose(const Shape& inputShape, uint32_t outputDepth, uint32_t filterSize, uint32_t stride, uint32_t padding, ActivationBase* activation, EDataFormat dataFormat, const string& name)
        : SingleLayer(__FUNCTION__, inputShape, activation, name)
    {
        m_FilterSize = filterSize;
        m_OutputDepth = outputDepth;
        m_Stride = stride;
        m_Padding = padding;
        m_DataFormat = dataFormat;
    }

    //////////////////////////////////////////////////////////////////////////
    Conv2DTranspose::~Conv2DTranspose()
    {
        delete m_KernelInitializer;
        delete m_BiasInitializer;
    }

    //////////////////////////////////////////////////////////////////////////
    void Conv2DTranspose::Build(const vector<Shape>& inputShapes)
    {
        if (m_DataFormat == NCHW)
        {
            m_Kernels = new Variable(Shape(m_FilterSize, m_FilterSize, m_OutputDepth, inputShapes[0].Depth()), m_KernelInitializer, "kernels");
            m_Bias = new Variable(Shape(1, 1, m_OutputDepth), m_BiasInitializer, "bias");
        }
        else
        {
            m_Kernels = new Variable(Shape(m_FilterSize, m_FilterSize, m_OutputDepth, inputShapes[0].Len(0)), m_KernelInitializer, "kernels");
            m_Bias = new Variable(Shape(m_OutputDepth), m_BiasInitializer, "bias");
        }

        m_Built = true;
    }

    //////////////////////////////////////////////////////////////////////////
    vector<TensorLike*> Conv2DTranspose::InternalCall(const vector<TensorLike*>& inputs, TensorLike* training)
    {
        TensorLike* output = conv2d_transpose(inputs[0], m_Kernels, m_Stride, m_Padding, m_DataFormat);
        if (m_UseBias)
            output = add(output, m_Bias);
        if (m_Activation)
            output = m_Activation->Build(output);
        return { output };
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
        m_Kernels = new Variable(*sourceDeconv.m_Kernels);
        m_Bias = new Variable(*sourceDeconv.m_Bias);
        m_UseBias = sourceDeconv.m_UseBias;
        m_FilterSize = sourceDeconv.m_FilterSize;
        m_OutputDepth = sourceDeconv.m_OutputDepth;
        m_Stride = sourceDeconv.m_Stride;
        m_Padding = sourceDeconv.m_Padding;
    }

    //////////////////////////////////////////////////////////////////////////
    void Conv2DTranspose::Parameters(vector<Variable*>& params, bool onlyTrainable)
    {
        if (onlyTrainable && !m_Trainable)
            return;

        params.push_back(m_Kernels);

        if (m_UseBias)
            params.push_back(m_Bias);
    }

    //////////////////////////////////////////////////////////////////////////
    void Conv2DTranspose::SerializedParameters(vector<SerializedParameter>& params)
    {
        params.push_back({ m_Kernels, { DepthAxis, BatchAxis, HeightAxis, WidthAxis } });

        if (m_UseBias)
        {
            if (m_DataFormat == NCHW)
                params.push_back({ m_Bias, { DepthAxis, HeightAxis, WidthAxis } });
            else
                params.push_back({ m_Bias });
        }
    }

    //////////////////////////////////////////////////////////////////////////
    Neuro::Tensor& Conv2DTranspose::Kernels()
    {
        return m_Kernels->Output();
    }

    //////////////////////////////////////////////////////////////////////////
    Neuro::Tensor& Conv2DTranspose::Bias()
    {
        return m_Bias->Output();
    }

    //////////////////////////////////////////////////////////////////////////
    void Conv2DTranspose::CopyParametersTo(LayerBase& target, float tau) const
    {
        __super::CopyParametersTo(target, tau);

        auto& targetConv = static_cast<Conv2DTranspose&>(target);
        m_Kernels->Output().CopyTo(targetConv.m_Kernels->Output(), tau);
        m_Bias->Output().CopyTo(targetConv.m_Bias->Output(), tau);
    }

    //////////////////////////////////////////////////////////////////////////
    uint32_t Conv2DTranspose::ParamsNum() const
    {
        return m_FilterSize * m_FilterSize * m_OutputDepth + (m_UseBias ? m_OutputDepth : 0);
    }

    //////////////////////////////////////////////////////////////////////////
    Conv2DTranspose* Conv2DTranspose::KernelInitializer(InitializerBase* initializer)
    {
        delete m_KernelInitializer;
        m_KernelInitializer = initializer;
        return this;
    }

    //////////////////////////////////////////////////////////////////////////
    Conv2DTranspose* Conv2DTranspose::BiasInitializer(InitializerBase* initializer)
    {
        delete m_BiasInitializer;
        m_BiasInitializer = initializer;
        return this;
    }

    //////////////////////////////////////////////////////////////////////////
    Conv2DTranspose* Conv2DTranspose::UseBias(bool useBias)
    {
        m_UseBias = useBias;
        return this;
    }
}
