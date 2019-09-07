#include "Layers/Pooling2D.h"

namespace Neuro
{
    //////////////////////////////////////////////////////////////////////////
    Pooling2D::Pooling2D(LayerBase* inputLayer, uint32_t filterSize, uint32_t stride, uint32_t padding, EPoolingMode mode, const string& name)
        : Pooling2D(__FUNCTION__, inputLayer, filterSize, stride, padding, mode, name)
    {
    }

    //////////////////////////////////////////////////////////////////////////
    Pooling2D::Pooling2D(uint32_t filterSize, uint32_t stride, uint32_t padding, EPoolingMode mode, const string& name)
        : Pooling2D(__FUNCTION__, filterSize, stride, padding, mode, name)
    {
    }

    //////////////////////////////////////////////////////////////////////////
    Pooling2D::Pooling2D(Shape inputShape, uint32_t filterSize, uint32_t stride, uint32_t padding, EPoolingMode mode, const string& name)
        : Pooling2D(__FUNCTION__, inputShape, filterSize, stride, padding, mode, name)
    {
    }

    //////////////////////////////////////////////////////////////////////////
    Pooling2D::Pooling2D(const string& constructorName, Shape inputShape, uint32_t filterSize, uint32_t stride, uint32_t padding, EPoolingMode mode, const string& name)
        : LayerBase(constructorName, inputShape, Tensor::GetPooling2DOutputShape(inputShape, filterSize, filterSize, stride, padding, padding), nullptr, name)
    {
        m_Mode = mode;
        m_FilterSize = filterSize;
        m_Stride = stride;
        m_Padding = padding;
    }

    //////////////////////////////////////////////////////////////////////////
    Pooling2D::Pooling2D(const string& constructorName, LayerBase* inputLayer, uint32_t filterSize, uint32_t stride, uint32_t padding, EPoolingMode mode, const string& name)
        : LayerBase(constructorName, inputLayer, Tensor::GetPooling2DOutputShape(inputLayer->OutputShape(), filterSize, filterSize, stride, padding, padding), nullptr, name)
    {
        m_Mode = mode;
        m_FilterSize = filterSize;
        m_Stride = stride;
        m_Padding = padding;
    }

    //////////////////////////////////////////////////////////////////////////
    Pooling2D::Pooling2D(const string& constructorName, uint32_t filterSize, uint32_t stride, uint32_t padding, EPoolingMode mode, const string& name)
        : LayerBase(constructorName, Shape(), nullptr, name)
    {
        m_Mode = mode;
        m_FilterSize = filterSize;
        m_Stride = stride;
        m_Padding = padding;
    }

    //////////////////////////////////////////////////////////////////////////
    LayerBase* Pooling2D::GetCloneInstance() const
    {
        return new Pooling2D();
    }

    //////////////////////////////////////////////////////////////////////////
    void Pooling2D::OnClone(const LayerBase& source)
    {
        __super::OnClone(source);

        auto sourcePool = static_cast<const Pooling2D&>(source);
        m_Mode = sourcePool.m_Mode;
        m_FilterSize = sourcePool.m_FilterSize;
        m_Stride = sourcePool.m_Stride;
    }

    //////////////////////////////////////////////////////////////////////////
    void Pooling2D::OnLink()
    {
        m_OutputShapes[0] = Tensor::GetPooling2DOutputShape(InputLayer()->OutputShape(), m_FilterSize, m_FilterSize, m_Stride, m_Padding, m_Padding);
    }

    //////////////////////////////////////////////////////////////////////////
    void Pooling2D::FeedForwardInternal(bool training)
    {
        m_Inputs[0]->Pool2D(m_FilterSize, m_Stride, m_Mode, m_Padding, m_Outputs[0]);
    }

    //////////////////////////////////////////////////////////////////////////
    void Pooling2D::BackPropInternal(vector<Tensor>& outputGradients)
    {
        m_Inputs[0]->Pool2DGradient(m_Outputs[0], *m_Inputs[0], outputGradients[0], m_FilterSize, m_Stride, m_Mode, m_Padding, m_InputGradients[0]);
    }

    //////////////////////////////////////////////////////////////////////////
    MaxPooling2D::MaxPooling2D(LayerBase* inputLayer, uint32_t filterSize, uint32_t stride, uint32_t padding, const string& name)
        : Pooling2D(__FUNCTION__, inputLayer, filterSize, stride, padding, EPoolingMode::Max, name)
    {
    }

    //////////////////////////////////////////////////////////////////////////
    MaxPooling2D::MaxPooling2D(uint32_t filterSize, uint32_t stride, uint32_t padding, const string& name)
        : Pooling2D(__FUNCTION__, filterSize, stride, padding, EPoolingMode::Max, name)
    {
    }

    //////////////////////////////////////////////////////////////////////////
    MaxPooling2D::MaxPooling2D(Shape inputShape, uint32_t filterSize, uint32_t stride, uint32_t padding, const string& name)
        : Pooling2D(__FUNCTION__, inputShape, filterSize, stride, padding, EPoolingMode::Max, name)
    {
    }

    //////////////////////////////////////////////////////////////////////////
    AvgPooling2D::AvgPooling2D(LayerBase* inputLayer, uint32_t filterSize, uint32_t stride, uint32_t padding, const string& name)
        : Pooling2D(__FUNCTION__, inputLayer, filterSize, stride, padding, EPoolingMode::Avg, name)
    {
    }

    //////////////////////////////////////////////////////////////////////////
    AvgPooling2D::AvgPooling2D(uint32_t filterSize, uint32_t stride, uint32_t padding, const string& name)
        : Pooling2D(__FUNCTION__, filterSize, stride, padding, EPoolingMode::Avg, name)
    {
    }

    //////////////////////////////////////////////////////////////////////////
    AvgPooling2D::AvgPooling2D(Shape inputShape, uint32_t filterSize, uint32_t stride, uint32_t padding, const string& name)
        : Pooling2D(__FUNCTION__, inputShape, filterSize, stride, padding, EPoolingMode::Avg, name)
    {
    }
}
