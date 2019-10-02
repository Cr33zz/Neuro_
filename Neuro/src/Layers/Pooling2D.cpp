#include "Layers/Pooling2D.h"
#include "ComputationalGraph/Ops.h"

namespace Neuro
{
    //////////////////////////////////////////////////////////////////////////
    Pooling2D::Pooling2D(LayerBase* inputLayer, uint32_t filterSize, uint32_t stride, uint32_t padding, EPoolingMode mode, EDataFormat dataFormat, const string& name)
        : Pooling2D(__FUNCTION__, inputLayer, filterSize, stride, padding, mode, dataFormat, name)
    {
    }

    //////////////////////////////////////////////////////////////////////////
    Pooling2D::Pooling2D(uint32_t filterSize, uint32_t stride, uint32_t padding, EPoolingMode mode, EDataFormat dataFormat, const string& name)
        : Pooling2D(__FUNCTION__, filterSize, stride, padding, mode, dataFormat, name)
    {
    }

    //////////////////////////////////////////////////////////////////////////
    Pooling2D::Pooling2D(Shape inputShape, uint32_t filterSize, uint32_t stride, uint32_t padding, EPoolingMode mode, EDataFormat dataFormat, const string& name)
        : Pooling2D(__FUNCTION__, inputShape, filterSize, stride, padding, mode, dataFormat, name)
    {
    }

    //////////////////////////////////////////////////////////////////////////
    Pooling2D::Pooling2D(const string& constructorName, Shape inputShape, uint32_t filterSize, uint32_t stride, uint32_t padding, EPoolingMode mode, EDataFormat dataFormat, const string& name)
        : SingleLayer(constructorName, inputShape, Tensor::GetPooling2DOutputShape(inputShape, filterSize, filterSize, stride, padding, padding, dataFormat), nullptr, name)
    {
        m_Mode = mode;
        m_FilterSize = filterSize;
        m_Stride = stride;
        m_Padding = padding;
        m_DataFormat = dataFormat;
    }

    //////////////////////////////////////////////////////////////////////////
    Pooling2D::Pooling2D(const string& constructorName, LayerBase* inputLayer, uint32_t filterSize, uint32_t stride, uint32_t padding, EPoolingMode mode, EDataFormat dataFormat, const string& name)
        : SingleLayer(constructorName, inputLayer, Tensor::GetPooling2DOutputShape(inputLayer->OutputShape(), filterSize, filterSize, stride, padding, padding, dataFormat), nullptr, name)
    {
        m_Mode = mode;
        m_FilterSize = filterSize;
        m_Stride = stride;
        m_Padding = padding;
        m_DataFormat = dataFormat;
    }

    //////////////////////////////////////////////////////////////////////////
    Pooling2D::Pooling2D(const string& constructorName, uint32_t filterSize, uint32_t stride, uint32_t padding, EPoolingMode mode, EDataFormat dataFormat, const string& name)
        : SingleLayer(constructorName, Shape(), nullptr, name)
    {
        m_Mode = mode;
        m_FilterSize = filterSize;
        m_Stride = stride;
        m_Padding = padding;
        m_DataFormat = dataFormat;
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
    void Pooling2D::OnLinkInput(const vector<LayerBase*>& inputLayers)
    {
        __super::OnLinkInput(inputLayers);

        m_OutputsShapes[0] = Tensor::GetPooling2DOutputShape(inputLayers[0]->OutputShape(), m_FilterSize, m_FilterSize, m_Stride, m_Padding, m_Padding, m_DataFormat);
    }

    //////////////////////////////////////////////////////////////////////////
    void Pooling2D::InitOps(TensorLike* training, bool initValues)
    {
        m_OutputOps[0] = pool2d(m_InputOps[0], m_FilterSize, m_Stride, m_Padding, m_Mode, m_DataFormat);
    }

    //////////////////////////////////////////////////////////////////////////
    MaxPooling2D::MaxPooling2D(LayerBase* inputLayer, uint32_t filterSize, uint32_t stride, uint32_t padding, EDataFormat dataFormat, const string& name)
        : Pooling2D(__FUNCTION__, inputLayer, filterSize, stride, padding, EPoolingMode::Max, dataFormat, name)
    {
    }

    //////////////////////////////////////////////////////////////////////////
    MaxPooling2D::MaxPooling2D(uint32_t filterSize, uint32_t stride, uint32_t padding, EDataFormat dataFormat, const string& name)
        : Pooling2D(__FUNCTION__, filterSize, stride, padding, EPoolingMode::Max, dataFormat, name)
    {
    }

    //////////////////////////////////////////////////////////////////////////
    MaxPooling2D::MaxPooling2D(Shape inputShape, uint32_t filterSize, uint32_t stride, uint32_t padding, EDataFormat dataFormat, const string& name)
        : Pooling2D(__FUNCTION__, inputShape, filterSize, stride, padding, EPoolingMode::Max, dataFormat, name)
    {
    }

    //////////////////////////////////////////////////////////////////////////
    AvgPooling2D::AvgPooling2D(LayerBase* inputLayer, uint32_t filterSize, uint32_t stride, uint32_t padding, EDataFormat dataFormat, const string& name)
        : Pooling2D(__FUNCTION__, inputLayer, filterSize, stride, padding, EPoolingMode::Avg, dataFormat, name)
    {
    }

    //////////////////////////////////////////////////////////////////////////
    AvgPooling2D::AvgPooling2D(uint32_t filterSize, uint32_t stride, uint32_t padding, EDataFormat dataFormat, const string& name)
        : Pooling2D(__FUNCTION__, filterSize, stride, padding, EPoolingMode::Avg, dataFormat, name)
    {
    }

    //////////////////////////////////////////////////////////////////////////
    AvgPooling2D::AvgPooling2D(Shape inputShape, uint32_t filterSize, uint32_t stride, uint32_t padding, EDataFormat dataFormat, const string& name)
        : Pooling2D(__FUNCTION__, inputShape, filterSize, stride, padding, EPoolingMode::Avg, dataFormat, name)
    {
    }
}
