#include "Layers/Pooling2D.h"
#include "ComputationalGraph/Ops.h"
#include "Activations.h"

namespace Neuro
{
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
        : SingleLayer(constructorName, inputShape, nullptr, name)
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
    vector<TensorLike*> Pooling2D::InternalCall(const vector<TensorLike*>& inputNodes, TensorLike* training)
    {
        return { pool2d(inputNodes[0], m_FilterSize, m_Stride, m_Padding, m_Mode, m_DataFormat) };
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
