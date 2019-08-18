#include "Layers/Pooling.h"

namespace Neuro
{
    //////////////////////////////////////////////////////////////////////////
    Pooling::Pooling(LayerBase* inputLayer, int filterSize, int stride, Tensor::EPoolType type, const string& name)
        : LayerBase(__FUNCTION__, inputLayer, GetOutShape(inputLayer->OutputShape(), filterSize, filterSize, stride), nullptr, name)
    {
        Type = type;
        FilterSize = filterSize;
        Stride = stride;
    }

    //////////////////////////////////////////////////////////////////////////
    Pooling::Pooling(Shape inputShape, int filterSize, int stride, Tensor::EPoolType type, const string& name)
        : LayerBase(__FUNCTION__, inputShape, GetOutShape(inputShape, filterSize, filterSize, stride), nullptr, name)
    {
        Type = type;
        FilterSize = filterSize;
        Stride = stride;
    }

    //////////////////////////////////////////////////////////////////////////
    Pooling::Pooling()
    {

    }

    //////////////////////////////////////////////////////////////////////////
    LayerBase* Pooling::GetCloneInstance() const
    {
        return new Pooling();
    }

    //////////////////////////////////////////////////////////////////////////
    void Pooling::OnClone(const LayerBase& source)
    {
        __super::OnClone(source);

        auto sourcePool = static_cast<const Pooling&>(source);
        Type = sourcePool.Type;
        FilterSize = sourcePool.FilterSize;
        Stride = sourcePool.Stride;
    }

    //////////////////////////////////////////////////////////////////////////
    void Pooling::FeedForwardInternal(bool training)
    {
        m_Inputs[0]->Pool(FilterSize, Stride, Type, Tensor::EPaddingType::Valid, m_Output);
    }

    //////////////////////////////////////////////////////////////////////////
    void Pooling::BackPropInternal(Tensor& outputGradient)
    {
        m_Inputs[0]->PoolGradient(m_Output, *m_Inputs[0], outputGradient, FilterSize, Stride, Type, Tensor::EPaddingType::Valid, m_InputsGradient[0]);
    }

    //////////////////////////////////////////////////////////////////////////
    Shape Pooling::GetOutShape(const Shape& inputShape, int filterWidth, int filterHeight, int stride)
    {
        return Shape((int)floor((float)(inputShape.Width() - filterWidth) / stride + 1),
            (int)floor((float)(inputShape.Height() - filterHeight) / stride + 1),
            inputShape.Depth());
    }

    //////////////////////////////////////////////////////////////////////////
    const char* Pooling::ClassName() const
    {
        return "Pooling";
    }
}
