#include "Layers/UpSampling.h"

namespace Neuro
{
    //////////////////////////////////////////////////////////////////////////
    UpSampling::UpSampling(LayerBase* inputLayer, int filterSize, int stride, const string& name)
        : LayerBase(__FUNCTION__, inputLayer, GetOutShape(inputLayer->OutputShape(), filterSize, filterSize, stride), nullptr, name)
    {
        FilterSize = filterSize;
        Stride = stride;
    }

    //////////////////////////////////////////////////////////////////////////
    UpSampling::UpSampling(Shape inputShape, int filterSize, int stride, const string& name)
        : LayerBase(__FUNCTION__, inputShape, GetOutShape(inputShape, filterSize, filterSize, stride), nullptr, name)
    {
        FilterSize = filterSize;
        Stride = stride;
    }

    //////////////////////////////////////////////////////////////////////////
    UpSampling::UpSampling()
    {
    }

    //////////////////////////////////////////////////////////////////////////
    LayerBase* UpSampling::GetCloneInstance() const
    {
        return new UpSampling();
    }

    //////////////////////////////////////////////////////////////////////////
    void UpSampling::OnClone(const LayerBase& source)
    {
        __super::OnClone(source);

        auto sourceUpSampling = static_cast<const UpSampling&>(source);
        FilterSize = sourceUpSampling.FilterSize;
        Stride = sourceUpSampling.Stride;
    }

    //////////////////////////////////////////////////////////////////////////
    void UpSampling::FeedForwardInternal(bool training)
    {
        //m_Inputs[0]->Pool(FilterSize, Stride, Type, EPaddingMode::Valid, m_Output);
    }

    //////////////////////////////////////////////////////////////////////////
    void UpSampling::BackPropInternal(Tensor& outputGradient)
    {
        //m_Inputs[0]->PoolGradient(m_Output, *m_Inputs[0], outputGradient, FilterSize, Stride, Type, EPaddingMode::Valid, m_InputsGradient[0]);
    }

    //////////////////////////////////////////////////////////////////////////
    Shape UpSampling::GetOutShape(const Shape& inputShape, int filterWidth, int filterHeight, int stride)
    {
        return Shape((int)floor((float)(inputShape.Width() - filterWidth) / stride + 1),
                     (int)floor((float)(inputShape.Height() - filterHeight) / stride + 1),
                     inputShape.Depth());
    }
}
