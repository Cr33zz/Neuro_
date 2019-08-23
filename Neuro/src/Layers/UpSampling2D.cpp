#include "Layers/UpSampling2D.h"

namespace Neuro
{
    //////////////////////////////////////////////////////////////////////////
    UpSampling2D::UpSampling2D(LayerBase* inputLayer, int scaleFactor, const string& name)
        : LayerBase(__FUNCTION__, inputLayer, Shape(inputLayer->OutputShape().Width() * scaleFactor, inputLayer->OutputShape().Height() * scaleFactor, inputLayer->OutputShape().Depth(), inputLayer->OutputShape().Batch()), nullptr, name)
    {
        m_ScaleFactor = scaleFactor;
    }

    //////////////////////////////////////////////////////////////////////////
    UpSampling2D::UpSampling2D(Shape inputShape, int scaleFactor, const string& name)
        : LayerBase(__FUNCTION__, inputShape, Shape(inputShape.Width() * scaleFactor, inputShape.Height() * scaleFactor, inputShape.Depth(), inputShape.Batch()), nullptr, name)
    {
        m_ScaleFactor = scaleFactor;
    }

    //////////////////////////////////////////////////////////////////////////
    UpSampling2D::UpSampling2D()
    {
    }

    //////////////////////////////////////////////////////////////////////////
    LayerBase* UpSampling2D::GetCloneInstance() const
    {
        return new UpSampling2D();
    }

    //////////////////////////////////////////////////////////////////////////
    void UpSampling2D::OnClone(const LayerBase& source)
    {
        __super::OnClone(source);

        auto sourceUpSampling = static_cast<const UpSampling2D&>(source);
        m_ScaleFactor = sourceUpSampling.m_ScaleFactor;
    }

    //////////////////////////////////////////////////////////////////////////
    void UpSampling2D::FeedForwardInternal(bool training)
    {
        m_Inputs[0]->UpSample2D(m_ScaleFactor, m_Output);
    }

    //////////////////////////////////////////////////////////////////////////
    void UpSampling2D::BackPropInternal(Tensor& outputGradient)
    {
        outputGradient.UpSample2DGradient(outputGradient, m_ScaleFactor, m_InputsGradient[0]);
    }
}
