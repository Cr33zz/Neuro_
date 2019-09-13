#include "Layers/UpSampling2D.h"

namespace Neuro
{
    //////////////////////////////////////////////////////////////////////////
    UpSampling2D::UpSampling2D(LayerBase* inputLayer, uint32_t scaleFactor, const string& name)
        : SingleLayer(__FUNCTION__, inputLayer, Shape(inputLayer->OutputShape().Width() * scaleFactor, inputLayer->OutputShape().Height() * scaleFactor, inputLayer->OutputShape().Depth(), inputLayer->OutputShape().Batch()), nullptr, name)
    {
        m_ScaleFactor = scaleFactor;
    }

    //////////////////////////////////////////////////////////////////////////
    UpSampling2D::UpSampling2D(uint32_t scaleFactor, const string& name)
        : SingleLayer(__FUNCTION__, Shape(), nullptr, name)
    {
        m_ScaleFactor = scaleFactor;
    }

    //////////////////////////////////////////////////////////////////////////
    UpSampling2D::UpSampling2D(Shape inputShape, uint32_t scaleFactor, const string& name)
        : SingleLayer(__FUNCTION__, inputShape, Shape(inputShape.Width() * scaleFactor, inputShape.Height() * scaleFactor, inputShape.Depth(), inputShape.Batch()), nullptr, name)
    {
        m_ScaleFactor = scaleFactor;
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
    void UpSampling2D::OnLink(LayerBase* layer, bool input)
    {
        __super::OnLink(layer, input);

        if (input)
            m_OutputShapes[0] = Shape(InputShape().Width() * m_ScaleFactor, InputShape().Height() * m_ScaleFactor, InputShape().Depth(), InputShape().Batch());
    }

    //////////////////////////////////////////////////////////////////////////
    void UpSampling2D::FeedForwardInternal(bool training)
    {
        m_Inputs[0]->UpSample2D(m_ScaleFactor, m_Outputs[0]);
    }

    //////////////////////////////////////////////////////////////////////////
    void UpSampling2D::BackPropInternal(vector<Tensor>& outputsGradient)
    {
        outputsGradient[0].UpSample2DGradient(outputsGradient[0], m_ScaleFactor, m_InputsGradient[0]);
    }
}
