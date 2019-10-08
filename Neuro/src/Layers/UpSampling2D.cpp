#include "Layers/UpSampling2D.h"
#include "ComputationalGraph/Ops.h"

namespace Neuro
{
    //////////////////////////////////////////////////////////////////////////
    UpSampling2D::UpSampling2D(uint32_t scaleFactor, const string& name)
        : SingleLayer(__FUNCTION__, Shape(), nullptr, name)
    {
        m_ScaleFactor = scaleFactor;
    }

    //////////////////////////////////////////////////////////////////////////
    UpSampling2D::UpSampling2D(const Shape& inputShape, uint32_t scaleFactor, const string& name)
        : SingleLayer(__FUNCTION__, inputShape, nullptr, name)
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
    vector<TensorLike*> UpSampling2D::InternalCall(const vector<TensorLike*>& inputNodes, TensorLike* training)
    {
        return { upsample2d(inputNodes[0], m_ScaleFactor) };
    }
}
