#include "Layers/Dropout.h"
#include "Tools.h"
#include "ComputationalGraph/Ops.h"

namespace Neuro
{
    //////////////////////////////////////////////////////////////////////////
    Dropout::Dropout(LayerBase* inputLayer, float p, const string& name)
        : SingleLayer(__FUNCTION__, inputLayer, inputLayer->OutputShape(), nullptr, name), m_Prob(p)
    {
    }

    //////////////////////////////////////////////////////////////////////////
    Dropout::Dropout(const Shape& inputShape, float p, const string& name)
        : SingleLayer(__FUNCTION__, inputShape, inputShape, nullptr, name), m_Prob(p)
    {
    }

    //////////////////////////////////////////////////////////////////////////
    Dropout::Dropout(float p, const string& name)
        : SingleLayer(__FUNCTION__, Shape(), nullptr, name), m_Prob(p)
    {
    }

    //////////////////////////////////////////////////////////////////////////
    LayerBase* Dropout::GetCloneInstance() const
    {
        return new Dropout();
    }

    //////////////////////////////////////////////////////////////////////////
    void Dropout::OnLinkInput(const vector<LayerBase*>& inputLayers)
    {
        __super::OnLinkInput(inputLayers);

        m_OutputsShapes[0] = m_InputShape;
    }

    //////////////////////////////////////////////////////////////////////////
    void Dropout::InternalCall(TensorLike* training, bool initValues)
    {
        m_OutputNodes[0] = dropout(m_InputNodes[0], m_Prob, training);
    }
}
