#include "Layers/Dropout.h"
#include "Tools.h"

namespace Neuro
{
    //////////////////////////////////////////////////////////////////////////
    Dropout::Dropout(LayerBase* inputLayer, float p, const string& name)
        : LayerBase(__FUNCTION__, inputLayer, inputLayer->OutputShape(), nullptr, name), m_Prob(p)
    {
    }

    //////////////////////////////////////////////////////////////////////////
    Dropout::Dropout(const Shape& inputShape, float p, const string& name)
        : LayerBase(__FUNCTION__, inputShape, inputShape, nullptr, name), m_Prob(p)
    {
    }

    //////////////////////////////////////////////////////////////////////////
    Dropout::Dropout(float p, const string& name)
        : LayerBase(__FUNCTION__, Shape(), nullptr, name), m_Prob(p)
    {
    }

    //////////////////////////////////////////////////////////////////////////
    LayerBase* Dropout::GetCloneInstance() const
    {
        return new Dropout();
    }

    //////////////////////////////////////////////////////////////////////////
    void Dropout::OnLink(LayerBase* layer, bool input)
    {
        __super::OnLink(layer, input);

        if (input)
            m_OutputShapes[0] = m_InputShapes[0];
    }

    //////////////////////////////////////////////////////////////////////////
    void Dropout::FeedForwardInternal(bool training)
    {
        if (training)
        {
            if (m_Mask.GetShape() != m_Inputs[0]->GetShape())
                m_Mask = Tensor(m_Inputs[0]->GetShape());

            m_Inputs[0]->Dropout(m_Prob, m_Mask, m_Outputs[0]);
        }
        else
        {
            m_Inputs[0]->CopyTo(m_Outputs[0]);
        }
    }

    //////////////////////////////////////////////////////////////////////////
    void Dropout::BackPropInternal(vector<Tensor>& outputsGradient)
    {
        outputsGradient[0].DropoutGradient(outputsGradient[0], m_Mask, m_InputsGradient[0]);
    }
}
