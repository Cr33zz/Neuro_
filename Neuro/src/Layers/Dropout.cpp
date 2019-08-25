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
    Dropout::Dropout()
    {
    }

    //////////////////////////////////////////////////////////////////////////
    LayerBase* Dropout::GetCloneInstance() const
    {
        return new Dropout();
    }

    //////////////////////////////////////////////////////////////////////////
    void Dropout::FeedForwardInternal(bool training)
    {
        if (training)
        {
            if (m_Mask.GetShape() != m_Inputs[0]->GetShape())
                m_Mask = Tensor(m_Inputs[0]->GetShape());

            m_Inputs[0]->Dropout(m_Prob, m_Mask, m_Output);
        }
        else
        {
            m_Inputs[0]->CopyTo(m_Output);
        }
    }

    //////////////////////////////////////////////////////////////////////////
    void Dropout::BackPropInternal(Tensor& outputGradient)
    {
        outputGradient.DropoutGradient(outputGradient, m_Mask, m_InputsGradient[0]);
    }
}
