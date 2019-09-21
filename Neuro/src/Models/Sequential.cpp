#include <algorithm>
#include <iomanip>
#include <sstream>

#include "Layers/LayerBase.h"
#include "Models/Sequential.h"

namespace Neuro
{
	//////////////////////////////////////////////////////////////////////////
	Sequential::Sequential(const string& name, int seed)
        : ModelBase(__FUNCTION__, name, seed)
	{
	}

    //////////////////////////////////////////////////////////////////////////
    Sequential::~Sequential()
    {
        for (auto layer : m_Layers)
            delete layer;
    }

    //////////////////////////////////////////////////////////////////////////
    LayerBase* Sequential::GetCloneInstance() const
    {
        return new Sequential(0);
    }

    //////////////////////////////////////////////////////////////////////////
	void Sequential::OnClone(const LayerBase& source)
	{
        __super::OnClone(source);

        auto& sourceSequence = static_cast<const Sequential&>(source);
        for (auto layer : sourceSequence.m_Layers)
            m_Layers.push_back(layer->Clone());
	}

    //////////////////////////////////////////////////////////////////////////
    void Sequential::OnLink(LayerBase* layer, bool input)
    {
        if (input)
            m_Layers.front()->OnLink(layer, input);
        else
            m_Layers.back()->OnLink(layer, input);
    }

	//////////////////////////////////////////////////////////////////////////
    const tensor_ptr_vec_t& Sequential::FeedForward(const const_tensor_ptr_vec_t& inputs, bool training)
	{
        Init();
		assert(inputs.size() == 1);

		for (size_t i = 0; i < m_Layers.size(); ++i)
			m_Layers[i]->FeedForward(i == 0 ? inputs[0] : m_Layers[i - 1]->Output(), training);

        return m_Layers.back()->Outputs();
	}

	//////////////////////////////////////////////////////////////////////////
    const tensor_ptr_vec_t& Sequential::BackProp(const tensor_ptr_vec_t& outputsGradient)
	{
		assert(outputsGradient.size() == 1);

        const tensor_ptr_vec_t* lastOutputsGradient = &outputsGradient;
		for (int i = (int)m_Layers.size() - 1; i >= 0; --i)
			lastOutputsGradient = &m_Layers[i]->BackProp(*lastOutputsGradient);

        return m_Layers[0]->InputsGradient();
	}

    //////////////////////////////////////////////////////////////////////////
    int Sequential::InputOffset(const LayerBase* inputLayer) const
    {
        return m_Layers[0]->InputOffset(inputLayer);
    }

	//////////////////////////////////////////////////////////////////////////
	void Sequential::AddLayer(LayerBase* layer)
	{
        assert(!m_Layers.empty() || layer->HasInputShape()); // first added layer must have input shape specified
        assert(!layer->InputLayer() || layer->InputLayer() == m_Layers.back()); // if layer being added has input layer it must be the last one in the sequence

        if (m_Layers.empty())
        {
            m_ModelInputLayers.resize(1);
            m_ModelInputLayers[0] = layer;
        }

        if (!m_Layers.empty() && !layer->InputLayer())
            layer->LinkInput(m_Layers.back());
		
        m_ModelOutputLayers.resize(1);
        m_ModelOutputLayers[0] = layer;

		m_Layers.push_back(layer);
	}
}
