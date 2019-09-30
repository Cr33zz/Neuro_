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
    void Sequential::OnLinkInput(const vector<LayerBase*>& inputLayers)
    {
        assert(inputLayers.size() == 1);
        m_Layers.front()->OnLinkInput(inputLayers);
    }

    //////////////////////////////////////////////////////////////////////////
    void Sequential::OnLinkOutput(LayerBase* outputLayer)
    {
        m_Layers.back()->OnLinkOutput(outputLayer);
    }

	//////////////////////////////////////////////////////////////////////////
    const tensor_ptr_vec_t& Sequential::FeedForward(const const_tensor_ptr_vec_t& inputs, bool training)
	{
        Init();
		assert(inputs.size() == 1);

        m_FeedForwardTimer.Start();

		for (size_t i = 0; i < m_Layers.size(); ++i)
			m_Layers[i]->FeedForward(i == 0 ? inputs[0] : m_Layers[i - 1]->Output(), training);

        m_FeedForwardTimer.Stop();

        return m_Layers.back()->Outputs();
	}

	//////////////////////////////////////////////////////////////////////////
    const tensor_ptr_vec_t& Sequential::BackProp(const tensor_ptr_vec_t& outputsGradient)
	{
		assert(outputsGradient.size() == 1);

        m_BackPropTimer.Start();

        const tensor_ptr_vec_t* lastOutputsGradient = &outputsGradient;
		for (int i = (int)m_Layers.size() - 1; i >= 0; --i)
			lastOutputsGradient = &m_Layers[i]->BackProp(*lastOutputsGradient);

        m_BackPropTimer.Stop();

        return m_Layers[0]->InputsGradient();
	}

    //////////////////////////////////////////////////////////////////////////
    int Sequential::InputOffset(const LayerBase* inputLayer) const
    {
        return m_Layers[0]->InputOffset(inputLayer);
    }

    //////////////////////////////////////////////////////////////////////////
    LayerBase* Sequential::LinkImpl(const vector<LayerBase*>& inputLayers)
    {
        assert(inputLayers.size() == 1);
        OnLinkInput(inputLayers);
        inputLayers[0]->OnLinkOutput(m_ModelInputLayers[0]);
        return this;
    }

    //////////////////////////////////////////////////////////////////////////
	void Sequential::AddLayer(LayerBase* layer)
	{
        //assert(!m_Layers.empty() || layer->HasInputShape()); // first added layer must have input shape specified
        assert(!layer->InputLayer() || layer->InputLayer() == m_Layers.back()); // if layer being added has input layer it must be the last one in the sequence

        if (m_Layers.empty())
        {
            m_ModelInputLayers.resize(1);
            m_ModelInputLayers[0] = layer;
        }

        if (!m_Layers.empty() && !layer->InputLayer())
            layer->Link(m_Layers.back());
		
        m_ModelOutputLayers.resize(1);
        m_ModelOutputLayers[0] = layer;

		m_Layers.push_back(layer);
	}
}
