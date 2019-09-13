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
        __super::OnLink(layer, input);

        if (input)
            m_Layers.front()->OnLink(layer, true);
        else
            m_Layers.back()->OnLink(layer, false);
    }

	//////////////////////////////////////////////////////////////////////////
	void Sequential::FeedForwardInternal(bool training)
	{
		assert(Inputs().size() == 1);

		for (size_t i = 0; i < m_Layers.size(); ++i)
			m_Layers[i]->FeedForward(i == 0 ? Inputs()[0] : &(m_Layers[i - 1]->Output()), training);
	}

	//////////////////////////////////////////////////////////////////////////
    void Sequential::BackPropInternal(vector<Tensor>& outputsGradient)
	{
		assert(outputsGradient.size() == 1);

        vector<Tensor>* lastOutputsGradient = &outputsGradient;
		for (int i = (int)m_Layers.size() - 1; i >= 0; --i)
			lastOutputsGradient = &m_Layers[i]->BackProp(*lastOutputsGradient);
	}

    //////////////////////////////////////////////////////////////////////////
	const vector<LayerBase*>& Sequential::ModelOutputLayers() const
	{
		return m_ModelOutputLayers;
	}

	//////////////////////////////////////////////////////////////////////////
    uint32_t Sequential::OutputLayersCount() const
	{
		return 1;
	}

	//////////////////////////////////////////////////////////////////////////
	const vector<LayerBase*>& Sequential::Layers() const
	{
		return m_Layers;
	}

	//////////////////////////////////////////////////////////////////////////
	LayerBase* Sequential::Layer(int i)
	{
		return m_Layers[i];
	}

	//////////////////////////////////////////////////////////////////////////
	LayerBase* Sequential::LastLayer() const
	{
		return m_Layers.back();
	}

	//////////////////////////////////////////////////////////////////////////
	int Sequential::LayersCount() const
	{
		return (int)m_Layers.size();
	}

	//////////////////////////////////////////////////////////////////////////
	void Sequential::AddLayer(LayerBase* layer)
	{
        assert(!m_Layers.empty() || layer->HasInputShape()); // first added layer must have input shape specified
        assert(!layer->InputLayer() || layer->InputLayer() == m_Layers.back()); // if layer being added has input layer it must be the last one in the sequence

        if (!m_Layers.empty() && !layer->InputLayer())
            layer->Link(m_Layers.back());
		
        m_ModelOutputLayers.resize(1);
        m_ModelOutputLayers[0] = layer;

		m_Layers.push_back(layer);
	}
}
