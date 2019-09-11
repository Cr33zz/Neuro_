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
		assert(m_Inputs.size() == 1);

		for (size_t i = 0; i < m_Layers.size(); ++i)
			m_Layers[i]->FeedForward(i == 0 ? m_Inputs[0] : &(m_Layers[i - 1]->Output()), training);

        m_Outputs[0] = m_OutputLayers[0]->Output();
	}

	//////////////////////////////////////////////////////////////////////////
    void Sequential::BackPropInternal(vector<Tensor>& outputsGradient)
	{
		assert(outputsGradient.size() == 1);

        vector<Tensor>* lastOutputsGradient = &outputsGradient;
		for (int i = (int)m_Layers.size() - 1; i >= 0; --i)
			lastOutputsGradient = &m_Layers[i]->BackProp(*lastOutputsGradient);

        m_InputsGradient[0] = m_Layers[0]->InputGradient();
	}

    //////////////////////////////////////////////////////////////////////////
	const vector<LayerBase*>& Sequential::OutputLayers() const
	{
		return m_OutputLayers;
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

        if (m_Layers.empty())
            m_InputShapes.push_back(layer->InputShape());

        if (!m_Layers.empty() && !layer->InputLayer())
            layer->Link(m_Layers.back());
		
        m_OutputLayers.resize(1);
		m_OutputLayers[0] = layer;
        m_OutputShapes[0] = layer->OutputShape();

		m_Layers.push_back(layer);
	}
}
