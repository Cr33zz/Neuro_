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
