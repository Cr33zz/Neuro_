#include <algorithm>
#include <iomanip>
#include <sstream>

#include "Models/Flow.h"
#include "Layers/LayerBase.h"

namespace Neuro
{
	//////////////////////////////////////////////////////////////////////////
	Flow::Flow(const vector<LayerBase*>& inputLayers, const vector<LayerBase*>& outputLayers, const string& name, int seed)
        : ModelBase(__FUNCTION__, name, seed)
	{
        m_ModelInputLayers = inputLayers;
		m_ModelOutputLayers = outputLayers;

        for (auto modelOutputLayer : m_ModelOutputLayers)
            m_OutputsShapes.insert(m_OutputsShapes.end(), modelOutputLayer->OutputShapes().begin(), modelOutputLayer->OutputShapes().end());
        
        vector<LayerBase*> visited;
        for (auto modelInputLayer : m_ModelInputLayers)
            ProcessLayer(modelInputLayer, visited);

        m_ReversedOrder.resize(m_Order.size());
        reverse_copy(m_Order.begin(), m_Order.end(), m_ReversedOrder.begin());
	}

	//////////////////////////////////////////////////////////////////////////
    Flow::~Flow()
    {
        for (auto layer : m_Order)
            delete layer;
        for (auto inputGrad : m_InputsGradient)
            delete inputGrad;
    }

    //////////////////////////////////////////////////////////////////////////
    LayerBase* Flow::GetCloneInstance() const
    {
        return new Flow();
    }

    //////////////////////////////////////////////////////////////////////////
    LayerBase* Flow::LinkImpl(const vector<LayerBase*>& inputLayers)
    {
        OnLinkInput(inputLayers);
        for (size_t i = 0; i < m_ModelInputLayers.size(); ++i)
            inputLayers[i]->OnLinkOutput(m_ModelInputLayers[i]);
        return this;
    }

    //////////////////////////////////////////////////////////////////////////
	void Flow::ProcessLayer(LayerBase* layer, vector<LayerBase*>& visited)
	{
		bool allInputLayersVisited = true;

		for (auto inLayer : layer->InputLayers())
		{
			if (find(visited.begin(), visited.end(), inLayer) == visited.end())
			{
				allInputLayersVisited = false;
				break;
			}
		}

		if (!allInputLayersVisited)
			return;

		m_Order.push_back(layer);
		visited.push_back(layer);

		for (auto outLayer : layer->OutputLayers())
			ProcessLayer(outLayer, visited);
	}

	//////////////////////////////////////////////////////////////////////////
	void Flow::OnClone(const LayerBase& source)
	{
        __super::OnClone(source);

        auto& sourceFlow = static_cast<const Flow&>(source);

        m_OutputsShapes = sourceFlow.m_OutputsShapes;

		// clone is not a frequently used functionality so I'm not too concerned about its performance

		// make clones first and store then in dictionary
		map<string, LayerBase*> clones;
		for (auto layer : sourceFlow.m_Order)
		{
			auto clone = layer->Clone();
			clones[clone->Name()] = clone;
		}

		// then connect them in the same manner as in original network and clone order
		for (auto layer : sourceFlow.m_Order)
		{
			auto layerClone = clones[layer->Name()];
			for (auto inLayer : layer->InputLayers())
                layerClone->Link(clones[inLayer->Name()]);

			m_Order.push_back(layerClone);
		}

		m_ReversedOrder.resize(m_Order.size());
		reverse_copy(m_Order.begin(), m_Order.end(), m_ReversedOrder.begin());

        for (auto layer : sourceFlow.m_ModelInputLayers)
            m_ModelInputLayers.push_back(clones[layer->Name()]);

        for (auto layer : sourceFlow.m_ModelOutputLayers)
            m_ModelOutputLayers.push_back(clones[layer->Name()]);
	}

    //////////////////////////////////////////////////////////////////////////
    void Flow::OnInit(TensorLike* training, bool initValues)
    {
        __super::OnInit(training, initValues);

        for (auto modelOutputLayer : m_ModelOutputLayers)
            m_Outputs.insert(m_Outputs.end(), modelOutputLayer->Outputs().begin(), modelOutputLayer->Outputs().end());
    }

    //////////////////////////////////////////////////////////////////////////
    void Flow::OnLinkInput(const vector<LayerBase*>& inputLayers)
    {
        assert(m_ModelInputLayers.size() == inputLayers.size());
        for (size_t i = 0; i < m_ModelInputLayers.size(); ++i)
            m_ModelInputLayers[i]->OnLinkInput({ inputLayers[i] });
    }

    //////////////////////////////////////////////////////////////////////////
    void Flow::OnLinkOutput(LayerBase* outputLayer)
    {
        assert(m_ModelOutputLayers.size() == 1);
        m_ModelOutputLayers[0]->OnLinkOutput(outputLayer);
    }
}
