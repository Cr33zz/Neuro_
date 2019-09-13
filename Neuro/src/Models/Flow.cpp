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
        
        vector<LayerBase*> visited;
        for (auto inputLayer : m_ModelInputLayers)
            ProcessLayer(inputLayer, visited);

        m_ReversedOrder.resize(m_Order.size());
        reverse_copy(m_Order.begin(), m_Order.end(), m_ReversedOrder.begin());
	}

	//////////////////////////////////////////////////////////////////////////
    Flow::~Flow()
    {
        for (auto layer : m_Order)
            delete layer;
    }

    //////////////////////////////////////////////////////////////////////////
    LayerBase* Flow::GetCloneInstance() const
    {
        return new Flow();
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
	void Flow::FeedForwardInternal(bool training)
	{
		for (size_t i = 0; i < m_ModelInputLayers.size(); ++i)
			m_ModelInputLayers[i]->FeedForward(Inputs()[i], training);

		for (auto layer : m_Order)
		{
            auto& layerInputLayers = layer->InputLayers();

			// layers with no input layers have are already been fed forward
			if (layerInputLayers.size() == 0)
				continue;

			tensor_ptr_vec_t ins(layerInputLayers.size());
			for (size_t i = 0; i < layerInputLayers.size(); ++i)
				ins[i] = &(layerInputLayers[i]->Outputs()[0]);

			layer->FeedForward(ins, training);
		}
	}

	//////////////////////////////////////////////////////////////////////////
	void Flow::BackPropInternal(vector<Tensor>& outputsGradient)
	{
        for (uint32_t i = 0; i < (int)m_ModelOutputLayers.size(); ++i)
        {
            vector<Tensor> grads = { outputsGradient[i] };
            m_ModelOutputLayers[i]->BackProp(grads);
        }

		for (auto layer : m_ReversedOrder)
		{
            auto& layerOutputLayers = layer->OutputLayers();

			// layers with no input layers have are already been fed forward
			if (layerOutputLayers.size() == 0)
				continue;

			Tensor avgInputGradient(layer->OutputShapes()[0]);
			for (size_t i = 0; i < layerOutputLayers.size(); ++i)
			{
                auto& otherInputLayers = layerOutputLayers[i]->InputLayers();

				// we need to find this layer index in output layer's inputs to grab proper delta (it could be cached)
				for (size_t j = 0; j < otherInputLayers.size(); ++j)
				{
					if (otherInputLayers[j] == layer)
					{
						avgInputGradient.Add(layerOutputLayers[i]->InputsGradient()[j], avgInputGradient);
						break;
					}
				}
			}

            avgInputGradient.Div((float)layerOutputLayers.size(), avgInputGradient);
            vector<Tensor> avgGrads = { avgInputGradient };
            layer->BackProp(avgGrads);
		}
	}

	//////////////////////////////////////////////////////////////////////////
	const vector<LayerBase*>& Flow::ModelOutputLayers() const
	{
		return m_ModelOutputLayers;
	}

	//////////////////////////////////////////////////////////////////////////
	uint32_t Flow::OutputLayersCount() const
	{
		return (uint32_t)m_ModelOutputLayers.size();
	}

	//////////////////////////////////////////////////////////////////////////
	void Flow::OnClone(const LayerBase& source)
	{
        __super::OnClone(source);

        auto& sourceFlow = static_cast<const Flow&>(source);

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
			{
				auto inLayerClone = clones[inLayer->Name()];
				layerClone->InputLayers().push_back(inLayerClone);
				inLayerClone->OutputLayers().push_back(layerClone);
			}

			m_Order.push_back(layerClone);
		}

		m_ReversedOrder.resize(m_Order.size());
		reverse_copy(m_Order.begin(), m_Order.end(), m_ReversedOrder.begin());

        for (auto layer : sourceFlow.m_ModelInputLayers)
		{
			auto layerClone = clones[layer->Name()];
            m_ModelInputLayers.push_back(layerClone);
		}

        for (auto layer : sourceFlow.m_ModelOutputLayers)
		{
			auto layerClone = clones[layer->Name()];
            m_ModelOutputLayers.push_back(layerClone);
		}
	}

	//////////////////////////////////////////////////////////////////////////
	const vector<LayerBase*>& Flow::Layers() const
	{
		return m_Order;
	}
}
