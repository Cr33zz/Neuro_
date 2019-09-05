#include <algorithm>
#include <iomanip>
#include <sstream>

#include "Models/Flow.h"
#include "Layers/LayerBase.h"

namespace Neuro
{
	//////////////////////////////////////////////////////////////////////////
	Flow::Flow(const vector<LayerBase*>& inputLayers, const vector<LayerBase*>& outputLayers)
	{
		m_InputLayers = inputLayers;
		m_OutputLayers = outputLayers;

        vector<LayerBase*> visited;
        for (auto inputLayer : m_InputLayers)
            ProcessLayer(inputLayer, visited);

        m_ReversedOrder.resize(m_Order.size());
        reverse_copy(m_Order.begin(), m_Order.end(), m_ReversedOrder.begin());
	}

	//////////////////////////////////////////////////////////////////////////
	Flow::Flow()
	{
	}

    //////////////////////////////////////////////////////////////////////////
    Flow::~Flow()
    {
        for (auto layer : m_Order)
            delete layer;
    }

    //////////////////////////////////////////////////////////////////////////
	void Flow::ProcessLayer(LayerBase* layer, vector<LayerBase*>& visited)
	{
		bool allInputLayersVisited = true;

		for (auto inLayer : layer->m_InputLayers)
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

		for (auto outLayer : layer->m_OutputLayers)
			ProcessLayer(outLayer, visited);
	}

	//////////////////////////////////////////////////////////////////////////
	void Flow::FeedForward(const tensor_ptr_vec_t& inputs, bool training)
	{
		for (uint i = 0; i < (int)m_InputLayers.size(); ++i)
			m_InputLayers[i]->FeedForward(inputs[i], training);

		for (auto layer : m_Order)
		{
			// layers with no input layers have are already been fed forward
			if (layer->m_InputLayers.size() == 0)
				continue;

			tensor_ptr_vec_t ins(layer->m_InputLayers.size());
			for (uint i = 0; i < (int)layer->m_InputLayers.size(); ++i)
				ins[i] = &(layer->m_InputLayers[i]->m_Output);

			layer->FeedForward(ins, training);
		}
	}

	//////////////////////////////////////////////////////////////////////////
	void Flow::BackProp(vector<Tensor>& deltas)
	{
		for (uint i = 0; i < (int)m_OutputLayers.size(); ++i)
			m_OutputLayers[i]->BackProp(deltas[i]);

		for (auto layer : m_ReversedOrder)
		{
			// layers with no input layers have are already been fed forward
			if (layer->m_OutputLayers.size() == 0)
				continue;

			Tensor avgDelta(layer->m_OutputShape);
			for (uint i = 0; i < (int)layer->m_OutputLayers.size(); ++i)
			{
				// we need to find this layer index in output layer's inputs to grab proper delta (it could be cached)
				for (int j = 0; j < (int)layer->m_OutputLayers[i]->m_InputLayers.size(); ++j)
				{
					if (layer->m_OutputLayers[i]->m_InputLayers[j] == layer)
					{
						avgDelta.Add(layer->m_OutputLayers[i]->m_InputsGradient[j], avgDelta);
						break;
					}
				}
			}

			avgDelta.Div((float)layer->m_OutputLayers.size(), avgDelta);

			layer->BackProp(avgDelta);
		}
	}

	//////////////////////////////////////////////////////////////////////////
	tensor_ptr_vec_t Flow::GetOutputs() const
	{
		tensor_ptr_vec_t outputs(m_OutputLayers.size());
		for (uint i = 0; i < (int)m_OutputLayers.size(); ++i)
			outputs[i] = &(m_OutputLayers[i]->m_Output);
		return outputs;
	}

	//////////////////////////////////////////////////////////////////////////
	const vector<LayerBase*>& Flow::GetOutputLayers() const
	{
		return m_OutputLayers;
	}

	//////////////////////////////////////////////////////////////////////////
	int Flow::GetOutputLayersCount() const
	{
		return (int)m_OutputLayers.size();
	}

	//////////////////////////////////////////////////////////////////////////
	ModelBase* Flow::Clone() const
	{
		// clone is not a frequently used functionality so I'm not too concerned about its performance

		// make clones first and store then in dictionary
		map<string, LayerBase*> clones;

		for (auto layer : m_Order)
		{
			auto clone = layer->Clone();
			clones[clone->Name()] = clone;
		}

		// then connect them in the same manner as in original network and clone order
		auto flowClone = new Flow();

		for (auto layer : m_Order)
		{
			auto layerClone = clones[layer->Name()];
			for (auto inLayer : layer->InputLayers())
			{
				auto inLayerClone = clones[inLayer->Name()];
				layerClone->m_InputLayers.push_back(inLayerClone);
				inLayerClone->m_OutputLayers.push_back(layerClone);
			}

			flowClone->m_Order.push_back(layerClone);
		}

		flowClone->m_ReversedOrder.resize(flowClone->m_Order.size());
		reverse_copy(flowClone->m_Order.begin(), flowClone->m_Order.end(), flowClone->m_ReversedOrder.begin());

		for (auto layer : m_InputLayers)
		{
			auto layerClone = clones[layer->Name()];
			flowClone->m_InputLayers.push_back(layerClone);
		}

		for (auto layer : m_OutputLayers)
		{
			auto layerClone = clones[layer->Name()];
			flowClone->m_OutputLayers.push_back(layerClone);
		}

		return flowClone;
	}

	//////////////////////////////////////////////////////////////////////////
	const vector<LayerBase*>& Flow::GetLayers() const
	{
		return m_Order;
	}
}
