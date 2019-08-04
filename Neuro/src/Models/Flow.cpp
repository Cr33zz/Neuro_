#include <iomanip>
#include <sstream>

#include "Models/Flow.h"
#include "Layers/LayerBase.h"

namespace Neuro
{
	//////////////////////////////////////////////////////////////////////////
	Flow::Flow(const vector<LayerBase*>& inputLayers, const vector<LayerBase*>& outputLayers)
	{
		InputLayers = inputLayers;
		OutputLayers = outputLayers;
	}

	//////////////////////////////////////////////////////////////////////////
	Flow::Flow()
	{
	}

	//////////////////////////////////////////////////////////////////////////
	void Flow::ProcessLayer(LayerBase* layer, vector<LayerBase*>& visited)
	{
		bool allInputLayersVisited = true;

		for (auto inLayer : layer->InputLayers)
		{
			if (find(visited.begin(), visited.end(), inLayer) == visited.end())
			{
				allInputLayersVisited = false;
				break;
			}
		}

		if (!allInputLayersVisited)
			return;

		Order.push_back(layer);
		visited.push_back(layer);

		for (auto outLayer : layer->OutputLayers)
			ProcessLayer(outLayer, visited);
	}

	//////////////////////////////////////////////////////////////////////////
	void Flow::FeedForward(const tensor_ptr_vec_t& inputs)
	{
		for (int i = 0; i < (int)InputLayers.size(); ++i)
			InputLayers[i]->FeedForward(inputs[i]);

		for (auto layer : Order)
		{
			// layers with no input layers have are already been fed forward
			if (layer->InputLayers.size() == 0)
				continue;

			tensor_ptr_vec_t ins(layer->InputLayers.size());
			for (int i = 0; i < (int)layer->InputLayers.size(); ++i)
				ins[i] = &(layer->InputLayers[i]->Output);

			layer->FeedForward(ins);
		}
	}

	//////////////////////////////////////////////////////////////////////////
	void Flow::BackProp(vector<Tensor>& deltas)
	{
		for (int i = 0; i < (int)OutputLayers.size(); ++i)
			OutputLayers[i]->BackProp(deltas[i]);

		for (auto layer : ReversedOrder)
		{
			// layers with no input layers have are already been fed forward
			if (layer->OutputLayers.size() == 0)
				continue;

			Tensor avgDelta(layer->OutputShape);
			for (int i = 0; i < (int)layer->OutputLayers.size(); ++i)
			{
				// we need to find this layer index in output layer's inputs to grab proper delta (it could be cached)
				for (int j = 0; j < (int)layer->OutputLayers[i]->InputLayers.size(); ++j)
				{
					if (layer->OutputLayers[i]->InputLayers[j] == layer)
					{
						avgDelta.Add(layer->OutputLayers[i]->InputsGradient[j], avgDelta);
						break;
					}
				}
			}

			avgDelta.Div((float)layer->OutputLayers.size(), avgDelta);

			layer->BackProp(avgDelta);
		}
	}

	//////////////////////////////////////////////////////////////////////////
	tensor_ptr_vec_t Flow::GetOutputs() const
	{
		tensor_ptr_vec_t outputs(OutputLayers.size());
		for (int i = 0; i < (int)OutputLayers.size(); ++i)
			outputs[i] = &(OutputLayers[i]->Output);
		return outputs;
	}

	//////////////////////////////////////////////////////////////////////////
	const vector<LayerBase*>& Flow::GetOutputLayers() const
	{
		return OutputLayers;
	}

	//////////////////////////////////////////////////////////////////////////
	int Flow::GetOutputLayersCount() const
	{
		return (int)OutputLayers.size();
	}

	//////////////////////////////////////////////////////////////////////////
	void Flow::Optimize()
	{
		vector<LayerBase*> visited;

		for (auto inputLayer : InputLayers)
			ProcessLayer(inputLayer, visited);

		ReversedOrder.resize(Order.size());
		reverse_copy(Order.begin(), Order.end(), ReversedOrder.begin());
	}

	//////////////////////////////////////////////////////////////////////////
	ModelBase* Flow::Clone() const
	{
		// clone is not a frequently used functionality so I'm not too concerned about its performance

		// make clones first and store then in dictionary
		map<string, LayerBase*> clones;

		for (auto layer : Order)
		{
			auto clone = layer->Clone();
			clones[clone->Name] = clone;
		}

		// then connect them in the same manner as in original network and clone order
		auto flowClone = new Flow();

		for (auto layer : Order)
		{
			auto layerClone = clones[layer->Name];
			for (auto inLayer : layer->InputLayers)
			{
				auto inLayerClone = clones[inLayer->Name];
				layerClone->InputLayers.push_back(inLayerClone);
				inLayerClone->OutputLayers.push_back(layerClone);
			}

			flowClone->Order.push_back(layerClone);
		}

		flowClone->ReversedOrder.resize(flowClone->Order.size());
		reverse_copy(flowClone->Order.begin(), flowClone->Order.end(), flowClone->ReversedOrder.begin());

		for (auto layer : InputLayers)
		{
			auto layerClone = clones[layer->Name];
			flowClone->InputLayers.push_back(layerClone);
		}

		for (auto layer : OutputLayers)
		{
			auto layerClone = clones[layer->Name];
			flowClone->OutputLayers.push_back(layerClone);
		}

		return flowClone;
	}

	//////////////////////////////////////////////////////////////////////////
	const vector<LayerBase*>& Flow::GetLayers() const
	{
		return Order;
	}

	//////////////////////////////////////////////////////////////////////////
	string Flow::Summary() const
	{
		stringstream ss;
		int totalParams = 0;
		ss << "_________________________________________________________________\n";
		ss << "Layer                        Output Shape              Param #\n";
		ss << "=================================================================\n";

		for (auto layer : Order)
		{
			totalParams += layer->GetParamsNum();
			ss << setw(29) << layer->Name << " (" << typeid(layer).name() << ")" << setw(26) << "(" << layer->OutputShape.Width() << ", " << layer->OutputShape.Height() << ", " << layer->OutputShape.Depth() << ")" << setw(13) << layer->GetParamsNum() << "\n";
			for (int i = 1; i < (int)layer->InputLayers.size(); ++i)
				ss << setw(68 + layer->InputLayers[i]->Name.length()) << layer->InputLayers[i]->Name << "\n";
			ss << "_________________________________________________________________\n";
		}

		ss << "Total params: " << totalParams;
		return ss.str();
	}

}
