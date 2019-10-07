#include <algorithm>
#include <iomanip>
#include <sstream>
#include <unordered_set>

#include "ComputationalGraph/TensorLike.h"
#include "Models/Flow.h"
#include "Layers/LayerBase.h"

namespace Neuro
{
	//////////////////////////////////////////////////////////////////////////
	Flow::Flow(const vector<LayerBase*>& inputLayers, const vector<LayerBase*>& outputLayers, const string& name, int seed)
        : ModelBase(__FUNCTION__, name, seed)
	{
        unordered_set<LayerBase*> visited;
        for (auto layer : outputLayers)
            ProcessLayer(layer, visited);
	}

	//////////////////////////////////////////////////////////////////////////
    Flow::~Flow()
    {
    }

    //////////////////////////////////////////////////////////////////////////
    LayerBase* Flow::GetCloneInstance() const
    {
        return new Flow();
    }

    //////////////////////////////////////////////////////////////////////////
	void Flow::ProcessLayer(LayerBase* layer, unordered_set<LayerBase*>& visited)
	{
		bool allInputLayersVisited = true;

        auto& inputNodes = layer->InputNodes();

        for (size_t i = 0; i < inputNodes.size(); ++i)
		{
            assert(inputNodes[i]->m_Origin);
            ProcessLayer(inputNodes[i]->m_Origin->layer, visited);
		}

        if (visited.find(layer) != visited.end())
            return;

        visited.insert(layer);
        m_Layers.push_back(layer);
	}

	//////////////////////////////////////////////////////////////////////////
	void Flow::OnClone(const LayerBase& source)
	{
        __super::OnClone(source);

        auto& sourceFlow = static_cast<const Flow&>(source);

  //      m_OutputsShapes = sourceFlow.m_OutputsShapes;

		//// clone is not a frequently used functionality so I'm not too concerned about its performance

		//// make clones first and store then in dictionary
		//map<string, LayerBase*> clones;
		//for (auto layer : sourceFlow.m_Order)
		//{
		//	auto clone = layer->Clone();
		//	clones[clone->Name()] = clone;
		//}

		//// then connect them in the same manner as in original network and clone order
		//for (auto layer : sourceFlow.m_Order)
		//{
		//	auto layerClone = clones[layer->Name()];
		//	for (auto inLayer : layer->InputLayers())
  //              layerClone->Link(clones[inLayer->Name()]);

		//	m_Order.push_back(layerClone);
		//}

		//m_ReversedOrder.resize(m_Order.size());
		//reverse_copy(m_Order.begin(), m_Order.end(), m_ReversedOrder.begin());

  //      for (auto layer : sourceFlow.m_ModelInputLayers)
  //          m_ModelInputLayers.push_back(clones[layer->Name()]);

  //      for (auto layer : sourceFlow.m_ModelOutputLayers)
  //          m_ModelOutputLayers.push_back(clones[layer->Name()]);
	}
}
