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
    const tensor_ptr_vec_t& Flow::FeedForward(const const_tensor_ptr_vec_t& inputs, bool training)
	{
        Init();

        if (m_InputsGradient.size() != inputs.size())
        {
            m_InputsGradient.resize(inputs.size());
            for (auto i = 0; i < inputs.size(); ++i)
                m_InputsGradient[i] = new Tensor(Name() + "/input_" + to_string(i) + "_grad");
        }

        for (size_t i = 0; i < inputs.size(); ++i)
            m_InputsGradient[i]->Resize(inputs[i]->GetShape());

        for (size_t i = 0; i < m_ModelInputLayers.size(); ++i)
            m_ModelInputLayers[i]->FeedForward(inputs[i], training);

		for (auto layer : m_Order)
		{
            auto& layerInputLayers = layer->InputLayers();

			// layers with no input layers have are already been fed forward
			if (layerInputLayers.size() == 0)
				continue;

            const_tensor_ptr_vec_t currInputs;
            for (size_t i = 0; i < layerInputLayers.size(); ++i)
            {
                auto& inputLayerOutputs = layerInputLayers[i]->Outputs();
                currInputs.insert(currInputs.end(), inputLayerOutputs.begin(), inputLayerOutputs.end());
            }

			layer->FeedForward(currInputs, training);
		}

        return m_Outputs;
	}

	//////////////////////////////////////////////////////////////////////////
    const tensor_ptr_vec_t& Flow::BackProp(const tensor_ptr_vec_t& outputsGradient)
	{
        size_t lastOutputGradIdx = 0;
        for (uint32_t i = 0; i < (int)m_ModelOutputLayers.size(); ++i)
        {
            size_t layerOutsNum = m_ModelOutputLayers[i]->Outputs().size();
            tensor_ptr_vec_t outGrad;
            outGrad.insert(outGrad.end(), outputsGradient.begin() + lastOutputGradIdx, outputsGradient.begin() + lastOutputGradIdx + layerOutsNum);
            m_ModelOutputLayers[i]->BackProp(outGrad);
            lastOutputGradIdx += layerOutsNum;
        }

		for (auto layer : m_ReversedOrder)
		{
            // output layers were already processed
			if (find(m_ModelOutputLayers.begin(), m_ModelOutputLayers.end(), layer) != m_ModelOutputLayers.end())
				continue;

            size_t layerOutsNum = layer->Outputs().size();
            auto& layerOutsShapes = layer->OutputShapes();

            // we need to average gradient for layers whose output was used by multiple layers. 
            // keep in mind that layer may have multiple outputs so we need to average gradient for each output separately
			vector<Tensor> tmpAvgOutputGradient(layerOutsNum);
            tensor_ptr_vec_t avgOutputGradient(layerOutsNum);
            for (size_t o = 0; o < tmpAvgOutputGradient.size(); ++o)
            {
                tmpAvgOutputGradient[o].Resize(layerOutsShapes[o]);
                tmpAvgOutputGradient[o].Zero();
                avgOutputGradient[o] = &tmpAvgOutputGradient[o];
            }

            auto& layerOutputLayers = layer->OutputLayers();

			for (size_t i = 0; i < layerOutputLayers.size(); ++i)
			{
                auto outputLayer = layerOutputLayers[i];
                auto outputLayerInputGrad = outputLayer->InputsGradient();
                int offset = outputLayer->InputOffset(layer);

                for (size_t o = 0; o < tmpAvgOutputGradient.size(); ++o)
                    tmpAvgOutputGradient[o].Add(*outputLayerInputGrad[offset + o], tmpAvgOutputGradient[o]);
			}

            // average
            for (size_t o = 0; o < tmpAvgOutputGradient.size(); ++o)
                tmpAvgOutputGradient[o].Div((float)layerOutputLayers.size(), tmpAvgOutputGradient[o]);

            layer->BackProp(avgOutputGradient);
		}

        // there is no point calculating inputs gradient if this layer in not linked to any previous layer;
        // additionally, unlinked model can have different input shapes for each internal input layer
        if (HasInputLayers())
        {
            auto& inputShape = InputShape();
            for (auto i = 0; i < m_InputsGradient.size(); ++i)
                m_InputsGradient[i]->Resize(Shape::From(inputShape, outputsGradient[0]->Batch()));

            for (auto modelInputLayer : m_ModelInputLayers)
            {
                for (auto i = 0; i < m_InputsGradient.size(); ++i)
                    m_InputsGradient[i]->Add(*modelInputLayer->InputsGradient()[i], *m_InputsGradient[i]);
            }

            for (auto i = 0; i < m_InputsGradient.size(); ++i)
                m_InputsGradient[i]->Div((float)m_ModelInputLayers.size(), *m_InputsGradient[i]);
        }

        return m_InputsGradient;
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
    void Flow::OnInit()
    {
        __super::OnInit();

        for (auto modelOutputLayer : m_ModelOutputLayers)
            m_Outputs.insert(m_Outputs.end(), modelOutputLayer->Outputs().begin(), modelOutputLayer->Outputs().end());
    }

    //////////////////////////////////////////////////////////////////////////
    int Flow::InputOffset(const LayerBase* inputLayer) const
    {
        int offset = 0;
        
        for (auto modelInputLayer : m_ModelInputLayers)
        {
            int localOffset = modelInputLayer->InputOffset(inputLayer);
            if (localOffset >= 0)
                return offset + localOffset;

            offset -= localOffset; // InputOffset will return total number of inputs as negative value
        }
        
        return -offset;
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
