#include <algorithm>
#include <iomanip>
#include <sstream>

#include "Models/Sequential.h"
#include "Layers/Input.h"
#include "ComputationalGraph/Placeholder.h"

namespace Neuro
{
	//////////////////////////////////////////////////////////////////////////
	Sequential::Sequential(const string& name, int seed)
        : Flow(__FUNCTION__, name, seed)
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
        return nullptr; // new Sequential(0);
    }

    //////////////////////////////////////////////////////////////////////////
	void Sequential::OnClone(const LayerBase& source)
	{
        __super::OnClone(source);

        auto& sourceSequence = static_cast<const Sequential&>(source);
        /*for (auto layer : sourceSequence.m_Layers)
            m_Layers.push_back(layer->Clone());*/
	}

    //////////////////////////////////////////////////////////////////////////
	void Sequential::AddLayer(LayerBase* layer)
	{
        m_Built = false;
        if (m_Layers.empty())
        {
            bool setInputs = false;

            if (dynamic_cast<Input*>(layer))
            {
                NEURO_ASSERT(layer->m_InboundNodes.back()->output_tensors.size() == 1, "");
                setInputs = true;
            }
            else
            {
                LayerBase* firstLayer = layer;

                // we have to dig through layer containers until we get the very first one
                while (ModelBase* modelLayer = dynamic_cast<ModelBase*>(layer))
                {
                    NEURO_ASSERT(!modelLayer->Layers().empty(), "Cannot add an empty model to a `Sequential` model.");
                    firstLayer = modelLayer->Layer(0);
                }

                if (!dynamic_cast<Input*>(layer) && layer->ExpectedInputShape().IsValid())
                {
                    auto inputLayer = new Input(layer->ExpectedInputShape());
                    layer->Call(inputLayer->Outputs());
                    setInputs = true;
                }
            }
            
            if (setInputs)
            {
                NEURO_ASSERT(layer->m_InboundNodes.back()->output_tensors.size() == 1, "All layers in a Sequential model should have a single output tensor. For multi-output layers, use the Flow.");
                m_Outputs = layer->m_InboundNodes.back()->output_tensors;
                m_Inputs = GetSourceInputs(m_Outputs[0]);
            }
        }
        else if (!m_Outputs.empty())
        {
            auto outputs = layer->Call(m_Outputs, m_TrainingPlaceholder);

            NEURO_ASSERT(outputs.size() == 1, "All layers in a Sequential model should have a single output tensor. For multi-output layers, use Flow.");
            m_Outputs = outputs;
        }
        
        if (!m_Inputs.empty())
            Build({});
        else
            m_Layers.push_back(layer);
	}

    void Sequential::Build(const vector<Shape>& inputShapes)
    {
        if (!inputShapes.empty() && m_Inputs.empty())
        {
            auto inputLayer = new Input(inputShapes[0]);
            auto x = inputLayer->Outputs();
            m_Inputs = x;

            for (auto layer : m_Layers)
                x = layer->Call(x);

            m_Outputs = x;
        }

        if (!m_Inputs.empty())
        {
            InitGraph(m_Inputs, m_Outputs);
            m_Built = true;
        }
    }
}
