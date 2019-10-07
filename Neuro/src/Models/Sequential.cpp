#include <algorithm>
#include <iomanip>
#include <sstream>

#include "ComputationalGraph/Placeholder.h"
#include "Layers/LayerBase.h"
#include "Models/Sequential.h"
#include "Layers/Input.h"

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
	void Sequential::AddLayer(LayerBase* layer)
	{
        m_Built = false;
        if (m_Layers.empty())
        {
            LayerBase* firstLayer = layer;

            // we have to dig through layer containers until we get the very first one
            while (ModelBase* modelLayer = dynamic_cast<ModelBase*>(layer))
                firstLayer = modelLayer->Layer(0);
            
            m_InputNodes = firstLayer->m_InputNodes;
            
        }
        else
        {
            layer->Init(m_Layers.back()->OutputNodes(), m_TrainingPlaceholder);
        }
        
        m_Layers.push_back(layer);
	}
}
