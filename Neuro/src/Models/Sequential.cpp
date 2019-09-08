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
	void Sequential::FeedForwardInternal(bool training)
	{
		assert(m_Inputs.size() == 1);

		for (int l = 0; l < (int)m_Layers.size(); ++l)
			m_Layers[l]->FeedForward(l == 0 ? m_Inputs[0] : &(m_Layers[l - 1]->Output()), training);

        m_Outputs[0] = m_OutputLayers[0]->Output();
	}

	//////////////////////////////////////////////////////////////////////////
    void Sequential::BackPropInternal(vector<Tensor>& outputGradients)
	{
		assert(outputGradients.size() == 1);

        vector<Tensor>* gradients = &outputGradients;
		for (int l = (int)m_Layers.size() - 1; l >= 0; --l)
			gradients = &m_Layers[l]->BackProp(*gradients);
	}

    //////////////////////////////////////////////////////////////////////////
	const vector<LayerBase*>& Sequential::GetOutputLayers() const
	{
		return m_OutputLayers;
	}

	//////////////////////////////////////////////////////////////////////////
    uint32_t Sequential::GetOutputLayersCount() const
	{
		return 1;
	}

	//////////////////////////////////////////////////////////////////////////
	const std::vector<LayerBase*>& Sequential::GetLayers() const
	{
		return m_Layers;
	}

    //////////////////////////////////////////////////////////////////////////
	void Sequential::SaveStateXml(string filename) const
	{
		/*XmlDocument doc = new XmlDocument();
		XmlElement modelElem = doc.CreateElement("Sequential");

		for (int l = 0; l < Layers.Count; l++)
		{
			XmlElement layerElem = doc.CreateElement(Layers[l].GetType().Name);
			Layers[l].SerializeParameters(layerElem);
			modelElem.AppendChild(layerElem);
		}

		doc.AppendChild(modelElem);
		doc.Save(filename);*/
	}

	//////////////////////////////////////////////////////////////////////////
	void Sequential::LoadStateXml(string filename)
	{
		/*XmlDocument doc = new XmlDocument();
		doc.Load(filename);
		XmlElement modelElem = doc.FirstChild as XmlElement;

		for (int l = 0; l < Layers.Count; l++)
		{
			XmlElement layerElem = modelElem.ChildNodes.Item(l) as XmlElement;
			Layers[l].DeserializeParameters(layerElem);
		}*/
	}

	//////////////////////////////////////////////////////////////////////////
	LayerBase* Sequential::GetLayer(int i)
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
		
        m_OutputLayers.resize(1);
		m_OutputLayers[0] = layer;
        m_OutputShapes[0] = layer->OutputShape();

        if (!m_Layers.empty() && !layer->InputLayer())
            layer->Link(m_Layers.back());

		m_Layers.push_back(layer);
	}
}
