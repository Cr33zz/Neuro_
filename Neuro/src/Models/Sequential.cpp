#include <iomanip>
#include <sstream>

#include "Layers/LayerBase.h"
#include "Models/Sequential.h"

namespace Neuro
{

	//////////////////////////////////////////////////////////////////////////
	Sequential::Sequential()
	{
	}

    //////////////////////////////////////////////////////////////////////////
    Sequential::~Sequential()
    {
        for (auto layer : m_Layers)
            delete layer;
    }

    //////////////////////////////////////////////////////////////////////////
	ModelBase* Sequential::Clone() const
	{
		Sequential* clone = new Sequential();
		for(auto layer : m_Layers)
			clone->m_Layers.push_back(layer->Clone());
		return clone;
	}

	//////////////////////////////////////////////////////////////////////////
	void Sequential::FeedForward(const tensor_ptr_vec_t& inputs, bool training)
	{
		//if (inputs.size() > 1) throw new Exception("Only single input is allowed for sequential model.");

		for (int l = 0; l < (int)m_Layers.size(); ++l)
			m_Layers[l]->FeedForward(l == 0 ? inputs[0] : &(m_Layers[l - 1]->Output()), training);
	}

	//////////////////////////////////////////////////////////////////////////
	void Sequential::BackProp(vector<Tensor>& deltas)
	{
		//if (deltas.Length > 1) throw new Exception("Only single delta is allowed for sequential model.");

		Tensor* delta = &deltas[0];
		for (int l = (int)m_Layers.size() - 1; l >= 0; --l)
			delta = &m_Layers[l]->BackProp(*delta)[0];
	}

	//////////////////////////////////////////////////////////////////////////
	tensor_ptr_vec_t Sequential::GetOutputs() const
	{
		return { &(LastLayer()->Output()) };
	}

	//////////////////////////////////////////////////////////////////////////
	const std::vector<LayerBase*>& Sequential::GetOutputLayers() const
	{
		return m_OutputLayers;
	}

	//////////////////////////////////////////////////////////////////////////
	int Sequential::GetOutputLayersCount() const
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
		m_OutputLayers.resize(1);
		m_OutputLayers[0] = layer;
		m_Layers.push_back(layer);
	}
}
