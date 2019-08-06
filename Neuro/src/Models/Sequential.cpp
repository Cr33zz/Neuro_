#include <iomanip>
#include <sstream>

#include "Layers/LayerBase.h"
#include "Models/Sequential.h"

namespace Neuro
{
	//////////////////////////////////////////////////////////////////////////
	ModelBase* Sequential::Clone() const
	{
		Sequential* clone = new Sequential();
		for(auto layer : Layers)
			clone->Layers.push_back(layer->Clone());
		return clone;
	}

	//////////////////////////////////////////////////////////////////////////
	void Sequential::FeedForward(const tensor_ptr_vec_t& inputs)
	{
		//if (inputs.size() > 1) throw new Exception("Only single input is allowed for sequential model.");

		for (int l = 0; l < (int)Layers.size(); ++l)
			Layers[l]->FeedForward(l == 0 ? inputs[0] : &(Layers[l - 1]->Output));
	}

	//////////////////////////////////////////////////////////////////////////
	void Sequential::BackProp(vector<Tensor>& deltas)
	{
		//if (deltas.Length > 1) throw new Exception("Only single delta is allowed for sequential model.");

		Tensor& delta = deltas[0];
		for (int l = (int)Layers.size() - 1; l >= 0; --l)
			delta = Layers[l]->BackProp(delta)[0];
	}

	//////////////////////////////////////////////////////////////////////////
	tensor_ptr_vec_t Sequential::GetOutputs() const
	{
		return { &(GetLastLayer()->Output) };
	}

	//////////////////////////////////////////////////////////////////////////
	const std::vector<LayerBase*>& Sequential::GetOutputLayers() const
	{
		return OutputLayers;
	}

	//////////////////////////////////////////////////////////////////////////
	int Sequential::GetOutputLayersCount() const
	{
		return 1;
	}

	//////////////////////////////////////////////////////////////////////////
	const std::vector<LayerBase*>& Sequential::GetLayers() const
	{
		return Layers;
	}

	//////////////////////////////////////////////////////////////////////////
	std::string Sequential::Summary() const
	{
		stringstream ss;
		int totalParams = 0;
		ss << "_________________________________________________________________\n";
		ss << "Layer                        Output Shape              Param #\n";
		ss << "=================================================================\n";

		for (auto layer : Layers)
		{
			totalParams += layer->GetParamsNum();
			ss << left << setw(29) << (layer->Name + "(" + layer->ClassName() + ")") << setw(26) << "(" + to_string(layer->OutputShape.Width()) + ", " + to_string(layer->OutputShape.Height()) + ", " + to_string(layer->OutputShape.Depth()) + ")" << setw(13) << layer->GetParamsNum() << "\n";
			ss << "_________________________________________________________________\n";
		}

		ss << "Total params: " << totalParams;
		return ss.str();
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
		return Layers[i];
	}

	//////////////////////////////////////////////////////////////////////////
	LayerBase* Sequential::GetLastLayer() const
	{
		return Layers.back();
	}

	//////////////////////////////////////////////////////////////////////////
	int Sequential::LayersCount() const
	{
		return (int)Layers.size();
	}

	//////////////////////////////////////////////////////////////////////////
	void Sequential::AddLayer(LayerBase* layer)
	{
		OutputLayers.resize(1);
		OutputLayers[0] = layer;
		Layers.push_back(layer);
	}
}
