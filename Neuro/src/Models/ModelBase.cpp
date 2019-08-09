#include "Models/ModelBase.h"
#include "Layers/LayerBase.h"

namespace Neuro
{
	//////////////////////////////////////////////////////////////////////////
	const LayerBase* ModelBase::GetLayer(const string& name) const
	{
		for (auto layer : GetLayers())
		{
			if (layer->Name() == name)
				return layer;
		}
		return nullptr;
	}

	//////////////////////////////////////////////////////////////////////////
	vector<ParametersAndGradients> ModelBase::GetParametersAndGradients()
	{
		vector<ParametersAndGradients> result;

		for(auto layer : GetLayers())
		{
			layer->GetParametersAndGradients(result);
		}

		return result;
	}

}
