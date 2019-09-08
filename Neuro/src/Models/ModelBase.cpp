#include <iomanip>
#include <sstream>

#include "Models/ModelBase.h"
#include "Layers/LayerBase.h"

namespace Neuro
{
    //////////////////////////////////////////////////////////////////////////
    ModelBase::ModelBase(const string& constructorName, const string& name)
        : LayerBase(constructorName, Shape(), nullptr, name)
    {
        // output shape will be established when layers are added
    }

    //////////////////////////////////////////////////////////////////////////
    string ModelBase::Summary() const
    {
        stringstream ss;
        int totalParams = 0;
        ss << "_________________________________________________________________\n";
        ss << "Layer                        Output Shape              Param #   \n";
        ss << "=================================================================\n";

        for (auto layer : GetLayers())
        {
            totalParams += layer->GetParamsNum();
            ss << left << setw(29) << (layer->Name() + "(" + layer->ClassName() + ")");
            ss << setw(26) << layer->OutputShape().ToString();
            ss << setw(13) << layer->GetParamsNum() << "\n";
            if (layer->InputLayers().size() > 1)
            {
                for (int i = 0; i < (int)layer->InputLayers().size(); ++i)
                    ss << layer->InputLayers()[i]->Name() << "\n";
            }
            ss << "_________________________________________________________________\n";
        }

        ss << "Total params: " << totalParams << "\n";
        return ss.str();
    }

    //////////////////////////////////////////////////////////////////////////
    string ModelBase::TrainSummary() const
    {
        stringstream ss;
        ss.precision(2);
        ss << fixed;
        int totalParams = 0;
        ss << "_____________________________________________________________________________\n";
        ss << "Layer                        FeedFwd     BackProp    ActFeedFwd  ActBackProp \n";
        ss << "=============================================================================\n";

        for (auto layer : GetLayers())
        {
            ss << left << setw(29) << (layer->Name() + "(" + layer->ClassName() + ")");
            ss << setw(12) << (to_string(layer->FeedForwardTime() * 0.001f) + "s");
            ss << setw(12) << (to_string(layer->BackPropTime() * 0.001f) + "s");
            ss << setw(12) << (to_string(layer->ActivationTime() * 0.001f) + "s");
            ss << setw(12) << (to_string(layer->ActivationBackPropTime() * 0.001f) + "s") << "\n";
            ss << "_____________________________________________________________________________\n";
        }

        return ss.str();
    }

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
    uint32_t ModelBase::GetParamsNum() const
    {
        uint32_t paramsNum = 0;
        for (auto layer : GetLayers())
            paramsNum += layer->GetParamsNum();
        return paramsNum;
    }

    //////////////////////////////////////////////////////////////////////////
    void ModelBase::GetParametersAndGradients(vector<ParametersAndGradients>& paramsAndGrads)
    {
        for (auto layer : GetLayers())
            layer->GetParametersAndGradients(paramsAndGrads);
    }
}
