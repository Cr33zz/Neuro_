#pragma once

#include <string>
#include <vector>

#include "Types.h"
#include "Layers/LayerBase.h"
#include "ParametersAndGradients.h"

namespace Neuro
{
	using namespace std;

	class Tensor;

    class ModelBase : public LayerBase
    {
	public:
        virtual void Optimize() {}
        virtual const vector<LayerBase*>& GetLayers() const = 0;
        virtual const vector<LayerBase*>& GetOutputLayers() const = 0;
        virtual uint32_t GetOutputLayersCount() const = 0;

        string Summary() const;
        string TrainSummary() const;

        const LayerBase* GetLayer(const string& name) const;
        
        virtual void SaveStateXml(string filename) const { }
        virtual void LoadStateXml(string filename) { }
        
        virtual uint32_t GetParamsNum() const;
        virtual void GetParametersAndGradients(vector<ParametersAndGradients>& paramsAndGrads);

    protected:
        ModelBase() {}
        ModelBase(const string& constructorName, const string& name = "");
	};
}
